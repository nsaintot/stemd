//! The HTTP routes for sessions. See [`crate::sessions`] and [`crate::cache`].
//!
//! ```text
//! POST   /v2/sessions                          open: track length and output;
//!                                              body is optional PCM from sample 0
//! PUT    /v2/sessions/{id}/pcm?offset=N        PCM placed at track sample N
//! PUT    /v2/sessions/{id}/window              playhead and margins, JSON
//! GET    /v2/sessions/{id}                     what has arrived, what is ready;
//!                                              ?wait= with If-None-Match long-polls
//! GET    /v2/sessions/{id}/stems/{name}        ?from=&to= in track samples, or the whole track
//! GET    /v2/sessions/{id}/pieces/{name}/{k}   piece k, encoded, as stored
//! DELETE /v2/sessions/{id}                     end the session
//! ```

use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, Query, State};
use axum::http::{self, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use stemd_core::{Audio, Normalisation, StemFormat};

use crate::api::error::ApiError;
use crate::api::params::{CreateParams, default_channels, default_rate, parse_output};
use crate::api::stem::{audio_response, send_whole};
use crate::cache::{Cache, TrackInfo};
use crate::output::{Output, output_span};
use crate::sessions::{
    Claim, JoinError, OpenError, Request, Session, Sessions, TrackSpec, UploadError, WindowRequest,
};
use crate::state::AppState;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v2/sessions", post(open))
        .route("/v2/sessions/{id}", get(view).delete(end))
        .route("/v2/sessions/{id}/pcm", put(upload))
        .route("/v2/sessions/{id}/window", put(window))
        .route("/v2/sessions/{id}/stems/{name}", get(stem))
        .route("/v2/sessions/{id}/pieces/{name}/{k}", get(piece))
}

#[derive(Debug, Deserialize)]
struct OpenParams {
    /// The track length in samples, at `sample_rate`. Required unless `track` is set.
    #[serde(default)]
    frames: Option<usize>,
    /// A track id. An open track is joined; for a stored track, the session opens on its pieces.
    #[serde(default)]
    track: Option<String>,
    /// The client's key for the file the audio came from; see `GET /v2/separations?file=`.
    #[serde(default)]
    file: Option<String>,
    /// Must be the model's rate; see `sample_rate` in `/v2/health`.
    #[serde(default = "default_rate")]
    sample_rate: u32,
    #[serde(default = "default_channels")]
    channels: usize,
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    output_format: Option<String>,
    #[serde(default)]
    output_sample_rate: Option<String>,
    #[serde(default)]
    include_derived: Option<bool>,
    #[serde(default)]
    parts: Option<String>,
    #[serde(default)]
    dsp_mode: Option<String>,
    #[serde(default)]
    dsp_filter: Option<String>,
    /// The mean of the mono mixdown over the whole track.
    #[serde(default)]
    mean: Option<f32>,
    /// The standard deviation of the mono mixdown over the whole track.
    #[serde(default)]
    std: Option<f32>,
    /// The largest absolute sample of the mix.
    #[serde(default)]
    peak: Option<f32>,
}

impl OpenParams {
    fn output_params(&self) -> CreateParams {
        CreateParams {
            sample_rate: self.sample_rate,
            channels: self.channels,
            format: self.format.clone(),
            output_format: self.output_format.clone(),
            output_sample_rate: self.output_sample_rate.clone(),
            include_derived: self.include_derived,
            parts: self.parts.clone(),
            dsp_mode: self.dsp_mode.clone(),
            dsp_filter: self.dsp_filter.clone(),
            file: None,
        }
    }
}

fn session(state: &AppState, id: &str) -> Result<Arc<Session>, ApiError> {
    let session = state
        .queue
        .sessions()
        .get(id)
        .ok_or_else(ApiError::not_found)?;
    session.touch();
    Ok(session)
}

fn view_of(state: &AppState, session: &Session) -> Response {
    let view = state.queue.view_of(session);
    ([(http::header::ETAG, etag(view.rev))], Json(view)).into_response()
}

/// The `ETag` of a view at `rev`.
fn etag(rev: u64) -> String {
    format!("\"{rev:016x}\"")
}

/// The longest a long poll waits.
const MAX_WAIT: Duration = Duration::from_secs(25);

/// Decode a PCM body and place it at `offset`, off the async runtime.
async fn place(
    state: &Arc<AppState>,
    session: &Arc<Session>,
    offset: usize,
    body: Bytes,
) -> Result<(), ApiError> {
    let session = Arc::clone(session);
    let state = Arc::clone(state);
    tokio::task::spawn_blocking(move || -> Result<(), ApiError> {
        let spec = session.track().spec;
        let audio =
            Audio::from_interleaved(&body, session.in_format, spec.channels, spec.sample_rate)
                .map_err(|e| ApiError::bad_request(format!("{e}")))?;
        state
            .queue
            .sessions()
            .upload(&session, offset, &audio)
            .map_err(|e| match e {
                UploadError::Invalid(_) => ApiError::bad_request(format!("{e}")),
                UploadError::Differs(..) => ApiError::new(StatusCode::CONFLICT, format!("{e}")),
                UploadError::Closed(_) => ApiError::new(StatusCode::GONE, format!("{e}")),
            })
    })
    .await
    .map_err(|e| ApiError::internal(format!("placing the upload: {e}")))?
}

async fn open(
    State(state): State<Arc<AppState>>,
    Query(params): Query<OpenParams>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let (in_format, output, model_rate) = parse_output(&params.output_params(), &state)?;
    let info = state.queue.info();
    check_shape(&params, &output, model_rate, &info)?;
    if let Some(key) = &params.file {
        crate::api::stem::check_file_key(key)?;
    }
    let norm = parse_levels(&params)?;
    let (frames, stored) = match &params.track {
        Some(id)
            if state
                .queue
                .sessions()
                .open_track(id, output.parts)
                .is_some() =>
        {
            return join(&state, &params, id, output, in_format, body).await;
        }
        Some(id) => {
            let (info, claim) = stored_claim(&state, &params, &output, id).await?;
            (info.frames, Some((info, claim)))
        }
        None => match params.frames.filter(|&f| f > 0) {
            Some(frames) => (frames, None),
            None => return Err(ApiError::bad_request("frames must be at least 1")),
        },
    };
    let secs = frames as f64 / f64::from(model_rate);
    let max_secs = state.max_track_secs();
    if secs > max_secs {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "track is {:.1} minutes; this server accepts up to {:.0}",
                secs / 60.0,
                max_secs / 60.0
            ),
        ));
    }
    let grid = session_grid(&state, &info, frames)?;
    let spec = TrackSpec {
        frames,
        channels: params.channels,
        sample_rate: model_rate,
        grid,
        normalises: state.queue.normalises(),
        parts: output.parts,
    };
    let claim = match stored {
        Some((info, claim)) if info == TrackInfo::from(&spec) => Some(claim),
        Some((_, claim)) => return Err(unknown_track(&claim.id)),
        None => None,
    };
    let claimed = claim.as_ref().map(|c| c.id.clone());

    let session = state
        .queue
        .sessions()
        .open(Request {
            output: output.clone(),
            in_format,
            track: spec,
            norm,
            peak: params.peak,
            cache_id: state.switcher.cache_id(),
            claim,
        })
        .map_err(open_error)?;
    session.set_file(params.file.clone());
    tracing::info!(
        session = %session.id,
        "opened for {secs:.1}s of audio, {} segments of {} samples",
        grid.len(),
        grid.segment(),
    );
    if let Some(id) = claimed {
        tracing::info!(session = %session.id, "opened on stored pieces of {id}");
    }

    if !body.is_empty()
        && let Err(err) = place(&state, &session, 0, body).await
    {
        state.queue.sessions().remove(&session.id, "upload refused");
        return Err(err);
    }
    state.queue.wake();
    Ok((StatusCode::CREATED, view_of(&state, &session)).into_response())
}

/// A 400 unless the output cuts into pieces and the upload has the model's rate and channels.
fn check_shape(
    params: &OpenParams,
    output: &Output,
    model_rate: u32,
    info: &stemd_core::BackendInfo,
) -> Result<(), ApiError> {
    if output.format == StemFormat::Mp3 {
        return Err(ApiError::bad_request(
            "mp3 does not cut into pieces; ask for flac, wav, s16le or f32le",
        ));
    }
    if params.sample_rate != model_rate {
        return Err(ApiError::bad_request(format!(
            "sample_rate: expected {model_rate} Hz, got {}",
            params.sample_rate
        )));
    }
    if params.channels != info.channels {
        return Err(ApiError::bad_request(format!(
            "{}: expected {} channels, got {}",
            info.model, info.channels, params.channels
        )));
    }
    Ok(())
}

/// The segment grid of a `frames`-sample track, or a 409 when segments cannot run out of order.
fn session_grid(
    state: &AppState,
    info: &stemd_core::BackendInfo,
    frames: usize,
) -> Result<stemd_core::Grid, ApiError> {
    let Some(segmenting) = state.queue.segmenting() else {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            format!("{} separates whole tracks only", info.model),
        ));
    };
    let grid = segmenting
        .grid(frames)
        .map_err(|e| ApiError::bad_request(format!("{e:#}")))?;
    if !grid.order_free() {
        return Err(ApiError::new(
            StatusCode::CONFLICT,
            "this server's --overlap is above 0.5, so segments cannot run out of order",
        ));
    }
    Ok(grid)
}

/// The highest `peak` a session accepts. A larger peak gives a gain that stores near-silence.
const MAX_PEAK: f32 = 16.0;

/// The client's normalisation from `mean` and `std`, after checking those and `peak`.
fn parse_levels(params: &OpenParams) -> Result<Option<Normalisation>, ApiError> {
    let norm = match (params.mean, params.std) {
        (Some(mean), Some(std)) if mean.is_finite() && std.is_finite() && std >= 0.0 => {
            Some(Normalisation::new(mean, std))
        }
        (None, None) => None,
        _ => {
            return Err(ApiError::bad_request(
                "mean and std must be sent together, as finite numbers, with std at least 0",
            ));
        }
    };
    if params.peak.is_some_and(|p| !(0.0..=MAX_PEAK).contains(&p)) {
        return Err(ApiError::bad_request(format!(
            "peak must be a number from 0 to {MAX_PEAK}"
        )));
    }
    Ok(norm)
}

/// The error for a session that the server has no room for.
fn open_error(e: OpenError) -> ApiError {
    let status = match e {
        OpenError::TooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
        OpenError::TooMany(_) | OpenError::Full { .. } => StatusCode::TOO_MANY_REQUESTS,
    };
    ApiError::new(status, format!("{e}"))
}

/// The 404 for a `track=` id that the server neither has open nor holds pieces of.
fn unknown_track(id: &str) -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        format!("no open or stored track {id}"),
    )
    .with_code("unknown_separation")
}

/// The stored grid of track `id` for `output`, and the claim that a session on its pieces opens
/// with.
async fn stored_claim(
    state: &Arc<AppState>,
    params: &OpenParams,
    output: &Output,
    id: &str,
) -> Result<(TrackInfo, Claim), ApiError> {
    if crate::api::stem::check_id(id).is_err() {
        return Err(unknown_track(id));
    }
    let (cache, cache_id) = (Arc::clone(&state.cache), state.switcher.cache_id());
    let (output, claimed, gain) = (output.clone(), id.to_owned(), output.gain(params.peak));
    let found = tokio::task::spawn_blocking(move || {
        crate::held::claimable(&cache, &cache_id, &output, &claimed, gain)
    })
    .await
    .map_err(|e| ApiError::internal(format!("looking up the stored pieces: {e}")))?
    .ok_or_else(|| unknown_track(id))?;
    if params.frames.is_some_and(|f| f != found.info.frames) {
        return Err(ApiError::bad_request(format!(
            "track {id} is {} samples long",
            found.info.frames
        )));
    }
    let claim = Claim {
        id: id.to_owned(),
        gain: found.gain,
    };
    Ok((found.info, claim))
}

/// Open a session on the open track `id`, which needs no upload.
async fn join(
    state: &Arc<AppState>,
    params: &OpenParams,
    id: &str,
    output: crate::output::Output,
    in_format: stemd_core::PcmFormat,
    body: Bytes,
) -> Result<Response, ApiError> {
    if !body.is_empty() {
        return Err(ApiError::bad_request(
            "a session joining a track takes no upload",
        ));
    }
    let sessions = state.queue.sessions();
    let track = sessions
        .open_track(id, output.parts)
        .ok_or_else(|| unknown_track(id))?;
    if params.frames.is_some_and(|f| f != track.spec.frames) {
        return Err(ApiError::bad_request(format!(
            "track {id} is {} samples long",
            track.spec.frames
        )));
    }
    let peak = params.peak;
    let joined = {
        let sessions = Arc::clone(sessions);
        tokio::task::spawn_blocking(move || sessions.join(output, in_format, peak, track))
            .await
            .map_err(|e| ApiError::internal(format!("join failed: {e}")))?
    };
    let session = joined.map_err(|e| match e {
        JoinError::TooMany(e) => ApiError::new(StatusCode::TOO_MANY_REQUESTS, format!("{e}")),
        JoinError::Gone => unknown_track(id),
    })?;
    tracing::info!(session = %session.id, "joined track {id}");
    state.queue.wake();
    Ok((StatusCode::CREATED, view_of(state, &session)).into_response())
}

#[derive(Debug, Deserialize)]
struct UploadParams {
    offset: usize,
}

async fn upload(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(params): Query<UploadParams>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let session = session(&state, &id)?;
    if body.is_empty() {
        return Err(ApiError::bad_request("empty payload"));
    }
    place(&state, &session, params.offset, body).await?;
    state.queue.wake();
    Ok(view_of(&state, &session))
}

async fn window(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    body: Result<Json<WindowRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let session = session(&state, &id)?;
    let Json(request) = body.map_err(|e| ApiError::bad_request(e.body_text()))?;
    request
        .check()
        .map_err(|e| ApiError::bad_request(format!("{e:#}")))?;
    state
        .queue
        .sessions()
        .set_window(&session, &request)
        .map_err(|e| ApiError::new(StatusCode::GONE, format!("{e:#}")))?;
    state.queue.wake();
    Ok(view_of(&state, &session))
}

#[derive(Debug, Deserialize)]
struct ViewParams {
    /// Seconds to hold the answer while the view matches `If-None-Match`.
    #[serde(default)]
    wait: Option<u64>,
}

async fn view(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(params): Query<ViewParams>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let session = session(&state, &id)?;
    let known = headers
        .get(http::header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    // Wait at most half the idle timeout, so a client that only polls is not reaped during a poll.
    let idle = state.queue.sessions().idle_timeout();
    let wait = Duration::from_secs(params.wait.unwrap_or(0))
        .min(MAX_WAIT)
        .min(idle / 2);
    let deadline = tokio::time::Instant::now() + wait;
    let mut changes = state.queue.sessions().subscribe();
    loop {
        changes.borrow_and_update();
        let view = state.queue.view_of(&session);
        let tag = etag(view.rev);
        if known.as_deref() != Some(tag.as_str()) {
            session.touch();
            return Ok(([(http::header::ETAG, tag)], Json(view)).into_response());
        }
        if tokio::time::timeout_at(deadline, changes.changed())
            .await
            .is_err()
        {
            session.touch();
            return Ok((StatusCode::NOT_MODIFIED, [(http::header::ETAG, tag)]).into_response());
        }
    }
}

async fn end(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    if !state.queue.sessions().remove(&id, "ended by the client") {
        return Err(ApiError::not_found());
    }
    // Wake the worker so it drops what the track held on the device.
    state.queue.wake();
    tracing::info!(session = %id, "ended by the client");
    Ok(StatusCode::NO_CONTENT.into_response())
}

/// A range of track samples; both left out means the whole track.
#[derive(Debug, Deserialize)]
struct RangeParams {
    #[serde(default)]
    from: Option<usize>,
    #[serde(default)]
    to: Option<usize>,
}

/// Encode `from..to` of `name` from memory or stored pieces; a whole-track read stores the file.
pub(crate) fn encode_range(
    sessions: &Sessions,
    session: &Session,
    name: &str,
    from: usize,
    to: usize,
) -> Result<Response, ApiError> {
    let in_memory = from >= to || session.track().ready_covers(&(from..to));
    let (excerpt, gain) = if in_memory {
        (session.read(name, from, to), session.gain)
    } else {
        (sessions.read_stored(session, name, from, to), 1.0)
    };
    let excerpt = excerpt.map_err(|e| ApiError::new(StatusCode::CONFLICT, format!("{e:#}")))?;
    let format = session.output.format;
    let (bytes, clipped) = crate::whole::encode(&excerpt.audio, format, gain, !in_memory)
        .map_err(|e| ApiError::internal(format!("encoding {name}: {e:#}")))?;
    if from == 0
        && to == session.track().spec.frames
        && let Err(err) =
            sessions.consolidate(session, name, &bytes, excerpt.audio.frames(), clipped)
    {
        tracing::warn!(session = %session.id, "storing {name} whole: {err:#}");
    }
    let len = bytes.len() as u64;
    Ok(audio_response(
        format,
        Body::from(bytes),
        len,
        excerpt.from,
        excerpt.audio.frames(),
        session.gain,
        clipped,
    ))
}

async fn stem(
    State(state): State<Arc<AppState>>,
    Path((id, name)): Path<(String, String)>,
    Query(range): Query<RangeParams>,
) -> Result<Response, ApiError> {
    let session = session(&state, &id)?;
    let frames = session.track().spec.frames;
    let (from, to) = match (range.from, range.to) {
        (None, None) => (0, frames),
        (Some(from), Some(to)) => (from, to),
        _ => {
            return Err(ApiError::bad_request(
                "send both from and to, or neither for the whole track",
            ));
        }
    };
    if from == 0
        && to == frames
        && let Some(whole) = state.queue.sessions().whole(&session, &name)
    {
        return send_whole(session.output.format, session.gain, whole).await;
    }
    tokio::task::spawn_blocking(move || {
        encode_range(state.queue.sessions(), &session, &name, from, to)
    })
    .await
    .map_err(|e| ApiError::internal(format!("reading the range: {e}")))?
}

async fn piece(
    State(state): State<Arc<AppState>>,
    Path((id, name, k)): Path<(String, String, usize)>,
) -> Result<Response, ApiError> {
    let session = session(&state, &id)?;
    if !session.output.served().contains(&name.as_str()) {
        return Err(ApiError::not_found());
    }
    // The stem is stored whole, so the client can fetch it in one read.
    let whole = state
        .queue
        .sessions()
        .whole(&session, &name)
        .map(|_| format!("/v2/sessions/{id}/stems/{name}"));
    let mut response = piece_of(state, session, name, k).await?;
    if let Some(value) = whole.and_then(|url| url.parse().ok()) {
        response.headers_mut().insert("x-stemd-whole", value);
    }
    Ok(response)
}

async fn piece_of(
    state: Arc<AppState>,
    session: Arc<Session>,
    name: String,
    k: usize,
) -> Result<Response, ApiError> {
    let track = session.track();
    if k >= track.spec.grid.len() {
        return Err(ApiError::not_found());
    }
    let samples = track.piece(k);
    let (model, out) = (
        u64::from(track.spec.sample_rate),
        u64::from(session.output.rate.hz()),
    );
    let span = output_span(&samples, model, out);
    let (from, frames) = (span.start, span.len());

    // Send a stored piece as it is; otherwise encode the piece from what is ready in memory.
    let stored = state
        .queue
        .sessions()
        .track_dir(&session)
        .and_then(|dir| Cache::lookup(&dir, &name, k, session.output.format.extension()));
    // A piece whose stem was stored whole after the lookup is cut from the whole file below.
    let bytes = match stored {
        Some(path) => match tokio::fs::read(&path).await {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                return Err(ApiError::internal(format!(
                    "reading {}: {e}",
                    path.display()
                )));
            }
        },
        None => None,
    };
    if let Some(bytes) = bytes {
        let len = bytes.len() as u64;
        return Ok(audio_response(
            session.output.format,
            Body::from(bytes),
            len,
            from,
            frames,
            session.gain,
            None,
        ));
    }
    tokio::task::spawn_blocking(move || {
        encode_range(
            state.queue.sessions(),
            &session,
            &name,
            samples.start,
            samples.end,
        )
    })
    .await
    .map_err(|e| ApiError::internal(format!("encoding the piece: {e}")))?
}
