//! HTTP surface for sessions. See [`crate::sessions`] and [`crate::cache`].
//!
//! ```text
//! POST   /v2/sessions                          open: track length and output;
//!                                              body is optional PCM from sample 0
//! PUT    /v2/sessions/{id}/pcm?offset=N        PCM placed at track sample N
//! PUT    /v2/sessions/{id}/window              needle and margins, JSON
//! GET    /v2/sessions/{id}                     what has arrived, what is ready;
//!                                              ?wait= with If-None-Match long-polls
//! GET    /v2/sessions/{id}/stems/{name}        ?from=&to= in track samples, or the whole track
//! GET    /v2/sessions/{id}/pieces/{name}/{k}   piece k, encoded, as stored
//! DELETE /v2/sessions/{id}                     end it
//! ```

use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use stemd_core::{Audio, Normalisation, StemFormat};

use crate::api::{ApiError, AppState, CreateParams, default_channels, default_rate, parse_output};
use crate::cache::Cache;
use crate::sessions::{
    JoinError, OpenError, Request, Session, Sessions, TrackSpec, UploadError, WindowRequest,
    output_index, served,
};
use crate::whole::Whole;

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
    /// Track length in samples, at `sample_rate`. Required unless `track` is set.
    #[serde(default)]
    frames: Option<usize>,
    /// The id of an open track to join instead of uploading.
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
    /// Mean of the mono mixdown over the whole track.
    #[serde(default)]
    mean: Option<f32>,
    /// Standard deviation of the same.
    #[serde(default)]
    std: Option<f32>,
    /// Largest absolute sample of the mix.
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
    ([(header::ETAG, etag(view.rev))], Json(view)).into_response()
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
    if output.format == StemFormat::Mp3 {
        return Err(ApiError::bad_request(
            "mp3 does not cut into pieces; ask for flac, wav, s16le or f32le",
        ));
    }
    if params.sample_rate != model_rate {
        return Err(ApiError::bad_request(format!(
            "a session takes the model's rate, {model_rate} Hz, not {}",
            params.sample_rate
        )));
    }
    let info = state.queue.info();
    if params.channels != info.channels {
        return Err(ApiError::bad_request(format!(
            "{} takes {} channels, not {}",
            info.model, info.channels, params.channels
        )));
    }
    if let Some(key) = &params.file {
        crate::separations::check_file_key(key)?;
    }
    if let Some(id) = &params.track {
        return join(&state, &params, id, output, in_format, body).await;
    }
    let Some(frames) = params.frames.filter(|&f| f > 0) else {
        return Err(ApiError::bad_request("frames must be at least 1"));
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
    if params.peak.is_some_and(|p| !p.is_finite() || p < 0.0) {
        return Err(ApiError::bad_request(
            "peak must be a finite number, at least 0",
        ));
    }

    let session = state
        .queue
        .sessions()
        .open(Request {
            output: output.clone(),
            in_format,
            track: TrackSpec {
                frames,
                channels: params.channels,
                sample_rate: model_rate,
                grid,
                normalises: state.queue.normalises(),
                parts: output.parts,
            },
            norm,
            peak: params.peak,
            model_id: state.switcher.cache_id(),
        })
        .map_err(|e| {
            let status = match e {
                OpenError::TooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
                OpenError::TooMany(_) | OpenError::Full { .. } => StatusCode::TOO_MANY_REQUESTS,
            };
            ApiError::new(status, format!("{e}"))
        })?;
    session.set_file(params.file.clone());
    tracing::info!(
        session = %session.id,
        "opened for {secs:.1}s of audio, {} segments of {} samples",
        grid.len(),
        grid.segment(),
    );

    if !body.is_empty()
        && let Err(err) = place(&state, &session, 0, body).await
    {
        state.queue.sessions().remove(&session.id, "upload refused");
        return Err(err);
    }
    state.queue.wake();
    Ok((StatusCode::CREATED, view_of(&state, &session)).into_response())
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
    let track = sessions.open_track(id, output.parts).ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            format!("no open track {id}; open a session with frames and upload it"),
        )
        .with_code("unknown_separation")
    })?;
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
        JoinError::Gone => ApiError::new(
            StatusCode::NOT_FOUND,
            format!("no open track {id}; open a session with frames and upload it"),
        )
        .with_code("unknown_separation"),
    })?;
    session.set_file(params.file.clone());
    sessions.record_file(&session, id);
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
        .get(header::IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    // Half the idle timeout at most, so a client that only polls is not reaped mid-poll.
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
            return Ok(([(header::ETAG, tag)], Json(view)).into_response());
        }
        if tokio::time::timeout_at(deadline, changes.changed())
            .await
            .is_err()
        {
            session.touch();
            return Ok((StatusCode::NOT_MODIFIED, [(header::ETAG, tag)]).into_response());
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
    // Lets the worker drop what the track held on the device.
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

/// Headers every range and piece carries.
pub(crate) fn audio_response(
    format: StemFormat,
    body: Body,
    len: u64,
    from: usize,
    frames: usize,
    gain: f32,
    clipped: Option<usize>,
) -> Response {
    let mut response = Response::builder()
        .header(header::CONTENT_TYPE, format.content_type())
        .header(header::CONTENT_LENGTH, len)
        .header("x-stemd-from", from)
        .header("x-stemd-frames", frames)
        .header("x-stemd-gain", gain.to_string());
    if let Some(clipped) = clipped {
        response = response.header("x-stemd-clipped", clipped);
    }
    response
        .body(body)
        .expect("response builder inputs are valid")
}

/// Open a stored whole file and give its length; one deleted since its lookup is `410`.
pub(crate) async fn open_whole(whole: &Whole) -> Result<(tokio::fs::File, u64), ApiError> {
    let path = whole.path.display();
    let file = tokio::fs::File::open(&whole.path).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            crate::separations::evicted()
        } else {
            ApiError::internal(format!("opening {path}: {e}"))
        }
    })?;
    let len = file
        .metadata()
        .await
        .map_err(|e| ApiError::internal(format!("reading {path}: {e}")))?
        .len();
    Ok((file, len))
}

/// Send a stored whole file as it is.
pub(crate) async fn send_whole(
    format: StemFormat,
    gain: f32,
    whole: Whole,
) -> Result<Response, ApiError> {
    let (file, len) = open_whole(&whole).await?;
    Ok(audio_response(
        format,
        Body::from_stream(tokio_util::io::ReaderStream::new(file)),
        len,
        0,
        whole.meta.frames,
        gain,
        whole.meta.clipped,
    ))
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
    let bytes = stemd_core::encode_stem(&excerpt.audio, format, gain)
        .map_err(|e| ApiError::internal(format!("encoding {name}: {e:#}")))?;
    let clipped =
        (!format.has_headroom()).then(|| crate::whole::clipped(&excerpt.audio, gain, !in_memory));
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
    if !served(&session.output).contains(&name.as_str()) {
        return Err(ApiError::not_found());
    }
    // A stem stored whole: the client can fetch it in one read.
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
    let from = output_index(samples.start, model, out);
    let frames = output_index(samples.end, model, out) - from;

    // Stored: sent as it is. Otherwise encoded from what is ready in memory.
    let stored = state
        .queue
        .sessions()
        .track_dir(&session)
        .and_then(|dir| Cache::lookup(&dir, &name, k, session.output.format.extension()));
    // A piece stored whole since the lookup is cut from the whole file below.
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

#[cfg(test)]
mod tests {
    use axum::http::{Method, StatusCode};
    use axum::response::IntoResponse;

    use super::open_whole;
    use crate::testkit::{call, state};
    use crate::whole::{Meta, Whole};

    #[tokio::test]
    async fn a_whole_file_deleted_since_its_lookup_is_410() {
        let whole = Whole {
            path: std::env::temp_dir().join("stemd-no-such-stem.flac"),
            meta: Meta {
                frames: 1,
                clipped: None,
                header: 0,
                block: 0,
                offsets: Vec::new(),
            },
        };
        let err = open_whole(&whole).await.expect_err("no such file");
        assert_eq!(err.into_response().status(), StatusCode::GONE);
    }

    #[tokio::test]
    async fn an_open_past_the_memory_budget_is_413_alone_and_429_beside_others() {
        let state = state();
        let app = crate::api::router(std::sync::Arc::clone(&state));
        let uri = "/v2/sessions?frames=30000";
        let sessions = state.queue.sessions();
        let first = call(&app, Method::POST, uri, Vec::new()).await;
        assert_eq!(first.status, StatusCode::CREATED, "{}", first.json);
        let id = first.json["id"].as_str().unwrap();
        let one = sessions.get(id).unwrap().track().spec.memory();

        sessions.set_memory_budget(2 * one - 1);
        let beside = call(&app, Method::POST, uri, Vec::new()).await;
        assert_eq!(
            beside.status,
            StatusCode::TOO_MANY_REQUESTS,
            "{}",
            beside.json
        );
        sessions.set_memory_budget(one - 1);
        let alone = call(&app, Method::POST, uri, Vec::new()).await;
        assert_eq!(
            alone.status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "{}",
            alone.json
        );
    }

    #[tokio::test]
    async fn a_malformed_window_is_a_400_with_a_json_error() {
        let app = crate::api::router(state());
        let opened = call(&app, Method::POST, "/v2/sessions?frames=30000", Vec::new()).await;
        assert_eq!(opened.status, StatusCode::CREATED, "{}", opened.json);
        let uri = format!(
            "/v2/sessions/{}/window",
            opened.json["id"].as_str().unwrap()
        );
        for body in [
            &b"not json"[..],
            br#"{"regions": [{"at": "soon"}]}"#,
            br#"{}"#,
        ] {
            let answer = call(&app, Method::PUT, &uri, body.to_vec()).await;
            assert_eq!(
                answer.status,
                StatusCode::BAD_REQUEST,
                "{:?}",
                String::from_utf8_lossy(body)
            );
            assert!(answer.json["error"].is_string(), "{}", answer.json);
        }
        let fine = call(&app, Method::PUT, &uri, br#"{"regions": []}"#.to_vec()).await;
        assert_eq!(fine.status, StatusCode::OK);
    }
}
