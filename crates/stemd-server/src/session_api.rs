//! HTTP surface for sessions. See [`crate::sessions`] and [`crate::pieces`].
//!
//! ```text
//! POST   /v1/sessions                          open: track length and output;
//!                                              body is optional PCM from sample 0
//! PUT    /v1/sessions/{id}/pcm?offset=N        PCM placed at track sample N
//! PUT    /v1/sessions/{id}/window              needle and margins, JSON
//! GET    /v1/sessions/{id}                     what has arrived, what is ready
//! GET    /v1/sessions/{id}/stems/{name}        ?from=&to= in track samples
//! GET    /v1/sessions/{id}/pieces/{name}/{k}   piece k, encoded, as stored
//! DELETE /v1/sessions/{id}                     end it
//! ```

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde::Deserialize;
use stemd_core::{Audio, Normalisation, StemFormat};

use crate::api::{ApiError, AppState, CreateParams, default_channels, default_rate, parse_output};
use crate::pieces::type_dir;
use crate::sessions::{
    Request, Session, TooMany, TrackSpec, UploadError, WindowRequest, output_index,
};

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/sessions", post(open))
        .route("/v1/sessions/{id}", get(view).delete(end))
        .route("/v1/sessions/{id}/pcm", put(upload))
        .route("/v1/sessions/{id}/window", put(window))
        .route("/v1/sessions/{id}/stems/{name}", get(stem))
        .route("/v1/sessions/{id}/pieces/{name}/{k}", get(piece))
}

#[derive(Debug, Deserialize)]
struct OpenParams {
    /// Track length in samples, at `sample_rate`.
    frames: usize,
    /// Must be the model's rate; see `sample_rate` in `/v1/health`.
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
    let pieces = state.queue.sessions().cached_for(session);
    Json(session.view(state.queue.sessions().idle_timeout(), pieces)).into_response()
}

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
    if params.frames == 0 {
        return Err(ApiError::bad_request("frames must be at least 1"));
    }
    let secs = params.frames as f64 / f64::from(model_rate);
    if secs > state.max_track_secs {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "track is {:.1} minutes; this server accepts up to {:.0}",
                secs / 60.0,
                state.max_track_secs / 60.0
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
        .grid(params.frames)
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
            output,
            in_format,
            track: TrackSpec {
                frames: params.frames,
                channels: params.channels,
                sample_rate: model_rate,
                grid,
                normalises: state.queue.normalises(),
                parts: output.parts,
            },
            norm,
            peak: params.peak,
            model_id: state.switcher.published_identity(),
        })
        .map_err(|e: TooMany| ApiError::new(StatusCode::TOO_MANY_REQUESTS, format!("{e}")))?;
    tracing::info!(
        session = %session.id,
        "opened for {secs:.1}s of audio, {} segments of {} samples",
        grid.len(),
        grid.segment(),
    );

    if !body.is_empty()
        && let Err(err) = place(&state, &session, 0, body).await
    {
        state.queue.sessions().remove(&session.id);
        return Err(err);
    }
    state.queue.wake();
    Ok((StatusCode::CREATED, view_of(&state, &session)).into_response())
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
    Json(request): Json<WindowRequest>,
) -> Result<Response, ApiError> {
    let session = session(&state, &id)?;
    session
        .set_window(&request)
        .map_err(|e| ApiError::new(StatusCode::GONE, format!("{e:#}")))?;
    state.queue.wake();
    Ok(view_of(&state, &session))
}

async fn view(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let session = session(&state, &id)?;
    Ok(view_of(&state, &session))
}

async fn end(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    if !state.queue.sessions().remove(&id) {
        return Err(ApiError::not_found());
    }
    // Lets the worker drop what the track held on the device.
    state.queue.wake();
    tracing::info!(session = %id, "ended by the client");
    Ok(StatusCode::NO_CONTENT.into_response())
}

#[derive(Debug, Deserialize)]
struct RangeParams {
    from: usize,
    to: usize,
}

/// Headers every range and piece carries.
fn audio_response(
    format: StemFormat,
    bytes: Vec<u8>,
    from: usize,
    frames: usize,
    gain: f32,
    clipped: Option<usize>,
) -> Response {
    let mut response = Response::builder()
        .header(header::CONTENT_TYPE, format.content_type())
        .header(header::CONTENT_LENGTH, bytes.len())
        .header("x-stemd-from", from)
        .header("x-stemd-frames", frames)
        .header("x-stemd-gain", gain.to_string());
    if let Some(clipped) = clipped {
        response = response.header("x-stemd-clipped", clipped);
    }
    response
        .body(Body::from(bytes))
        .expect("response builder inputs are valid")
}

/// Read `from..to` of `name` and encode it.
fn encode_range(
    session: &Session,
    name: &str,
    from: usize,
    to: usize,
) -> Result<Response, ApiError> {
    let excerpt = session
        .read(name, from, to)
        .map_err(|e| ApiError::new(StatusCode::CONFLICT, format!("{e:#}")))?;
    let format = session.output.format;
    let bytes = stemd_core::encode_stem(&excerpt.audio, format, session.gain)
        .map_err(|e| ApiError::internal(format!("encoding {name}: {e:#}")))?;
    let clipped = (!format.has_headroom()).then(|| {
        excerpt
            .audio
            .data
            .iter()
            .flatten()
            .filter(|v| (*v * session.gain).abs() > 1.0)
            .count()
    });
    Ok(audio_response(
        format,
        bytes,
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
    tokio::task::spawn_blocking(move || encode_range(&session, &name, range.from, range.to))
        .await
        .map_err(|e| ApiError::internal(format!("reading the range: {e}")))?
}

async fn piece(
    State(state): State<Arc<AppState>>,
    Path((id, name, k)): Path<(String, String, usize)>,
) -> Result<Response, ApiError> {
    let session = session(&state, &id)?;
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
    let stored = match (state.queue.sessions().pieces(), track.verified()) {
        (Some(pieces), Some(track_id)) => pieces.lookup(
            &track.model_id,
            &type_dir(&session.output, session.gain),
            &track_id,
            &name,
            k,
            session.output.format.extension(),
        ),
        _ => None,
    };
    if let Some(path) = stored {
        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| ApiError::internal(format!("reading {}: {e}", path.display())))?;
        return Ok(audio_response(
            session.output.format,
            bytes,
            from,
            frames,
            session.gain,
            None,
        ));
    }
    tokio::task::spawn_blocking(move || encode_range(&session, &name, samples.start, samples.end))
        .await
        .map_err(|e| ApiError::internal(format!("encoding the piece: {e}")))?
}
