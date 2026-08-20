//! HTTP surface.
//!
//! ```text
//! GET    /v1/health                     runtime, model, queue and cache state
//! POST   /v1/jobs                       raw interleaved PCM body -> 202 + job,
//!                                       or 200 + a finished one on a cache hit
//! GET    /v1/jobs/{id}                  progress, then the result object
//! GET    /v1/jobs/{id}/stems/{name}     raw interleaved PCM stream
//! DELETE /v1/jobs/{id}                  stop the job, drop the handle
//! GET    /v1/logs                       recent log lines
//! ```
//!
//! Upload is raw PCM, no decode/encode
//!
//! Nothing here deletes stems: `DELETE` drops a handle, and the two rules in
//! [`crate::cache`] are the only things that free disk.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Context as _;
use axum::Json;
use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Router, body::Bytes};
use serde::{Deserialize, Serialize};
use stemd_core::{Audio, DspMode, OutputRate, PcmFormat, Stage, resample};
use tokio_util::io::ReaderStream;

use crate::cache::{Cache, Ingredients, Output};
use crate::discovery::Advertiser;
use crate::jobs::JobStore;
use crate::logbuf::LogBuffer;
use crate::models::Preset;
use crate::queue::{Queue, QueuedWork};
use crate::settings::SettingsStore;

pub struct AppState {
    pub store: Arc<JobStore>,
    pub cache: Arc<Cache>,
    pub queue: Arc<Queue>,
    pub switcher: Arc<crate::switch::Switcher>,
    pub logs: LogBuffer,
    /// Files dropped on the window, and what became of them.
    pub drops: Arc<crate::drops::Drops>,
    /// What a request that names no output format or rate gets. Live: the window
    /// changes these while the server is running.
    pub settings: Arc<SettingsStore>,
    /// Longest track accepted, from `--max-track-minutes`. Advertised in
    /// `/v1/health` so a client can refuse locally instead of uploading a track
    /// it will be told to take back.
    pub max_track_secs: f64,
    /// Present unless --no-mdns. Shared so every exit path can withdraw the
    /// advertisement rather than relying on Drop, which macOS skips on Cmd-Q.
    pub advertiser: Option<Arc<Advertiser>>,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/{id}", get(get_job))
        .route("/v1/jobs/{id}", delete(delete_job))
        .route("/v1/jobs/{id}/stems/{name}", get(get_stem))
        .route("/v1/logs", get(get_logs))
        .with_state(state)
}

#[derive(Debug, Serialize)]
struct Health {
    version: &'static str,
    backend: String,
    model: String,
    device: String,
    /// The rate the model runs at. An upload at any other rate is converted to
    /// this one on arrival, so this is what the model saw rather than what a
    /// client has to send.
    sample_rate: u32,
    channels: usize,
    /// Stems served over the wire at the native rate. A job that asks for a
    /// different `output_sample_rate` also gets the derived part, which it
    /// could not reconstruct itself at that rate: read the `stems` of the job
    /// result rather than assuming this list.
    stems: Vec<String>,
    /// Which of the window's two presets is loaded, if either.
    #[serde(skip_serializing_if = "Option::is_none")]
    preset: Option<&'static str>,
    /// Exactly which weights are loaded and how they are arranged: the pinned
    /// digest for a preset, or `custom:<artefact>` otherwise.
    ///
    /// Key a client-side cache on this, not on `model`: several artefacts share one
    /// model name, so `model` cannot tell you whether stems on disk came from the
    /// weights now loaded.
    ///
    /// It covers the model and nothing about this run, so it survives the same
    /// server restarting on a different accelerator or at a different precision.
    /// The server's own cache key is narrower in what it will accept and wider in
    /// what it covers, since precision and overlap do change the audio; a client
    /// keeping stems under this id is choosing not to distinguish those.
    model_id: String,
    completed_jobs: usize,
    queue_depth: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    running_job: Option<String>,
    /// Separations held on disk, ready to be returned without running again.
    cached_tracks: usize,
    cached_bytes: u64,
    cache_max_bytes: u64,
    /// Longest track this server will accept. Check before uploading.
    max_track_seconds: f64,
    /// Rates `output_sample_rate` accepts. The model's own rate, 44100, is the
    /// only one that costs no conversion.
    output_sample_rates: Vec<u32>,
    /// Values `dsp_mode` accepts. `0` is the default and converts any rate pair;
    /// the rest are fixed filters covering one pair each, listed under
    /// `dsp_mode_pairs`. Ask for one only if you know why you need it.
    dsp_modes: Vec<u8>,
    /// The `[from, to]` a numbered mode is limited to, keyed by the mode. Mode 0
    /// is absent because it has no such limit.
    dsp_mode_pairs: BTreeMap<u8, [u32; 2]>,
    /// What a job that names neither gets. Set in the window, so read them here
    /// rather than assuming: they persist across launches and can change while
    /// this server is running.
    default_output_format: String,
    default_output_sample_rate: u32,
    /// The part `include_derived=true` adds. Not shipped by default: a client
    /// holding the mix rebuilds it as `mix` minus the stems above.
    derived_stem: &'static str,
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Health> {
    // Asked of the queue rather than kept alongside it: the worker owns the
    // separator, so it is the only thing that can say which one is loaded
    // without racing a switch that has not been applied yet.
    let info = state.queue.info();
    let cache = state.cache.stats();
    let defaults = state.settings.get();
    Json(Health {
        version: env!("CARGO_PKG_VERSION"),
        backend: info.backend,
        model: info.model,
        device: info.device,
        sample_rate: info.sample_rate,
        channels: info.channels,
        stems: info.stems,
        preset: state.switcher.current().map(Preset::label),
        model_id: state.switcher.published_identity(),
        completed_jobs: state.store.len(),
        queue_depth: state.queue.depth(),
        running_job: state.queue.running(),
        cached_tracks: cache.tracks,
        cached_bytes: cache.bytes,
        cache_max_bytes: cache.max_bytes,
        max_track_seconds: state.max_track_secs,
        output_sample_rates: OutputRate::ALL.iter().map(|r| r.hz()).collect(),
        dsp_modes: DspMode::ALL.iter().map(|m| m.id()).collect(),
        dsp_mode_pairs: DspMode::ALL
            .iter()
            .filter_map(|m| m.only_pair().map(|(from, to)| (m.id(), [from, to])))
            .collect(),
        default_output_format: defaults.format.to_string(),
        default_output_sample_rate: defaults.rate.hz(),
        derived_stem: stemd_core::DERIVED,
    })
}

#[derive(Debug, Deserialize)]
struct CreateParams {
    #[serde(default = "default_rate")]
    sample_rate: u32,
    #[serde(default = "default_channels")]
    channels: usize,
    /// Encoding of the uploaded body.
    #[serde(default)]
    format: Option<String>,
    /// Container for the stems returned: `flac`, `s16le` or `f32le`. Defaults to
    /// `default_output_format` from `/v1/health`.
    #[serde(default)]
    output_format: Option<String>,
    /// Rate the stems are converted to: 24000, 44100, 48000 or 96000. Defaults
    /// to `default_output_sample_rate` from `/v1/health`.
    #[serde(default)]
    output_sample_rate: Option<String>,
    /// Ship the derived part as well. Off by default: a client holding the mix
    /// can rebuild it for free, and not sending it keeps the transfer at two
    /// stems.
    #[serde(default)]
    include_derived: Option<bool>,
    /// Filter the conversion to `output_sample_rate` runs through. `0`, the
    /// default, converts any pair. See `dsp_modes` in `/v1/health`.
    #[serde(default)]
    dsp_mode: Option<String>,
}

const fn default_rate() -> u32 {
    44100
}

const fn default_channels() -> usize {
    2
}

/// Duration of an upload, from its length. Exact for raw PCM, and available
/// without decoding it.
fn track_seconds(bytes: usize, params: &CreateParams, format: PcmFormat) -> f64 {
    let stride = params.channels * format.bytes_per_sample();
    if stride == 0 || params.sample_rate == 0 {
        return 0.0;
    }
    (bytes / stride) as f64 / f64::from(params.sample_rate)
}

/// Whether `dsp` covers the conversion this job will actually run.
///
/// A mode that covers one rate pair is refused on any other rather than ignored:
/// a client asks for one because it has to match that filter, and quietly giving
/// it a different one is the failure it was trying to avoid. The uploaded rate
/// does not enter into it, since the conversion starts from the model's rate
/// whatever arrived.
fn check_dsp_mode(dsp: DspMode, model_rate: u32, out_rate: OutputRate) -> Result<(), String> {
    match dsp.only_pair() {
        Some((from, to)) if from != model_rate || to != out_rate.hz() => Err(format!(
            "dsp_mode {dsp} converts {from} to {to} Hz; this job converts {model_rate} to {} Hz",
            out_rate.hz()
        )),
        _ => Ok(()),
    }
}

/// A submission whose query string has been parsed and checked against the body.
#[derive(Clone, Copy)]
struct Submission {
    in_format: PcmFormat,
    output: Output,
    channels: usize,
    /// Rate the body is at, which the model's need not match.
    sample_rate: u32,
    /// Rate the model runs at, read once here so the whole request works from
    /// one answer even if the window switches models underneath it.
    model_rate: u32,
}

impl Submission {
    /// Reject anything malformed before the upload is hashed or decoded.
    fn parse(params: &CreateParams, body: &Bytes, state: &AppState) -> Result<Self, ApiError> {
        let in_format: PcmFormat = params
            .format
            .as_deref()
            .unwrap_or("s16le")
            .parse()
            .map_err(|e| ApiError::bad_request(format!("{e}")))?;
        // Read once: the window can change these between two requests, and a job
        // whose format and rate came from different instants is not one anybody
        // asked for.
        let defaults = state.settings.get();
        let out_format = match params.output_format.as_deref() {
            Some(s) => s
                .parse()
                .map_err(|e| ApiError::bad_request(format!("{e}")))?,
            None => defaults.format,
        };
        let out_rate = match params.output_sample_rate.as_deref() {
            Some(s) => s
                .parse()
                .map_err(|e| ApiError::bad_request(format!("{e}")))?,
            None => defaults.rate,
        };
        let dsp: DspMode = match params.dsp_mode.as_deref() {
            Some(s) => s
                .parse()
                .map_err(|e| ApiError::bad_request(format!("{e}")))?,
            None => DspMode::default(),
        };

        let model_rate = state.queue.info().sample_rate;
        check_dsp_mode(dsp, model_rate, out_rate).map_err(ApiError::bad_request)?;

        // Said out loud, because the way this goes wrong is inaudible from here
        // and unmistakable at the other end. A client rebuilding the derived
        // part subtracts these stems from a mix it resampled itself, and the two
        // filters have to be the same one: mode 1 carries 64 samples of group
        // delay that the general resampler does not, which at 1.5 kHz is a third
        // of a cycle, so the vocals come back rather than cancelling. Not an
        // error, because plenty of clients ask for a rate and subtract nothing.
        if dsp == DspMode::General
            && let Some(other) = DspMode::for_pair(model_rate, out_rate.hz())
        {
            tracing::info!(
                "converting {model_rate} to {} Hz with the general resampler; \
                 dsp_mode {other} is this server's copy of one client's own \
                 filter for that pair, and this job did not ask for it",
                out_rate.hz()
            );
        }

        // Refused rather than quietly honoured at a rate the format can carry:
        // a client that asked for 96 kHz mp3 and got 48 has no way to notice.
        if !out_format.carries(out_rate) {
            return Err(ApiError::bad_request(format!(
                "{out_format} has no {} Hz mode; it carries {}",
                out_rate.hz(),
                out_format
                    .rates()
                    .map(|r| r.hz().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }

        if body.is_empty() {
            return Err(ApiError::bad_request("empty payload"));
        }
        // Validated before they are used as divisors below.
        if params.channels == 0 {
            return Err(ApiError::bad_request("channels must be at least 1"));
        }
        if params.sample_rate == 0 {
            return Err(ApiError::bad_request("sample_rate must be at least 1"));
        }

        // Length is exact from the byte count, so an over-long track is refused
        // before it is hashed or decoded rather than after it has been paid for.
        let secs = track_seconds(body.len(), params, in_format);
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

        Ok(Self {
            in_format,
            output: Output {
                format: out_format,
                rate: out_rate,
                derived: params.include_derived.unwrap_or(false),
                dsp,
            },
            channels: params.channels,
            sample_rate: params.sample_rate,
            model_rate,
        })
    }

    fn ingredients<'a>(&self, pcm: &'a [u8], model: &'a str) -> Ingredients<'a> {
        Ingredients {
            pcm,
            sample_rate: self.sample_rate,
            channels: self.channels,
            in_format: self.in_format,
            out_format: self.output.format,
            out_rate: self.output.rate,
            include_derived: self.output.derived,
            dsp: self.output.dsp,
            model,
        }
    }
}

/// Hash the upload into a cache key, off the async runtime.
///
/// Hashing a five-minute upload takes about 160 ms, which would stall every other
/// connection this runtime thread is serving. It happens before decoding because
/// the samples on the wire are what identifies the work, so a hit skips converting
/// a hundred megabytes to float.
async fn hash_upload(
    body: Bytes,
    submission: Submission,
    model: String,
) -> Result<(String, Bytes), ApiError> {
    tokio::task::spawn_blocking(move || {
        let key = crate::cache::key(&submission.ingredients(&body, &model));
        (key, body)
    })
    .await
    .map_err(|e| ApiError::internal(format!("hashing the upload: {e}")))
}

/// Decode the upload to planar float at the model's rate, off the async runtime.
///
/// The upload may be at any rate; the model works at exactly one, so anything
/// else is converted here rather than carried to the worker to be refused there.
/// The conversion is the general resampler: a client that wants a particular
/// filter is asking about the stems on the way out, not about the mix on the way
/// in, which the model consumes and never returns.
///
/// The job is already visible to other requests by now, so a bad payload has to
/// reach a terminal stage rather than vanish: a joiner is polling it.
async fn decode_upload(
    body: Bytes,
    submission: Submission,
    job: &crate::jobs::Job,
) -> Result<Audio, ApiError> {
    let model_rate = submission.model_rate;
    let decoded = tokio::task::spawn_blocking(move || -> anyhow::Result<Audio> {
        let audio = Audio::from_interleaved(
            &body,
            submission.in_format,
            submission.channels,
            submission.sample_rate,
        )?;
        if audio.sample_rate == model_rate {
            return Ok(audio);
        }
        let began = std::time::Instant::now();
        let converted = resample::to_rate(&audio, model_rate)
            .with_context(|| format!("converting the upload from {} Hz", audio.sample_rate))?;
        tracing::info!(
            "upload at {} Hz, converted to the model's {model_rate} Hz in {:.2?}",
            audio.sample_rate,
            began.elapsed()
        );
        Ok(converted)
    })
    .await
    .map_err(|e| ApiError::internal(format!("decoding the upload: {e}")))?;

    decoded.map_err(|err| {
        job.fail(format!("{err:#}"));
        ApiError::bad_request(format!("{err:#}"))
    })
}

/// A job as the client should see it right now.
///
/// Queue positions shift as the worker drains it, so the live one is resolved
/// at read time rather than trusting what was stored at submit.
fn live_view(state: &AppState, job: &crate::jobs::Job) -> crate::jobs::JobView {
    let mut view = job.view();
    if view.progress.stage == Stage::Queued
        && let Some(progress) = state.queue.queued_progress_for(&job.id)
    {
        view.progress = progress;
    }
    view
}

async fn create_job(
    State(state): State<Arc<AppState>>,
    Query(params): Query<CreateParams>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let submission = Submission::parse(&params, &body, &state)?;
    let (key, body) = hash_upload(body, submission, state.switcher.identity()).await?;

    // Claim the key before the decode, not after: `claim` registers the job
    // under the same lock that looks for one, so simultaneous requests for the
    // same track collapse onto a single separation instead of racing through
    // the gap a decode would leave between lookup and insert.
    let job = match state.store.claim(&key) {
        Err(existing) => {
            tracing::debug!(
                job = %existing.id,
                "joining a separation already under way for {}",
                crate::cache::short(&key)
            );
            return Ok((StatusCode::ACCEPTED, Json(live_view(&state, &existing))).into_response());
        }
        Ok(job) => job,
    };

    if let Some(entry) = state.cache.get(&key) {
        tracing::debug!(
            job = %job.id,
            "serving {} from cache, separated in {:.2}s",
            crate::cache::short(&key),
            entry.separation_secs
        );
        job.complete(entry, true);
        return Ok((StatusCode::OK, Json(job.view())).into_response());
    }

    let mix = decode_upload(body, submission, &job).await?;
    enqueue(&state, &job, mix, submission)?;
    Ok((StatusCode::ACCEPTED, Json(job.view())).into_response())
}

/// Hand the decoded mix to the worker, failing the job if the queue is full.
fn enqueue(
    state: &AppState,
    job: &Arc<crate::jobs::Job>,
    mix: Audio,
    submission: Submission,
) -> Result<(), ApiError> {
    let duration = mix.duration_secs();
    let position = state
        .queue
        .submit(QueuedWork {
            job: Arc::clone(job),
            mix,
            output: submission.output,
        })
        .map_err(|full| {
            job.fail(format!("{full}"));
            ApiError::new(StatusCode::TOO_MANY_REQUESTS, format!("{full}"))
        })?;

    tracing::info!(
        job = %job.id,
        "queued {duration:.1}s of audio at position {position} ({} ch @ {} Hz, {} in, \
         {} parts out at {} Hz)",
        submission.channels,
        submission.sample_rate,
        submission.in_format,
        submission.output.parts(),
        submission.output.rate,
    );
    Ok(())
}

async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let job = state.store.get(&id).ok_or_else(ApiError::not_found)?;
    Ok(Json(live_view(&state, &job)).into_response())
}

async fn delete_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let job = state.store.get(&id).ok_or_else(ApiError::not_found)?;

    // Two decks can share one job. Cancelling on the first `DELETE` would take
    // the separation out from under the other one, so only the last waiter
    // letting go actually stops the work.
    let remaining = job.release();
    if remaining > 0 {
        tracing::debug!(job = %id, "released, {remaining} waiter(s) still want it");
        return Ok(StatusCode::NO_CONTENT.into_response());
    }

    // Cancel before removing: `cancel` matches on the running job's id, and the
    // worker reports what it actually stopped. Removing first would only widen
    // the window in which a re-submission of the same track cannot join.
    let cancelled = state.queue.cancel(&id);
    state.store.remove(&id);
    if cancelled {
        tracing::info!(job = %id, "cancelled");
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}

async fn get_stem(
    State(state): State<Arc<AppState>>,
    Path((id, name)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let job = state.store.get(&id).ok_or_else(ApiError::not_found)?;
    let entry =
        job.entry.lock().clone().ok_or_else(|| {
            ApiError::new(StatusCode::CONFLICT, "job has not finished separating")
        })?;
    let path = entry
        .stem(&name)
        .ok_or_else(ApiError::not_found)?
        .path
        .clone();

    let file = tokio::fs::File::open(&path).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            // Reaped between the lookup and the open. Everything here is
            // reproducible, so say so plainly and let the client resubmit.
            ApiError::new(
                StatusCode::GONE,
                "these stems have been reaped; resubmit the track",
            )
        } else {
            ApiError::internal(format!("opening {}: {e}", path.display()))
        }
    })?;
    let len = file
        .metadata()
        .await
        .map(|m| m.len())
        .map_err(|e| ApiError::internal(format!("stat {}: {e}", path.display())))?;

    // Only once the bytes are provably readable. Marking on request would let a
    // failed fetch count as consumption and keep an entry nobody can use.
    entry.mark_fetched(&name);

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, entry.format.content_type())
        .header(header::CONTENT_LENGTH, len)
        .body(Body::from_stream(ReaderStream::new(file)))
        .expect("response builder inputs are valid"))
}

#[derive(Debug, Deserialize)]
struct LogParams {
    #[serde(default = "default_log_limit")]
    limit: usize,
}

const fn default_log_limit() -> usize {
    200
}

async fn get_logs(
    State(state): State<Arc<AppState>>,
    Query(params): Query<LogParams>,
) -> impl IntoResponse {
    Json(state.logs.recent(params.limit))
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    fn not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "not found")
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({"error": self.message})),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(channels: usize, sample_rate: u32) -> CreateParams {
        CreateParams {
            sample_rate,
            channels,
            format: None,
            output_format: None,
            output_sample_rate: None,
            include_derived: None,
            dsp_mode: None,
        }
    }

    #[test]
    fn track_length_is_exact_from_the_byte_count() {
        // 10 s of 44.1k stereo s16 = 44100 * 10 * 2 * 2 bytes.
        let bytes = 44100 * 10 * 2 * 2;
        let secs = track_seconds(bytes, &params(2, 44100), PcmFormat::S16le);
        assert!((secs - 10.0).abs() < 1e-9, "got {secs}");

        // The same audio as f32 is twice the bytes and the same duration.
        let secs = track_seconds(bytes * 2, &params(2, 44100), PcmFormat::F32le);
        assert!((secs - 10.0).abs() < 1e-9, "got {secs}");
    }

    /// Mode 0 fits every job. Mode 1 fits exactly one, and says which when it
    /// does not, so a client that asked for the wrong output rate is told rather
    /// than handed the general filter under the name it asked for.
    #[test]
    fn a_numbered_dsp_mode_is_refused_outside_its_own_rate_pair() {
        for rate in OutputRate::ALL {
            assert!(check_dsp_mode(DspMode::General, 44_100, rate).is_ok());
        }

        assert!(check_dsp_mode(DspMode::Mode1, 44_100, OutputRate::Hz96000).is_ok());

        let err = check_dsp_mode(DspMode::Mode1, 44_100, OutputRate::Hz48000)
            .expect_err("48 kHz is not mode 1's pair");
        assert!(err.contains("44100 to 96000"), "{err}");
        assert!(err.contains("44100 to 48000"), "{err}");

        // A model at another rate takes mode 1 out of reach entirely, and the
        // message has to say so rather than blaming the output rate.
        let err = check_dsp_mode(DspMode::Mode1, 48_000, OutputRate::Hz96000)
            .expect_err("mode 1 starts at 44.1 kHz");
        assert!(err.contains("48000 to 96000"), "{err}");
    }

    #[test]
    fn a_zero_stride_cannot_divide_by_zero() {
        // Both come straight off the query string, so both are attacker-chosen.
        assert_eq!(
            track_seconds(1024, &params(0, 44100), PcmFormat::S16le),
            0.0
        );
        assert_eq!(track_seconds(1024, &params(2, 0), PcmFormat::S16le), 0.0);
    }
}
