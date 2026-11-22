//! HTTP surface.
//!
//! ```text
//! GET    /v2/health                     runtime, model, queue and cache state
//! POST   /v2/jobs                       raw interleaved PCM body -> 202 + job,
//!                                       or 200 + a finished one on a cache hit
//! GET    /v2/jobs/{id}                  progress, then the result object
//! GET    /v2/jobs/{id}/stems/{name}     raw interleaved PCM stream
//! DELETE /v2/jobs/{id}                  stop the job, drop the handle
//! POST   /v2/jobs/{id}/join             one more waiter on a job found by lookup
//! GET    /v2/logs                       recent log lines
//! PUT    /v2/filters/{id}               store a conversion filter, see [`crate::filters`]
//! GET    /v2/filters/{id}               a stored filter's summary
//! ```
//!
//! Upload is raw PCM, no decode/encode
//!
//! Nothing here deletes stems: `DELETE` drops a handle, and the two rules in
//! [`crate::cache`] are the only things that free disk.

use crate::separations::OutputView;
use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Context as _;
use axum::Json;
use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Router, body::Bytes};
use serde::{Deserialize, Serialize};
use stemd_core::{Audio, DspMode, OutputRate, Parts, PcmFormat, Stage, resample};
use tokio_util::io::ReaderStream;

use crate::cache::Cache;
use crate::filters::{Dsp, Filters};
use crate::jobs::{Entry, Jobs};
use crate::logbuf::LogBuffer;
use crate::models::Preset;
use crate::output::Output;
use crate::queue::{Queue, QueuedWork};
use crate::settings::SettingsFile;

pub struct AppState {
    pub jobs: Arc<Jobs>,
    /// The one cache of separated stems; the same as the sessions'.
    pub cache: Arc<Cache>,
    pub queue: Arc<Queue>,
    pub switcher: Arc<crate::switch::Switcher>,
    pub logs: LogBuffer,
    /// Files dropped on the window, and what became of them.
    pub drops: Arc<crate::drops::Drops>,
    /// What a request that names no output format or rate gets. Live: the window
    /// changes these while the server is running.
    pub settings: Arc<SettingsFile>,
    /// Whether the API answers, and its mDNS advertisement. Shared so every
    /// exit path can withdraw the advertisement: macOS skips Drop on Cmd-Q.
    pub serving: crate::serving::Serving,
    /// Conversion filters clients uploaded.
    pub filters: Arc<Filters>,
    /// Empty the cache at quit, from `--ephemeral`.
    pub ephemeral: bool,
    /// Where the API listens.
    pub endpoint: crate::endpoint::Endpoint,
}

impl AppState {
    /// Longest track accepted: the saved length, never past what this machine's memory holds.
    pub fn max_track_secs(&self) -> f64 {
        self.settings
            .get()
            .max_track_secs(crate::disk::longest_track_minutes())
    }

    /// Largest request body to buffer: the longest track at the widest format.
    pub fn max_body_bytes(&self) -> usize {
        crate::startup::max_body_bytes(
            self.max_track_secs(),
            crate::startup::highest_rate(),
            self.queue.info().channels,
        )
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v2/health", get(health))
        .route("/v2/jobs", post(create_job))
        .route("/v2/jobs/{id}", get(get_job))
        .route("/v2/jobs/{id}", delete(delete_job))
        .route("/v2/jobs/{id}/join", post(join_job))
        .route("/v2/jobs/{id}/stems/{name}", get(get_stem))
        .route("/v2/logs", get(get_logs))
        .route("/v2/filters/{id}", put(put_filter))
        .route("/v2/filters/{id}", get(get_filter))
        .merge(crate::session_api::routes())
        .merge(crate::separations::routes())
        .merge(crate::v1::routes())
        .route_layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            refuse_when_off,
        ))
        .layer(axum::middleware::from_fn(crate::v1::deprecate))
        .with_state(state)
}

/// 503 for every request while the server is turned off in the window.
async fn refuse_when_off(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if state.serving.is_on() {
        next.run(request).await
    } else {
        ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "the server is turned off in the Stemd window",
        )
        .into_response()
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct Health {
    version: &'static str,
    backend: String,
    model: String,
    device: String,
    /// The rate the model runs at. An upload at any other rate is converted to
    /// this one on arrival, so this is what the model saw rather than what a
    /// client has to send.
    sample_rate: u32,
    channels: usize,
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
    /// `dsp_mode_pairs`. The numbered modes are deprecated in favor of
    /// `dsp_filter`.
    dsp_modes: Vec<u8>,
    /// The `[from, to]` a numbered mode is limited to, keyed by the mode. Mode 0
    /// is absent because it has no such limit.
    dsp_mode_pairs: BTreeMap<u8, [u32; 2]>,
    /// Parts a job can list in `parts`. Which one the player rebuilds follows from
    /// the request: whatever it leaves out.
    parts: Vec<String>,
    /// The output of a request that names no format, rate or parts. Format and rate
    /// are set in the window and can change while the server runs.
    default_output: OutputView,
    /// Behaviors a client can test for, see [`FEATURES`].
    features: &'static [&'static str],
    /// Median seconds the loaded model takes to run one segment, `null` until one ran.
    segment_run_seconds: Option<f64>,
    /// How a session can upload, or `null` when the model cannot run segment by segment.
    session: Option<SessionMethods>,
}

/// The upload methods of a session on the loaded model.
#[derive(Debug, Serialize)]
struct SessionMethods {
    methods: BTreeMap<&'static str, Method>,
}

/// What a method needs before a segment can run ahead of the upload's end.
#[derive(Debug, Serialize)]
struct Method {
    /// Open parameters the method needs.
    requires: Vec<&'static str>,
    /// Open parameters that change the output but are never needed.
    optional: Vec<&'static str>,
}

impl SessionMethods {
    /// The methods of the loaded model, or `None` when it cannot run segment by segment.
    fn of(queue: &crate::queue::Queue) -> Option<Self> {
        queue.segmenting()?;
        let partial = if queue.normalises() {
            vec!["mean", "std"]
        } else {
            Vec::new()
        };
        let methods = BTreeMap::from([
            (
                "whole",
                Method {
                    requires: Vec::new(),
                    optional: vec!["peak"],
                },
            ),
            (
                "partial",
                Method {
                    requires: partial,
                    optional: vec!["peak"],
                },
            ),
        ]);
        Some(Self { methods })
    }
}

/// Behaviors added after protocol 2, listed in `/v2/health` and the TXT record.
pub const FEATURES: &[&str] = &[];

pub(crate) async fn health(State(state): State<Arc<AppState>>) -> Json<Health> {
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
        preset: state.switcher.current().map(Preset::label),
        model_id: state.switcher.published_identity(),
        completed_jobs: state.jobs.len(),
        queue_depth: state.queue.depth(),
        running_job: state.queue.running(),
        cached_tracks: cache.tracks,
        cached_bytes: cache.bytes,
        cache_max_bytes: cache.max_bytes,
        max_track_seconds: state.max_track_secs(),
        output_sample_rates: OutputRate::ALL.iter().map(|r| r.hz()).collect(),
        dsp_modes: DspMode::ALL.iter().map(|m| m.id()).collect(),
        dsp_mode_pairs: DspMode::ALL
            .iter()
            .filter_map(|m| m.only_pair().map(|(from, to)| (m.id(), [from, to])))
            .collect(),
        parts: info.parts,
        default_output: OutputView {
            format: defaults.format.to_string(),
            sample_rate: defaults.rate.hz(),
            gain: None,
            parts: Parts::DEFAULT.names().map(str::to_owned).collect(),
            derived: Parts::DEFAULT.derived(),
        },
        features: FEATURES,
        segment_run_seconds: state.queue.segment_run_seconds(),
        session: SessionMethods::of(&state.queue),
    })
}

#[derive(Debug, Deserialize)]
pub(crate) struct CreateParams {
    #[serde(default = "default_rate")]
    pub(crate) sample_rate: u32,
    #[serde(default = "default_channels")]
    pub(crate) channels: usize,
    /// Encoding of the uploaded body.
    #[serde(default)]
    pub(crate) format: Option<String>,
    /// Container for the stems returned: `flac`, `s16le` or `f32le`. Defaults to
    /// `default_output.format` from `/v2/health`.
    #[serde(default)]
    pub(crate) output_format: Option<String>,
    /// Rate the stems are converted to: 24000, 44100, 48000 or 96000. Defaults
    /// to `default_output.sample_rate` from `/v2/health`.
    #[serde(default)]
    pub(crate) output_sample_rate: Option<String>,
    /// Ship the derived part as well. Off by default: a client holding the mix
    /// can rebuild it for free, and not sending it keeps the transfer at two
    /// stems.
    #[serde(default)]
    pub(crate) include_derived: Option<bool>,
    /// Comma-separated parts to ship, such as `bass,other,vocals`. Defaults to
    /// `default_output.parts` from `/v2/health`. They must not overlap, and must leave out
    /// one part or none: that part is the one `include_derived` adds.
    #[serde(default)]
    pub(crate) parts: Option<String>,
    /// Filter the conversion to `output_sample_rate` runs through. `0`, the
    /// default, converts any pair. See `dsp_modes` in `/v2/health`.
    #[serde(default)]
    pub(crate) dsp_mode: Option<String>,
    /// Id of an uploaded filter the conversion runs through instead. See
    /// `PUT /v2/filters/{id}`.
    #[serde(default)]
    pub(crate) dsp_filter: Option<String>,
    /// The client's key for the file the audio came from, recorded against the track id.
    #[serde(default)]
    pub(crate) file: Option<String>,
}

pub(crate) const fn default_rate() -> u32 {
    44100
}

pub(crate) const fn default_channels() -> usize {
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
fn check_dsp(dsp: &Dsp, model_rate: u32, out_rate: OutputRate) -> Result<(), String> {
    let name = match dsp {
        Dsp::Mode(mode) => format!("dsp_mode {mode}"),
        Dsp::Fir(fir) => format!("dsp_filter {}", fir.id),
    };
    match dsp.only_pair() {
        Some((from, to)) if from != model_rate || to != out_rate.hz() => Err(format!(
            "{name} converts {from} to {to} Hz; this job converts {model_rate} to {} Hz",
            out_rate.hz()
        )),
        _ => Ok(()),
    }
}

/// Log once per run that a client still uses a numbered mode.
fn warn_numbered_mode_once(mode: DspMode) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            "a client asked for dsp_mode {mode}, which is deprecated; \
             it should upload its filter and send dsp_filter instead"
        );
    }
}

/// The filter a request names with `dsp_mode` or `dsp_filter`.
fn parse_dsp(params: &CreateParams, filters: &Filters) -> Result<Dsp, ApiError> {
    let mode: DspMode = match params.dsp_mode.as_deref() {
        Some(s) => s
            .parse()
            .map_err(|e| ApiError::bad_request(format!("{e}")))?,
        None => DspMode::default(),
    };
    let Some(id) = params.dsp_filter.as_deref() else {
        if mode != DspMode::General {
            warn_numbered_mode_once(mode);
        }
        return Ok(mode.into());
    };
    if mode != DspMode::General {
        return Err(ApiError::bad_request(
            "dsp_mode and dsp_filter both name a filter; send one",
        ));
    }
    if !crate::filters::is_id(id) {
        return Err(ApiError::bad_request(format!(
            "{id:?} is not a filter id: expected 64 lowercase hex digits"
        )));
    }
    filters.get(id).map(Dsp::Fir).ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            format!("no filter {id}; upload it with PUT /v2/filters/{id}"),
        )
        .with_code("unknown_filter")
    })
}

/// A submission whose query string has been parsed and checked against the body.
#[derive(Clone)]
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

/// The upload format and the output a request asks for, checked against the
/// loaded model.
pub(crate) fn parse_output(
    params: &CreateParams,
    state: &AppState,
) -> Result<(PcmFormat, Output, u32), ApiError> {
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
    let dsp = parse_dsp(params, &state.filters)?;

    let info = state.queue.info();
    let model_rate = info.sample_rate;
    let parts = match params.parts.as_deref() {
        Some(list) => Parts::parse(list).map_err(ApiError::bad_request)?,
        None => Parts::DEFAULT,
    };
    if let Some(missing) = parts.names().find(|p| !info.parts.iter().any(|q| q == p)) {
        return Err(ApiError::bad_request(format!(
            "{} cannot produce {missing}; it offers {:?}",
            info.model, info.parts
        )));
    }
    check_dsp(&dsp, model_rate, out_rate).map_err(ApiError::bad_request)?;

    // Said out loud, because the way this goes wrong is inaudible from here
    // and unmistakable at the other end. A client rebuilding the derived
    // part subtracts these stems from a mix it resampled itself, and the two
    // filters have to be the same one: mode 1 carries 64 samples of group
    // delay that the general resampler does not, which at 1.5 kHz is a third
    // of a cycle, so the vocals come back rather than cancelling. Not an
    // error, because plenty of clients ask for a rate and subtract nothing.
    if dsp.is_general()
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

    Ok((
        in_format,
        Output {
            format: out_format,
            rate: out_rate,
            derived: params.include_derived.unwrap_or(false),
            dsp,
            parts,
        },
        model_rate,
    ))
}

impl Submission {
    /// Reject anything malformed before the upload is hashed or decoded.
    fn parse(params: &CreateParams, body: &Bytes, state: &AppState) -> Result<Self, ApiError> {
        let (in_format, output, model_rate) = parse_output(params, state)?;

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

        Ok(Self {
            in_format,
            output,
            channels: params.channels,
            sample_rate: params.sample_rate,
            model_rate,
        })
    }
}

/// Decode the upload to planar float at the model's rate, off the async runtime.
///
/// The upload may be at any rate; the model works at exactly one, so anything
/// else is converted here rather than carried to the worker to be refused there.
/// The conversion is the general resampler: a client that wants a particular
/// filter is asking about the stems on the way out, not about the mix on the way
/// in, which the model consumes and never returns.
async fn decode_upload(body: Bytes, submission: Submission) -> Result<Audio, ApiError> {
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

    decoded.map_err(|err| ApiError::bad_request(format!("{err:#}")))
}

/// The track id of `mix`, off the async runtime.
async fn identify(mix: Audio) -> Result<(Audio, String), ApiError> {
    tokio::task::spawn_blocking(move || {
        let id = crate::sessions::track_id(&mix);
        (mix, id)
    })
    .await
    .map_err(|e| ApiError::internal(format!("hashing the upload: {e}")))
}

/// The job result for track `id` at `output` when the cache holds every stem
/// whole, at any gain, held for the client to fetch.
pub fn cached_entry(
    cache: &Arc<Cache>,
    cache_id: &str,
    output: &Output,
    id: &str,
) -> Option<Arc<Entry>> {
    let found = crate::separations::whole_dir(cache, cache_id, output, id)?;
    let _writing = cache.writing(&found.dir);
    let entry = Arc::new(Entry::stored(
        &found.dir,
        output,
        found.gain,
        found.info.channels,
        Cache::separation(&found.dir),
    )?);
    cache.hold(&entry);
    entry.present().then_some(entry)
}

/// [`cached_entry`], off the async runtime.
async fn cached_result(
    cache: &Arc<Cache>,
    cache_id: String,
    output: Output,
    id: String,
) -> Option<Arc<Entry>> {
    let cache = Arc::clone(cache);
    tokio::task::spawn_blocking(move || cached_entry(&cache, &cache_id, &output, &id))
        .await
        .ok()
        .flatten()
}

/// A job as the client should see it right now.
///
/// Queue positions shift as the worker drains it, so the live one is resolved
/// at read time rather than trusting what was stored at submit.
pub(crate) fn live_view(state: &AppState, job: &crate::jobs::Job) -> crate::jobs::JobView {
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
    let (status, job) = submit(&state, &params, body).await?;
    Ok((status, Json(live_view(&state, &job))).into_response())
}

/// Start a job, join the one running for the same track, or answer from the cache.
pub(crate) async fn submit(
    state: &Arc<AppState>,
    params: &CreateParams,
    body: Bytes,
) -> Result<(StatusCode, Arc<crate::jobs::Job>), ApiError> {
    if let Some(key) = &params.file {
        crate::separations::check_file_key(key)?;
    }
    let submission = Submission::parse(params, &body, state)?;
    let mix = decode_upload(body, submission.clone()).await?;
    let (mix, id) = identify(mix).await?;
    // Before the claim, so a lookup by the file key finds the job while it runs.
    if let Some(key) = &params.file
        && let Err(err) = state.cache.record_file(key, &id)
    {
        tracing::warn!("recording file key {key}: {err:#}");
    }
    let cache_id = state.switcher.cache_id();
    let key = crate::jobs::job_key(&id, &submission.output, &cache_id);

    // Claim before the lookup: `claim` registers the job under the same lock
    // that looks for one, so simultaneous requests for the same track collapse
    // onto a single separation.
    let job = match state.jobs.claim(&key, crate::jobs::MixShape::of(&mix)) {
        Err(existing) => {
            tracing::debug!(
                job = %existing.id,
                "joining a separation already under way for track {id}"
            );
            return Ok((StatusCode::ACCEPTED, existing));
        }
        Ok(job) => job,
    };

    if let Some(entry) = cached_result(
        &state.cache,
        cache_id,
        submission.output.clone(),
        id.clone(),
    )
    .await
    {
        tracing::debug!(job = %job.id, "serving track {id} from the cache");
        job.complete(entry, true);
        return Ok((StatusCode::OK, job));
    }

    // Stems are written to the cache before they are served, and a full disk
    // also leaves the system no room to swap.
    if let Err(full) = crate::disk::ensure_room(state.cache.root()) {
        job.fail(full.to_string());
        return Err(ApiError::new(
            StatusCode::INSUFFICIENT_STORAGE,
            full.to_string(),
        ));
    }

    enqueue(state, &job, mix, submission)?;
    Ok((StatusCode::ACCEPTED, job))
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
            output: submission.output.clone(),
            model_id: state.switcher.cache_id(),
            from_api: true,
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
    let job = state.jobs.get(&id).ok_or_else(ApiError::not_found)?;
    Ok(Json(live_view(&state, &job)).into_response())
}

/// Add a waiter to a job, so another client's DELETE leaves it running.
async fn join_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let job = state.jobs.get(&id).ok_or_else(ApiError::not_found)?;
    job.join();
    Ok(Json(live_view(&state, &job)).into_response())
}

pub(crate) async fn delete_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let job = state.jobs.get(&id).ok_or_else(ApiError::not_found)?;

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
    state.jobs.remove(&id);
    if cancelled {
        tracing::info!(job = %id, "cancelled");
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}

pub(crate) async fn get_stem(
    State(state): State<Arc<AppState>>,
    Path((id, name)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let job = state.jobs.get(&id).ok_or_else(ApiError::not_found)?;
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
            // Deleted from the cache between the lookup and the open.
            ApiError::new(
                StatusCode::GONE,
                "these stems have left the cache; resubmit the track",
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

    // Only once the bytes are provably readable, so a failed fetch keeps the
    // stems held.
    entry.mark_fetched(&name);

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, entry.output.format.content_type())
        .header(header::CONTENT_LENGTH, len)
        .body(Body::from_stream(ReaderStream::new(file)))
        .expect("response builder inputs are valid"))
}

#[derive(Debug, Deserialize)]
pub(crate) struct LogParams {
    #[serde(default = "default_log_limit")]
    limit: usize,
}

const fn default_log_limit() -> usize {
    200
}

pub(crate) async fn get_logs(
    State(state): State<Arc<AppState>>,
    Query(params): Query<LogParams>,
) -> impl IntoResponse {
    Json(state.logs.recent(params.limit))
}

/// Store a filter file under its SHA-256. `201` when new, `200` when already stored.
async fn put_filter(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    body: Bytes,
) -> Result<Response, ApiError> {
    if body.len() > crate::filters::MAX_FILE_BYTES {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "{} bytes is over the limit of {}",
                body.len(),
                crate::filters::MAX_FILE_BYTES
            ),
        ));
    }
    let filters = Arc::clone(&state.filters);
    let stored = tokio::task::spawn_blocking(move || filters.put(&id, &body))
        .await
        .map_err(|e| ApiError::internal(format!("storing the filter: {e}")))?;
    match stored {
        Ok((fir, created)) => {
            if created {
                tracing::info!(
                    "filter {} stored: {} to {} Hz, {} taps",
                    &fir.id[..12],
                    fir.poly.in_rate(),
                    fir.poly.out_rate(),
                    fir.poly.taps()
                );
            }
            let status = if created {
                StatusCode::CREATED
            } else {
                StatusCode::OK
            };
            Ok((status, Json(fir.summary())).into_response())
        }
        Err(crate::filters::PutError::Io(err)) => Err(ApiError::internal(format!("{err:#}"))),
        Err(err) => Err(ApiError::bad_request(err.to_string())),
    }
}

/// A stored filter's summary, or `404` with code `unknown_filter`.
async fn get_filter(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    state
        .filters
        .peek(&id)
        .map(|fir| Json(fir.summary()).into_response())
        .ok_or_else(|| {
            ApiError::new(StatusCode::NOT_FOUND, format!("no filter {id}"))
                .with_code("unknown_filter")
        })
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
    /// A stable name for an error a client acts on, sent as `code`.
    code: Option<&'static str>,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            code: None,
        }
    }

    pub(crate) const fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    pub(crate) fn not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "not found")
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = match self.code {
            Some(code) => serde_json::json!({"error": self.message, "code": code}),
            None => serde_json::json!({"error": self.message}),
        };
        (self.status, Json(body)).into_response()
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
            parts: None,
            dsp_mode: None,
            dsp_filter: None,
            file: None,
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
            assert!(check_dsp(&DspMode::General.into(), 44_100, rate).is_ok());
        }

        assert!(check_dsp(&DspMode::Mode1.into(), 44_100, OutputRate::Hz96000).is_ok());

        let err = check_dsp(&DspMode::Mode1.into(), 44_100, OutputRate::Hz48000)
            .expect_err("48 kHz is not mode 1's pair");
        assert!(err.contains("44100 to 96000"), "{err}");
        assert!(err.contains("44100 to 48000"), "{err}");

        // A model at another rate takes mode 1 out of reach entirely, and the
        // message has to say so rather than blaming the output rate.
        let err = check_dsp(&DspMode::Mode1.into(), 48_000, OutputRate::Hz96000)
            .expect_err("mode 1 starts at 44.1 kHz");
        assert!(err.contains("48000 to 96000"), "{err}");
    }

    #[test]
    fn a_job_names_an_uploaded_filter_by_its_id() {
        let dir = std::env::temp_dir().join(format!("stemd-api-filters-{}", std::process::id()));
        let filters = Filters::open(dir, 4).unwrap();
        let body = crate::filters::encode(44_100, 96_000, &[1.0; 320]);
        let id = crate::filters::id_of(&body);

        let mut p = params(2, 44_100);
        p.dsp_filter = Some(id.clone());
        let err = parse_dsp(&p, &filters).unwrap_err();
        assert_eq!(err.status, StatusCode::NOT_FOUND);
        assert_eq!(err.code, Some("unknown_filter"));

        filters.put(&id, &body).unwrap();
        let dsp = parse_dsp(&p, &filters).unwrap();
        assert_eq!(dsp.to_string(), format!("fir:{id}"));
        assert!(check_dsp(&dsp, 44_100, OutputRate::Hz96000).is_ok());
        let err = check_dsp(&dsp, 44_100, OutputRate::Hz48000).unwrap_err();
        assert!(err.contains(&format!("dsp_filter {id}")), "{err}");

        p.dsp_mode = Some("1".into());
        assert_eq!(
            parse_dsp(&p, &filters).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );

        p.dsp_mode = None;
        p.dsp_filter = Some("not-a-hash".into());
        assert_eq!(
            parse_dsp(&p, &filters).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );
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
