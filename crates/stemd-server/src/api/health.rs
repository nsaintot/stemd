//! `GET /v2/health` and `GET /v2/logs`.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Query, State};
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use stemd_core::{DspMode, OutputRate, Parts};

use crate::models::Preset;
use crate::output::OutputView;
use crate::state::AppState;

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
pub(super) struct SessionMethods {
    methods: BTreeMap<&'static str, Method>,
}

/// What a method needs before a segment can run ahead of the upload's end.
#[derive(Debug, Serialize)]
pub(super) struct Method {
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
