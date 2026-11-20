//! Protocol 1 under `/v1`, which is deprecated and may stop working in a future release.
//!
//! `/v1` keeps the 0.2.0 health and jobs. The handlers translate their requests
//! and answers onto the protocol 2 ones, so both protocols share every job and
//! stored stem. Sessions are protocol 2 only: `/v1/sessions` answers `410`.
//! "Protocol 1 (deprecated)" in `docs/api.md` lists the differences.
//!
//! Every answer under `/v1` carries `Deprecation: true` and a `Link` to the
//! API reference. The first request from each client address logs a warning.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use axum::body::Bytes;
use axum::extract::{ConnectInfo, Path, Query, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{Value, json};
use stemd_core::Progress;

use crate::api::{self, ApiError, AppState, CreateParams};
use crate::jobs::{Entry, Job};

/// The residual sent when the stems sum to the mix exactly, since JSON has no `-inf`.
const FLOOR_DB: f64 = -200.0;

/// The `Link` header on every protocol 1 answer.
const LINK: &str = "<https://github.com/nsaintot/stemd/blob/main/docs/api.md>; rel=\"deprecation\"; type=\"text/html\"";

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/{id}", get(get_job).delete(api::delete_job))
        .route("/v1/jobs/{id}/stems/{name}", get(api::get_stem))
        .route("/v1/logs", get(api::get_logs))
        .route("/v1/sessions", any(sessions_moved))
        .route("/v1/sessions/{*rest}", any(sessions_moved))
}

/// `410` for a protocol 1 session path.
async fn sessions_moved() -> ApiError {
    ApiError::new(
        StatusCode::GONE,
        "sessions are protocol 2 only; use /v2/sessions",
    )
    .with_code("protocol_1")
}

/// True for a protocol 1 path.
fn is_v1(path: &str) -> bool {
    path == "/v1" || path.starts_with("/v1/")
}

/// Mark a protocol 1 answer deprecated, and warn once per client address.
pub async fn deprecate(request: Request, next: Next) -> Response {
    if !is_v1(request.uri().path()) {
        return next.run(request).await;
    }
    if let Some(ConnectInfo(addr)) = request.extensions().get::<ConnectInfo<SocketAddr>>()
        && first_from(addr.ip())
    {
        tracing::warn!(
            "{} uses protocol 1 (/v1), which is deprecated and may stop working in a future release",
            addr.ip()
        );
    }
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert("deprecation", HeaderValue::from_static("true"));
    headers.insert(header::LINK, HeaderValue::from_static(LINK));
    response
}

/// True the first time `ip` is seen.
fn first_from(ip: IpAddr) -> bool {
    static SEEN: LazyLock<Mutex<Seen>> = LazyLock::new(Mutex::default);
    SEEN.lock().first(ip)
}

/// The most client addresses remembered; past it they are all forgotten.
const SEEN_MAX: usize = 1024;

/// The client addresses already warned about.
#[derive(Default)]
struct Seen(HashSet<IpAddr>);

impl Seen {
    /// True the first time `ip` is seen.
    fn first(&mut self, ip: IpAddr) -> bool {
        if self.0.len() >= SEEN_MAX && !self.0.contains(&ip) {
            self.0.clear();
        }
        self.0.insert(ip)
    }
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let stems = state.queue.info().stems;
    let Json(health) = api::health(State(state)).await;
    Json(health_of(
        serde_json::to_value(health).unwrap_or_default(),
        stems,
    ))
}

/// The 0.2.0 health fields from the protocol 2 health.
fn health_of(mut health: Value, stems: Vec<String>) -> Value {
    let Some(fields) = health.as_object_mut() else {
        return health;
    };
    let output = fields.remove("default_output").unwrap_or_default();
    fields.remove("session");
    fields.remove("segment_run_seconds");
    fields.insert("stems".into(), json!(stems));
    fields.insert("default_output_format".into(), output["format"].clone());
    fields.insert(
        "default_output_sample_rate".into(),
        output["sample_rate"].clone(),
    );
    fields.insert("derived_stem".into(), output["derived"].clone());
    fields.insert("default_parts".into(), output["parts"].clone());
    fields.remove("features");
    health
}

/// A job as 0.2.0 shows it.
#[derive(Debug, Serialize)]
struct JobView {
    id: String,
    progress: Progress,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<JobResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct JobResult {
    sample_rate: u32,
    channels: usize,
    frames: usize,
    format: String,
    stems: Vec<StemFile>,
    model_residual_db: f64,
    derived_stem: Option<&'static str>,
    separation_secs: f64,
    realtime_factor: f64,
    cached: bool,
}

#[derive(Debug, Serialize)]
struct StemFile {
    name: String,
    /// Server-local path of the stem file.
    path: PathBuf,
    url: String,
    bytes: u64,
    gain: f32,
}

/// The 0.2.0 result of job `id`, with the stem paths from `entry`.
fn result_of(id: &str, result: crate::jobs::JobResult, entry: Option<&Entry>) -> JobResult {
    JobResult {
        sample_rate: result.output.sample_rate,
        channels: result.channels,
        frames: result.frames,
        format: result.output.format,
        stems: result
            .stems
            .into_iter()
            .map(|stem| StemFile {
                path: entry
                    .and_then(|e| e.stem(&stem.name))
                    .map(|s| s.path.clone())
                    .unwrap_or_default(),
                url: format!("/v1/jobs/{id}/stems/{}", stem.name),
                name: stem.name,
                bytes: stem.bytes,
                gain: stem.gain,
            })
            .collect(),
        model_residual_db: result.model_residual_db.unwrap_or(FLOOR_DB),
        derived_stem: result.output.derived,
        separation_secs: result.separation_secs,
        // JSON has no infinity, and 0.2.0 clients need a number.
        realtime_factor: if result.realtime_factor.is_finite() {
            result.realtime_factor
        } else {
            0.0
        },
        cached: result.cached,
    }
}

fn job_view(state: &AppState, job: &Job) -> JobView {
    let view = api::live_view(state, job);
    let entry = job.entry.lock().clone();
    JobView {
        result: view
            .result
            .map(|result| result_of(&view.id, result, entry.as_deref())),
        id: view.id,
        progress: view.progress,
        error: view.error,
    }
}

async fn create_job(
    State(state): State<Arc<AppState>>,
    Query(params): Query<CreateParams>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let params = CreateParams {
        file: None,
        ..params
    };
    let (status, job) = api::submit(&state, &params, body).await?;
    job.mark_protocol_1();
    Ok((status, Json(job_view(&state, &job))).into_response())
}

async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let job = state.store.get(&id).ok_or_else(ApiError::not_found)?;
    Ok(Json(job_view(&state, &job)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{call, state};
    use axum::http::Method;

    fn keys(value: &Value) -> Vec<&str> {
        let mut keys: Vec<&str> = value
            .as_object()
            .expect("an object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        keys
    }

    fn sorted(mut keys: Vec<&str>) -> Vec<&str> {
        keys.sort_unstable();
        keys
    }

    #[tokio::test]
    async fn health_has_the_0_2_0_fields() {
        let app = api::router(state());
        let answer = call(&app, Method::GET, "/v1/health", Vec::new()).await;
        assert_eq!(answer.status, StatusCode::OK);
        assert_eq!(answer.headers["deprecation"], "true");
        assert_eq!(answer.headers[header::LINK], LINK);
        assert_eq!(
            keys(&answer.json),
            sorted(vec![
                "version",
                "backend",
                "model",
                "device",
                "sample_rate",
                "channels",
                "stems",
                "model_id",
                "completed_jobs",
                "queue_depth",
                "cached_tracks",
                "cached_bytes",
                "cache_max_bytes",
                "max_track_seconds",
                "output_sample_rates",
                "dsp_modes",
                "dsp_mode_pairs",
                "default_output_format",
                "default_output_sample_rate",
                "derived_stem",
                "parts",
                "default_parts",
            ])
        );
        assert_eq!(answer.json["stems"], json!(["harmonics", "vocals"]));
        assert_eq!(answer.json["derived_stem"], "drums");
        assert_eq!(answer.json["default_parts"], json!(["harmonics", "vocals"]));
        assert_eq!(answer.json["default_output_format"], "flac");
        assert_eq!(answer.json["default_output_sample_rate"], 44_100);

        let v2 = call(&app, Method::GET, "/v2/health", Vec::new()).await;
        assert!(v2.headers.get("deprecation").is_none());
    }

    #[tokio::test]
    async fn a_job_result_has_the_0_2_0_shape() {
        let state = state();
        let app = api::router(Arc::clone(&state));
        let job = state.store.create("track".into());
        job.complete(
            crate::testkit::publish(&state.cache, "track", 100, 1.0),
            false,
        );

        let answer = call(
            &app,
            Method::GET,
            &format!("/v1/jobs/{}", job.id),
            Vec::new(),
        )
        .await;
        assert_eq!(answer.status, StatusCode::OK);
        assert_eq!(keys(&answer.json), ["id", "progress", "result"]);
        let result = &answer.json["result"];
        assert_eq!(
            keys(result),
            sorted(vec![
                "sample_rate",
                "channels",
                "frames",
                "format",
                "stems",
                "model_residual_db",
                "derived_stem",
                "separation_secs",
                "realtime_factor",
                "cached",
            ])
        );
        assert_eq!(result["format"], "pcm16");
        assert_eq!(result["sample_rate"], 44_100);
        assert_eq!(result["derived_stem"], "drums");
        let stem = &result["stems"][0];
        assert_eq!(keys(stem), ["bytes", "gain", "name", "path", "url"]);
        assert_eq!(stem["url"], format!("/v1/jobs/{}/stems/harmonics", job.id));
        assert!(
            std::path::Path::new(stem["path"].as_str().unwrap()).is_file(),
            "{stem}"
        );

        let url = stem["url"].as_str().unwrap();
        let download = call(&app, Method::GET, url, Vec::new()).await;
        assert_eq!(download.status, StatusCode::OK);
        assert_eq!(download.headers["deprecation"], "true");

        let empty = call(&app, Method::POST, "/v1/jobs?format=s16le", Vec::new()).await;
        assert_eq!(empty.status, StatusCode::BAD_REQUEST);
        assert_eq!(empty.headers["deprecation"], "true");
    }

    #[tokio::test]
    async fn a_job_on_stems_a_session_stored_reads_the_floor_residual() {
        use crate::cache::{Consolidation, type_dir};
        use stemd_core::Audio;
        let state = state();
        let app = api::router(Arc::clone(&state));
        let frames = 4410;
        let wave = |scale: f32| -> Vec<f32> {
            (0..frames)
                .map(|i| scale * (i as f32 * 0.01).sin())
                .collect()
        };
        let mix = Audio::new(vec![wave(0.5), wave(0.4)], 44_100);
        let id = crate::sessions::track_id(&mix);
        let output = crate::testkit::output();
        let dir =
            state
                .cache
                .verified_dir(&state.switcher.cache_id(), &type_dir(&output, 1.0), &id);
        let task = Consolidation::at_44100(dir, output.format, frames);
        state.cache.write_info(&task.dir, &task.info).unwrap();
        for (stem, scale) in [("harmonics", 0.2), ("vocals", 0.15), ("drums", 0.1)] {
            let part = Audio::new(vec![wave(scale), wave(scale)], 44_100);
            let bytes = stemd_core::encode_stem(&part, output.format, 1.0).unwrap();
            state
                .cache
                .consolidate(&task, stem, &bytes, frames, Some(0))
                .unwrap();
        }
        let body: Vec<u8> = (0..frames)
            .flat_map(|i| [mix.data[0][i], mix.data[1][i]])
            .flat_map(f32::to_le_bytes)
            .collect();

        let answer = call(
            &app,
            Method::POST,
            "/v1/jobs?format=f32le&output_format=pcm16",
            body,
        )
        .await;
        assert_eq!(answer.status, StatusCode::OK, "{}", answer.json);
        let result = &answer.json["result"];
        assert_eq!(result["cached"], true);
        assert_eq!(result["model_residual_db"].as_f64(), Some(FLOOR_DB));
        assert_eq!(result["realtime_factor"].as_f64(), Some(0.0));

        let id = answer.json["id"].as_str().unwrap();
        let again = call(&app, Method::GET, &format!("/v1/jobs/{id}"), Vec::new()).await;
        let result = &again.json["result"];
        assert_eq!(result["model_residual_db"].as_f64(), Some(FLOOR_DB));
        assert_eq!(result["realtime_factor"].as_f64(), Some(0.0));
    }

    #[tokio::test]
    async fn sessions_and_filters_are_protocol_2_only() {
        let app = api::router(state());
        for (method, path) in [
            (Method::POST, "/v1/sessions?frames=30000"),
            (Method::GET, "/v1/sessions/abc"),
            (Method::PUT, "/v1/sessions/abc/window"),
        ] {
            let answer = call(&app, method, path, Vec::new()).await;
            assert_eq!(answer.status, StatusCode::GONE, "{path}");
            assert_eq!(answer.json["code"], "protocol_1", "{path}");
            assert!(
                answer.json["error"]
                    .as_str()
                    .unwrap()
                    .contains("/v2/sessions"),
                "{path}"
            );
            assert_eq!(answer.headers["deprecation"], "true", "{path}");
        }
        let filter = format!("/v1/filters/{}", "0".repeat(64));
        let answer = call(&app, Method::GET, &filter, Vec::new()).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_v1_answer_is_marked_deprecated() {
        let app = Router::new()
            .route("/v1/health", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(deprecate));
        for path in ["/v1/health", "/v1/nothing"] {
            let answer = call(&app, Method::GET, path, Vec::new()).await;
            assert_eq!(answer.headers["deprecation"], "true", "{path}");
            assert_eq!(answer.headers[header::LINK], LINK, "{path}");
        }
        let answer = call(&app, Method::GET, "/v10/health", Vec::new()).await;
        assert!(answer.headers.get("deprecation").is_none());
    }

    #[test]
    fn a_client_address_is_warned_about_once() {
        let ip = IpAddr::from([198, 51, 100, 1]);
        assert!(first_from(ip));
        assert!(!first_from(ip));
        assert!(first_from(IpAddr::from([198, 51, 100, 2])));
    }

    #[test]
    fn the_warned_addresses_are_forgotten_past_the_cap() {
        let mut seen = Seen::default();
        for n in 0..SEEN_MAX as u32 {
            assert!(seen.first(IpAddr::from(n.to_be_bytes())));
        }
        assert!(!seen.first(IpAddr::from(0u32.to_be_bytes())));
        assert!(seen.first(IpAddr::from([203, 0, 113, 1])));
        assert_eq!(seen.0.len(), 1);
    }
}
