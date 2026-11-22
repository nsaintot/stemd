//! Protocol 1 under `/v1`, which is deprecated and may stop working in a future release.
//!
//! `/v1` keeps the 0.2.0 health and jobs routes. The handlers translate requests and answers to and
//! from protocol 2, so both protocols share every job and stored stem. Sessions are protocol 2
//! only: `/v1/sessions` answers `410`. "Protocol 1 (deprecated)" in `docs/api.md` lists the
//! differences.
//!
//! Every answer under `/v1` carries `Deprecation: true` and a `Link` to the
//! API reference. The first request from each client address logs a warning.

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

use axum::body::Bytes;
use axum::extract::{ConnectInfo, Path, Query, Request, State};
use axum::http::{self, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{Value, json};
use stemd_core::Progress;

use crate::api::error::ApiError;
use crate::api::health::get_logs;
use crate::api::jobs::{delete_job, get_stem, live_view, submit};
use crate::api::params::CreateParams;
use crate::jobs::{Entry, Job};
use crate::state::AppState;

/// The residual sent when the stems sum to the mix exactly, since JSON has no `-inf`.
pub(crate) const FLOOR_DB: f64 = -200.0;

/// The `Link` header on every protocol 1 answer.
pub(crate) const LINK: &str = "<https://github.com/nsaintot/stemd/blob/main/docs/api.md>; rel=\"deprecation\"; type=\"text/html\"";

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/jobs", post(create_job))
        .route("/v1/jobs/{id}", get(get_job).delete(delete_job))
        .route("/v1/jobs/{id}/stems/{name}", get(get_stem))
        .route("/v1/logs", get(get_logs))
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
    headers.insert(http::header::LINK, HeaderValue::from_static(LINK));
    response
}

/// True the first time `ip` is seen.
fn first_from(ip: IpAddr) -> bool {
    static SEEN: LazyLock<Mutex<Seen>> = LazyLock::new(Mutex::default);
    SEEN.lock().first(ip)
}

/// The most client addresses remembered; past this count, the set is cleared.
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
    let Json(health) = crate::api::health::health(State(state)).await;
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
    /// The server-local path of the stem file.
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
    let view = live_view(state, job);
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
    let (status, job) = submit(&state, &params, body).await?;
    job.mark_protocol_1();
    Ok((status, Json(job_view(&state, &job))).into_response())
}

async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let job = state.jobs.get(&id).ok_or_else(ApiError::not_found)?;
    Ok(Json(job_view(&state, &job)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::call;
    use axum::http::Method;

    #[tokio::test]
    async fn a_v1_answer_is_marked_deprecated() {
        let app = Router::new()
            .route("/v1/health", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(deprecate));
        for path in ["/v1/health", "/v1/nothing"] {
            let answer = call(&app, Method::GET, path, Vec::new()).await;
            assert_eq!(answer.headers["deprecation"], "true", "{path}");
            assert_eq!(answer.headers[http::header::LINK], LINK, "{path}");
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
