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

use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post, put};

use crate::state::AppState;

pub mod error;
pub mod filters;
pub mod health;
pub mod jobs;
pub mod params;

pub use error::ApiError;
use filters::{get_filter, put_filter};
use health::{get_logs, health};
use jobs::{create_job, delete_job, get_job, get_stem, join_job};

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
