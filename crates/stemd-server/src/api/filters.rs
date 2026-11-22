//! The `/v2/filters` handlers.

use std::sync::Arc;

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use crate::state::AppState;

use super::error::ApiError;

/// Store a filter file under its SHA-256. `201` when new, `200` when already stored.
pub(super) async fn put_filter(
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
pub(super) async fn get_filter(
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
