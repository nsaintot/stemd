//! The JSON error every handler answers with.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

#[derive(Debug)]
pub struct ApiError {
    pub(super) status: StatusCode,
    pub(super) message: String,
    /// A stable name for an error a client acts on, sent as `code`.
    pub(super) code: Option<&'static str>,
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
