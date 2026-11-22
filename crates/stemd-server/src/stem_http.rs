//! Stem responses and id checks shared by the job, session and separation handlers.

use axum::body::Body;
use axum::http::{self, StatusCode};
use axum::response::Response;
use stemd_core::StemFormat;

use crate::api::error::ApiError;
use crate::ident::TRACK_ID_LEN;
use crate::whole::Whole;

/// A response with the headers that every range and piece carries.
pub fn audio_response(
    format: StemFormat,
    body: Body,
    len: u64,
    from: usize,
    frames: usize,
    gain: f32,
    clipped: Option<usize>,
) -> Response {
    let mut response = Response::builder()
        .header(http::header::CONTENT_TYPE, format.content_type())
        .header(http::header::CONTENT_LENGTH, len)
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

/// Open a stored whole file and return its length; a file deleted since its lookup gets `410`.
pub async fn open_whole(whole: &Whole) -> Result<(tokio::fs::File, u64), ApiError> {
    let path = whole.path.display();
    let file = tokio::fs::File::open(&whole.path).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            evicted()
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
pub async fn send_whole(format: StemFormat, gain: f32, whole: Whole) -> Result<Response, ApiError> {
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

/// `410` for stems evicted while a request read them.
pub fn evicted() -> ApiError {
    ApiError::new(StatusCode::GONE, "these stems have been evicted; ask again")
}

/// A 400 unless `id` has the shape of a track id.
pub fn check_id(id: &str) -> Result<(), ApiError> {
    check_key(id, "track id")
}

/// A 400 unless `key` is a file key: as long as a track id, lowercase hex.
pub fn check_file_key(key: &str) -> Result<(), ApiError> {
    check_key(key, "file key")
}

fn check_key(s: &str, what: &str) -> Result<(), ApiError> {
    if s.len() == TRACK_ID_LEN && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "{s} is not a {what}: {TRACK_ID_LEN} lowercase hex characters"
        )))
    }
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    use super::*;
    use crate::whole::Meta;

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

    #[test]
    fn only_a_track_id_is_looked_up() {
        assert!(check_id("f9c29cbfe3466b674f9903d9").is_ok());
        for bad in [
            "",
            "..",
            "F9C29CBFE3466B674F9903D9",
            "f9c29cbfe3466b674f9903d",
            "f9c29cbfe3466b674f9903d9/",
        ] {
            assert!(check_id(bad).is_err(), "{bad:?}");
        }
    }
}
