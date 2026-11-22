//! Lookups by track id: what the server holds for one output, each stored stem, and every stem in
//! one download.
//!
//! ```text
//! GET /v2/separations/{track id}?<output params>                state and stems
//! GET /v2/separations/{track id}/stems/{name}?<output params>   one stem, stored whole
//! GET /v2/separations/{track id}/bundle?<output params>         every stem in one file
//! ```
//!
//! The output params are those of `POST /v2/jobs`: `output_format`,
//! `output_sample_rate`, `parts`, `include_derived`, `dsp_mode`, `dsp_filter`.

use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{self, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use stemd_core::StemFormat;
use tokio::io::AsyncReadExt;

use crate::api::error::ApiError;
use crate::api::params::{CreateParams, parse_output};
use crate::held::{Lookup, TypeDir, derivable, derive, lookup, whole_dir};
use crate::output::{Output, sources};
use crate::state::AppState;
use crate::stem_http::{check_file_key, check_id, evicted, open_whole, send_whole};

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v2/separations", get(view_by_file))
        .route("/v2/separations/{id}", get(view))
        .route("/v2/separations/{id}/stems/{name}", get(stem))
        .route("/v2/separations/{id}/bundle", get(bundle))
}

/// The bundle's format code for `format`; `mp3` has none.
pub fn format_code(format: StemFormat) -> Option<u16> {
    match format {
        StemFormat::Flac => Some(0),
        StemFormat::Wav => Some(1),
        StemFormat::Pcm16 => Some(2),
        StemFormat::Pcm32 => Some(3),
        StemFormat::Mp3 => None,
    }
}

/// The number of bytes before the first index entry.
pub const BUNDLE_HEADER: usize = 24;
/// The number of bytes per index entry.
pub const BUNDLE_ENTRY: usize = 32;

/// One stem of a bundle.
pub struct Entry {
    pub sources: u16,
    pub gain: f32,
    pub bytes: u64,
    pub clipped: Option<usize>,
}

/// The header and index of a bundle whose stems follow it in `entries` order.
pub fn bundle_index(
    format: StemFormat,
    frames: usize,
    rate: u32,
    channels: usize,
    entries: &[Entry],
) -> Result<Vec<u8>> {
    let u32_of = |v: usize, what: &str| u32::try_from(v).with_context(|| format!("{what} {v}"));
    let code = format_code(format).with_context(|| format!("{format} has no bundle"))?;
    let mut out = Vec::with_capacity(BUNDLE_HEADER + BUNDLE_ENTRY * entries.len());
    out.extend_from_slice(b"STMB");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&code.to_le_bytes());
    out.extend_from_slice(&u32_of(entries.len(), "stems")?.to_le_bytes());
    out.extend_from_slice(&u32_of(frames, "frames")?.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&u32_of(channels, "channels")?.to_le_bytes());
    let mut offset = (BUNDLE_HEADER + BUNDLE_ENTRY * entries.len()) as u64;
    for entry in entries {
        let clipped = match entry.clipped {
            Some(n) => u32_of(n, "clipped samples")?,
            None => u32::MAX,
        };
        out.extend_from_slice(&entry.sources.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&entry.gain.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&entry.bytes.to_le_bytes());
        out.extend_from_slice(&clipped.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        offset += entry.bytes;
    }
    Ok(out)
}

fn unknown(id: &str) -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        format!("no separation of {id} for this output; open a session or submit a job"),
    )
    .with_code("unknown_separation")
}

/// The output `params` ask for. `mp3` is stored for jobs only, so it is refused.
pub(crate) fn output_of(state: &AppState, params: &CreateParams) -> Result<Output, ApiError> {
    let (_, output, _) = parse_output(params, state)?;
    if output.format == StemFormat::Mp3 {
        return Err(ApiError::bad_request(
            "mp3 is served to jobs only; ask for flac, wav, s16le or f32le",
        ));
    }
    Ok(output)
}

async fn view(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(params): Query<CreateParams>,
) -> Result<Response, ApiError> {
    check_id(&id)?;
    answer_for(&state, id, &params).await
}

/// The lookup for the track that a file key names; an unknown or ambiguous key gets
/// `unknown_separation`.
async fn view_by_file(
    State(state): State<Arc<AppState>>,
    Query(params): Query<CreateParams>,
) -> Result<Response, ApiError> {
    let key = params
        .file
        .clone()
        .ok_or_else(|| ApiError::bad_request("name a track id in the path, or file="))?;
    check_file_key(&key)?;
    let id = state.cache.file_track(&key).ok_or_else(|| unknown(&key))?;
    answer_for(&state, id, &params).await
}

/// The lookup answer for track `id` and the output `params` name.
async fn answer_for(
    state: &Arc<AppState>,
    id: String,
    params: &CreateParams,
) -> Result<Response, ApiError> {
    let output = output_of(state, params)?;
    let model_id = state.switcher.cache_id();
    let (cache, sessions) = (Arc::clone(&state.cache), Arc::clone(state.queue.sessions()));
    let jobs = Arc::clone(&state.jobs);
    let found = tokio::task::spawn_blocking(move || {
        lookup(&cache, &sessions, &jobs, &model_id, &output, &id).ok_or_else(|| unknown(&id))
    })
    .await
    .map_err(|e| ApiError::internal(format!("looking up the separation: {e}")))??;
    let found = Lookup {
        model_id: state.switcher.published_identity(),
        ..found
    };
    Ok(Json(found).into_response())
}

/// The stored output `params` ask for, converted first when it is derivable.
async fn resolve(
    state: &Arc<AppState>,
    id: String,
    params: &CreateParams,
) -> Result<TypeDir, ApiError> {
    check_id(&id)?;
    let output = output_of(state, params)?;
    let model_id = state.switcher.cache_id();
    let (cache, sessions) = (Arc::clone(&state.cache), Arc::clone(state.queue.sessions()));
    let jobs = Arc::clone(&state.jobs);
    tokio::task::spawn_blocking(move || {
        if let Some(best) = whole_dir(&cache, &model_id, &output, &id) {
            return Ok(best);
        }
        if let Some((source, gain)) = derivable(&cache, &model_id, &output, &id) {
            return derive(&cache, &model_id, &id, &source, &output, gain)
                .map_err(|e| ApiError::internal(format!("converting: {e:#}")));
        }
        match lookup(&cache, &sessions, &jobs, &model_id, &output, &id) {
            Some(held) => Err(ApiError::new(
                StatusCode::CONFLICT,
                format!("{id}: expected stored, got {}", held.state.as_str()),
            )),
            None => Err(unknown(&id)),
        }
    })
    .await
    .map_err(|e| ApiError::internal(format!("resolving the separation: {e}")))?
}

async fn stem(
    State(state): State<Arc<AppState>>,
    Path((id, name)): Path<(String, String)>,
    Query(params): Query<CreateParams>,
) -> Result<Response, ApiError> {
    let found = resolve(&state, id, &params).await?;
    if !found.output.served().contains(&name.as_str()) {
        return Err(ApiError::not_found());
    }
    let whole = found.whole(&name).ok_or_else(evicted)?;
    send_whole(found.output.format, found.gain, whole).await
}

async fn bundle(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(params): Query<CreateParams>,
) -> Result<Response, ApiError> {
    let found = resolve(&state, id, &params).await?;
    let mut files = Vec::new();
    let mut entries = Vec::new();
    let mut frames = 0;
    for name in found.output.served() {
        let whole = found.whole(name).ok_or_else(evicted)?;
        let (file, bytes) = open_whole(&whole).await?;
        frames = whole.meta.frames;
        entries.push(Entry {
            sources: sources(name),
            gain: found.gain,
            bytes,
            clipped: whole.meta.clipped,
        });
        files.push(file);
    }
    let index = bundle_index(
        found.output.format,
        frames,
        found.output.rate.hz(),
        found.info.channels,
        &entries,
    )
    .map_err(|e| ApiError::internal(format!("{e:#}")))?;
    let len = index.len() as u64 + entries.iter().map(|e| e.bytes).sum::<u64>();
    let mut reader: Box<dyn tokio::io::AsyncRead + Send + Unpin> =
        Box::new(std::io::Cursor::new(index));
    for file in files {
        reader = Box::new(reader.chain(file));
    }
    Response::builder()
        .header(http::header::CONTENT_TYPE, "application/octet-stream")
        .header(http::header::CONTENT_LENGTH, len)
        .header("x-stemd-frames", frames)
        .body(Body::from_stream(tokio_util::io::ReaderStream::new(reader)))
        .map_err(|e| ApiError::internal(format!("{e}")))
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn a_part_holds_its_sources_as_bits() {
        assert_eq!(sources("drums"), 1);
        assert_eq!(sources("bass"), 2);
        assert_eq!(sources("other"), 4);
        assert_eq!(sources("vocals"), 8);
        assert_eq!(sources("harmonics"), sources("bass") | sources("other"));
        assert_eq!(sources("guitar"), 0);
    }

    #[test]
    fn the_bundle_index_places_each_stem_after_the_last() {
        let entries = [
            Entry {
                sources: 6,
                gain: 0.5,
                bytes: 100,
                clipped: Some(3),
            },
            Entry {
                sources: 8,
                gain: 0.5,
                bytes: 40,
                clipped: None,
            },
        ];
        let index = bundle_index(StemFormat::Flac, 1000, 96_000, 2, &entries).unwrap();
        assert_eq!(index.len(), BUNDLE_HEADER + 2 * BUNDLE_ENTRY);
        assert_eq!(&index[0..4], b"STMB");
        let u16_at = |at: usize| u16::from_le_bytes([index[at], index[at + 1]]);
        let u32_at = |at: usize| u32::from_le_bytes(index[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(index[at..at + 8].try_into().unwrap());
        assert_eq!(u16_at(4), 1, "version");
        assert_eq!(u16_at(6), 0, "flac");
        assert_eq!(u32_at(8), 2, "stems");
        assert_eq!(u32_at(12), 1000, "frames");
        assert_eq!(u32_at(16), 96_000, "rate");
        assert_eq!(u32_at(20), 2, "channels");
        let (a, b) = (BUNDLE_HEADER, BUNDLE_HEADER + BUNDLE_ENTRY);
        assert_eq!(u16_at(a), 6);
        assert_eq!(
            f32::from_le_bytes(index[a + 4..a + 8].try_into().unwrap()),
            0.5
        );
        assert_eq!(
            u64_at(a + 8),
            index.len() as u64,
            "the first stem follows the index"
        );
        assert_eq!(u64_at(a + 16), 100);
        assert_eq!(u32_at(a + 24), 3);
        assert_eq!(u16_at(b), 8);
        assert_eq!(u64_at(b + 8), index.len() as u64 + 100);
        assert_eq!(u32_at(b + 24), u32::MAX, "no clip count");
        assert!(bundle_index(StemFormat::Mp3, 1000, 44_100, 2, &entries).is_err());
    }
}
