//! Separations by track id: what the server holds for one output, its stems,
//! and every stem in one download.
//!
//! ```text
//! GET /v2/separations/{track id}?<output params>                state and stems
//! GET /v2/separations/{track id}/stems/{name}?<output params>   one stem, stored whole
//! GET /v2/separations/{track id}/bundle?<output params>         every stem in one file
//! ```
//!
//! The output params are those of `POST /v2/jobs`: `output_format`,
//! `output_sample_rate`, `parts`, `include_derived`, `dsp_mode`, `dsp_filter`.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde::Serialize;
use stemd_core::StemFormat;
use tokio::io::AsyncReadExt;

use crate::api::{ApiError, AppState, CreateParams, parse_output};
use crate::cache::{Cache, Consolidation, TrackInfo, stems_of, type_dir, type_prefix};
use crate::jobs::{Jobs, MixShape, job_key};
use crate::output::Output;
use crate::session_api::{open_whole, send_whole};
use crate::sessions::{Sessions, TRACK_ID_LEN, gain, served};
use crate::whole::Whole;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v2/separations", get(view_by_file))
        .route("/v2/separations/{id}", get(view))
        .route("/v2/separations/{id}/stems/{name}", get(stem))
        .route("/v2/separations/{id}/bundle", get(bundle))
}

/// What the server holds for one track and output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Held {
    /// Every stem is stored whole.
    Stored,
    /// Every stem is stored whole in another container of the same samples.
    Derivable,
    /// Every piece is stored, and the whole files are being written.
    Consolidating,
    /// A job for this output is running; `job` is its id.
    Separating,
    /// The mix is in memory: a session can join it without an upload.
    Open,
    /// Some pieces are stored, and the mix is not in memory.
    Partial,
}

/// How a stem is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Storage {
    Whole,
    Pieces,
    None,
}

/// The output stems are in, as sessions, jobs and lookups report it.
#[derive(Debug, Clone, Serialize)]
pub struct OutputView {
    pub format: String,
    pub sample_rate: u32,
    /// Multiply by `1.0 / gain`; absent for an open track and for a job, whose stems carry it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gain: Option<f32>,
    pub parts: Vec<String>,
    /// The part the client rebuilds as `mix` minus the parts, when one is left out.
    pub derived: Option<&'static str>,
}

impl OutputView {
    pub fn of(output: &Output, gain: Option<f32>) -> Self {
        Self {
            format: output.format.to_string(),
            sample_rate: output.rate.hz(),
            gain,
            parts: output.parts.names().map(str::to_owned).collect(),
            derived: output.parts.derived(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StemState {
    pub name: &'static str,
    /// The sources the stem holds, see [`sources`].
    pub sources: u16,
    pub stored: Storage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
}

/// The answer to `GET /v2/separations/{id}`.
#[derive(Debug, Clone, Serialize)]
pub struct Lookup {
    pub track: String,
    pub model_id: String,
    /// Track length at the model's rate.
    pub frames: usize,
    pub sample_rate: u32,
    pub channels: usize,
    pub output: OutputView,
    pub state: Held,
    /// The stored output a `derivable` one is converted from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<OutputView>,
    /// The running job, when `state` is `separating`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job: Option<String>,
    pub stems: Vec<StemState>,
}

/// Bits of [`sources`]: `drums`, `bass`, `other`, `vocals`.
const SOURCE_BITS: [(&str, u16); 4] = [("drums", 1), ("bass", 2), ("other", 4), ("vocals", 8)];

/// The sources a part holds, one bit each: `1` drums, `2` bass, `4` other,
/// `8` vocals. `harmonics` is `bass` and `other`, so `6`.
pub fn sources(part: &str) -> u16 {
    match part {
        "harmonics" => 2 | 4,
        name => SOURCE_BITS
            .iter()
            .find(|(n, _)| *n == name)
            .map_or(0, |(_, bit)| *bit),
    }
}

/// A track directory of one output at one gain.
#[derive(Debug, Clone)]
pub struct Found {
    pub dir: PathBuf,
    pub output: Output,
    pub gain: f32,
    pub info: TrackInfo,
}

impl Found {
    /// The whole file of `stem`, when it holds the whole track.
    pub fn whole(&self, stem: &str) -> Option<Whole> {
        let frames = self.info.frames_at(self.output.rate.hz());
        crate::whole::find_track(&self.dir, stem, self.output.format.extension(), frames)
    }

    fn all_whole(&self) -> bool {
        stems_of(&self.output)
            .iter()
            .all(|s| self.whole(s).is_some())
    }

    fn consolidation(&self) -> Consolidation {
        Consolidation {
            dir: self.dir.clone(),
            format: self.output.format,
            rate: self.output.rate.hz(),
            info: self.info,
            stems: stems_of(&self.output),
        }
    }

    fn storage(&self, stem: &str) -> (Storage, Option<u64>) {
        if let Some(whole) = self.whole(stem) {
            let bytes = std::fs::metadata(&whole.path).map(|m| m.len()).ok();
            return (Storage::Whole, bytes);
        }
        let any_piece = std::fs::read_dir(self.dir.join(stem))
            .map(|mut entries| entries.next().is_some())
            .unwrap_or(false);
        let storage = if any_piece {
            Storage::Pieces
        } else {
            Storage::None
        };
        (storage, None)
    }

    /// True when any stem has a piece or a whole file.
    fn holds_any(&self) -> bool {
        stems_of(&self.output)
            .iter()
            .any(|s| self.storage(s).0 != Storage::None)
    }
}

/// `output`'s track directories for `id` at any gain: all whole, then all pieces, then the rest.
fn found(cache: &Cache, model_id: &str, output: &Output, id: &str) -> Vec<Found> {
    let prefix = type_prefix(output);
    let mut out: Vec<Found> = cache
        .types(model_id)
        .into_iter()
        .filter_map(|name| {
            let gain: f32 = name.strip_prefix(&prefix)?.parse().ok()?;
            let dir = cache.verified_dir(model_id, &name, id);
            let info = Cache::info(&dir)?;
            Some(Found {
                dir,
                output: output.clone(),
                gain,
                info,
            })
        })
        .collect();
    out.sort_by_key(|f| {
        if f.all_whole() {
            0
        } else if Cache::needs_consolidation(&f.consolidation()) {
            1
        } else {
            2
        }
    });
    out
}

/// The outputs whose whole files convert to `output`'s samples, each with the gain it gives.
fn sources_of(output: &Output) -> Vec<(Output, Option<f32>)> {
    if output.format.has_headroom() {
        return Vec::new();
    }
    let with = |format: StemFormat| Output {
        format,
        ..output.clone()
    };
    let mut out: Vec<(Output, Option<f32>)> =
        [StemFormat::Flac, StemFormat::Wav, StemFormat::Pcm16]
            .into_iter()
            .filter(|f| *f != output.format)
            .map(|f| (with(f), None))
            .collect();
    // A 16-bit read of `pcm32` samples, as a session without `peak` makes it.
    out.push((with(StemFormat::Pcm32), Some(gain(output, None))));
    out
}

/// A track directory of `output` for `id` with every stem stored whole, at any gain.
pub fn whole_dir(cache: &Cache, model_id: &str, output: &Output, id: &str) -> Option<Found> {
    let best = found(cache, model_id, output, id)
        .into_iter()
        .find(Found::all_whole)?;
    Cache::touch(&best.dir);
    Some(best)
}

/// A stored output that converts to `output`, and the gain the result has.
fn derivable(cache: &Cache, model_id: &str, output: &Output, id: &str) -> Option<(Found, f32)> {
    sources_of(output).into_iter().find_map(|(source, gain)| {
        let found = found(cache, model_id, &source, id)
            .into_iter()
            .find(Found::all_whole)?;
        let gain = gain.unwrap_or(found.gain);
        Some((found, gain))
    })
}

/// What the server holds for `id` at `output`, queueing consolidation once every piece is stored.
pub fn lookup(
    cache: &Arc<Cache>,
    sessions: &Sessions,
    jobs: &Jobs,
    model_id: &str,
    output: &Output,
    id: &str,
) -> Option<Lookup> {
    let served = served(output);
    let stem_states = |found: Option<&Found>| -> Vec<StemState> {
        served
            .iter()
            .map(|&name| {
                let (stored, bytes) = found.map_or((Storage::None, None), |f| f.storage(name));
                StemState {
                    name,
                    sources: sources(name),
                    stored,
                    bytes,
                }
            })
            .collect()
    };
    let answer = |state: Held, found: Option<&Found>, mix: MixShape, gain: Option<f32>| Lookup {
        track: id.to_owned(),
        model_id: model_id.to_owned(),
        frames: mix.frames,
        sample_rate: mix.sample_rate,
        channels: mix.channels,
        output: OutputView::of(output, gain),
        state,
        from: None,
        job: None,
        stems: stem_states(found),
    };

    let exact = found(cache, model_id, output, id);
    if let Some(best) = exact.first() {
        Cache::touch(&best.dir);
        if best.all_whole() {
            return Some(answer(
                Held::Stored,
                Some(best),
                best.info.into(),
                Some(best.gain),
            ));
        }
        let task = best.consolidation();
        if Cache::needs_consolidation(&task) && cache.consolidate_later(task) {
            return Some(answer(
                Held::Consolidating,
                Some(best),
                best.info.into(),
                Some(best.gain),
            ));
        }
    }
    if let Some((source, gain)) = derivable(cache, model_id, output, id) {
        Cache::touch(&source.dir);
        let mut lookup = answer(Held::Derivable, None, source.info.into(), Some(gain));
        lookup.from = Some(OutputView::of(&source.output, Some(source.gain)));
        return Some(lookup);
    }
    if let Some(job) = jobs.running(&job_key(id, output, model_id)) {
        let mut lookup = answer(Held::Separating, exact.first(), job.mix, None);
        lookup.job = Some(job.id.clone());
        return Some(lookup);
    }
    if let Some(track) = sessions.open_track(id, output.parts) {
        let info = TrackInfo::from(&track.spec);
        return Some(answer(Held::Open, exact.first(), info.into(), None));
    }
    let best = exact.iter().find(|f| f.holds_any())?;
    Some(answer(
        Held::Partial,
        Some(best),
        best.info.into(),
        Some(best.gain),
    ))
}

/// Convert `source` into `output` for track `id`, skipping stems already stored.
pub fn derive(
    cache: &Cache,
    model_id: &str,
    id: &str,
    source: &Found,
    output: &Output,
    gain: f32,
) -> Result<Found> {
    let dir = cache.verified_dir(model_id, &type_dir(output, gain), id);
    // A second request for the same output waits here, then finds the stems stored.
    let converting = cache.converting(&dir);
    let _held = converting.lock();
    cache.write_info(&dir, &source.info)?;
    let target = Found {
        dir,
        output: output.clone(),
        gain,
        info: source.info,
    };
    let rate = output.rate.hz();
    for stem in stems_of(output) {
        if target.whole(stem).is_some() {
            continue;
        }
        let whole = source
            .whole(stem)
            .with_context(|| format!("{stem} is not stored whole"))?;
        let bytes = std::fs::read(&whole.path)
            .with_context(|| format!("reading {}", whole.path.display()))?;
        let audio = crate::cache::decode(bytes, source.output.format, source.info.channels, rate)
            .with_context(|| format!("decoding {}", whole.path.display()))?;
        let (bytes, clipped) = if source.output.format.has_headroom() {
            (
                stemd_core::encode_stem(&audio, output.format, gain)?,
                Some(crate::whole::clipped(&audio, gain, false)),
            )
        } else {
            (
                stemd_core::encode_stem(&audio, output.format, 1.0)?,
                whole.meta.clipped,
            )
        };
        cache.consolidate(
            &target.consolidation(),
            stem,
            &bytes,
            audio.frames(),
            clipped,
        )?;
    }
    tracing::info!(track = %id, "converted {} to {}", source.output.format, output.format);
    Ok(target)
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

/// Bytes before the first index entry.
pub const BUNDLE_HEADER: usize = 24;
/// Bytes per index entry.
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

/// A 400 unless `id` has the shape of a track id.
fn check_id(id: &str) -> Result<(), ApiError> {
    if is_key(id) {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "{id} is not a track id: {TRACK_ID_LEN} lowercase hex characters"
        )))
    }
}

/// Fail unless `key` is a file key: as long as a track id, lowercase hex.
pub fn check_file_key(key: &str) -> Result<(), ApiError> {
    if is_key(key) {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "{key} is not a file key: {TRACK_ID_LEN} lowercase hex characters"
        )))
    }
}

fn is_key(s: &str) -> bool {
    s.len() == TRACK_ID_LEN && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn unknown(id: &str) -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        format!("no separation of {id} for this output; open a session or submit a job"),
    )
    .with_code("unknown_separation")
}

/// The output `params` ask for. `mp3` is stored for jobs only, so it is refused.
fn output_of(state: &AppState, params: &CreateParams) -> Result<Output, ApiError> {
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

/// The lookup of the track a file key names; an unknown or ambiguous key is `unknown_separation`.
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
) -> Result<Found, ApiError> {
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
                format!(
                    "{id} is {} for this output, not stored whole",
                    serde_json::to_value(held.state)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_owned))
                        .unwrap_or_default()
                ),
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
    if !served(&found.output).contains(&name.as_str()) {
        return Err(ApiError::not_found());
    }
    let whole = found.whole(&name).ok_or_else(evicted)?;
    send_whole(found.output.format, found.gain, whole).await
}

/// `410` for stems evicted while a request read them.
pub(crate) fn evicted() -> ApiError {
    ApiError::new(StatusCode::GONE, "these stems have been evicted; ask again")
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
    for name in served(&found.output) {
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
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header(header::CONTENT_LENGTH, len)
        .header("x-stemd-frames", frames)
        .body(Body::from_stream(tokio_util::io::ReaderStream::new(reader)))
        .map_err(|e| ApiError::internal(format!("{e}")))
}

#[cfg(test)]
mod tests {
    use axum::http::Method;
    use serde_json::json;

    use super::*;
    use crate::testkit::{Answer, call, state};

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

    const TRACK: &str = "f9c29cbfe3466b674f9903d9";
    const FILE: &str = "3b0c55e19a7d42f08c6e1d2a";

    /// The output a request with `query` names.
    fn output_for(state: &AppState, query: serde_json::Value) -> Output {
        let params: CreateParams = serde_json::from_value(query).expect("params");
        output_of(state, &params).expect("an output")
    }

    /// A queued job for `TRACK` at `output`, with `FILE` naming the track.
    fn queued_job(state: &AppState, output: &Output) -> Arc<crate::jobs::Job> {
        let key = job_key(TRACK, output, &state.switcher.cache_id());
        let mix = MixShape {
            frames: 44_100,
            sample_rate: 44_100,
            channels: 2,
        };
        let Ok(job) = state.jobs.claim(&key, mix) else {
            panic!("a job already holds {key}");
        };
        state.cache.record_file(FILE, TRACK).expect("recorded");
        job
    }

    /// `GET /v2/separations?file=FILE` for 16-bit stems.
    async fn by_file(app: &Router) -> Answer {
        let uri = format!("/v2/separations?file={FILE}&output_format=s16le");
        call(app, Method::GET, &uri, Vec::new()).await
    }

    #[tokio::test]
    async fn a_lookup_by_file_key_while_a_job_runs_answers_its_job() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let job = queued_job(
            &state,
            &output_for(&state, json!({"output_format": "s16le"})),
        );
        let answer = by_file(&app).await;
        assert_eq!(answer.status, StatusCode::OK, "{}", answer.json);
        assert_eq!(answer.json["state"], "separating");
        assert_eq!(answer.json["job"], job.id.as_str());
        assert_eq!(answer.json["track"], TRACK);
        assert_eq!(answer.json["frames"], 44_100);
        assert!(
            answer.json["output"].get("gain").is_none(),
            "{}",
            answer.json
        );
    }

    #[tokio::test]
    async fn a_lookup_after_the_job_ends_answers_stored_without_a_job() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let body: Vec<u8> = (0..44_100 * 2)
            .flat_map(|i: i32| (((i * 37) % 2000 - 1000) as i16 * 8).to_le_bytes())
            .collect();
        let uri = format!("/v2/jobs?file={FILE}&format=s16le&output_format=s16le");
        let posted = call(&app, Method::POST, &uri, body).await;
        assert!(posted.status.is_success(), "{}", posted.json);
        let id = posted.json["id"].as_str().expect("a job id").to_owned();
        let early = by_file(&app).await;
        assert!(
            early.json["state"] == "stored" || early.json["job"] == id.as_str(),
            "{}",
            early.json
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let job = call(&app, Method::GET, &format!("/v2/jobs/{id}"), Vec::new()).await;
            let stage = &job.json["progress"]["stage"];
            if stage == "done" {
                break;
            }
            assert!(stage != "failed", "{}", job.json);
            assert!(std::time::Instant::now() < deadline, "the job never ended");
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let answer = by_file(&app).await;
        assert_eq!(answer.json["state"], "stored", "{}", answer.json);
        assert!(answer.json.get("job").is_none(), "{}", answer.json);
    }

    #[tokio::test]
    async fn a_running_job_for_another_output_is_not_reported() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        queued_job(
            &state,
            &output_for(&state, json!({"output_format": "flac"})),
        );
        let answer = by_file(&app).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND, "{}", answer.json);
    }

    #[tokio::test]
    async fn a_failed_job_is_no_longer_reported_as_separating() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let job = queued_job(
            &state,
            &output_for(&state, json!({"output_format": "s16le"})),
        );
        assert_eq!(by_file(&app).await.json["state"], "separating");
        job.fail("the model failed");
        let answer = by_file(&app).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND, "{}", answer.json);
    }

    #[tokio::test]
    async fn after_a_join_only_the_last_delete_cancels_the_job() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let job = queued_job(
            &state,
            &output_for(&state, json!({"output_format": "s16le"})),
        );
        let uri = format!("/v2/jobs/{}", job.id);
        let joined = call(&app, Method::POST, &format!("{uri}/join"), Vec::new()).await;
        assert_eq!(joined.status, StatusCode::OK, "{}", joined.json);

        let submitter = call(&app, Method::DELETE, &uri, Vec::new()).await;
        assert_eq!(submitter.status, StatusCode::NO_CONTENT);
        let running = call(&app, Method::GET, &uri, Vec::new()).await;
        assert_eq!(running.status, StatusCode::OK, "{}", running.json);
        assert_eq!(by_file(&app).await.json["state"], "separating");

        let joiner = call(&app, Method::DELETE, &uri, Vec::new()).await;
        assert_eq!(joiner.status, StatusCode::NO_CONTENT);
        let gone = call(&app, Method::GET, &uri, Vec::new()).await;
        assert_eq!(gone.status, StatusCode::NOT_FOUND, "{}", gone.json);
        assert_eq!(by_file(&app).await.status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_join_on_an_unknown_job_is_not_found() {
        let app = crate::api::router(state());
        let answer = call(&app, Method::POST, "/v2/jobs/nope-1/join", Vec::new()).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND, "{}", answer.json);
    }

    #[tokio::test]
    async fn a_join_answers_the_job_view() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let job = queued_job(
            &state,
            &output_for(&state, json!({"output_format": "s16le"})),
        );
        let uri = format!("/v2/jobs/{}/join", job.id);
        let answer = call(&app, Method::POST, &uri, Vec::new()).await;
        assert_eq!(answer.status, StatusCode::OK, "{}", answer.json);
        assert_eq!(answer.json["id"], job.id.as_str());
        assert_eq!(
            answer.json["progress"]["stage"], "queued",
            "{}",
            answer.json
        );
    }
}
