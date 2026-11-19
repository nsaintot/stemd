//! Sessions: tracks separated in pieces, in the order their clients ask for.
//!
//! A [`Track`] holds one track's audio and what has been separated of it. A
//! [`Session`] is one client's view of a track, a deck: its output format, its
//! window around the playhead and its idle timer.
//!
//! A session starts on a track of its own. When its whole upload has arrived the
//! server hashes it into a [`track_id`]; if another session's track already has
//! that id, the session moves onto it, so two decks playing one song share the
//! work and the memory from then on.
//!
//! | Track field | Meaning                                                  |
//! |-------------|----------------------------------------------------------|
//! | `arrived`   | track samples uploaded so far                            |
//! | `ready`     | samples whose every covering segment has run             |
//! | `verdict`   | the track id, once the whole upload has been hashed      |
//!
//! Samples are counted at the model's rate throughout. Piece `k` is the samples
//! `[k * stride, (k + 1) * stride)`.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use stemd_core::{Audio, Grid, Normalisation, Parts, PcmFormat, Stems};

use crate::cache::Output;
use crate::pieces::{Consolidation, Pieces, TrackInfo, stems_of, type_dir};
use crate::whole::Whole;

/// Samples of context resampled either side of a range read at another rate.
pub const RESAMPLE_CONTEXT: usize = 8192;

/// Hex characters of a track id.
pub const TRACK_ID_LEN: usize = 24;

/// Changes when the track id's definition does, so old ids stop matching.
const TRACK_ID_EPOCH: &str = "stemd-track-1";

/// A track's identity: SHA-256 over its samples at the model's rate.
///
/// Length-prefixed fields `stemd-track-1`, the rate and the channel count, then
/// every frame's samples as little-endian f32, channel by channel within a frame.
/// The first [`TRACK_ID_LEN`] hex characters.
pub fn track_id(audio: &Audio) -> String {
    let mut hasher = Sha256::new();
    for field in [
        TRACK_ID_EPOCH.to_owned(),
        audio.sample_rate.to_string(),
        audio.channels().to_string(),
    ] {
        hasher.update(u32::try_from(field.len()).unwrap_or(u32::MAX).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    let mut buffer = Vec::with_capacity(1 << 16);
    for i in 0..audio.frames() {
        for channel in &audio.data {
            buffer.extend_from_slice(&channel[i].to_le_bytes());
        }
        if buffer.len() >= 1 << 16 {
            hasher.update(&buffer);
            buffer.clear();
        }
    }
    hasher.update(&buffer);
    let mut id = format!("{:x}", hasher.finalize());
    id.truncate(TRACK_ID_LEN);
    id
}

/// Priorities below this run before whole-track jobs, the rest when no job waits.
pub const BEFORE_JOBS: u8 = 50;

/// The highest priority a region may ask for.
pub const MAX_PRIORITY: u8 = 99;

/// The most regions one window may hold.
pub const MAX_REGIONS: usize = 32;

/// Which planned segments the worker takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Priorities below [`BEFORE_JOBS`]: ahead of whole-track jobs.
    BeforeJobs,
    /// The rest: when no job is waiting.
    WhenIdle,
}

/// Samples around `at` a client wants separated, and how urgently: `0` first.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Region {
    /// Track sample the region is centred on, such as the needle or a cue.
    pub at: usize,
    #[serde(default)]
    pub ahead: usize,
    #[serde(default)]
    pub behind: usize,
    pub priority: u8,
}

/// What a client wants separated next. Replaces the session's last window.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WindowRequest {
    pub regions: Vec<Region>,
}

impl WindowRequest {
    /// Fail on too many regions or a priority past [`MAX_PRIORITY`].
    pub fn check(&self) -> Result<()> {
        if self.regions.len() > MAX_REGIONS {
            bail!(
                "{} regions; a window holds up to {MAX_REGIONS}",
                self.regions.len()
            );
        }
        if let Some(region) = self.regions.iter().find(|r| r.priority > MAX_PRIORITY) {
            bail!("priority {} is past {MAX_PRIORITY}", region.priority);
        }
        Ok(())
    }
}

/// Where a planned segment stands: the lower runs first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rank {
    pub priority: u8,
    /// Samples from `at`: `0` under it, `2d + 1` for `d` samples behind, so ahead wins a tie.
    pub distance: usize,
}

impl Rank {
    pub const fn class(self) -> Class {
        if self.priority < BEFORE_JOBS {
            Class::BeforeJobs
        } else {
            Class::WhenIdle
        }
    }
}

/// Sorted, disjoint, half-open sample ranges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Spans(Vec<Range<usize>>);

impl Spans {
    /// Add `range`, merging it with any it touches.
    pub fn insert(&mut self, range: Range<usize>) {
        if range.is_empty() {
            return;
        }
        let (mut start, mut end) = (range.start, range.end);
        self.0.retain(|r| {
            if r.end < start || r.start > end {
                return true;
            }
            start = start.min(r.start);
            end = end.max(r.end);
            false
        });
        let at = self.0.partition_point(|r| r.start < start);
        self.0.insert(at, start..end);
    }

    /// True when one span holds all of `range`.
    pub fn covers(&self, range: &Range<usize>) -> bool {
        range.is_empty()
            || self
                .span_of(range.start)
                .is_some_and(|r| r.end >= range.end)
    }

    /// The span holding sample `i`.
    pub fn span_of(&self, i: usize) -> Option<&Range<usize>> {
        let at = self.0.partition_point(|r| r.end <= i);
        self.0.get(at).filter(|r| r.start <= i)
    }

    /// The parts of `range` already in a span.
    pub fn overlaps(&self, range: &Range<usize>) -> Vec<Range<usize>> {
        self.0
            .iter()
            .map(|r| r.start.max(range.start)..r.end.min(range.end))
            .filter(|r| !r.is_empty())
            .collect()
    }

    pub fn as_pairs(&self) -> Vec<[usize; 2]> {
        self.0.iter().map(|r| [r.start, r.end]).collect()
    }

    /// Samples in all spans together.
    pub fn total(&self) -> usize {
        self.0.iter().map(|r| r.end - r.start).sum()
    }
}

/// The segments a window asks for, each at its best rank, in the order to run them.
pub fn plan(grid: &Grid, request: &WindowRequest) -> Vec<(usize, Rank)> {
    let frames = grid.length().unsigned_abs() as usize;
    if frames == 0 {
        return Vec::new();
    }
    let as_i32 = |v: usize| i32::try_from(v.min(frames)).unwrap_or(i32::MAX);
    let mut best: BTreeMap<usize, Rank> = BTreeMap::new();
    for region in &request.regions {
        let at = region.at.min(frames - 1);
        let from = at.saturating_sub(region.behind);
        let to = at.saturating_add(region.ahead).max(at + 1);
        for k in grid.covering(as_i32(from), as_i32(to)) {
            let placement = grid.placement(k);
            let start = placement.offset.unsigned_abs() as usize;
            let end = placement.end().unsigned_abs() as usize;
            let distance = if start > at {
                start - at
            } else if end <= at {
                2 * (at - end) + 1
            } else {
                0
            };
            let rank = Rank {
                priority: region.priority,
                distance,
            };
            best.entry(k)
                .and_modify(|r| *r = (*r).min(rank))
                .or_insert(rank);
        }
    }
    let mut order: Vec<(usize, Rank)> = best.into_iter().collect();
    order.sort_by_key(|&(k, rank)| (rank, k));
    order
}

/// The transfer gain for every stem of an output.
///
/// A stem's own peak is known only once all of it is separated, so this leaves
/// 3 dB of headroom over the mix's peak, or over full scale when it is unknown.
pub fn gain(output: &Output, peak: Option<f32>) -> f32 {
    if output.format.has_headroom() {
        return 1.0;
    }
    let ceiling = peak.unwrap_or(1.0) * std::f32::consts::SQRT_2;
    if ceiling <= 1.0 { 1.0 } else { 1.0 / ceiling }
}

/// The parts an output serves: those asked for, then the derived one if it ships.
pub fn served(output: &Output) -> Vec<&'static str> {
    let derived = output
        .ships_derived()
        .then(|| output.parts.derived())
        .flatten();
    output.parts.names().chain(derived).collect()
}

/// What a track is, fixed when it is opened.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackSpec {
    pub frames: usize,
    pub channels: usize,
    /// The model's rate.
    pub sample_rate: u32,
    pub grid: Grid,
    /// True when the model needs the whole mix's normalisation.
    pub normalises: bool,
    /// The parts separated, which decide what the models run.
    pub parts: Parts,
}

impl From<&TrackSpec> for TrackInfo {
    fn from(spec: &TrackSpec) -> Self {
        Self {
            frames: spec.frames,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            stride: spec.grid.stride().unsigned_abs() as usize,
            pieces: spec.grid.len(),
        }
    }
}

/// What became of the whole upload once it arrived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The samples hash to this id, and any statistics the client sent match.
    Verified(String),
    /// The client's mean and std are not the mix's, so the stems are not the
    /// ones a whole-track run gives, and nothing of this track is shared.
    Unpublishable(String),
}

struct TrackState {
    mix: Audio,
    arrived: Spans,
    norm: Option<Normalisation>,
    /// True when `norm` came from a client and has not been checked.
    norm_from_client: bool,
    stems: Vec<(&'static str, Audio)>,
    ready: Spans,
    done: Vec<bool>,
    verdict: Option<Verdict>,
    /// Pieces already written, by output type.
    written: HashSet<(String, usize)>,
    closed: Option<String>,
}

pub struct Track {
    /// Identifies the track while it is open; it stays the same when sessions join.
    pub uid: u64,
    /// The `model_id` that separates it.
    pub model_id: String,
    pub spec: TrackSpec,
    state: Mutex<TrackState>,
}

impl Track {
    fn new(uid: u64, model_id: String, spec: TrackSpec, norm: Option<Normalisation>) -> Self {
        let silent = || {
            Audio::new(
                vec![vec![0.0; spec.frames]; spec.channels],
                spec.sample_rate,
            )
        };
        Self {
            uid,
            model_id,
            state: Mutex::new(TrackState {
                mix: silent(),
                arrived: Spans::default(),
                norm: norm.filter(|_| spec.normalises),
                norm_from_client: norm.is_some() && spec.normalises,
                stems: spec.parts.names().map(|name| (name, silent())).collect(),
                ready: Spans::default(),
                done: vec![false; spec.grid.len()],
                verdict: None,
                written: HashSet::new(),
                closed: None,
            }),
            spec,
        }
    }

    fn check_open(state: &TrackState) -> Result<()> {
        match &state.closed {
            Some(reason) => bail!("this track is closed: {reason}"),
            None => Ok(()),
        }
    }

    /// Samples piece `k` covers.
    pub fn piece(&self, k: usize) -> Range<usize> {
        let stride = self.spec.grid.stride().unsigned_abs() as usize;
        (k * stride).min(self.spec.frames)..((k + 1) * stride).min(self.spec.frames)
    }

    /// Place `audio` at track sample `offset`.
    ///
    /// Samples another session already uploaded must be the same to the bit.
    /// Returns the verdict once this upload completes the track.
    pub fn upload(&self, offset: usize, audio: &Audio) -> Result<Option<Verdict>, UploadError> {
        let mut state = self.state.lock();
        Self::check_open(&state).map_err(UploadError::Closed)?;
        let end = offset
            .checked_add(audio.frames())
            .filter(|&end| end <= self.spec.frames)
            .ok_or_else(|| {
                UploadError::Invalid(format!(
                    "{} samples at {offset} run past the track's {}",
                    audio.frames(),
                    self.spec.frames
                ))
            })?;
        for seen in state.arrived.overlaps(&(offset..end)) {
            for (have, given) in state.mix.data.iter().zip(&audio.data) {
                let given = &given[seen.start - offset..seen.end - offset];
                if have[seen.clone()]
                    .iter()
                    .zip(given)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    return Err(UploadError::Differs(seen.start, seen.end));
                }
            }
        }
        let was_complete = state.arrived.covers(&(0..self.spec.frames));
        for (dst, src) in state.mix.data.iter_mut().zip(&audio.data) {
            dst[offset..end].copy_from_slice(src);
        }
        state.arrived.insert(offset..end);
        if was_complete || !state.arrived.covers(&(0..self.spec.frames)) {
            return Ok(None);
        }

        // The whole track is here: settle its statistics and its identity.
        let computed = Normalisation::of(&state.mix);
        let actual = track_id(&state.mix);
        let client_norm_differs = self.spec.normalises
            && state.norm_from_client
            && state.norm.is_some_and(|n| {
                n.mean().to_bits() != computed.mean().to_bits()
                    || n.std().to_bits() != computed.std().to_bits()
            });
        if self.spec.normalises && state.norm.is_none() {
            state.norm = Some(computed);
        }
        let verdict = if client_norm_differs {
            Verdict::Unpublishable(actual)
        } else {
            Verdict::Verified(actual)
        };
        state.verdict = Some(verdict.clone());
        Ok(Some(verdict))
    }

    /// The track id, once the whole upload has hashed to it and may be shared.
    pub fn verified(&self) -> Option<String> {
        match &self.state.lock().verdict {
            Some(Verdict::Verified(id)) => Some(id.clone()),
            _ => None,
        }
    }

    /// True when segment `k` can run now.
    fn runnable(&self, state: &TrackState, k: usize) -> bool {
        let window = self.spec.grid.window(k);
        let input = window.from.unsigned_abs() as usize..window.to.unsigned_abs() as usize;
        state.closed.is_none()
            && !state.done[k]
            && state.arrived.covers(&input)
            && (!self.spec.normalises || state.norm.is_some())
    }

    pub fn is_done(&self, k: usize) -> bool {
        self.state.lock().done.get(k).copied().unwrap_or(false)
    }

    /// Segment `k`'s input and the normalisation to run it with.
    pub fn input(&self, k: usize) -> Result<(Audio, Normalisation)> {
        let state = self.state.lock();
        Self::check_open(&state)?;
        let window = self.spec.grid.window(k);
        let (from, to) = (
            window.from.unsigned_abs() as usize,
            window.to.unsigned_abs() as usize,
        );
        // Models trained on raw audio ignore it.
        let norm = state.norm.unwrap_or_else(|| Normalisation::new(0.0, 1.0));
        Ok((excerpt(&state.mix, from, to), norm))
    }

    /// The mix over `from..to`.
    pub fn mix(&self, from: usize, to: usize) -> Audio {
        excerpt(&self.state.lock().mix, from, to)
    }

    /// Record that segment `k` has run.
    pub fn done(&self, k: usize) {
        if let Some(d) = self.state.lock().done.get_mut(k) {
            *d = true;
        }
    }

    /// Copy the final stems over `from..to` in.
    pub fn store(&self, from: usize, to: usize, stems: &Stems) -> Result<()> {
        let mut state = self.state.lock();
        for (name, buffer) in &mut state.stems {
            let audio = stems
                .stem(name)
                .with_context(|| format!("this separation has no {name}"))?;
            for (dst, src) in buffer.data.iter_mut().zip(&audio.data) {
                dst[from..to].copy_from_slice(src);
            }
        }
        state.ready.insert(from..to);
        Ok(())
    }

    pub fn close(&self, reason: impl Into<String>) {
        let mut state = self.state.lock();
        if state.closed.is_none() {
            state.closed = Some(reason.into());
        }
    }

    pub fn closed(&self) -> Option<String> {
        self.state.lock().closed.clone()
    }

    /// True when piece `k` at `output` can be written: the piece is ready, and
    /// so is the context a rate conversion needs, and it was not written yet.
    pub fn piece_writable(&self, k: usize, type_dir: &str, converts: bool) -> bool {
        let state = self.state.lock();
        if state.closed.is_some() || state.written.contains(&(type_dir.to_owned(), k)) {
            return false;
        }
        let piece = self.piece(k);
        let needed = if converts {
            piece.start.saturating_sub(RESAMPLE_CONTEXT)
                ..(piece.end + RESAMPLE_CONTEXT).min(self.spec.frames)
        } else {
            piece
        };
        state.ready.covers(&needed)
    }

    /// True when every sample of `range` is ready in memory.
    pub fn ready_covers(&self, range: &Range<usize>) -> bool {
        self.state.lock().ready.covers(range)
    }

    /// True when every sample of the track is ready or in one of the `stored` pieces.
    pub fn complete_with(&self, stored: &[usize]) -> bool {
        let mut spans = self.state.lock().ready.clone();
        for &k in stored {
            spans.insert(self.piece(k));
        }
        spans.covers(&(0..self.spec.frames))
    }

    pub fn mark_written(&self, k: usize, type_dir: &str) {
        self.state.lock().written.insert((type_dir.to_owned(), k));
    }

    /// Part `name` over `from..to` at `output`'s rate.
    ///
    /// `from..to` are track samples at the model's rate and must be ready. At
    /// another rate the range is converted with up to [`RESAMPLE_CONTEXT`]
    /// samples either side, taken from what is ready around it.
    pub fn read(&self, output: &Output, name: &str, from: usize, to: usize) -> Result<Excerpt> {
        let (audio, start) = {
            let state = self.state.lock();
            Self::check_open(&state)?;
            if from >= to || !state.ready.covers(&(from..to)) {
                bail!("samples {from}..{to} are not ready");
            }
            if !served(output).contains(&name) {
                bail!("this session serves {:?}, not {name}", served(output));
            }
            let model = self.spec.sample_rate;
            let out = output.rate.hz();
            let (start, end) = if model == out {
                (from, to)
            } else {
                let span = state.ready.span_of(from).cloned().unwrap_or(from..to);
                let step = (model / gcd(model, out)) as usize;
                let start = from.saturating_sub(RESAMPLE_CONTEXT).max(span.start) / step * step;
                (start, (to + RESAMPLE_CONTEXT).min(span.end))
            };
            (Self::part(&state, name, start, end), start)
        };

        let model = u64::from(self.spec.sample_rate);
        let out = u64::from(output.rate.hz());
        if model == out {
            return Ok(Excerpt { audio, from });
        }
        let converted = output.dsp.convert(&audio, output.rate.hz())?;
        let offset = output_index(start, model, out);
        let a = output_index(from, model, out) - offset;
        let b = (output_index(to, model, out) - offset).min(converted.frames());
        Ok(Excerpt {
            audio: excerpt(&converted, a.min(b), b),
            from: output_index(from, model, out),
        })
    }

    /// Part `name` over `from..to` at the model's rate, the derived one included.
    fn part(state: &TrackState, name: &str, from: usize, to: usize) -> Audio {
        if let Some((_, audio)) = state.stems.iter().find(|(n, _)| *n == name) {
            return excerpt(audio, from, to);
        }
        let parts: Vec<Audio> = state
            .stems
            .iter()
            .map(|(_, a)| excerpt(a, from, to))
            .collect();
        stemd_core::stems::remainder(&excerpt(&state.mix, from, to), parts.iter())
    }

    /// Pieces ready in memory or `stored`, samples uploaded, and samples ready or stored.
    fn counts(&self, stored: &HashSet<usize>) -> (usize, usize, usize) {
        let state = self.state.lock();
        let mut covered = state.ready.clone();
        for &k in stored {
            covered.insert(self.piece(k));
        }
        let ready = (0..self.spec.grid.len())
            .filter(|&k| covered.covers(&self.piece(k)))
            .count();
        (ready, state.arrived.total(), covered.total())
    }

    fn summary(&self) -> TrackSummary {
        let state = self.state.lock();
        TrackSummary {
            arrived: state.arrived.as_pairs(),
            ready: state.ready.as_pairs(),
            needs_stats: self.spec.normalises && state.norm.is_none(),
            closed: state.closed.clone(),
        }
    }
}

struct TrackSummary {
    arrived: Vec<[usize; 2]>,
    ready: Vec<[usize; 2]>,
    needs_stats: bool,
    closed: Option<String>,
}

/// Why an upload was refused.
#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    #[error("{0}")]
    Invalid(String),
    #[error("samples {0}..{1} differ from what was already uploaded for this track")]
    Differs(usize, usize),
    #[error("{0:#}")]
    Closed(anyhow::Error),
}

/// Output-rate sample `j` of the track sits at `j * model / out`, so model-rate
/// sample `i` maps to the first output sample at or after it.
pub fn output_index(i: usize, model: u64, out: u64) -> usize {
    (i as u64 * out).div_ceil(model) as usize
}

/// A session as its client sees it.
#[derive(Debug, Serialize)]
pub struct View {
    pub id: String,
    /// `open`, or `closed` with `error` saying why.
    pub state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Changes whenever another field does.
    pub rev: u64,
    pub track: TrackView,
    pub output: crate::separations::OutputView,
    pub grid: GridView,
    pub arrived: Vec<[usize; 2]>,
    /// Samples separated in memory.
    pub ready: Vec<[usize; 2]>,
    pub stems: Vec<StemView>,
    /// True when every sample is ready or stored, so any range reads.
    pub complete: bool,
    /// Segments asked for and not run yet.
    pub planned: usize,
    /// Set while segments wait on something other than the worker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waiting_for: Option<&'static str>,
    pub idle_timeout_secs: u64,
}

#[derive(Debug, Serialize)]
pub struct TrackView {
    /// The track id, `null` until the whole upload has been hashed.
    pub id: Option<String>,
    pub frames: usize,
    /// The model's rate.
    pub sample_rate: u32,
    pub channels: usize,
}

#[derive(Debug, Serialize)]
pub struct GridView {
    pub segment_frames: i32,
    pub stride_frames: i32,
    pub segments: usize,
    /// Median seconds of the last [`crate::queue::SPEED_KEPT`] segments, `null` until one ran.
    pub segment_run_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StemView {
    pub name: &'static str,
    /// See [`crate::separations::sources`].
    pub sources: u16,
    pub stored: crate::separations::Storage,
    /// The whole file's size, when stored whole.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// Stored pieces as half-open spans of piece indices, when stored as pieces.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pieces: Vec<[usize; 2]>,
}

/// Sorted piece indices as half-open spans.
fn spans_of(pieces: &[usize]) -> Vec<[usize; 2]> {
    let mut out: Vec<[usize; 2]> = Vec::new();
    for &k in pieces {
        match out.last_mut() {
            Some(last) if last[1] == k => last[1] = k + 1,
            _ => out.push([k, k + 1]),
        }
    }
    out
}

/// What the window shows of a session. It carries no track id or name.
#[derive(Debug, Clone, PartialEq)]
pub struct Activity {
    /// Counts sessions from 1 since the server started.
    pub number: u64,
    pub frames: usize,
    pub sample_rate: u32,
    /// Pieces in the track.
    pub pieces: usize,
    /// Pieces ready in memory or in the piece store.
    pub ready: usize,
    /// Samples uploaded.
    pub arrived: usize,
    /// Samples ready in memory or in a stored piece.
    pub covered: usize,
    /// Planned segments not run yet that run before jobs: priority below [`BEFORE_JOBS`].
    pub high: usize,
    /// Planned segments not run yet that wait for jobs.
    pub normal: usize,
    /// True when a planned segment can run now.
    pub runnable: bool,
}

/// A session that ended, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ended {
    pub number: u64,
    pub reason: String,
    pub at: Instant,
}

/// Ended sessions kept for the window.
const ENDED_KEPT: usize = 8;

/// Where [`Sessions::read_stored`] takes a run of samples from.
enum Source {
    Memory,
    Piece(std::path::PathBuf),
    Whole,
}

/// A range read back from a track, at the output rate.
pub struct Excerpt {
    pub audio: Audio,
    /// First output-rate sample.
    pub from: usize,
}

struct SessionState {
    plan: VecDeque<(usize, Rank)>,
    /// When the worker last took one of this session's segments.
    last_served: Option<Instant>,
    last_seen: Instant,
    closed: Option<String>,
}

pub struct Session {
    pub id: String,
    /// Counts sessions from 1 since the server started.
    pub number: u64,
    pub output: Output,
    pub in_format: PcmFormat,
    pub gain: f32,
    /// The track this session plays, which changes once when its upload
    /// completes and another session's track already holds the same audio.
    track: Mutex<Arc<Track>>,
    /// Pieces of this output already on disk, which no segment needs to remake.
    cached: Mutex<HashSet<usize>>,
    /// The client's file key, recorded against the track id once the upload is hashed.
    file: Mutex<Option<String>>,
    state: Mutex<SessionState>,
}

impl Session {
    fn new(
        id: String,
        number: u64,
        output: Output,
        in_format: PcmFormat,
        peak: Option<f32>,
        track: Arc<Track>,
    ) -> Self {
        let now = Instant::now();
        Self {
            id,
            number,
            gain: gain(&output, peak),
            output,
            in_format,
            track: Mutex::new(track),
            cached: Mutex::new(HashSet::new()),
            file: Mutex::new(None),
            state: Mutex::new(SessionState {
                plan: VecDeque::new(),
                last_served: None,
                last_seen: now,
                closed: None,
            }),
        }
    }

    pub fn track(&self) -> Arc<Track> {
        Arc::clone(&self.track.lock())
    }

    pub fn set_file(&self, key: Option<String>) {
        *self.file.lock() = key;
    }

    /// Record the pieces of this output on disk now, and drop the planned
    /// segments they make unnecessary.
    pub fn set_cached(&self, pieces: impl IntoIterator<Item = usize>) {
        *self.cached.lock() = pieces.into_iter().collect();
        let mut state = self.state.lock();
        let plan = std::mem::take(&mut state.plan);
        state.plan = plan
            .into_iter()
            .filter(|(k, _)| !self.covered(*k))
            .collect();
    }

    /// True when every piece segment `k` feeds is on disk. At another output rate,
    /// a piece also needs the segments within [`RESAMPLE_CONTEXT`] of it.
    fn covered(&self, k: usize) -> bool {
        let cached = self.cached.lock();
        let track = self.track();
        let margin = if self.output.rate.hz() == track.spec.sample_rate {
            0
        } else {
            RESAMPLE_CONTEXT
        };
        let as_i32 = |v: usize| i32::try_from(v).unwrap_or(i32::MAX);
        (0..track.spec.grid.len()).all(|j| {
            let piece = track.piece(j);
            cached.contains(&j)
                || !track
                    .spec
                    .grid
                    .covering(
                        as_i32(piece.start.saturating_sub(margin)),
                        as_i32(piece.end + margin),
                    )
                    .contains(&k)
        })
    }

    /// Why the session or its track is closed.
    pub fn closed(&self) -> Option<String> {
        self.state
            .lock()
            .closed
            .clone()
            .or_else(|| self.track().closed())
    }

    /// Record a request from the client.
    pub fn touch(&self) {
        self.state.lock().last_seen = Instant::now();
    }

    /// Replace the plan with what `request` asks for, less what has run.
    pub fn set_window(&self, request: &WindowRequest) -> Result<()> {
        if let Some(reason) = self.closed() {
            bail!("this session is closed: {reason}");
        }
        let track = self.track();
        let planned: VecDeque<(usize, Rank)> = plan(&track.spec.grid, request)
            .into_iter()
            .filter(|(k, _)| !track.is_done(*k) && !self.covered(*k))
            .collect();
        let mut state = self.state.lock();
        let now = Instant::now();
        state.last_seen = now;
        state.plan = planned;
        Ok(())
    }

    /// The best runnable segment of `class`, its rank, and when this session was last served.
    fn next(&self, class: Class) -> Option<(Arc<Track>, usize, Rank, Option<Instant>)> {
        let track = self.track();
        let mut state = self.state.lock();
        if state.closed.is_some() {
            return None;
        }
        let held = track.state.lock();
        state.plan.retain(|(k, _)| !held.done[*k]);
        let (k, rank) = state
            .plan
            .iter()
            .find(|(k, rank)| rank.class() == class && track.runnable(&held, *k))
            .copied()?;
        drop(held);
        Some((track, k, rank, state.last_served))
    }

    pub fn close(&self, reason: impl Into<String>) {
        let mut state = self.state.lock();
        if state.closed.is_none() {
            state.closed = Some(reason.into());
        }
        state.plan.clear();
    }

    /// True when a planned segment can run now.
    fn has_runnable(&self) -> bool {
        let track = self.track();
        let state = self.state.lock();
        let held = track.state.lock();
        state.plan.iter().any(|(k, _)| track.runnable(&held, *k))
    }

    /// True when nothing can run and the client has been quiet for `timeout`.
    fn idle(&self, now: Instant, timeout: Duration) -> bool {
        !self.has_runnable() && now.duration_since(self.state.lock().last_seen) > timeout
    }

    /// Part `name` over `from..to`, at the output rate.
    pub fn read(&self, name: &str, from: usize, to: usize) -> Result<Excerpt> {
        if let Some(reason) = self.closed() {
            bail!("this session is closed: {reason}");
        }
        self.track().read(&self.output, name, from, to)
    }

    /// What the window shows of this session. Takes only short locks.
    pub fn activity(&self) -> Activity {
        let track = self.track();
        let stored = self.cached.lock().clone();
        let (ready, arrived, covered) = track.counts(&stored);
        let (high, normal, runnable) = {
            let state = self.state.lock();
            let held = track.state.lock();
            let mut counts = (0, 0, false);
            for &(k, rank) in state.plan.iter().filter(|(k, _)| !held.done[*k]) {
                match rank.class() {
                    Class::BeforeJobs => counts.0 += 1,
                    Class::WhenIdle => counts.1 += 1,
                }
                counts.2 |= track.runnable(&held, k);
            }
            counts
        };
        Activity {
            number: self.number,
            frames: track.spec.frames,
            sample_rate: track.spec.sample_rate,
            pieces: track.spec.grid.len(),
            ready,
            arrived,
            covered,
            high,
            normal,
            runnable,
        }
    }

    /// The view, with `rev` left at 0.
    fn view(
        &self,
        idle: Duration,
        pieces: &[usize],
        stems: Vec<StemView>,
        segment_run_seconds: Option<f64>,
    ) -> View {
        let track = self.track();
        let summary = track.summary();
        let (planned, waiting) = {
            let state = self.state.lock();
            let held = track.state.lock();
            // A finished segment stays in the plan until the worker next takes from it.
            let left: Vec<usize> = state
                .plan
                .iter()
                .map(|(k, _)| *k)
                .filter(|&k| !held.done[k])
                .collect();
            let waiting = left.iter().any(|&k| !track.runnable(&held, k));
            (left.len(), waiting)
        };
        let closed = self.state.lock().closed.clone().or(summary.closed);
        let spec = &track.spec;
        View {
            id: self.id.clone(),
            state: if closed.is_some() { "closed" } else { "open" },
            error: closed,
            rev: 0,
            track: TrackView {
                id: track.verified(),
                frames: spec.frames,
                sample_rate: spec.sample_rate,
                channels: spec.channels,
            },
            output: crate::separations::OutputView::of(&self.output, Some(self.gain)),
            grid: GridView {
                segment_frames: spec.grid.segment(),
                stride_frames: spec.grid.stride(),
                segments: spec.grid.len(),
                segment_run_seconds,
            },
            arrived: summary.arrived,
            ready: summary.ready,
            stems,
            complete: track.complete_with(pieces),
            planned,
            waiting_for: match (waiting, summary.needs_stats) {
                (true, true) => Some("the whole upload, or mean and std"),
                (true, false) => Some("upload"),
                _ => None,
            },
            idle_timeout_secs: idle.as_secs(),
        }
    }
}

pub fn excerpt(audio: &Audio, from: usize, to: usize) -> Audio {
    Audio::new(
        audio.data.iter().map(|c| c[from..to].to_vec()).collect(),
        audio.sample_rate,
    )
}

const fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[derive(Debug, thiserror::Error)]
#[error("{0} sessions are open, the most this server allows")]
pub struct TooMany(pub usize);

/// Why a session could not join a track.
#[derive(Debug, thiserror::Error)]
pub enum JoinError {
    #[error(transparent)]
    TooMany(#[from] TooMany),
    /// The track was dropped or closed after it was found.
    #[error("the track closed")]
    Gone,
}

/// What a client asks for when it opens a session.
pub struct Request {
    pub output: Output,
    pub in_format: PcmFormat,
    pub track: TrackSpec,
    pub norm: Option<Normalisation>,
    pub peak: Option<f32>,
    pub model_id: String,
}

/// Every open session and the tracks they play.
pub struct Sessions {
    sessions: Mutex<BTreeMap<String, Arc<Session>>>,
    /// By [`Track::uid`].
    tracks: Mutex<BTreeMap<u64, Arc<Track>>>,
    /// Tracks whose whole upload has been hashed, by id and part set.
    by_id: Mutex<BTreeMap<String, u64>>,
    limits: Mutex<(Duration, usize)>,
    pieces: Mutex<Option<Arc<Pieces>>>,
    /// The last [`ENDED_KEPT`] sessions to end, oldest first.
    ended: Mutex<VecDeque<Ended>>,
    seq: AtomicU64,
    /// Counts changes any view can show, for long polls.
    changes: tokio::sync::watch::Sender<u64>,
}

impl Default for Sessions {
    fn default() -> Self {
        Self::new(Duration::from_secs(30), 4)
    }
}

impl Sessions {
    pub fn new(idle: Duration, max: usize) -> Self {
        Self {
            sessions: Mutex::new(BTreeMap::new()),
            tracks: Mutex::new(BTreeMap::new()),
            by_id: Mutex::new(BTreeMap::new()),
            limits: Mutex::new((idle, max)),
            pieces: Mutex::new(None),
            ended: Mutex::new(VecDeque::new()),
            seq: AtomicU64::new(0),
            changes: tokio::sync::watch::Sender::new(0),
        }
    }

    /// Record that a view may have changed, and wake the long polls.
    pub fn changed(&self) {
        self.changes.send_modify(|n| *n = n.wrapping_add(1));
    }

    /// A receiver that wakes on every [`Sessions::changed`].
    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<u64> {
        self.changes.subscribe()
    }

    /// What `session`'s client sees now, with its `rev`.
    pub fn view_of(&self, session: &Session, segment_run_seconds: Option<f64>) -> View {
        use std::hash::{Hash, Hasher};
        let mut view = session.view(
            self.idle_timeout(),
            &self.cached_for(session),
            self.stems_for(session),
            segment_run_seconds,
        );
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        serde_json::to_vec(&view)
            .unwrap_or_default()
            .hash(&mut hasher);
        view.rev = hasher.finish();
        view
    }

    /// How each stem `session` serves is stored.
    pub fn stems_for(&self, session: &Session) -> Vec<StemView> {
        use crate::separations::{Storage, sources};
        let track = session.track();
        let dir = self.track_dir(session);
        let extension = session.output.format.extension();
        served(&session.output)
            .into_iter()
            .map(|name| {
                let mut stem = StemView {
                    name,
                    sources: sources(name),
                    stored: Storage::None,
                    bytes: None,
                    pieces: Vec::new(),
                };
                if let Some(whole) = self.whole(session, name) {
                    stem.stored = Storage::Whole;
                    stem.bytes = std::fs::metadata(&whole.path).map(|m| m.len()).ok();
                } else if let Some(dir) = &dir {
                    let stored = Pieces::stem_pieces(dir, name, extension, track.spec.grid.len());
                    if !stored.is_empty() {
                        stem.stored = Storage::Pieces;
                        stem.pieces = spans_of(&stored);
                    }
                }
                stem
            })
            .collect()
    }

    pub fn set_limits(&self, idle: Duration, max: usize) {
        *self.limits.lock() = (idle, max);
    }

    pub fn idle_timeout(&self) -> Duration {
        self.limits.lock().0
    }

    /// Use `pieces` as the cache, and keep the tracks of open sessions from eviction.
    pub fn set_pieces(self: &Arc<Self>, pieces: Arc<Pieces>) {
        let sessions = Arc::downgrade(self);
        pieces.set_in_use(Box::new(move || {
            sessions
                .upgrade()
                .map(|s| s.open_track_ids())
                .unwrap_or_default()
        }));
        let sessions = Arc::downgrade(self);
        pieces.set_on_change(Box::new(move || {
            if let Some(sessions) = sessions.upgrade() {
                sessions.changed();
            }
        }));
        *self.pieces.lock() = Some(pieces);
    }

    /// The ids of the verified tracks open sessions play.
    fn open_track_ids(&self) -> HashSet<String> {
        let tracks: Vec<Arc<Track>> = self.tracks.lock().values().cloned().collect();
        tracks.iter().filter_map(|t| t.verified()).collect()
    }

    pub fn pieces(&self) -> Option<Arc<Pieces>> {
        self.pieces.lock().clone()
    }

    /// The track directory of `session`'s output, once its track id is known.
    pub fn track_dir(&self, session: &Session) -> Option<PathBuf> {
        let track = session.track();
        let type_dir = type_dir(&session.output, session.gain);
        Some(
            self.pieces()?
                .verified_dir(&track.model_id, &type_dir, &track.verified()?),
        )
    }

    /// The open track whose whole upload hashed to `id`, separated into `parts`.
    pub fn open_track(&self, id: &str, parts: Parts) -> Option<Arc<Track>> {
        let uid = *self.by_id.lock().get(&format!("{id}-{parts}"))?;
        self.tracks
            .lock()
            .get(&uid)
            .filter(|t| t.closed().is_none())
            .cloned()
    }

    /// Open a session on a track of its own.
    pub fn open(&self, request: Request) -> Result<Arc<Session>, TooMany> {
        let max = self.limits.lock().1;
        let mut sessions = self.sessions.lock();
        if sessions.len() >= max {
            return Err(TooMany(sessions.len()));
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let track = Arc::new(Track::new(
            seq,
            request.model_id.clone(),
            request.track,
            request.norm,
        ));
        self.tracks.lock().insert(seq, Arc::clone(&track));
        let session = Arc::new(Session::new(
            format!("s{seq}-{nanos:08x}"),
            seq + 1,
            request.output,
            request.in_format,
            request.peak,
            track,
        ));
        sessions.insert(session.id.clone(), Arc::clone(&session));
        drop(sessions);
        self.changed();
        Ok(session)
    }

    /// Open a session on `track`, already uploaded and hashed, so its client uploads nothing.
    pub fn join(
        &self,
        output: Output,
        in_format: PcmFormat,
        peak: Option<f32>,
        track: Arc<Track>,
    ) -> Result<Arc<Session>, JoinError> {
        let max = self.limits.lock().1;
        let mut sessions = self.sessions.lock();
        if sessions.len() >= max {
            return Err(JoinError::TooMany(TooMany(sessions.len())));
        }
        if !self.tracks.lock().contains_key(&track.uid) || track.closed().is_some() {
            return Err(JoinError::Gone);
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let session = Arc::new(Session::new(
            format!("s{seq}-{nanos:08x}"),
            seq + 1,
            output,
            in_format,
            peak,
            Arc::clone(&track),
        ));
        sessions.insert(session.id.clone(), Arc::clone(&session));
        drop(sessions);
        // Writes this output's pieces of what is ready already.
        self.store_pieces(&track, 0..track.spec.frames);
        session.set_cached(self.cached_for(&session));
        self.changed();
        Ok(session)
    }

    /// Place `audio` at `offset` in `session`'s track.
    ///
    /// The upload that completes the track hashes it (a few hundred ms) and files it under that id,
    /// merging it into an open track with the same audio if there is one.
    pub fn upload(
        &self,
        session: &Session,
        offset: usize,
        audio: &Audio,
    ) -> Result<(), UploadError> {
        let track = session.track();
        match track.upload(offset, audio)? {
            None => {}
            Some(Verdict::Unpublishable(id)) => tracing::warn!(
                track = %id,
                "the client's mean and std are not the mix's; this track is not shared"
            ),
            Some(Verdict::Verified(id)) => {
                let track = self.settle(&track, &id);
                self.store_pieces(&track, 0..track.spec.frames);
                self.record_file(session, &id);
                for other in self.of_track(&track) {
                    other.set_cached(self.cached_for(&other));
                }
            }
        }
        self.changed();
        Ok(())
    }

    /// Record `session`'s file key against `track_id`, when it sent one.
    pub fn record_file(&self, session: &Session, track_id: &str) {
        let key = session.file.lock().clone();
        if let (Some(pieces), Some(key)) = (self.pieces(), key)
            && let Err(err) = pieces.record_file(&key, track_id)
        {
            tracing::warn!(session = %session.id, "recording file key {key}: {err:#}");
        }
    }

    /// Replace `session`'s plan, against the pieces on disk now: a file deleted
    /// outside the server is separated again.
    pub fn set_window(&self, session: &Session, request: &WindowRequest) -> Result<()> {
        request.check()?;
        session.set_cached(self.cached_for(session));
        session.set_window(request)?;
        self.changed();
        Ok(())
    }

    /// The pieces of `session`'s output stored under its track's id.
    pub fn cached_for(&self, session: &Session) -> Vec<usize> {
        let Some(dir) = self.track_dir(session) else {
            return Vec::new();
        };
        Pieces::cached(
            &dir,
            &served(&session.output),
            session.output.format.extension(),
            session.track().spec.grid.len(),
        )
    }

    /// Encode and write the pieces near `touched` that are complete in `track`, once it is hashed.
    pub fn store_pieces(&self, track: &Track, touched: Range<usize>) {
        let (Some(pieces), Some(id)) = (self.pieces(), track.verified()) else {
            return;
        };
        // Pieces are a cache, so a nearly full disk skips them and the job goes on.
        if crate::disk::low(pieces.root()) {
            tracing::debug!(track = %id, "disk nearly full, not writing pieces");
            return;
        }
        let info = TrackInfo::from(&track.spec);
        let first = touched.start.saturating_sub(RESAMPLE_CONTEXT) / info.stride;
        let last = (touched.end + RESAMPLE_CONTEXT)
            .div_ceil(info.stride)
            .min(info.pieces);

        let mut outputs: Vec<(String, Output, f32)> = Vec::new();
        for session in self.of_track(track) {
            let name = type_dir(&session.output, session.gain);
            if !outputs.iter().any(|(n, _, _)| *n == name) {
                let mut output = session.output.clone();
                output.derived = true;
                outputs.push((name, output, session.gain));
            }
        }
        for (name, output, gain) in outputs {
            let converts = output.rate.hz() != track.spec.sample_rate;
            let dir = pieces.verified_dir(&track.model_id, &name, &id);
            if let Err(err) = pieces.write_info(&dir, &info) {
                tracing::warn!(track = %id, "describing {name}: {err:#}");
                continue;
            }
            for k in first..last {
                if !track.piece_writable(k, &name, converts) {
                    continue;
                }
                let piece = track.piece(k);
                let written = stems_of(&output).into_iter().try_for_each(|stem| {
                    let excerpt = track.read(&output, stem, piece.start, piece.end)?;
                    let bytes = stemd_core::encode_stem(&excerpt.audio, output.format, gain)?;
                    pieces.write_verified(&dir, stem, k, output.format.extension(), &bytes)
                });
                match written {
                    Ok(()) => track.mark_written(k, &name),
                    Err(err) => tracing::warn!(track = %id, "piece {k}: {err:#}"),
                }
            }
            let task = Consolidation {
                dir,
                format: output.format,
                rate: output.rate.hz(),
                info,
                stems: stems_of(&output),
            };
            if Pieces::needs_consolidation(&task) {
                pieces.consolidate_later(task);
            }
        }
    }

    /// Part `name` over `from..to` at the output rate and the transfer scale, from memory or disk.
    pub fn read_stored(
        &self,
        session: &Session,
        name: &str,
        from: usize,
        to: usize,
    ) -> Result<Excerpt> {
        if let Some(reason) = session.closed() {
            bail!("this session is closed: {reason}");
        }
        let output = &session.output;
        if !served(output).contains(&name) {
            bail!("this session serves {:?}, not {name}", served(output));
        }
        let track = session.track();
        if from >= to || to > track.spec.frames {
            bail!(
                "{from}..{to} is not a range of this {}-sample track",
                track.spec.frames
            );
        }
        let stride = track.spec.grid.stride().unsigned_abs() as usize;
        let whole = self.whole(session, name);
        let dir = self.track_dir(session);
        let lookup = |k: usize| {
            if whole.is_some() {
                return Some(Source::Whole);
            }
            Pieces::lookup(dir.as_ref()?, name, k, output.format.extension()).map(Source::Piece)
        };

        // A chunk comes from memory when it is ready with the context a rate
        // conversion needs, else from disk, else from memory with less context.
        // Consecutive chunks from memory or from the whole file are read as one range.
        let converts = output.rate.hz() != track.spec.sample_rate;
        let mut runs: Vec<(Range<usize>, Source)> = Vec::new();
        for k in from / stride..to.div_ceil(stride) {
            let piece = track.piece(k);
            let chunk = piece.start.max(from)..piece.end.min(to);
            let context = if converts {
                chunk.start.saturating_sub(RESAMPLE_CONTEXT)
                    ..(chunk.end + RESAMPLE_CONTEXT).min(track.spec.frames)
            } else {
                chunk.clone()
            };
            let source = if track.ready_covers(&context) {
                Source::Memory
            } else if let Some(source) = lookup(k) {
                source
            } else if track.ready_covers(&chunk) {
                Source::Memory
            } else {
                bail!(
                    "samples {}..{} are neither ready nor stored",
                    chunk.start,
                    chunk.end
                );
            };
            match (runs.last_mut(), source) {
                (Some((run, Source::Memory)), Source::Memory)
                | (Some((run, Source::Whole)), Source::Whole) => run.end = chunk.end,
                (_, source) => runs.push((chunk, source)),
            }
        }

        let (model, out) = (
            u64::from(track.spec.sample_rate),
            u64::from(output.rate.hz()),
        );
        let start = output_index(from, model, out);
        let frames = output_index(to, model, out) - start;
        let mut data = vec![Vec::with_capacity(frames); track.spec.channels];
        for (run, source) in runs {
            let at = output_index(run.start, model, out);
            let len = output_index(run.end, model, out) - at;
            let cut = |whole: &Whole| {
                crate::whole::read(
                    whole,
                    output.format,
                    track.spec.channels,
                    output.rate.hz(),
                    at,
                    at + len,
                )
            };
            let audio = match source {
                Source::Memory => {
                    let mut audio = track.read(output, name, run.start, run.end)?.audio;
                    for v in audio.data.iter_mut().flatten() {
                        *v *= session.gain;
                    }
                    audio
                }
                Source::Whole => cut(whole.as_ref().context("the whole file went away")?)?,
                Source::Piece(path) => match std::fs::read(&path) {
                    Ok(bytes) => {
                        let k = run.start / stride;
                        let piece_at = output_index(track.piece(k).start, model, out);
                        let piece_len = output_index(track.piece(k).end, model, out) - piece_at;
                        let audio = crate::pieces::decode(
                            bytes,
                            output.format,
                            track.spec.channels,
                            output.rate.hz(),
                        )
                        .with_context(|| format!("decoding {}", path.display()))?;
                        if audio.frames() != piece_len {
                            bail!(
                                "{} holds {} samples, not {piece_len}",
                                path.display(),
                                audio.frames()
                            );
                        }
                        excerpt(&audio, at - piece_at, at - piece_at + len)
                    }
                    // Stored whole since the lookup.
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => cut(&self
                        .whole(session, name)
                        .with_context(|| format!("{} went away", path.display()))?)?,
                    Err(err) => {
                        return Err(err).with_context(|| format!("reading {}", path.display()));
                    }
                },
            };
            if audio.frames() != len {
                bail!(
                    "samples {}..{} read back {} samples, not {len}",
                    run.start,
                    run.end,
                    audio.frames()
                );
            }
            for (dst, src) in data.iter_mut().zip(audio.data) {
                dst.extend(src);
            }
        }
        Ok(Excerpt {
            audio: Audio::new(data, output.rate.hz()),
            from: start,
        })
    }

    /// The whole file of `name` for `session`'s output, when it holds the whole track.
    pub fn whole(&self, session: &Session, name: &str) -> Option<Whole> {
        if session.closed().is_some() || !served(&session.output).contains(&name) {
            return None;
        }
        let dir = self.track_dir(session)?;
        let frames = TrackInfo::from(&session.track().spec).frames_at(session.output.rate.hz());
        let whole =
            crate::whole::find_track(&dir, name, session.output.format.extension(), frames)?;
        Pieces::touch(&dir);
        Some(whole)
    }

    /// Store `bytes`, the whole of `name` just encoded for `session`, in place
    /// of its pieces. Does nothing for a track that is not shared.
    pub fn consolidate(
        &self,
        session: &Session,
        name: &str,
        bytes: &[u8],
        frames: usize,
        clipped: Option<usize>,
    ) -> Result<()> {
        let track = session.track();
        let (Some(pieces), Some(dir), Some(id)) =
            (self.pieces(), self.track_dir(session), track.verified())
        else {
            return Ok(());
        };
        if crate::disk::low(pieces.root()) {
            return Ok(());
        }
        let task = Consolidation {
            dir,
            format: session.output.format,
            rate: session.output.rate.hz(),
            info: TrackInfo::from(&track.spec),
            stems: stems_of(&session.output),
        };
        pieces.consolidate(&task, name, bytes, frames, clipped)?;
        tracing::info!(track = %id, "stored {name} whole");
        Ok(())
    }

    /// File `track`, whose whole upload hashed to `id`, under that id.
    ///
    /// When another open track already holds the same audio and parts, every
    /// session on `track` moves onto that one and `track` closes. Returns the
    /// track the sessions play now.
    pub fn settle(&self, track: &Arc<Track>, id: &str) -> Arc<Track> {
        let key = format!("{id}-{}", track.spec.parts);
        let sessions = self.sessions.lock();
        let mut tracks = self.tracks.lock();
        let mut by_id = self.by_id.lock();
        let existing = by_id
            .get(&key)
            .and_then(|uid| tracks.get(uid))
            .filter(|t| t.uid != track.uid && t.closed().is_none() && t.spec == track.spec)
            .cloned();
        let Some(existing) = existing else {
            by_id.insert(key, track.uid);
            return Arc::clone(track);
        };
        for session in sessions.values() {
            let mut held = session.track.lock();
            if held.uid == track.uid {
                *held = Arc::clone(&existing);
            }
        }
        track.close(format!("merged into the open track {id}"));
        tracks.remove(&track.uid);
        tracing::info!("track {id}: a second session joined the one already open");
        existing
    }

    /// Every open session, as the window shows it, in the order they opened.
    pub fn activity(&self) -> Vec<Activity> {
        let open: Vec<Arc<Session>> = self.sessions.lock().values().cloned().collect();
        let mut rows: Vec<Activity> = open.iter().map(|s| s.activity()).collect();
        rows.sort_by_key(|a| a.number);
        rows
    }

    /// The sessions that ended last, oldest first.
    pub fn ended(&self) -> Vec<Ended> {
        self.ended.lock().iter().cloned().collect()
    }

    fn record_end(&self, number: u64, reason: String) {
        let mut ended = self.ended.lock();
        if ended.len() == ENDED_KEPT {
            ended.pop_front();
        }
        ended.push_back(Ended {
            number,
            reason,
            at: Instant::now(),
        });
    }

    pub fn get(&self, id: &str) -> Option<Arc<Session>> {
        self.sessions.lock().get(id).cloned()
    }

    /// End session `id`, saying why, such as "ended by the client". Returns
    /// false when there is no such session.
    pub fn remove(&self, id: &str, reason: &str) -> bool {
        let removed = self.sessions.lock().remove(id);
        if let Some(session) = &removed {
            session.close(reason);
            self.record_end(session.number, reason.to_owned());
        }
        self.drop_orphans();
        self.changed();
        removed.is_some()
    }

    pub fn len(&self) -> usize {
        self.sessions.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True while a session plays the track with this uid.
    pub fn has_track(&self, uid: u64) -> bool {
        self.tracks.lock().contains_key(&uid)
    }

    /// The sessions playing `track`.
    pub fn of_track(&self, track: &Track) -> Vec<Arc<Session>> {
        self.sessions
            .lock()
            .values()
            .filter(|s| s.track().uid == track.uid)
            .cloned()
            .collect()
    }

    /// Close and drop every session and track, saying why.
    pub fn close_all(&self, reason: &str) {
        let closed = std::mem::take(&mut *self.sessions.lock());
        for session in closed.values() {
            session.close(reason);
            self.record_end(session.number, reason.to_owned());
        }
        for track in std::mem::take(&mut *self.tracks.lock()).values() {
            track.close(reason);
        }
        self.by_id.lock().clear();
        if !closed.is_empty() {
            tracing::info!("closed {} sessions: {reason}", closed.len());
        }
        self.changed();
    }

    /// The best-ranked segment of `class` across sessions; at a tie, the one served longest ago.
    pub fn next(&self, class: Class) -> Option<(Arc<Track>, usize)> {
        let sessions: Vec<Arc<Session>> = self.sessions.lock().values().cloned().collect();
        let (session, (track, k, _, _)) = sessions
            .iter()
            .filter_map(|s| s.next(class).map(|next| (s, next)))
            .min_by_key(|(_, (_, k, rank, served))| (*rank, *served, *k))?;
        session.state.lock().last_served = Some(Instant::now());
        Some((track, k))
    }

    /// True when a session has a segment that can run now.
    pub fn has_runnable(&self) -> bool {
        let sessions: Vec<Arc<Session>> = self.sessions.lock().values().cloned().collect();
        sessions.iter().any(|s| s.has_runnable())
    }

    /// Drop idle sessions and the tracks nobody plays; return the number of sessions dropped.
    pub fn reap(&self) -> usize {
        let timeout = self.idle_timeout();
        let now = Instant::now();
        let mut reaped = Vec::new();
        self.sessions.lock().retain(|id, session| {
            let closed = session.closed();
            let keep = closed.is_none() && !session.idle(now, timeout);
            if !keep {
                session.close("idle");
                tracing::info!(session = %id, "ended after {:.0?} idle", timeout);
                reaped.push((session.number, closed.unwrap_or_else(|| "idle".to_owned())));
            }
            keep
        });
        let count = reaped.len();
        for (number, reason) in reaped {
            self.record_end(number, reason);
        }
        self.drop_orphans();
        if count > 0 {
            self.changed();
        }
        count
    }

    /// Drop the tracks no session plays.
    fn drop_orphans(&self) {
        // Held throughout, so a `join` cannot attach to a track while it is dropped.
        let sessions = self.sessions.lock();
        let used: HashSet<u64> = sessions.values().map(|s| s.track().uid).collect();
        let mut tracks = self.tracks.lock();
        tracks.retain(|uid, _| used.contains(uid));
        self.by_id.lock().retain(|_, uid| tracks.contains_key(uid));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stemd_core::{DspMode, OutputRate, StemFormat};

    #[test]
    fn spans_merge_what_touches_and_cover_only_within_one_span() {
        let mut spans = Spans::default();
        spans.insert(10..20);
        spans.insert(30..40);
        assert!(spans.covers(&(12..18)));
        assert!(!spans.covers(&(15..35)), "a gap is not covered");
        assert_eq!(spans.overlaps(&(15..35)), vec![15..20, 30..35]);
        spans.insert(20..30);
        assert_eq!(spans.as_pairs(), vec![[10, 40]]);
        assert!(spans.covers(&(15..35)));
        spans.insert(0..5);
        assert_eq!(spans.as_pairs(), vec![[0, 5], [10, 40]]);
        assert_eq!(spans.span_of(4), Some(&(0..5)));
        assert_eq!(spans.span_of(5), None);
    }

    fn grid() -> Grid {
        // Segments of 1000 every 750 samples.
        Grid::new(100_000, 1000, 0.25).expect("a grid")
    }

    fn region(at: usize, ahead: usize, behind: usize, priority: u8) -> Region {
        Region {
            at,
            ahead,
            behind,
            priority,
        }
    }

    fn window(regions: &[Region]) -> WindowRequest {
        WindowRequest {
            regions: regions.to_vec(),
        }
    }

    #[test]
    fn segments_run_nearest_at_first_with_behind_counted_twice() {
        let planned = plan(&grid(), &window(&[region(10_000, 3000, 1500, 0)]));
        let order: Vec<(usize, usize)> = planned.iter().map(|(k, r)| (*k, r.distance)).collect();
        assert_eq!(
            order,
            vec![
                (13, 0),
                (12, 1),
                (14, 500),
                (15, 1250),
                (11, 1501),
                (16, 2000),
                (17, 2750)
            ]
        );
        assert!(planned.iter().all(|(_, r)| r.priority == 0));
    }

    #[test]
    fn a_segment_takes_its_best_rank_across_regions() {
        let request = window(&[
            region(50_000, 1000, 0, 1),
            region(10_000, 1000, 0, 0),
            region(10_000, 0, 0, 7),
        ]);
        let planned = plan(&grid(), &request);
        let order: Vec<(usize, u8)> = planned.iter().map(|(k, r)| (*k, r.priority)).collect();
        assert_eq!(order, vec![(13, 0), (14, 0), (66, 1), (67, 1)]);
        assert_eq!(planned[0].1.class(), Class::BeforeJobs);
        let idle = plan(&grid(), &window(&[region(0, 0, 0, BEFORE_JOBS)]));
        assert_eq!(idle[0].1.class(), Class::WhenIdle);
    }

    #[test]
    fn a_plan_stays_inside_the_track() {
        let planned = plan(&grid(), &window(&[region(99_999, 50_000, 2000, 0)]));
        assert!(planned.iter().all(|(k, _)| *k < grid().len()));
        let mut ks: Vec<usize> = planned.iter().map(|(k, _)| *k).collect();
        ks.sort_unstable();
        ks.dedup();
        assert_eq!(ks.len(), planned.len(), "a segment is planned twice");
    }

    #[test]
    fn a_window_past_its_limits_is_refused() {
        assert!(window(&[region(0, 0, 0, MAX_PRIORITY)]).check().is_ok());
        assert!(
            window(&[region(0, 0, 0, MAX_PRIORITY + 1)])
                .check()
                .is_err()
        );
        assert!(
            window(&vec![region(0, 0, 0, 0); MAX_REGIONS + 1])
                .check()
                .is_err()
        );
    }

    fn output(format: StemFormat) -> Output {
        Output {
            format,
            rate: OutputRate::default(),
            derived: false,
            dsp: DspMode::default().into(),
            parts: Parts::DEFAULT,
        }
    }

    #[test]
    fn the_gain_leaves_three_decibels_over_the_mix_peak() {
        assert_eq!(gain(&output(StemFormat::Pcm32), Some(4.0)), 1.0);
        assert_eq!(gain(&output(StemFormat::Pcm16), Some(0.5)), 1.0);
        let full_scale = gain(&output(StemFormat::Flac), None);
        assert!((full_scale - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        let loud = gain(&output(StemFormat::Pcm16), Some(0.9));
        assert!((loud - 1.0 / (0.9 * std::f32::consts::SQRT_2)).abs() < 1e-6);
    }

    fn spec() -> TrackSpec {
        TrackSpec {
            frames: 100_000,
            channels: 2,
            sample_rate: 44_100,
            grid: grid(),
            normalises: true,
            parts: Parts::DEFAULT,
        }
    }

    fn audio(frames: usize, seed: f32) -> Audio {
        Audio::new(
            (0..2)
                .map(|c| (0..frames).map(|i| ((i + c) as f32 * seed).sin()).collect())
                .collect(),
            44_100,
        )
    }

    fn request() -> Request {
        Request {
            output: output(StemFormat::Pcm32),
            in_format: PcmFormat::F32le,
            track: spec(),
            norm: None,
            peak: None,
            model_id: "model".into(),
        }
    }

    fn settled(sessions: &Sessions, session: &Session, mix: &Audio) -> Arc<Track> {
        let track = session.track();
        match track.upload(0, mix).expect("upload") {
            Some(Verdict::Verified(id)) => sessions.settle(&track, &id),
            other => panic!("the upload came to {other:?}"),
        }
    }

    #[test]
    fn two_sessions_at_one_rank_take_turns_and_a_better_rank_goes_first() {
        let sessions = Sessions::new(Duration::from_secs(30), 4);
        let a = sessions.open(request()).expect("open");
        let b = sessions.open(request()).expect("open");
        for (i, session) in [&a, &b].into_iter().enumerate() {
            session
                .track()
                .upload(0, &audio(100_000, 0.01 + i as f32 / 100.0))
                .expect("upload");
            session
                .set_window(&window(&[region(10_000, 3000, 0, 0)]))
                .expect("window");
        }
        let mut order = Vec::new();
        for _ in 0..6 {
            let (track, k) = sessions.next(Class::BeforeJobs).expect("a segment");
            order.push(if track.uid == a.track().uid { 'a' } else { 'b' });
            track.done(k);
        }
        assert!(
            order.windows(2).all(|w| w[0] != w[1]),
            "the sessions did not take turns: {order:?}"
        );

        b.set_window(&window(&[region(50_000, 0, 0, 0)]))
            .expect("window");
        a.set_window(&window(&[region(60_000, 0, 0, 1)]))
            .expect("window");
        let (track, _) = sessions.next(Class::BeforeJobs).expect("a segment");
        assert_eq!(track.uid, b.track().uid, "priority 0 waited for priority 1");
        assert!(sessions.next(Class::WhenIdle).is_none());
    }

    #[test]
    fn a_track_id_covers_every_sample_and_the_geometry() {
        let a = audio(1000, 0.01);
        let mut b = a.clone();
        b.data[1][999] = f32::from_bits(b.data[1][999].to_bits() ^ 1);
        assert_ne!(track_id(&a), track_id(&b));
        let c = Audio::new(a.data.clone(), 48_000);
        assert_ne!(track_id(&a), track_id(&c));
    }

    #[test]
    fn two_sessions_on_the_same_audio_end_up_on_one_track() {
        let sessions = Sessions::default();
        let mix = audio(100_000, 0.01);
        let a = sessions.open(request()).expect("open");
        let b = sessions.open(request()).expect("open");
        assert!(
            !Arc::ptr_eq(&a.track(), &b.track()),
            "each starts on its own"
        );

        let first = settled(&sessions, &a, &mix);
        let merged_from = b.track();
        let second = settled(&sessions, &b, &mix);
        assert!(Arc::ptr_eq(&first, &second));
        assert!(Arc::ptr_eq(&a.track(), &b.track()));
        assert!(
            merged_from.closed().is_some(),
            "the track b left is still open"
        );
        assert!(b.closed().is_none(), "b went down with its old track");

        let c = sessions.open(request()).expect("open");
        let other = settled(&sessions, &c, &audio(100_000, 0.02));
        assert!(!Arc::ptr_eq(&other, &first), "different audio was merged");

        sessions.remove(&a.id, "ended by the client");
        assert!(sessions.has_track(first.uid), "b still plays it");
        sessions.remove(&b.id, "ended by the client");
        assert!(!sessions.has_track(first.uid));
    }

    #[test]
    fn an_upload_that_differs_from_what_the_track_holds_is_refused() {
        let sessions = Sessions::default();
        let a = sessions.open(request()).expect("open");
        let mix = audio(100_000, 0.01);
        let track = a.track();
        track.upload(0, &excerpt(&mix, 0, 60_000)).expect("first");
        track
            .upload(50_000, &excerpt(&mix, 50_000, 70_000))
            .expect("the same samples again");
        let err = track
            .upload(40_000, &audio(20_000, 0.02))
            .expect_err("different");
        assert!(matches!(err, UploadError::Differs(40_000, 60_000)), "{err}");
    }

    #[test]
    fn client_statistics_that_differ_from_the_mix_keep_the_track_unshared() {
        let sessions = Sessions::default();
        let mix = audio(100_000, 0.01);
        let mut wrong = request();
        wrong.norm = Some(Normalisation::new(0.5, 2.0));
        let session = sessions.open(wrong).expect("open");
        let verdict = session.track().upload(0, &mix).expect("upload");
        assert_eq!(verdict, Some(Verdict::Unpublishable(track_id(&mix))));
        assert_eq!(session.track().verified(), None);
    }

    #[test]
    fn a_range_not_ready_is_refused() {
        let sessions = Sessions::default();
        let session = sessions.open(request()).expect("open");
        let err = session
            .read("vocals", 0, 10)
            .err()
            .expect("nothing is ready");
        assert!(format!("{err}").contains("not ready"), "{err}");
    }

    #[test]
    fn pieces_at_another_rate_tile_the_output_without_gaps() {
        for out in [24_000u64, 48_000, 96_000] {
            for stride in [257_985usize, 441_600] {
                let ends: Vec<usize> = (0..10)
                    .map(|k| output_index(k * stride, 44_100, out))
                    .collect();
                for w in ends.windows(2) {
                    let expected = (stride as u64 * out) as f64 / 44_100.0;
                    let got = (w[1] - w[0]) as f64;
                    assert!(
                        (got - expected).abs() < 1.0,
                        "{out} Hz, stride {stride}: {got}"
                    );
                }
            }
        }
        assert_eq!(output_index(441_000, 44_100, 48_000), 480_000);
    }

    #[test]
    fn a_polyphase_read_to_the_end_reaches_the_last_output_sample() {
        let frames = 100_000;
        assert_ne!(frames % 147, 0);
        let mut req = request();
        req.output.rate = OutputRate::Hz96000;
        req.output.dsp = DspMode::Mode1.into();
        let sessions = Sessions::default();
        let session = sessions.open(req).expect("open");
        let track = session.track();
        track.upload(0, &audio(frames, 0.01)).expect("upload");
        track
            .store(0, frames, &crate::testkit::stems(frames).1)
            .expect("store");
        for from in [0, frames - 10_000] {
            let got = session.read("vocals", from, frames).expect("read");
            assert_eq!(
                got.audio.frames(),
                output_index(frames, 44_100, 96_000) - output_index(from, 44_100, 96_000),
                "from {from}"
            );
        }
    }

    #[test]
    fn the_activity_counts_uploads_plans_and_stored_pieces() {
        let sessions = Sessions::new(Duration::from_secs(30), 4);
        let session = sessions.open(request()).expect("open");
        let track = session.track();
        track
            .upload(0, &audio(40_000, 0.01))
            .expect("upload the first part");
        session
            .set_window(&window(&[region(0, 100_000, 0, 0)]))
            .expect("window");
        track.done(0);
        session.set_cached([1]);

        let [activity] = sessions.activity().try_into().expect("one session");
        assert_eq!(activity.number, 1);
        assert_eq!(activity.pieces, track.spec.grid.len());
        assert_eq!(activity.ready, 1, "the stored piece");
        assert_eq!(activity.arrived, 40_000);
        assert_eq!(activity.covered, track.piece(1).len(), "the stored piece");
        assert_eq!(activity.normal, 0);
        assert!(activity.high > 0, "{activity:?}");
        assert!(!activity.runnable, "runnable without the mix statistics");

        let mix = audio(100_000, 0.01);
        track
            .upload(40_000, &excerpt(&mix, 40_000, 100_000))
            .expect("upload the rest");
        let [activity] = sessions.activity().try_into().expect("one session");
        assert_eq!(activity.arrived, 100_000);
        assert!(activity.runnable, "{activity:?}");
    }

    #[test]
    fn an_ended_session_is_kept_with_its_reason() {
        let sessions = Sessions::new(Duration::from_millis(20), 4);
        let first = sessions.open(request()).expect("open");
        let second = sessions.open(request()).expect("open");
        let third = sessions.open(request()).expect("open");
        third.track().close("segment 3: out of memory");
        sessions.remove(&first.id, "ended by the client");
        std::thread::sleep(Duration::from_millis(40));
        sessions.reap();
        let fourth = sessions.open(request()).expect("open");
        sessions.close_all("the server switched models");
        let ended: Vec<(u64, String)> = sessions
            .ended()
            .into_iter()
            .map(|e| (e.number, e.reason))
            .collect();
        assert_eq!(
            ended,
            [
                (first.number, "ended by the client".to_owned()),
                (second.number, "idle".to_owned()),
                (third.number, "segment 3: out of memory".to_owned()),
                (fourth.number, "the server switched models".to_owned()),
            ]
        );
        assert!(sessions.activity().is_empty());
    }

    #[test]
    fn a_quiet_session_with_nothing_to_run_is_reaped() {
        let sessions = Sessions::new(Duration::from_millis(20), 4);
        let session = sessions.open(request()).expect("open");
        assert_eq!(sessions.reap(), 0, "reaped at once");
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(sessions.reap(), 1);
        assert!(session.closed().is_some());
        assert!(!sessions.has_track(session.track().uid));
    }
}
