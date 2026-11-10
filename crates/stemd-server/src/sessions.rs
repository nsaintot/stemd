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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use stemd_core::{Audio, Grid, Normalisation, Parts, PcmFormat, Stems, resample};

use crate::cache::Output;
use crate::pieces::{Pieces, type_dir};

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

/// How soon a client needs a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    /// Ahead of whole-track jobs.
    Critical,
    /// When nothing else is waiting.
    Bonus,
}

/// One side of a window around the needle.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Margin {
    pub frames: usize,
    pub priority: Priority,
}

/// What a client wants separated next.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct WindowRequest {
    /// Playhead, in track samples.
    pub needle: usize,
    pub ahead: Margin,
    pub behind: Margin,
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
}

/// The segments a window asks for, in the order to run them.
///
/// The segments under the needle come first, at the higher of the two margins'
/// priorities. Then both sides outward by distance, two from the side with the
/// higher priority for each one from the other; ahead wins a tie.
pub fn plan(grid: &Grid, request: &WindowRequest) -> Vec<(usize, Priority)> {
    let frames = grid.length().unsigned_abs() as usize;
    if frames == 0 {
        return Vec::new();
    }
    let needle = request.needle.min(frames - 1);
    let as_i32 = |v: usize| i32::try_from(v.min(frames)).unwrap_or(i32::MAX);

    let under: Vec<usize> = grid.covering(as_i32(needle), as_i32(needle + 1)).collect();
    let ahead: Vec<usize> = grid
        .covering(
            as_i32(needle),
            as_i32(needle.saturating_add(request.ahead.frames)),
        )
        .filter(|k| !under.contains(k))
        .collect();
    let behind: Vec<usize> = grid
        .covering(
            as_i32(needle.saturating_sub(request.behind.frames)),
            as_i32(needle),
        )
        .rev()
        .filter(|k| !under.contains(k))
        .collect();

    let first = request.ahead.priority.min(request.behind.priority);
    let mut order: Vec<(usize, Priority)> = under.into_iter().map(|k| (k, first)).collect();

    let ahead_leads = request.ahead.priority <= request.behind.priority;
    let (lead, lead_priority, follow, follow_priority) = if ahead_leads {
        (
            ahead,
            request.ahead.priority,
            behind,
            request.behind.priority,
        )
    } else {
        (
            behind,
            request.behind.priority,
            ahead,
            request.ahead.priority,
        )
    };
    let (mut lead, mut follow) = (lead.into_iter(), follow.into_iter());
    loop {
        let a = lead.next();
        let b = lead.next();
        let c = follow.next();
        if a.is_none() && c.is_none() {
            break;
        }
        order.extend([a, b].into_iter().flatten().map(|k| (k, lead_priority)));
        order.extend(c.map(|k| (k, follow_priority)));
    }
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
        let converted = resample::to_rate_with(&audio, output.rate.hz(), output.dsp)?;
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

    fn summary(&self) -> TrackSummary {
        let state = self.state.lock();
        TrackSummary {
            arrived: state.arrived.as_pairs(),
            ready: state.ready.as_pairs(),
            needs_stats: self.spec.normalises && state.norm.is_none(),
            verified: matches!(state.verdict, Some(Verdict::Verified(_))),
            closed: state.closed.clone(),
        }
    }
}

struct TrackSummary {
    arrived: Vec<[usize; 2]>,
    ready: Vec<[usize; 2]>,
    needs_stats: bool,
    verified: bool,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track: Option<String>,
    /// True once the whole upload has hashed to `track`.
    pub verified: bool,
    pub frames: usize,
    pub sample_rate: u32,
    pub output_sample_rate: u32,
    pub channels: usize,
    pub format: String,
    pub segment_frames: i32,
    pub stride_frames: i32,
    pub segments: usize,
    pub stems: Vec<StemView>,
    pub arrived: Vec<[usize; 2]>,
    pub ready: Vec<[usize; 2]>,
    /// Pieces on disk for this output, ready to fetch whole.
    pub pieces: Vec<usize>,
    /// Segments asked for and not run yet.
    pub planned: usize,
    /// Set while segments wait on something other than the worker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waiting_for: Option<&'static str>,
    pub idle_timeout_secs: u64,
}

#[derive(Debug, Serialize)]
pub struct StemView {
    pub name: &'static str,
    /// Multiply by `1.0 / gain` to restore the level.
    pub gain: f32,
}

/// A range read back from a track, at the output rate.
pub struct Excerpt {
    pub audio: Audio,
    /// First output-rate sample.
    pub from: usize,
}

struct SessionState {
    plan: VecDeque<(usize, Priority)>,
    planned_at: Instant,
    last_seen: Instant,
    closed: Option<String>,
}

pub struct Session {
    pub id: String,
    pub output: Output,
    pub in_format: PcmFormat,
    pub gain: f32,
    /// The track this session plays, which changes once when its upload
    /// completes and another session's track already holds the same audio.
    track: Mutex<Arc<Track>>,
    /// Pieces of this output already on disk, which no segment needs to remake.
    cached: Mutex<HashSet<usize>>,
    state: Mutex<SessionState>,
}

impl Session {
    fn new(
        id: String,
        output: Output,
        in_format: PcmFormat,
        peak: Option<f32>,
        track: Arc<Track>,
    ) -> Self {
        let now = Instant::now();
        Self {
            id,
            gain: gain(&output, peak),
            output,
            in_format,
            track: Mutex::new(track),
            cached: Mutex::new(HashSet::new()),
            state: Mutex::new(SessionState {
                plan: VecDeque::new(),
                planned_at: now,
                last_seen: now,
                closed: None,
            }),
        }
    }

    pub fn track(&self) -> Arc<Track> {
        Arc::clone(&self.track.lock())
    }

    /// Record the pieces of this output already on disk, and drop the planned
    /// segments they make unnecessary.
    pub fn set_cached(&self, pieces: impl IntoIterator<Item = usize>) {
        self.cached.lock().extend(pieces);
        let mut state = self.state.lock();
        let plan = std::mem::take(&mut state.plan);
        state.plan = plan
            .into_iter()
            .filter(|(k, _)| !self.covered(*k))
            .collect();
    }

    /// True when every piece segment `k` feeds is on disk already.
    fn covered(&self, k: usize) -> bool {
        let cached = self.cached.lock();
        let pieces = self.track().spec.grid.len();
        cached.contains(&k) && (k + 1 >= pieces || cached.contains(&(k + 1)))
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
        let planned: VecDeque<(usize, Priority)> = plan(&track.spec.grid, request)
            .into_iter()
            .filter(|(k, _)| !track.is_done(*k) && !self.covered(*k))
            .collect();
        let mut state = self.state.lock();
        let now = Instant::now();
        state.last_seen = now;
        state.planned_at = now;
        state.plan = planned;
        Ok(())
    }

    /// The first planned segment at `priority` that can run, and when the plan was set.
    fn next(&self, priority: Priority) -> Option<(Arc<Track>, usize, Instant)> {
        let track = self.track();
        let mut state = self.state.lock();
        if state.closed.is_some() {
            return None;
        }
        let held = track.state.lock();
        state.plan.retain(|(k, _)| !held.done[*k]);
        let k = state
            .plan
            .iter()
            .find(|(k, p)| *p == priority && track.runnable(&held, *k))
            .map(|(k, _)| *k)?;
        drop(held);
        Some((track, k, state.planned_at))
    }

    pub fn close(&self, reason: impl Into<String>) {
        let mut state = self.state.lock();
        if state.closed.is_none() {
            state.closed = Some(reason.into());
        }
        state.plan.clear();
    }

    /// True when nothing can run and the client has been quiet for `timeout`.
    fn idle(&self, now: Instant, timeout: Duration) -> bool {
        let track = self.track();
        let state = self.state.lock();
        let held = track.state.lock();
        let waiting = state.plan.iter().any(|(k, _)| track.runnable(&held, *k));
        !waiting && now.duration_since(state.last_seen) > timeout
    }

    /// Part `name` over `from..to`, at the output rate.
    pub fn read(&self, name: &str, from: usize, to: usize) -> Result<Excerpt> {
        if let Some(reason) = self.closed() {
            bail!("this session is closed: {reason}");
        }
        self.track().read(&self.output, name, from, to)
    }

    pub fn view(&self, idle: Duration, pieces: Vec<usize>) -> View {
        let track = self.track();
        let summary = track.summary();
        let (planned, waiting) = {
            let state = self.state.lock();
            let held = track.state.lock();
            let waiting = state.plan.iter().any(|(k, _)| !track.runnable(&held, *k));
            (state.plan.len(), waiting)
        };
        let closed = self.state.lock().closed.clone().or(summary.closed);
        let spec = &track.spec;
        View {
            id: self.id.clone(),
            state: if closed.is_some() { "closed" } else { "open" },
            error: closed,
            track: track.verified(),
            verified: summary.verified,
            frames: spec.frames,
            sample_rate: spec.sample_rate,
            output_sample_rate: self.output.rate.hz(),
            channels: spec.channels,
            format: self.output.format.to_string(),
            segment_frames: spec.grid.segment(),
            stride_frames: spec.grid.stride(),
            segments: spec.grid.len(),
            stems: served(&self.output)
                .into_iter()
                .map(|name| StemView {
                    name,
                    gain: self.gain,
                })
                .collect(),
            arrived: summary.arrived,
            ready: summary.ready,
            pieces,
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
    seq: AtomicU64,
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
            seq: AtomicU64::new(0),
        }
    }

    pub fn set_limits(&self, idle: Duration, max: usize) {
        *self.limits.lock() = (idle, max);
    }

    pub fn idle_timeout(&self) -> Duration {
        self.limits.lock().0
    }

    pub fn set_pieces(&self, pieces: Arc<Pieces>) {
        *self.pieces.lock() = Some(pieces);
    }

    pub fn pieces(&self) -> Option<Arc<Pieces>> {
        self.pieces.lock().clone()
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
            request.output,
            request.in_format,
            request.peak,
            track,
        ));
        sessions.insert(session.id.clone(), Arc::clone(&session));
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
                if let Some(pieces) = self.pieces() {
                    pieces.publish(self, &track, 0..track.spec.frames);
                }
                for other in self.of_track(&track) {
                    other.set_cached(self.cached_for(&other));
                }
            }
        }
        Ok(())
    }

    /// The pieces of `session`'s output stored under its track's id.
    pub fn cached_for(&self, session: &Session) -> Vec<usize> {
        let track = session.track();
        let (Some(pieces), Some(id)) = (self.pieces(), track.verified()) else {
            return Vec::new();
        };
        pieces.cached(
            &track.model_id,
            &type_dir(&session.output, session.gain),
            &id,
            &served(&session.output),
            session.output.format.extension(),
            track.spec.grid.len(),
        )
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

    pub fn get(&self, id: &str) -> Option<Arc<Session>> {
        self.sessions.lock().get(id).cloned()
    }

    pub fn remove(&self, id: &str) -> bool {
        let removed = self.sessions.lock().remove(id);
        if let Some(session) = &removed {
            session.close("ended by the client");
        }
        self.drop_orphans();
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
        }
        for track in std::mem::take(&mut *self.tracks.lock()).values() {
            track.close(reason);
        }
        self.by_id.lock().clear();
        if !closed.is_empty() {
            tracing::info!("closed {} sessions: {reason}", closed.len());
        }
    }

    /// The next segment to run at `priority`: the oldest window's first.
    pub fn next(&self, priority: Priority) -> Option<(Arc<Track>, usize)> {
        let sessions: Vec<Arc<Session>> = self.sessions.lock().values().cloned().collect();
        sessions
            .into_iter()
            .filter_map(|s| s.next(priority))
            .min_by_key(|(_, _, at)| *at)
            .map(|(track, k, _)| (track, k))
    }

    /// Drop idle sessions and the tracks nobody plays; return the number of sessions dropped.
    pub fn reap(&self) -> usize {
        let timeout = self.idle_timeout();
        let now = Instant::now();
        let reaped = {
            let mut sessions = self.sessions.lock();
            let before = sessions.len();
            sessions.retain(|id, session| {
                let keep = session.closed().is_none() && !session.idle(now, timeout);
                if !keep {
                    session.close("idle");
                    tracing::info!(session = %id, "ended after {:.0?} idle", timeout);
                }
                keep
            });
            before - sessions.len()
        };
        self.drop_orphans();
        reaped
    }

    /// Drop the tracks no session plays.
    fn drop_orphans(&self) {
        let used: HashSet<u64> = self
            .sessions
            .lock()
            .values()
            .map(|s| s.track().uid)
            .collect();
        self.tracks.lock().retain(|uid, _| used.contains(uid));
        let tracks = self.tracks.lock();
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

    fn window(needle: usize, ahead: (usize, Priority), behind: (usize, Priority)) -> WindowRequest {
        WindowRequest {
            needle,
            ahead: Margin {
                frames: ahead.0,
                priority: ahead.1,
            },
            behind: Margin {
                frames: behind.0,
                priority: behind.1,
            },
        }
    }

    #[test]
    fn a_plan_starts_under_the_needle_then_takes_two_ahead_for_one_behind() {
        use Priority::{Bonus, Critical};
        let planned = plan(&grid(), &window(10_000, (3000, Critical), (1500, Bonus)));
        let order: Vec<usize> = planned.iter().map(|(k, _)| *k).collect();
        assert_eq!(order, vec![13, 14, 15, 12, 16, 17, 11]);
        let priorities: Vec<Priority> = planned.iter().map(|(_, p)| *p).collect();
        assert_eq!(
            priorities,
            vec![
                Critical, Critical, Critical, Bonus, Critical, Critical, Bonus
            ]
        );
    }

    #[test]
    fn a_critical_behind_leads_a_bonus_ahead() {
        use Priority::{Bonus, Critical};
        let planned = plan(&grid(), &window(10_000, (1500, Bonus), (3000, Critical)));
        let order: Vec<usize> = planned.iter().map(|(k, _)| *k).collect();
        assert_eq!(order, vec![13, 12, 11, 14, 10, 9, 15]);
        assert_eq!(
            planned[0].1, Critical,
            "the needle takes the higher priority"
        );
    }

    #[test]
    fn a_plan_stays_inside_the_track() {
        let request = window(
            99_999,
            (50_000, Priority::Critical),
            (2000, Priority::Critical),
        );
        let planned = plan(&grid(), &request);
        assert!(planned.iter().all(|(k, _)| *k < grid().len()));
        let mut ks: Vec<usize> = planned.iter().map(|(k, _)| *k).collect();
        ks.sort_unstable();
        ks.dedup();
        assert_eq!(ks.len(), planned.len(), "a segment is planned twice");
    }

    fn output(format: StemFormat) -> Output {
        Output {
            format,
            rate: OutputRate::default(),
            derived: false,
            dsp: DspMode::default(),
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

        sessions.remove(&a.id);
        assert!(sessions.has_track(first.uid), "b still plays it");
        sessions.remove(&b.id);
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
