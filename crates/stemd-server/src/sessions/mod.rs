//! Sessions separate tracks in pieces, in the order their clients ask for.
//!
//! A [`Track`] holds one track's audio and the parts separated so far. A [`Session`] is one
//! client's (one deck's) view of a track: its output format, its window around the playhead and its
//! idle timer.
//!
//! A session starts on a track of its own. When its whole upload has arrived, the server hashes it
//! into a [`track_id`]. If another session's track already has that id, the session moves onto that
//! track, so two decks playing one song share the work and the memory from then on.
//!
//! | Track field | Meaning                                                  |
//! |-------------|----------------------------------------------------------|
//! | `arrived`   | track samples uploaded so far                            |
//! | `ready`     | samples for which every covering segment has run         |
//! | `verdict`   | the track id, once the whole upload has been hashed      |
//!
//! Samples are counted at the model's rate throughout. Piece `k` is the samples
//! `[k * stride, (k + 1) * stride)`.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use parking_lot::Mutex;
use stemd_core::{Audio, Normalisation, Parts, PcmFormat};

use crate::cache::Cache;
use crate::output::Output;

pub mod plan;
mod spans;
mod stored;
mod track;
mod view;

pub use plan::{Class, Rank, WindowRequest, plan};
pub use spans::Spans;
pub use track::{Track, TrackSpec, UploadError, Verdict};
pub use view::{Activity, Ended, GridView, StemView, TrackView, View};

/// Samples of context resampled on each side of a range that is read at another rate.
pub const RESAMPLE_CONTEXT: usize = 8192;

/// The number of ended sessions kept for the window.
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
    /// The first sample, at the output rate.
    pub from: usize,
}

struct SessionState {
    plan: VecDeque<(usize, Rank)>,
    /// The client's last window, planned again when the session moves onto another track.
    window: Option<WindowRequest>,
    /// When the worker last took one of this session's segments.
    last_served: Option<Instant>,
    last_seen: Instant,
    closed: Option<String>,
}

pub struct Session {
    pub id: String,
    /// The session's number, counting from 1 since the server started.
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
        gain: f32,
        track: Arc<Track>,
    ) -> Self {
        let now = Instant::now();
        Self {
            id,
            number,
            gain,
            output,
            in_format,
            track: Mutex::new(track),
            cached: Mutex::new(HashSet::new()),
            file: Mutex::new(None),
            state: Mutex::new(SessionState {
                plan: VecDeque::new(),
                window: None,
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

    /// True when every piece that segment `k` feeds is on disk. At another output rate, a piece
    /// also needs the segments within [`RESAMPLE_CONTEXT`] of it.
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

    /// An error naming why the session is closed, if it is.
    pub fn ensure_open(&self) -> Result<()> {
        if let Some(reason) = self.closed() {
            bail!("this session is closed: {reason}");
        }
        Ok(())
    }

    /// Record a request from the client.
    pub fn touch(&self) {
        self.state.lock().last_seen = Instant::now();
    }

    /// Replace the plan with the segments `request` asks for, minus those that have run.
    pub fn set_window(&self, request: &WindowRequest) -> Result<()> {
        self.ensure_open()?;
        let planned = self.plan_for(request);
        let mut state = self.state.lock();
        state.last_seen = Instant::now();
        state.plan = planned;
        state.window = Some(request.clone());
        Ok(())
    }

    /// Plan the last window again on the track this session plays now.
    fn replan(&self) {
        let Some(request) = self.state.lock().window.clone() else {
            return;
        };
        let planned = self.plan_for(&request);
        let mut state = self.state.lock();
        if state.closed.is_none() {
            state.plan = planned;
        }
    }

    /// The segments `request` asks for, minus those that have run.
    fn plan_for(&self, request: &WindowRequest) -> VecDeque<(usize, Rank)> {
        let track = self.track();
        plan(&track.spec.grid, request)
            .into_iter()
            .filter(|(k, _)| !track.is_done(*k) && !self.covered(*k))
            .collect()
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
        self.ensure_open()?;
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
            output: crate::output::OutputView::of(&self.output, Some(self.gain)),
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

const fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[derive(Debug, thiserror::Error)]
#[error("{0} sessions are open, the most this server allows")]
pub struct TooMany(pub usize);

/// Why a session could not open on a track of its own.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    #[error(transparent)]
    TooMany(#[from] TooMany),
    #[error(
        "this track needs {:.1} GB of memory; open tracks may hold {:.1} GB",
        gb(*.need),
        gb(*.budget)
    )]
    TooLarge { need: u64, budget: u64 },
    #[error(
        "open tracks hold {:.1} of {:.1} GB of memory; this track needs {:.1} GB",
        gb(*.held),
        gb(*.budget),
        gb(*.need)
    )]
    Full { need: u64, held: u64, budget: u64 },
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

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
    /// The stored track that the open names with `track=`, if any.
    pub claim: Option<Claim>,
}

/// A track id that an open names, and the gain its stored pieces were written at.
pub struct Claim {
    pub id: String,
    pub gain: f32,
}

/// Every open session and the tracks they play.
pub struct Sessions {
    sessions: Mutex<BTreeMap<String, Arc<Session>>>,
    /// Tracks keyed by [`Track::uid`].
    tracks: Mutex<BTreeMap<u64, Arc<Track>>>,
    /// The uids of tracks whose whole upload has been hashed, keyed by track id and part set.
    by_id: Mutex<BTreeMap<String, u64>>,
    limits: Mutex<(Duration, usize)>,
    /// Bytes the open tracks may hold together, by [`TrackSpec::memory`].
    memory_budget: AtomicU64,
    cache: Arc<Cache>,
    /// The last [`ENDED_KEPT`] sessions to end, oldest first.
    ended: Mutex<VecDeque<Ended>>,
    seq: AtomicU64,
    /// Counts the changes that any view can show, for long polls.
    changes: tokio::sync::watch::Sender<u64>,
}

impl Sessions {
    /// Create the sessions on `cache`, which keeps the tracks they play from eviction.
    pub fn new(idle: Duration, max: usize, cache: Arc<Cache>) -> Arc<Self> {
        let sessions = Arc::new(Self {
            sessions: Mutex::new(BTreeMap::new()),
            tracks: Mutex::new(BTreeMap::new()),
            by_id: Mutex::new(BTreeMap::new()),
            limits: Mutex::new((idle, max)),
            memory_budget: AtomicU64::new(crate::disk::physical_memory().unwrap_or(8 << 30) / 2),
            cache,
            ended: Mutex::new(VecDeque::new()),
            seq: AtomicU64::new(0),
            changes: tokio::sync::watch::Sender::new(0),
        });
        let weak = Arc::downgrade(&sessions);
        sessions.cache.set_in_use(Box::new(move || {
            weak.upgrade()
                .map(|s| s.open_track_ids())
                .unwrap_or_default()
        }));
        let weak = Arc::downgrade(&sessions);
        sessions.cache.set_on_change(Box::new(move || {
            if let Some(sessions) = weak.upgrade() {
                sessions.changed();
            }
        }));
        sessions
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

    pub fn set_limits(&self, idle: Duration, max: usize) {
        *self.limits.lock() = (idle, max);
    }

    #[cfg(test)]
    pub fn set_memory_budget(&self, bytes: u64) {
        self.memory_budget.store(bytes, Ordering::Relaxed);
    }

    pub fn idle_timeout(&self) -> Duration {
        self.limits.lock().0
    }

    /// The ids of the verified or claimed tracks that open sessions play.
    fn open_track_ids(&self) -> HashSet<String> {
        let tracks: Vec<Arc<Track>> = self.tracks.lock().values().cloned().collect();
        tracks.iter().filter_map(|t| t.stored_id()).collect()
    }

    pub fn cache(&self) -> &Arc<Cache> {
        &self.cache
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
    pub fn open(&self, request: Request) -> Result<Arc<Session>, OpenError> {
        let max = self.limits.lock().1;
        let mut sessions = self.sessions.lock();
        if sessions.len() >= max {
            return Err(TooMany(sessions.len()).into());
        }
        let need = request.track.memory();
        let budget = self.memory_budget.load(Ordering::Relaxed);
        if need > budget {
            return Err(OpenError::TooLarge { need, budget });
        }
        let held: u64 = self.tracks.lock().values().map(|t| t.spec.memory()).sum();
        if held + need > budget {
            return Err(OpenError::Full { need, held, budget });
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let gain = request
            .claim
            .as_ref()
            .map_or_else(|| request.output.gain(request.peak), |c| c.gain);
        let track = Arc::new(Track::new(
            seq,
            request.model_id.clone(),
            request.track,
            request.norm,
            request.claim.map(|c| c.id),
        ));
        self.tracks.lock().insert(seq, Arc::clone(&track));
        let session = Arc::new(Session::new(
            format!("s{seq}-{nanos:08x}"),
            seq + 1,
            request.output,
            request.in_format,
            gain,
            track,
        ));
        sessions.insert(session.id.clone(), Arc::clone(&session));
        drop(sessions);
        session.set_cached(self.cached_for(&session));
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
        let gain = output.gain(peak);
        let session = Arc::new(Session::new(
            format!("s{seq}-{nanos:08x}"),
            seq + 1,
            output,
            in_format,
            gain,
            Arc::clone(&track),
        ));
        sessions.insert(session.id.clone(), Arc::clone(&session));
        drop(sessions);
        // Write this output's pieces for the samples that are already ready.
        self.store_pieces(&track, 0..track.spec.frames);
        session.set_cached(self.cached_for(&session));
        self.changed();
        Ok(session)
    }

    /// Place `audio` at `offset` in `session`'s track.
    ///
    /// The upload that completes the track hashes it, which takes a few hundred ms, and files it
    /// under that id, merging it into an open track with the same audio.
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
            Some(Verdict::Mismatch(actual)) => self.refute(&track, &actual),
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

    /// End every session on `track`, whose upload hashed to `actual`, and record their file keys
    /// against `actual`.
    fn refute(&self, track: &Track, actual: &str) {
        let claim = track.claim.as_deref().unwrap_or_default();
        let reason = format!("upload is not track {claim}");
        tracing::warn!(track = %claim, "hash mismatch: upload is {actual}; sessions ended");
        track.close(reason.as_str());
        for session in self.of_track(track) {
            self.record_file(&session, actual);
            self.remove(&session.id, &reason);
        }
    }

    /// Record `session`'s file key against `track_id`, when it sent one.
    pub fn record_file(&self, session: &Session, track_id: &str) {
        let key = session.file.lock().clone();
        if let Some(key) = key
            && let Err(err) = self.cache.record_file(&key, track_id)
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

    /// File `track`, whose whole upload hashed to `id`, under that id.
    ///
    /// When another open track already holds the same audio and parts, every session on `track`
    /// moves onto that track, plans its window again there, and `track` closes. Returns the track
    /// the sessions play now.
    pub fn settle(&self, track: &Arc<Track>, id: &str) -> Arc<Track> {
        let (existing, moved) = self.merge(track, id);
        for session in moved {
            session.replan();
        }
        existing
    }

    /// File `track` under `id`, or move its sessions onto the open track with that id and return
    /// them.
    fn merge(&self, track: &Arc<Track>, id: &str) -> (Arc<Track>, Vec<Arc<Session>>) {
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
            return (Arc::clone(track), Vec::new());
        };
        let mut moved = Vec::new();
        for session in sessions.values() {
            let mut held = session.track.lock();
            if held.uid == track.uid {
                *held = Arc::clone(&existing);
                moved.push(Arc::clone(session));
            }
        }
        track.close(format!("merged into the open track {id}"));
        tracks.remove(&track.uid);
        tracing::info!("track {id}: a second session joined the one already open");
        (existing, moved)
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
                match &closed {
                    Some(reason) => tracing::info!(session = %id, "ended: {reason}"),
                    None => tracing::info!(session = %id, "ended after {:.0?} idle", timeout),
                }
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
        // Hold the lock throughout, so a `join` cannot attach to a track while it is being dropped.
        let sessions = self.sessions.lock();
        let used: HashSet<u64> = sessions.values().map(|s| s.track().uid).collect();
        let mut tracks = self.tracks.lock();
        tracks.retain(|uid, _| used.contains(uid));
        self.by_id.lock().retain(|_, uid| tracks.contains_key(uid));
    }
}

#[cfg(test)]
mod tests;
