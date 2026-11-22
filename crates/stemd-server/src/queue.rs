//! Job queue and the single separation worker.
//!
//! Separation is serialised: one segment at a time on one worker. Whole-track jobs
//! run FIFO, with a position to report, backpressure when saturated, and
//! cancellation that covers the running job as well as the waiting ones.
//!
//! Between segments the worker takes session work, in this order:
//!
//! 1. a staged model, when no job is mid-track;
//! 2. a high-priority session segment, also between two segments of a running job;
//! 3. the next waiting job;
//! 4. a normal-priority session segment.
//!
//! The worker builds every separator it runs. See [`BuildSeparator`].

use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use anyhow::Result;
use parking_lot::{Condvar, Mutex};
use stemd_core::{
    Audio, BackendInfo, Cancelled, Pass, Progress, Segmenting, Separate, Stage, Stems, whole_track,
};

use crate::cache::Output;

use crate::cache::Cache;
use crate::jobs::Job;
use crate::pieces::{Consolidation, Pieces, TrackInfo, stems_of, type_dir};
use crate::sessions::{
    Class, Session, Sessions, Track, View, excerpt, gain, output_index, track_id,
};

/// How a separator reaches the worker: as a recipe, never as a finished model.
///
/// MLX weights must be allocated on the thread that will evaluate them. The CUDA
/// backend keeps its stream registry in thread-local storage, so a model built on
/// one thread cannot be evaluated on another. Handing over a closure rather than
/// a built model makes that unrepresentable. See [`stemd_core::mlx`].
pub type BuildSeparator = Box<dyn FnOnce() -> Result<Box<dyn Separate>> + Send>;

pub struct QueuedWork {
    pub job: Arc<Job>,
    pub mix: Audio,
    /// Format, rate and part set the client asked for.
    pub output: Output,
    /// The `model_id` the job was submitted under, for the pieces it writes.
    pub model_id: String,
    /// Submitted over the API. Its stems are kept as session pieces too, and
    /// turning the server off cancels it. Off for a drop, whose stems end up
    /// only in its folder.
    pub from_api: bool,
    /// The client's file key, recorded against the track id when the pieces are written.
    pub file: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("queue is full ({depth} waiting)")]
pub struct QueueFull {
    pub depth: usize,
}

/// The job the worker has in flight, and the switch that stops it.
///
/// The flag is created fresh per job rather than reset between them, so a
/// `cancel` that raced the end of one job cannot leak onto the next: it sets a
/// flag on a `Running` the worker has already dropped.
struct Running {
    id: String,
    from_api: bool,
    /// See [`Pending::number`].
    number: Option<u64>,
    /// The job, for its progress in [`Queue::client_jobs`].
    job: Arc<Job>,
    cancel: Arc<AtomicBool>,
}

/// Segment times [`Queue::segment_run_seconds`] takes the median of.
pub const SPEED_KEPT: usize = 16;

/// The last [`SPEED_KEPT`] segment times.
#[derive(Debug, Default)]
struct Speed {
    recent: VecDeque<f64>,
}

impl Speed {
    /// Record that a segment took `secs`.
    fn record(&mut self, secs: f64) {
        if self.recent.len() == SPEED_KEPT {
            self.recent.pop_front();
        }
        self.recent.push_back(secs);
    }

    /// The median of the recent times, to the millisecond.
    fn median(&self) -> Option<f64> {
        let mut sorted: Vec<f64> = self.recent.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let median = match n {
            0 => return None,
            _ if n % 2 == 1 => sorted[n / 2],
            _ => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
        };
        Some((median * 1000.0).round() / 1000.0)
    }
}

/// A job waiting for the worker.
struct Pending {
    work: QueuedWork,
    /// Counts API jobs from 1 since the server started; `None` for a drop.
    number: Option<u64>,
}

/// An API job as the window shows it: no id, since a job id carries a prefix
/// of the track's hash.
#[derive(Clone, Debug)]
pub struct ClientJob {
    pub number: u64,
    pub running: bool,
    pub progress: Progress,
    /// A protocol 1 client waits on it.
    pub protocol_1: bool,
}

/// The running job, then the waiting ones. A job listed in both keeps only its
/// running entry.
fn running_first(running: Option<ClientJob>, waiting: Vec<ClientJob>) -> Vec<ClientJob> {
    let taken = running.as_ref().map(|r| r.number);
    running
        .into_iter()
        .chain(waiting.into_iter().filter(|w| Some(w.number) != taken))
        .collect()
}

/// A model waiting to be built, and the caller waiting to hear how it went.
struct Staged {
    build: BuildSeparator,
    /// What the installed model says about itself, or why it never loaded.
    /// Dropping this without sending is how a caller learns the worker went
    /// away underneath it.
    reply: mpsc::SyncSender<Result<BackendInfo>>,
}

struct Inner {
    pending: Mutex<VecDeque<Pending>>,
    /// The last API job's number.
    api_jobs: AtomicU64,
    ready: Condvar,
    running: Mutex<Option<Running>>,
    max_depth: usize,
    shutdown: AtomicBool,
    /// A separator waiting to be built and to replace the live one, collected between
    /// jobs. The worker owns its separator outright, so a switch requested
    /// mid-separation takes effect on the next track.
    incoming: Mutex<Option<Staged>>,
    /// What the separator the worker holds says about itself. Written by the worker
    /// as it installs, so it can never describe a model the worker has not got.
    ///
    /// `None` only between spawning the worker and its first build; [`Queue::start`]
    /// does not return until it is `Some`.
    info: Mutex<Option<BackendInfo>>,
    /// How the loaded separator cuts a track, written with `info`.
    segmenting: Mutex<Option<Segmenting>>,
    /// Whether the loaded separator needs the whole mix's normalisation.
    normalises: AtomicBool,
    sessions: Arc<Sessions>,
    /// Segment times on the loaded model.
    speed: Mutex<Speed>,
    /// Set while the worker runs a job or a segment.
    busy: AtomicBool,
    /// Set when the worker has left its loop and dropped its separator. The
    /// thread then parks and never exits: MLX's per-thread CUDA state throws
    /// from its destructor when a thread exits on Windows.
    left: AtomicBool,
}

pub struct Queue {
    inner: Arc<Inner>,
    /// Taken by whichever of [`Queue::stop`] or `Drop` gets there first.
    worker: Mutex<Option<JoinHandle<()>>>,
}

/// How often [`Queue::stop`] checks whether the worker is out yet.
const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(5);

/// How long [`Queue::stop`] waits for the worker to leave the separator.
///
/// The check is at segment boundaries, so it has to cover one segment on the
/// slowest preset with room for a loaded machine, and a model build already under
/// way when the quit arrives.
pub const STOP_GRACE: std::time::Duration = std::time::Duration::from_secs(10);

impl Queue {
    /// Spawn the worker thread and have it build the first separator.
    ///
    /// Blocks until it has, so a model that will not load is still a synchronous error
    /// at the call site. See [`BuildSeparator`] for why the worker allocates the
    /// weights.
    pub fn start(
        build: BuildSeparator,
        max_depth: usize,
        cache: Arc<Cache>,
    ) -> Result<(Self, BackendInfo)> {
        let inner = Arc::new(Inner {
            pending: Mutex::new(VecDeque::new()),
            api_jobs: AtomicU64::new(0),
            ready: Condvar::new(),
            running: Mutex::new(None),
            max_depth,
            shutdown: AtomicBool::new(false),
            incoming: Mutex::new(None),
            info: Mutex::new(None),
            segmenting: Mutex::new(None),
            normalises: AtomicBool::new(false),
            sessions: Arc::new(Sessions::default()),
            speed: Mutex::new(Speed::default()),
            busy: AtomicBool::new(false),
            left: AtomicBool::new(false),
        });

        let (reply, built) = mpsc::sync_channel(1);
        let worker_inner = Arc::clone(&inner);
        let worker = thread::Builder::new()
            .name("stemd-worker".into())
            .spawn(move || {
                // The first thing this thread does, and the reason it is handed
                // a recipe rather than a model.
                if let Some(separator) = worker_inner.build_here(build, &reply) {
                    worker_loop(&worker_inner, separator, &cache);
                }
                worker_inner.left.store(true, Ordering::SeqCst);
                loop {
                    thread::park();
                }
            })
            .expect("spawning the separation worker");

        match built.recv() {
            Ok(Ok(info)) => Ok((
                Self {
                    inner,
                    worker: Mutex::new(Some(worker)),
                },
                info,
            )),
            // The worker parks the moment the build fails; see `Inner::left`.
            Ok(Err(err)) => Err(err),
            Err(_) => {
                let _ = worker.join();
                anyhow::bail!("the separation worker died before it could load the model")
            }
        }
    }

    /// What the separator the worker currently holds says about itself.
    pub fn info(&self) -> BackendInfo {
        self.inner
            .info
            .lock()
            .clone()
            .expect("the worker publishes its model before `start` returns")
    }

    /// How the loaded separator cuts a track, or `None` when it cannot run
    /// segment by segment.
    pub fn segmenting(&self) -> Option<Segmenting> {
        *self.inner.segmenting.lock()
    }

    /// Whether the loaded separator needs the whole mix's normalisation.
    pub fn normalises(&self) -> bool {
        self.inner.normalises.load(Ordering::SeqCst)
    }

    pub fn sessions(&self) -> &Arc<Sessions> {
        &self.inner.sessions
    }

    /// Median seconds of the last [`SPEED_KEPT`] segments, `None` until one ran.
    pub fn segment_run_seconds(&self) -> Option<f64> {
        self.inner.speed.lock().median()
    }

    /// What `session`'s client sees now.
    pub fn view_of(&self, session: &Session) -> View {
        self.inner
            .sessions
            .view_of(session, self.segment_run_seconds())
    }

    /// When background consolidation may run: always on a GPU, on a CPU while the worker idles.
    pub fn consolidation_gate(&self) -> crate::pieces::Gate {
        let inner = Arc::downgrade(&self.inner);
        Box::new(move || {
            let Some(inner) = inner.upgrade() else {
                return true;
            };
            let cpu = inner
                .info
                .lock()
                .as_ref()
                .is_some_and(|info| info.device == "cpu");
            !cpu || (!inner.busy.load(Ordering::SeqCst)
                && inner.pending.lock().is_empty()
                && !inner.sessions.has_runnable())
        })
    }

    /// Tell the worker that session work may be waiting.
    pub fn wake(&self) {
        // Notify under `pending` so the worker cannot miss the wake between its check and its wait.
        let _parked = self.inner.pending.lock();
        self.inner.ready.notify_all();
    }

    /// Stop the worker: refuse new jobs, cancel the one in flight, and wait for it to
    /// come out of the separator. Returns false if it was still inside when `grace`
    /// ran out.
    ///
    /// Every exit path has to call this and wait for true. Ending the process with a
    /// forward pass running tears the GPU allocator down underneath it, from a
    /// destructor, where nothing catches the abort. The wait is one segment.
    pub fn stop(&self, grace: std::time::Duration) -> bool {
        {
            // Same reasoning as `Drop`: the worker checks `shutdown` while
            // holding `pending` and only then parks, so a notify sent inside
            // that window would reach nobody.
            let _parked = self.inner.pending.lock();
            self.inner.shutdown.store(true, Ordering::SeqCst);
        }
        // A switch blocked in `install` has to be let go too. Its reply channel
        // sits in `incoming`, which outlives the worker, so leaving it there
        // would park the switch thread for as long as the queue exists.
        if let Some(staged) = self.inner.incoming.lock().take() {
            let _ = staged
                .reply
                .send(Err(anyhow::anyhow!("the server is shutting down")));
        }
        // Released first: the worker takes `running` while holding `pending`, so
        // holding them in the other order here would deadlock.
        if let Some(running) = self.inner.running.lock().as_ref() {
            // How it stops, not that it stopped: the line worth reading is the
            // one the worker writes when it actually comes out of the model.
            tracing::debug!("stopping {} at its next segment", running.id);
            running.cancel.store(true, Ordering::Relaxed);
        }
        self.inner.ready.notify_all();

        let Some(worker) = self.worker.lock().take() else {
            return true;
        };
        // `JoinHandle` has no timed join, and an untimed one is exactly what
        // must not happen here: a worker wedged in the model would hang the quit
        // instead of crashing it, which is not an improvement.
        let deadline = std::time::Instant::now() + grace;
        // The thread only finishes if it panicked.
        while !self.inner.left.load(Ordering::SeqCst) && !worker.is_finished() {
            if std::time::Instant::now() >= deadline {
                tracing::warn!("the separation worker did not stop within {:.0?}", grace);
                return false;
            }
            thread::sleep(POLL_INTERVAL);
        }
        self.inner.left.load(Ordering::SeqCst)
    }

    /// Enqueue a job, returning its position (0 = next up).
    pub fn submit(&self, work: QueuedWork) -> Result<usize, QueueFull> {
        let mut pending = self.inner.pending.lock();
        if pending.len() >= self.inner.max_depth {
            return Err(QueueFull {
                depth: pending.len(),
            });
        }
        let position = pending.len();
        work.job
            .set_progress(queued_progress(position, position + 1));
        let number = work
            .from_api
            .then(|| self.inner.api_jobs.fetch_add(1, Ordering::Relaxed) + 1);
        pending.push_back(Pending { work, number });
        self.inner.ready.notify_one();
        Ok(position)
    }

    /// Stop a job, whether it is waiting or already under way. Returns false only if
    /// it is neither: finished, or never here.
    ///
    /// A waiting job is dropped outright. A running one stops at its next segment
    /// boundary, so this returns before the worker is free.
    pub fn cancel(&self, id: &str) -> bool {
        {
            let mut pending = self.inner.pending.lock();
            let before = pending.len();
            pending.retain(|p| p.work.job.id != id);
            if before != pending.len() {
                return true;
            }
        }
        // Lock released first. The worker holds `pending` while it takes
        // `running`, so holding them in the opposite order here would deadlock.
        let running = self.inner.running.lock();
        match running.as_ref() {
            Some(job) if job.id == id => {
                job.cancel.store(true, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }

    /// Cancel every job submitted over the API, waiting or running, and return
    /// how many. Drops keep their place.
    pub fn cancel_api_jobs(&self) -> usize {
        let dropped = {
            let mut pending = self.inner.pending.lock();
            let (api, kept): (VecDeque<_>, VecDeque<_>) =
                pending.drain(..).partition(|p| p.work.from_api);
            *pending = kept;
            api
        };
        for pending in &dropped {
            pending.work.job.cancelled();
        }
        // `pending` released first, as in `cancel`.
        let stopped = match self.inner.running.lock().as_ref() {
            Some(job) if job.from_api => {
                job.cancel.store(true, Ordering::Relaxed);
                1
            }
            _ => 0,
        };
        dropped.len() + stopped
    }

    pub fn depth(&self) -> usize {
        self.inner.pending.lock().len()
    }

    pub fn running(&self) -> Option<String> {
        self.inner.running.lock().as_ref().map(|r| r.id.clone())
    }

    /// API jobs, the running one first, then the waiting ones in queue order.
    pub fn client_jobs(&self) -> Vec<ClientJob> {
        // `pending` before `running`, never both held: the worker holds `pending`
        // while it takes `running`. A job taken in between shows up twice and
        // keeps its running entry.
        let waiting: Vec<ClientJob> = self
            .inner
            .pending
            .lock()
            .iter()
            .filter_map(|p| {
                p.number.map(|number| ClientJob {
                    number,
                    running: false,
                    progress: p.work.job.progress.lock().clone(),
                    protocol_1: p.work.job.protocol_1(),
                })
            })
            .collect();
        let running = self.inner.running.lock().as_ref().and_then(|r| {
            r.number.map(|number| ClientJob {
                number,
                running: true,
                progress: r.job.progress.lock().clone(),
                protocol_1: r.job.protocol_1(),
            })
        });
        running_first(running, waiting)
    }

    /// Have the worker build a new separator and run on it from now on.
    ///
    /// Blocks until the swap has happened: the track in flight first, then the build,
    /// so the caller learns what loaded. The old separator keeps running until the new
    /// one is in hand.
    pub fn install(&self, build: BuildSeparator) -> Result<BackendInfo> {
        let (reply, done) = mpsc::sync_channel(1);
        let superseded = {
            // Under `pending` for the reason `stop` gives: the worker checks
            // `incoming` while holding it and only then parks, so a notify sent
            // without this lock could reach nobody, and an idle worker parks
            // until a *job* arrives, which on a server nobody is using is never.
            let _parked = self.inner.pending.lock();
            // Read under that same lock, so a switch cannot slip in behind a
            // `stop` and then wait for a worker that has already gone.
            if self.inner.shutdown.load(Ordering::SeqCst) {
                anyhow::bail!("the server is shutting down");
            }
            self.inner.incoming.lock().replace(Staged { build, reply })
        };
        // Only reachable if two switches are requested faster than the worker
        // can take one, which `Switcher::request` refuses. Told, not dropped:
        // the caller is blocked on that channel.
        if let Some(staged) = superseded {
            let _ = staged
                .reply
                .send(Err(anyhow::anyhow!("superseded by a later switch")));
        }
        self.inner.ready.notify_one();

        match done.recv() {
            Ok(result) => result,
            // The sender goes with the worker, so this is the queue stopping
            // underneath a switch: on the way out, where there is nothing left
            // to switch to.
            Err(_) => anyhow::bail!("the separation worker stopped before the model could load"),
        }
    }

    /// Progress for a waiting job, with its live queue position filled in.
    ///
    /// Computed on read rather than pushed on every dequeue: positions shift for
    /// every waiter each time one is taken, and the queue is short enough that a
    /// scan on poll is cheaper than fanning out updates.
    pub fn queued_progress_for(&self, id: &str) -> Option<Progress> {
        let pending = self.inner.pending.lock();
        let position = pending.iter().position(|p| p.work.job.id == id)?;
        Some(queued_progress(position, pending.len()))
    }
}

impl Drop for Queue {
    /// A no-op once [`Queue::stop`] has run, which is the ordinary path. This
    /// covers a queue dropped without one: every test, and any early return
    /// between `start` and a server that can shut itself down.
    fn drop(&mut self) {
        self.stop(STOP_GRACE);
    }
}

fn queued_progress(position: usize, total: usize) -> Progress {
    Progress::counted(Stage::Queued, position as u32, total as u32).with_detail(if position == 0 {
        "next".to_owned()
    } else {
        format!("{position} ahead")
    })
}

/// A job the worker has taken, with the switch that stops it.
struct Claimed {
    work: QueuedWork,
    cancel: Arc<AtomicBool>,
}

/// What the worker found waiting for it.
enum Next {
    Install(Staged),
    Job(Claimed),
    Segment(Arc<Track>, usize),
    /// Woken with nothing to run: a chance to drop what closed sessions held.
    Tidy,
}

impl Inner {
    /// Block until there is something to do. `None` means shutdown.
    ///
    /// A staged model is taken ahead of a waiting job, because between jobs is the only
    /// safe point to swap, and while idle too, which is what lets [`Queue::install`]
    /// return on a server nobody is using. Registering as running happens under the
    /// same lock that took the job out of `pending`.
    fn next(&self) -> Option<Next> {
        let mut pending = self.pending.lock();
        let mut woken = false;
        loop {
            // Ahead of the swap as well as the job: building a model while the
            // process is tearing down puts a fresh allocation in front of the C
            // runtime's destructors, which is the crash `Queue::stop` exists to
            // prevent.
            if self.shutdown.load(Ordering::SeqCst) {
                return None;
            }
            if let Some(staged) = self.incoming.lock().take() {
                return Some(Next::Install(staged));
            }
            if let Some((track, k)) = self.sessions.next(Class::BeforeJobs) {
                return Some(Next::Segment(track, k));
            }
            if let Some(Pending { work, number }) = pending.pop_front() {
                let cancel = Arc::new(AtomicBool::new(false));
                *self.running.lock() = Some(Running {
                    id: work.job.id.clone(),
                    from_api: work.from_api,
                    number,
                    job: Arc::clone(&work.job),
                    cancel: Arc::clone(&cancel),
                });
                return Some(Next::Job(Claimed { work, cancel }));
            }
            if let Some((track, k)) = self.sessions.next(Class::WhenIdle) {
                return Some(Next::Segment(track, k));
            }
            // Tidy only after every check above, so a wake never skips waiting work.
            if woken {
                return Some(Next::Tidy);
            }
            self.ready.wait(&mut pending);
            woken = true;
        }
    }

    /// Build a separator on this thread and publish what it says about itself.
    /// Called only from the worker; see [`BuildSeparator`].
    ///
    /// `None` means the build failed, and whatever the worker already holds is still
    /// good.
    fn build_here(
        &self,
        build: BuildSeparator,
        reply: &mpsc::SyncSender<Result<BackendInfo>>,
    ) -> Option<Box<dyn Separate>> {
        match build() {
            Ok(separator) => {
                let info = separator.info();
                // Published before the reply, so a caller reading it the instant
                // `install` returns sees the model it just installed.
                *self.info.lock() = Some(info.clone());
                // Open sessions hold output from the old model. Close them before the
                // reply, so `install` returns with them closed.
                self.sessions.close_all("the server switched models");
                *self.speed.lock() = Speed::default();
                *self.segmenting.lock() = separator.segmenting();
                self.normalises
                    .store(separator.normalises(), Ordering::SeqCst);
                // The send may find nobody there: a switch superseded between
                // the build starting and finishing. Keep the model anyway: it is
                // the newer of the two, and `info` already says so.
                let _ = reply.send(Ok(info));
                Some(separator)
            }
            Err(err) => {
                let _ = reply.send(Err(err));
                None
            }
        }
    }
}

/// Take work until asked to stop.
///
/// Nothing a separator does may end this loop. Both callees are caught, so a track
/// that panics fails that track and a build that panics leaves the model already
/// loaded in place. Whether the separator is still sound cannot be known from
/// here, so it is reported at `error`.
fn worker_loop(inner: &Arc<Inner>, mut separator: Box<dyn Separate>, cache: &Cache) {
    // Each open track's partial output, on the device, by track uid. Held here
    // because MLX arrays belong to the thread that made them.
    let mut passes: HashMap<u64, Pass> = HashMap::new();
    while let Some(next) = inner.next() {
        match next {
            Next::Install(staged) => {
                // The reply is destructured out and kept clear of the caught
                // closure, so a panicking build can still be answered on it.
                // Letting it drop instead would tell `install` the worker had
                // died, which is exactly the thing that is no longer true.
                let Staged { build, reply } = staged;
                match catch_unwind(AssertUnwindSafe(|| inner.build_here(build, &reply))) {
                    Ok(Some(built)) => {
                        separator = built;
                        // Partial passes hold output from the old model.
                        passes.clear();
                        tracing::info!("switched to {}", separator.info().model);
                    }
                    // `build_here` has already sent whatever it failed with.
                    Ok(None) => {}
                    Err(payload) => {
                        let what = crate::panics::message(&*payload);
                        tracing::error!(
                            "the model build panicked: {what}. Staying on {}",
                            separator.info().model
                        );
                        let _ =
                            reply.send(Err(anyhow::anyhow!("the model build panicked: {what}")));
                    }
                }
            }
            Next::Job(claimed) => {
                inner.busy.store(true, Ordering::SeqCst);
                let pieces = inner.sessions.pieces();
                // The time between two calls is one job segment.
                let mut last: Option<std::time::Instant> = None;
                let separating = catch_unwind(AssertUnwindSafe(|| {
                    separate_one(separator.as_mut(), &claimed, cache, pieces, &mut |sep| {
                        let now = std::time::Instant::now();
                        if let Some(at) = last {
                            inner.speed.lock().record((now - at).as_secs_f64());
                        }
                        serve_high(inner, sep, &mut passes);
                        last = Some(std::time::Instant::now());
                        Ok(())
                    });
                }));
                if let Err(payload) = separating {
                    let what = crate::panics::message(&*payload);
                    tracing::error!(job = %claimed.work.job.id, "the separation panicked: {what}");
                    // Terminal, so a client polling this job stops waiting for
                    // something that is never going to arrive.
                    claimed
                        .work
                        .job
                        .fail(format!("the separation panicked: {what}"));
                }
                *inner.running.lock() = None;
                trim_after_job(cache);
            }
            Next::Segment(track, k) => {
                inner.busy.store(true, Ordering::SeqCst);
                run_segment(separator.as_mut(), inner, &track, k, &mut passes);
            }
            Next::Tidy => {}
        }
        inner.busy.store(false, Ordering::SeqCst);
        passes.retain(|uid, _| inner.sessions.has_track(*uid));
    }
}

/// Run every high-priority session segment that can run now.
fn serve_high(inner: &Inner, separator: &mut dyn Separate, passes: &mut HashMap<u64, Pass>) {
    while !inner.shutdown.load(Ordering::SeqCst)
        && let Some((track, k)) = inner.sessions.next(Class::BeforeJobs)
    {
        run_segment(separator, inner, &track, k, passes);
    }
}

/// Run one segment of a track, copy what it made final into the track, and
/// write the pieces that became complete.
///
/// A failure closes the track; the worker keeps running.
fn run_segment(
    separator: &mut dyn Separate,
    inner: &Inner,
    track: &Track,
    k: usize,
    passes: &mut HashMap<u64, Pass>,
) {
    let sessions = &inner.sessions;
    let began = std::time::Instant::now();
    let ran = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let (window, norm) = track.input(k)?;
        let pass = match passes.entry(track.uid) {
            std::collections::hash_map::Entry::Occupied(held) => held.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(separator.begin(track.spec.frames, track.spec.parts, norm)?)
            }
        };
        separator.run(pass, &window, k)?;
        track.done(k);
        let finals = newly_final(pass, k);
        for &(from, to) in &finals {
            let stems = separator.stems(pass, &track.mix(from, to), from, to)?;
            track.store(from, to, &stems)?;
        }
        if let (Some(first), Some(last)) = (finals.first(), finals.last()) {
            sessions.store_pieces(track, first.0..last.1);
        }
        Ok(())
    }));
    sessions.changed();
    let failure = match ran {
        Ok(Ok(())) => {
            let took = began.elapsed();
            inner.speed.lock().record(took.as_secs_f64());
            tracing::debug!(track = track.uid, "segment {k} in {took:.2?}");
            return;
        }
        Ok(Err(err)) => format!("segment {k}: {err:#}"),
        Err(payload) => format!(
            "segment {k} panicked: {}",
            crate::panics::message(&*payload)
        ),
    };
    tracing::warn!(track = track.uid, "{failure}");
    track.close(failure);
    sessions.changed();
    passes.remove(&track.uid);
}

/// Whether there is room to write a finished job's pieces: they hold a copy of
/// the mix and stems in memory while they encode, and land on disk.
fn pieces_affordable(pieces: &Pieces) -> bool {
    if crate::disk::low(pieces.root()) {
        tracing::info!("disk nearly full, not writing this job's pieces");
        return false;
    }
    if crate::disk::memory_pressure() {
        tracing::info!("memory is short, not writing this job's pieces");
        return false;
    }
    true
}

/// Hand back MLX's allocator cache after a job if keeping it is a risk: when
/// memory or disk is short, or it holds over half the machine's memory. Kept
/// otherwise, so the next track in a queue starts warm.
fn trim_after_job(cache: &Cache) {
    let held = stemd_core::memory::cached();
    let ceiling = crate::disk::physical_memory().map_or(8 << 30, |ram| ram / 2);
    let short = crate::disk::memory_pressure() || crate::disk::low(cache.root());
    if held as u64 > ceiling || (short && held > 0) {
        let freed = stemd_core::memory::release();
        tracing::info!(
            "returned {:.0} MB of MLX allocator cache after the job{}",
            freed as f64 / 1e6,
            if short {
                ", memory or disk is short"
            } else {
                ""
            }
        );
    }
}

/// Store a finished job's stems whole; a stem whose length is not the track's is written as pieces.
fn write_job_pieces(
    pieces: &Pieces,
    model_id: &str,
    mix: &Audio,
    stems: &Stems,
    output: Output,
    stride: usize,
    file: Option<String>,
) -> Result<()> {
    let id = track_id(mix);
    if let Some(key) = file
        && let Err(err) = pieces.record_file(&key, &id)
    {
        tracing::warn!("recording file key {key}: {err:#}");
    }
    let gain = gain(&output, mix.peak().ok());
    let name = type_dir(&output, gain);
    let mut all = output.clone();
    all.derived = true;
    let converted = crate::cache::to_output_rate(mix, stems, &all)?;
    let (model, out) = (u64::from(mix.sample_rate), u64::from(output.rate.hz()));
    let frames = mix.frames();
    let count = frames.div_ceil(stride);
    let info = TrackInfo {
        frames,
        sample_rate: mix.sample_rate,
        channels: mix.data.len(),
        stride,
        pieces: count,
    };
    let task = Consolidation {
        dir: pieces.verified_dir(model_id, &name, &id),
        format: output.format,
        rate: output.rate.hz(),
        info,
        stems: stems_of(&all),
    };
    let dir = &task.dir;
    pieces.write_info(dir, &info)?;
    let expected = info.frames_at(output.rate.hz());
    let mut whole = Vec::new();
    for (stem, audio) in &converted {
        if audio.frames() != expected {
            continue;
        }
        let bytes = stemd_core::encode_stem(audio, output.format, gain)?;
        let clipped = (!output.format.has_headroom()).then(|| {
            audio
                .data
                .iter()
                .flatten()
                .filter(|v| (*v * gain).abs() > 1.0)
                .count()
        });
        pieces.consolidate(&task, stem, &bytes, audio.frames(), clipped)?;
        whole.push(*stem);
    }
    if whole.len() == converted.len() {
        tracing::info!("track {id}: stored {} stems whole in {name}", whole.len());
        return Ok(());
    }
    for k in 0..count {
        let (from, to) = (k * stride, ((k + 1) * stride).min(frames));
        for (stem, audio) in converted.iter().filter(|(stem, _)| !whole.contains(stem)) {
            let a = output_index(from, model, out).min(audio.frames());
            let b = output_index(to, model, out).min(audio.frames());
            let bytes = stemd_core::encode_stem(&excerpt(audio, a, b), output.format, gain)?;
            pieces.write_verified(dir, stem, k, output.format.extension(), &bytes)?;
        }
    }
    tracing::info!("track {id}: wrote {count} pieces of {name}");
    Ok(())
}

/// The ranges inside segment `k`'s output that are final now that it has run.
fn newly_final(pass: &Pass, k: usize) -> Vec<(usize, usize)> {
    let grid = pass.grid();
    let own = grid.placement(k);
    let mut cuts = vec![own.offset, own.end()];
    for j in grid.covering(own.offset, own.end()) {
        let other = grid.placement(j);
        cuts.extend([other.offset, other.end()]);
    }
    cuts.retain(|c| (own.offset..=own.end()).contains(c));
    cuts.sort_unstable();
    cuts.dedup();
    cuts.windows(2)
        .map(|w| (w[0].unsigned_abs() as usize, w[1].unsigned_abs() as usize))
        .filter(|&(from, to)| pass.is_final(from, to))
        .collect()
}

/// Progress sink for one running job: forwards updates to the handle, and lets
/// the separator see a `DELETE` that arrived after it started.
struct JobSink {
    job: Arc<Job>,
    cancel: Arc<AtomicBool>,
}

impl stemd_core::ProgressSink for JobSink {
    fn update(&self, progress: Progress) {
        self.job.set_progress(progress);
    }

    fn cancelled(&self) -> bool {
        // Relaxed is enough: the flag orders no other data, and a segment of
        // slack either way is invisible against the segment it guards.
        self.cancel.load(Ordering::Relaxed)
    }
}

fn separate_one(
    separator: &mut dyn Separate,
    claimed: &Claimed,
    cache: &Cache,
    pieces: Option<Arc<Pieces>>,
    between: &mut dyn FnMut(&mut dyn Separate) -> Result<()>,
) {
    let work = &claimed.work;
    let job = &work.job;
    let started = std::time::Instant::now();
    let sink = JobSink {
        job: Arc::clone(job),
        cancel: Arc::clone(&claimed.cancel),
    };

    // Segment by segment where the separator can, so high-priority session work
    // waits at most one segment.
    let segmenting = separator.segmenting();
    let separated = if segmenting.is_some() {
        whole_track(separator, &work.mix, work.output.parts, &sink, |sep| {
            between(sep)
        })
    } else {
        separator.separate(&work.mix, work.output.parts, &sink)
    };
    match separated {
        Ok(stems) => {
            publish(job, work, &stems, cache, started.elapsed().as_secs_f64());
            // Off the worker: the job is done for its client, and the next one
            // should not wait on encoding.
            if let (Some(pieces), Some(segmenting)) = (pieces, segmenting)
                && work.from_api
                && work.output.format != stemd_core::StemFormat::Mp3
                && pieces_affordable(&pieces)
                && let Ok(grid) = segmenting.grid(work.mix.frames())
            {
                let (mix, output, model_id, file) = (
                    work.mix.clone(),
                    work.output.clone(),
                    work.model_id.clone(),
                    work.file.clone(),
                );
                let stride = grid.stride().unsigned_abs() as usize;
                std::thread::spawn(move || {
                    if let Err(err) =
                        write_job_pieces(&pieces, &model_id, &mix, &stems, output, stride, file)
                    {
                        tracing::warn!("writing a job's pieces: {err:#}");
                    }
                });
            }
        }
        // Routine, not a failure: the client said so. Logged at info with what
        // it cost, so a client skipping repeatedly is visible in the log without
        // looking like the model is breaking.
        Err(err) if Cancelled::caused(&err) => {
            tracing::info!(
                job = %job.id,
                "cancelled after {:.1}s of separation",
                started.elapsed().as_secs_f64()
            );
            job.cancelled();
        }
        Err(err) => job.fail(format!("{err:#}")),
    }
}

/// Write the finished stems into the cache and mark the job done.
fn publish(job: &Job, work: &QueuedWork, stems: &Stems, cache: &Cache, elapsed: f64) {
    job.set_progress(Progress::new(Stage::Writing));
    //  Timed and reported separately: `elapsed` stops before this runs, and
    //  `separation_secs` is the model's cost by definition. Writing can cost more than
    //  separating, and used to appear in no total anyone saw.
    let writing = std::time::Instant::now();
    match cache.publish(&job.key, &work.mix, stems, work.output.clone(), elapsed) {
        Ok(entry) => {
            tracing::info!(
                job = %job.id,
                "done in {elapsed:.2}s ({:.1}x realtime, model residual {:.1} dB), \
                 {:.2}s writing {} stems, cached as {}",
                entry.realtime_factor(),
                entry.model_residual_db,
                writing.elapsed().as_secs_f64(),
                entry.stems.len(),
                crate::cache::short(&entry.key)
            );
            job.complete(entry, false);
        }
        Err(err) => job.fail(format!("writing stems: {err:#}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::cache;
    use std::time::Duration;
    use stemd_core::{BackendInfo, DspMode, OutputRate, ProgressSink, StemFormat, Stems};

    /// The segment methods of a stub that only separates whole tracks.
    macro_rules! whole_tracks_only {
        () => {
            fn normalises(&self) -> bool {
                false
            }
            fn segmenting(&self) -> Option<stemd_core::Segmenting> {
                None
            }
            fn begin(
                &self,
                _frames: usize,
                _parts: stemd_core::Parts,
                _norm: stemd_core::Normalisation,
            ) -> anyhow::Result<Pass> {
                anyhow::bail!("whole tracks only")
            }
            fn run(&mut self, _pass: &mut Pass, _window: &Audio, _k: usize) -> anyhow::Result<()> {
                anyhow::bail!("whole tracks only")
            }
            fn stems(
                &self,
                _pass: &Pass,
                _mix: &Audio,
                _from: usize,
                _to: usize,
            ) -> anyhow::Result<Stems> {
                anyhow::bail!("whole tracks only")
            }
        };
    }

    /// A separator that records which model "ran", so a swap is observable
    /// without loading 300 MB of weights onto a GPU.
    struct Stub {
        name: &'static str,
        seen: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Separate for Stub {
        whole_tracks_only!();

        fn separate(
            &mut self,
            _mix: &Audio,
            _parts: stemd_core::Parts,
            _sink: &dyn ProgressSink,
        ) -> anyhow::Result<Stems> {
            self.seen.lock().push(self.name);
            anyhow::bail!("stub");
        }

        fn info(&self) -> BackendInfo {
            BackendInfo {
                backend: "stub".into(),
                model: self.name.into(),
                sample_rate: 44100,
                channels: 2,
                device: "stub".into(),
                stems: vec![],
                parts: vec![],
            }
        }
    }

    /// A separator that is already built, wrapped as the recipe the worker
    /// takes. Fine here and nowhere real: a stub allocates nothing, so which
    /// thread it was made on is not a fact about it.
    fn ready(separator: impl Separate + 'static) -> BuildSeparator {
        Box::new(move || Ok(Box::new(separator) as Box<dyn Separate>))
    }

    fn start(separator: impl Separate + 'static, cache: Arc<Cache>) -> Queue {
        Queue::start(ready(separator), 8, cache)
            .expect("a stub always builds")
            .0
    }

    /// Where each half of a separator's life happened, in order.
    type Threads = Arc<Mutex<Vec<(&'static str, thread::ThreadId)>>>;

    /// A separator that records the thread it ran on, built by a recipe that
    /// records the thread it was built on.
    struct Traced(Threads);

    impl Separate for Traced {
        whole_tracks_only!();

        fn separate(
            &mut self,
            _mix: &Audio,
            _parts: stemd_core::Parts,
            _sink: &dyn ProgressSink,
        ) -> anyhow::Result<Stems> {
            self.0.lock().push(("ran", thread::current().id()));
            anyhow::bail!("stub produces no stems")
        }

        fn info(&self) -> BackendInfo {
            BackendInfo {
                backend: "traced".into(),
                model: "traced".into(),
                sample_rate: 44100,
                channels: 2,
                device: "stub".into(),
                stems: vec![],
                parts: vec![],
            }
        }
    }

    fn traced(threads: &Threads) -> BuildSeparator {
        let threads = Arc::clone(threads);
        Box::new(move || {
            threads.lock().push(("built", thread::current().id()));
            Ok(Box::new(Traced(threads)) as Box<dyn Separate>)
        })
    }

    /// Poll until `seen` holds `count` names. `Stub` records under a plain mutex
    /// with nothing to wait on, and the deadline is what turns a swap that never
    /// happens into a failure rather than a hung suite.
    fn ran(seen: &Arc<Mutex<Vec<&'static str>>>, count: usize) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while seen.lock().len() < count {
            if std::time::Instant::now() >= deadline {
                return false;
            }
            thread::sleep(POLL_INTERVAL);
        }
        true
    }

    fn job(store: &crate::jobs::JobStore) -> Arc<Job> {
        job_for(store, "test-key")
    }

    /// Two jobs in one test need two keys: `claim` collapses a repeated key into
    /// the separation already in flight, which is the whole point of it.
    fn job_for(store: &crate::jobs::JobStore, key: &str) -> Arc<Job> {
        store.create(key.into())
    }

    #[derive(Default)]
    struct Segments {
        started: usize,
        cancelled: usize,
    }

    type Shared = Arc<(Mutex<Segments>, Condvar)>;

    /// A separator shaped like the real one in the way that matters here: it
    /// works in segments and only checks for cancellation between them. Lets the
    /// plumbing be tested without 300 MB of weights on a GPU.
    struct Segmented {
        state: Shared,
        /// Segments before it would finish unaided. Deliberately long, so a test
        /// waiting on the *next* job cannot pass by outlasting this one instead.
        segments: usize,
    }

    impl Separate for Segmented {
        whole_tracks_only!();

        fn separate(
            &mut self,
            _mix: &Audio,
            _parts: stemd_core::Parts,
            sink: &dyn ProgressSink,
        ) -> anyhow::Result<Stems> {
            bump(&self.state, |s| s.started += 1);
            for _ in 0..self.segments {
                if sink.cancelled() {
                    bump(&self.state, |s| s.cancelled += 1);
                    return Err(Cancelled.into());
                }
                thread::sleep(Duration::from_millis(2));
            }
            anyhow::bail!("stub produces no stems")
        }

        fn info(&self) -> BackendInfo {
            BackendInfo {
                backend: "segmented".into(),
                model: "segmented".into(),
                sample_rate: 44100,
                channels: 2,
                device: "stub".into(),
                stems: vec![],
                parts: vec![],
            }
        }
    }

    /// Like [`Segmented`], but it reports progress as it goes, which is the
    /// thing under test here. `Segmented` only sleeps, so a job running on it
    /// looks identical whether its progress is being published or not.
    struct Reporting {
        state: Shared,
        segments: usize,
    }

    impl Separate for Reporting {
        whole_tracks_only!();

        fn separate(
            &mut self,
            _mix: &Audio,
            _parts: stemd_core::Parts,
            sink: &dyn ProgressSink,
        ) -> anyhow::Result<Stems> {
            bump(&self.state, |s| s.started += 1);
            for done in 0..self.segments {
                if sink.cancelled() {
                    bump(&self.state, |s| s.cancelled += 1);
                    return Err(Cancelled.into());
                }
                sink.update(stemd_core::Progress {
                    stage: stemd_core::Stage::Separating,
                    completed: done as u32,
                    total: self.segments as u32,
                    fraction: done as f32 / self.segments as f32,
                    detail: None,
                });
                thread::sleep(Duration::from_millis(4));
            }
            anyhow::bail!("stub produces no stems")
        }

        fn info(&self) -> BackendInfo {
            BackendInfo {
                backend: "reporting".into(),
                model: "reporting".into(),
                sample_rate: 44100,
                channels: 2,
                device: "stub".into(),
                stems: vec![],
                parts: vec![],
            }
        }
    }

    /// A separator that unwinds instead of separating.
    ///
    /// Which is what a panic inside a forward pass looks like from here, and the
    /// worker has no other way to leave its loop except being asked to.
    struct Panics {
        state: Shared,
    }

    impl Separate for Panics {
        whole_tracks_only!();

        fn separate(
            &mut self,
            _mix: &Audio,
            _parts: stemd_core::Parts,
            _sink: &dyn ProgressSink,
        ) -> anyhow::Result<Stems> {
            bump(&self.state, |s| s.started += 1);
            panic!("the model came apart mid-pass");
        }

        fn info(&self) -> BackendInfo {
            BackendInfo {
                backend: "panics".into(),
                model: "panics".into(),
                sample_rate: 44100,
                channels: 2,
                device: "stub".into(),
                stems: vec![],
                parts: vec![],
            }
        }
    }

    /// One bad track has to cost that track and nothing else: a panic in a separation
    /// must not end the worker and leave the queue accepting work it will never run.
    #[test]
    fn a_panic_in_one_separation_does_not_strand_the_jobs_behind_it() {
        let state = shared();
        let queue = start(
            Panics {
                state: Arc::clone(&state),
            },
            cache(u64::MAX, Duration::from_secs(60)),
        );
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));

        let first = job_for(&store, "first");
        submit(&queue, Arc::clone(&first));
        assert!(
            wait_until(&state, Duration::from_secs(5), |s| s.started == 1),
            "the worker never picked the first job up"
        );

        // Told, rather than left waiting. A client polling this job has no other
        // way to learn that the thing it is waiting for will never happen.
        assert!(
            settled(&first, Duration::from_secs(5)),
            "the job that panicked never reached a terminal state"
        );

        // And the queue is still a queue.
        let second = job_for(&store, "second");
        submit(&queue, Arc::clone(&second));
        assert!(
            wait_until(&state, Duration::from_secs(5), |s| s.started == 2),
            "the worker died with the first job and never took the second"
        );
    }

    /// Poll until the job stops being something a client would sit and wait on.
    fn settled(job: &Job, within: Duration) -> bool {
        let deadline = std::time::Instant::now() + within;
        loop {
            if matches!(
                job.view().progress.stage,
                Stage::Done | Stage::Failed | Stage::Cancelled
            ) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            thread::sleep(POLL_INTERVAL);
        }
    }

    fn shared() -> Shared {
        Arc::new((Mutex::new(Segments::default()), Condvar::new()))
    }

    fn bump(state: &Shared, change: impl FnOnce(&mut Segments)) {
        let (counts, changed) = &**state;
        change(&mut counts.lock());
        changed.notify_all();
    }

    /// Block until the counts satisfy `done`, or give up. Returns whether it
    /// happened: polling on a sleep would either be flaky or slow, and the
    /// timeouts here are what separate "cancelled promptly" from "eventually".
    fn wait_until(state: &Shared, within: Duration, done: impl Fn(&Segments) -> bool) -> bool {
        let (counts, changed) = &**state;
        let deadline = std::time::Instant::now() + within;
        let mut counts = counts.lock();
        while !done(&counts) {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                return false;
            }
            changed.wait_for(&mut counts, left);
        }
        true
    }

    /// Long enough (~10 s) that nothing below can pass by simply waiting it out.
    const NEVER_FINISHES: usize = 5_000;

    fn submit(queue: &Queue, job: Arc<Job>) {
        queue
            .submit(QueuedWork {
                job,
                mix: Audio::new(vec![vec![0.0; 16], vec![0.0; 16]], 44100),
                output: Output {
                    format: StemFormat::Pcm32,
                    rate: OutputRate::default(),
                    derived: false,
                    dsp: DspMode::default().into(),
                    parts: stemd_core::Parts::DEFAULT,
                },
                model_id: "test".into(),
                from_api: true,
                file: None,
            })
            .expect("queue has room");
    }

    /// Quitting during a separation must not end the process with the worker still
    /// inside the model. Stopping has to cost a segment, not a track: a shutdown that
    /// waits out a six-minute separation is a hang, and one that does not wait is a
    /// crash in GPU teardown.
    #[test]
    fn stopping_costs_a_segment_rather_than_a_whole_track() {
        let state = shared();
        let queue = start(
            Segmented {
                state: Arc::clone(&state),
                segments: NEVER_FINISHES,
            },
            cache(u64::MAX, Duration::from_secs(60)),
        );
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        submit(&queue, job(&store));
        assert!(
            wait_until(&state, Duration::from_secs(5), |s| s.started == 1),
            "the worker never picked the job up"
        );

        let began = std::time::Instant::now();
        let stopped = queue.stop(Duration::from_secs(10));
        let took = began.elapsed();

        assert!(stopped, "the worker was still in the model after 10 s");
        assert_eq!(
            state.0.lock().cancelled,
            1,
            "the job in flight was left to run instead of being cancelled"
        );
        assert!(
            took < Duration::from_secs(2),
            "stopping took {took:?}; the job it cancelled had ~10 s left to run, \
             so this waited for the track rather than the segment"
        );
    }

    /// A switch requested mid-track must not stop the track reporting progress. The
    /// swap waits for the track, and the window's dialog sits over a bar that has to
    /// keep moving.
    #[test]
    fn a_switch_staged_mid_track_does_not_freeze_its_progress() {
        let state = shared();
        let queue = start(
            Reporting {
                state: Arc::clone(&state),
                segments: NEVER_FINISHES,
            },
            cache(u64::MAX, Duration::from_secs(60)),
        );
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let job = job(&store);
        submit(&queue, Arc::clone(&job));
        assert!(
            wait_until(&state, Duration::from_secs(5), |s| s.started == 1),
            "the worker never picked the job up"
        );

        let advanced = |from: f32| {
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while job.progress.lock().fraction <= from {
                if std::time::Instant::now() >= deadline {
                    return false;
                }
                thread::sleep(POLL_INTERVAL);
            }
            true
        };
        assert!(advanced(0.0), "progress never started moving");
        let at_request = job.progress.lock().fraction;

        thread::scope(|scope| {
            let staged = scope.spawn(|| {
                queue.install(ready(Reporting {
                    state: shared(),
                    segments: 1,
                }))
            });
            assert!(
                advanced(at_request),
                "progress stopped at {at_request} once a switch was staged; the \
                 track is still running, so the window would show a dialog over \
                 a frozen bar"
            );
            // Let the switch through, or the scope waits on a worker that is
            // still separating a track with NEVER_FINISHES segments left.
            queue.stop(Duration::from_secs(10));
            let _ = staged.join();
        });
    }

    /// `Drop` stops the queue too, so the ordinary path stops it twice. The
    /// second must be a no-op rather than a wait or a panic.
    #[test]
    fn stopping_an_already_stopped_queue_is_harmless() {
        let queue = start(
            Segmented {
                state: shared(),
                segments: NEVER_FINISHES,
            },
            cache(u64::MAX, Duration::from_secs(60)),
        );

        assert!(queue.stop(Duration::from_secs(10)));
        let began = std::time::Instant::now();
        assert!(queue.stop(Duration::from_secs(10)), "a second stop failed");
        assert!(began.elapsed() < Duration::from_millis(100));
        // And a third, from `Drop`, as the queue goes out of scope here.
    }

    #[test]
    fn a_running_job_stops_between_segments_rather_than_running_to_the_end() {
        let state = shared();
        let queue = start(
            Segmented {
                state: Arc::clone(&state),
                segments: NEVER_FINISHES,
            },
            cache(u64::MAX, Duration::from_secs(60)),
        );
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let job = job(&store);
        let id = job.id.clone();
        submit(&queue, Arc::clone(&job));

        assert!(
            wait_until(&state, Duration::from_secs(5), |s| s.started == 1),
            "the worker never picked the job up"
        );
        assert!(
            queue.cancel(&id),
            "a job already under way must still be cancellable"
        );
        assert!(
            wait_until(&state, Duration::from_secs(5), |s| s.cancelled == 1),
            "the separator was never told to stop"
        );

        // Terminal, and not a failure: nothing went wrong, so there is no error
        // for a client to read back.
        assert!(
            wait_until(&state, Duration::from_secs(5), |_| job
                .progress
                .lock()
                .stage
                .is_terminal()),
            "the job never reached a terminal stage"
        );
        assert_eq!(job.progress.lock().stage, Stage::Cancelled);
        assert!(job.error.lock().is_none(), "a cancellation is not an error");
    }

    #[test]
    fn an_abandoned_track_does_not_hold_up_the_one_cued_behind_it() {
        // The reason cancellation covers the running job at all. Separation is
        // serialised, so without it a deck that skips mid-analysis makes the
        // track it actually wants wait out a full separation of the one it threw
        // away: the entire latency budget, spent on stems nobody will fetch.
        let state = shared();
        let queue = start(
            Segmented {
                state: Arc::clone(&state),
                segments: NEVER_FINISHES,
            },
            cache(u64::MAX, Duration::from_secs(60)),
        );
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));

        let skipped = job_for(&store, "skipped");
        let cued = job_for(&store, "cued");
        let (skipped_id, cued_id) = (skipped.id.clone(), cued.id.clone());
        submit(&queue, skipped);
        submit(&queue, cued);

        assert!(
            wait_until(&state, Duration::from_secs(5), |s| s.started == 1),
            "the first track never started"
        );
        // Guards what this is actually testing: the skipped track must be out of
        // `pending` and under way, or the cancel below takes the easy path.
        assert_eq!(
            queue.running().as_deref(),
            Some(skipped_id.as_str()),
            "the skipped track should be running, not still queued"
        );
        assert!(queue.cancel(&skipped_id));

        // Well inside the ~10 s the abandoned track would otherwise take, so
        // this fails rather than passes slowly if cancellation stops working.
        assert!(
            wait_until(&state, Duration::from_secs(5), |s| s.started == 2),
            "the cued track waited for the abandoned one to finish"
        );

        // Leave nothing running: the worker is joined on drop.
        queue.cancel(&cued_id);
    }

    #[test]
    fn cancelling_what_is_neither_waiting_nor_running_reports_it() {
        let queue = start(
            Stub {
                name: "stub",
                seen: Arc::new(Mutex::new(Vec::new())),
            },
            cache(u64::MAX, Duration::from_secs(60)),
        );
        assert!(!queue.cancel("no-such-job"));
    }

    #[test]
    fn a_job_after_an_install_runs_on_the_model_that_was_installed() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let queue = start(
            Stub {
                name: "first",
                seen: Arc::clone(&seen),
            },
            cache(u64::MAX, Duration::from_secs(60)),
        );

        submit(&queue, job_for(&store, "one"));
        assert!(ran(&seen, 1), "the worker never ran the first job");

        // Returns once the swap has happened, on an idle queue as well as a busy
        // one. The worker used to look at what was staged only after claiming a
        // job, so an install on a server nobody is using waited for a track that
        // may never be submitted.
        queue
            .install(ready(Stub {
                name: "second",
                seen: Arc::clone(&seen),
            }))
            .expect("a stub always builds");
        assert_eq!(queue.info().model, "second");

        submit(&queue, job_for(&store, "two"));
        assert!(ran(&seen, 2), "the worker never ran the second job");
        assert_eq!(&*seen.lock(), &["first", "second"]);
    }

    /// Whatever thread builds a separator is the thread that runs it, on both paths
    /// that install one. MLX's CUDA backend keeps its stream registry in thread-local
    /// storage, so weights allocated on one thread cannot be evaluated on another.
    /// Asserted as an arrangement rather than as a backend error, so it needs no GPU.
    #[test]
    fn the_thread_that_builds_a_model_is_the_thread_that_runs_it() {
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let threads: Threads = Arc::default();

        let (queue, _) = Queue::start(
            traced(&threads),
            8,
            cache(u64::MAX, Duration::from_secs(60)),
        )
        .expect("a stub always builds");
        submit(&queue, job_for(&store, "one"));
        queue
            .install(traced(&threads))
            .expect("a stub always builds");
        submit(&queue, job_for(&store, "two"));

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while threads.lock().len() < 4 {
            assert!(
                std::time::Instant::now() < deadline,
                "only {:?} happened",
                threads.lock()
            );
            thread::sleep(POLL_INTERVAL);
        }

        let seen = threads.lock().clone();
        assert_eq!(
            seen.iter().map(|(what, _)| *what).collect::<Vec<_>>(),
            ["built", "ran", "built", "ran"]
        );
        let worker = seen[0].1;
        assert!(
            seen.iter().all(|(_, on)| *on == worker),
            "a model was built on one thread and run on another: {seen:?}"
        );
        assert_ne!(
            worker,
            thread::current().id(),
            "the stub was built on the test's own thread, so this proves nothing"
        );
    }

    /// A model that will not load has to fail the caller rather than the first
    /// track to arrive. `startup::load_model` turns this error into a fallback
    /// to the default preset; a launch that "succeeded" onto a worker with no
    /// model would instead fail every separation for the life of the process.
    #[test]
    fn a_model_that_will_not_load_fails_the_start() {
        let err = Queue::start(
            Box::new(|| anyhow::bail!("no such weights")),
            8,
            cache(u64::MAX, Duration::from_secs(60)),
        )
        .map(|_| ())
        .expect_err("a build that fails must not hand back a queue");
        assert!(format!("{err:#}").contains("no such weights"), "{err:#}");
    }

    /// A switch to a model that will not load leaves the one that does. The
    /// window reports the failure and offers "Keep current model", which is only
    /// truthful if the current model is in fact still there.
    #[test]
    fn an_install_that_fails_keeps_the_model_that_was_running() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let queue = start(
            Stub {
                name: "first",
                seen: Arc::clone(&seen),
            },
            cache(u64::MAX, Duration::from_secs(60)),
        );

        let err = queue
            .install(Box::new(|| anyhow::bail!("no such weights")))
            .map(|_| ())
            .expect_err("a build that fails must be reported to whoever asked");
        assert!(format!("{err:#}").contains("no such weights"), "{err:#}");

        assert_eq!(queue.info().model, "first", "the old model was unpublished");
        submit(&queue, job_for(&store, "after"));
        assert!(ran(&seen, 1), "the worker stopped running jobs");
        assert_eq!(&*seen.lock(), &["first"]);
    }

    /// Quitting during a switch has to let the switch thread go. It is blocked on a
    /// reply channel that lives on the queue rather than the worker, so stopping only
    /// the worker would park it until the process ended.
    #[test]
    fn stopping_releases_a_switch_waiting_on_the_worker() {
        let state = shared();
        let seen: Arc<Mutex<Vec<&'static str>>> = Arc::default();
        let queue = Arc::new(start(
            Segmented {
                state: Arc::clone(&state),
                segments: NEVER_FINISHES,
            },
            cache(u64::MAX, Duration::from_secs(60)),
        ));
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        submit(&queue, job(&store));
        assert!(
            wait_until(&state, Duration::from_secs(5), |s| s.started == 1),
            "the worker never picked the job up"
        );

        let waiting = {
            let (queue, seen) = (Arc::clone(&queue), Arc::clone(&seen));
            thread::spawn(move || queue.install(ready(Stub { name: "b", seen })))
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while queue.inner.incoming.lock().is_none() {
            assert!(
                std::time::Instant::now() < deadline,
                "the install never staged"
            );
            thread::sleep(POLL_INTERVAL);
        }

        assert!(queue.stop(Duration::from_secs(10)));
        waiting
            .join()
            .expect("the install thread panicked")
            .map(|_| ())
            .expect_err("a switch cut short by the quit must not report success");
        queue
            .install(ready(Stub { name: "c", seen }))
            .map(|_| ())
            .expect_err("a switch requested after the stop must be refused, not parked");
    }

    /// Two switches faster than the worker can take one. `Switcher::request`
    /// refuses that, so this is about the queue not leaving a caller blocked on
    /// a reply that will never come.
    #[test]
    fn a_superseded_install_tells_the_caller_rather_than_leaving_it_waiting() {
        let state = shared();
        let seen: Arc<Mutex<Vec<&'static str>>> = Arc::default();
        let queue = Arc::new(start(
            Segmented {
                state: Arc::clone(&state),
                segments: NEVER_FINISHES,
            },
            cache(u64::MAX, Duration::from_secs(60)),
        ));
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let job = job(&store);
        let id = job.id.clone();
        submit(&queue, job);
        assert!(
            wait_until(&state, Duration::from_secs(5), |s| s.started == 1),
            "the worker never picked the job up"
        );

        // Both stage rather than being taken, because the worker is in the
        // model. Spawned, because the first would otherwise block the test.
        let install = |name: &'static str| {
            let (queue, seen) = (Arc::clone(&queue), Arc::clone(&seen));
            thread::spawn(move || queue.install(ready(Stub { name, seen })))
        };
        let first = install("b");
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while queue.inner.incoming.lock().is_none() {
            assert!(
                std::time::Instant::now() < deadline,
                "the first install never staged"
            );
            thread::sleep(POLL_INTERVAL);
        }
        let second = install("c");

        // Returns as soon as the second replaces it, without waiting for the
        // worker, which is the whole claim.
        let err = first
            .join()
            .expect("the install thread panicked")
            .map(|_| ())
            .expect_err("a superseded install must not report success");
        assert!(format!("{err:#}").contains("superseded"), "{err:#}");

        queue.cancel(&id);
        second
            .join()
            .expect("the install thread panicked")
            .expect("the surviving install must still be applied");
        assert_eq!(queue.info().model, "c");
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;

    const KEY: &str = "0123456789abcdef01234567";
    use crate::sessions::{Region, Request, Session, TrackSpec, WindowRequest};
    use crate::testkit::cache;
    use std::time::Duration;
    use stemd_core::toy::Toy;
    use stemd_core::{DspMode, Normalisation, OutputRate, Parts, PcmFormat, Silent, StemFormat};

    /// MLX runs on one thread at a time; each queue here has its own worker.
    static GPU: Mutex<()> = Mutex::new(());

    const SEGMENT: i32 = 4410;

    fn track(frames: usize, seed: u32) -> Audio {
        let mut state = seed;
        let data = (0..2)
            .map(|_| {
                (0..frames)
                    .map(|_| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                    })
                    .collect()
            })
            .collect();
        Audio::new(data, stemd_core::toy::RATE)
    }

    fn output(format: StemFormat) -> Output {
        Output {
            format,
            rate: OutputRate::default(),
            derived: true,
            dsp: DspMode::default().into(),
            parts: Parts::DEFAULT,
        }
    }

    fn start(toy: Toy) -> Queue {
        let build: BuildSeparator = Box::new(move || Ok(Box::new(toy) as Box<dyn Separate>));
        Queue::start(build, 8, cache(1 << 30, Duration::from_secs(60)))
            .expect("the toy builds")
            .0
    }

    /// A piece store no other test touches.
    fn with_pieces(queue: &Queue) -> Arc<Pieces> {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "stemd-pieces-test-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let pieces = Pieces::open(root, 1 << 30).expect("a piece store");
        queue.sessions().set_pieces(Arc::clone(&pieces));
        pieces
    }

    /// A piece store whose stems are never stored whole in the background.
    fn with_held_pieces(queue: &Queue) -> Arc<Pieces> {
        let pieces = with_pieces(queue);
        pieces.set_gate(Box::new(|| false));
        pieces
    }

    fn open_as(
        queue: &Queue,
        frames: usize,
        norm: Option<Normalisation>,
        format: StemFormat,
    ) -> Arc<Session> {
        let grid = queue
            .segmenting()
            .expect("the toy segments")
            .grid(frames)
            .expect("a grid");
        queue
            .sessions()
            .open(Request {
                output: output(format),
                in_format: PcmFormat::F32le,
                track: TrackSpec {
                    frames,
                    channels: 2,
                    sample_rate: stemd_core::toy::RATE,
                    grid,
                    normalises: queue.normalises(),
                    parts: Parts::DEFAULT,
                },
                norm,
                peak: None,
                model_id: "toy-00000000".into(),
            })
            .expect("room for a session")
    }

    fn open(queue: &Queue, frames: usize, norm: Option<Normalisation>) -> Arc<Session> {
        open_as(queue, frames, norm, StemFormat::Pcm32)
    }

    fn whole(session: &Session) -> WindowRequest {
        WindowRequest {
            regions: vec![Region {
                at: 0,
                ahead: session.track().spec.frames,
                behind: 0,
                priority: 0,
            }],
        }
    }

    fn ready_within(session: &Session, from: usize, to: usize, within: Duration) -> bool {
        let deadline = std::time::Instant::now() + within;
        while std::time::Instant::now() < deadline {
            if session.read("vocals", from, to).is_ok() {
                return true;
            }
            thread::sleep(POLL_INTERVAL);
        }
        false
    }

    fn bits(audio: &Audio) -> Vec<u32> {
        audio.data.iter().flatten().map(|v| v.to_bits()).collect()
    }

    fn upload(queue: &Queue, session: &Session, offset: usize, audio: &Audio) {
        queue
            .sessions()
            .upload(session, offset, audio)
            .expect("uploading");
    }

    #[test]
    fn a_session_uploaded_backwards_matches_a_whole_track_run() {
        let _gpu = GPU.lock();
        let frames = 31_337;
        let mix = track(frames, 1);
        let mut direct = Toy::new(SEGMENT, 0.25);
        let want = direct
            .separate(&mix, Parts::DEFAULT, &Silent)
            .expect("separating");

        let queue = start(Toy::new(SEGMENT, 0.25));
        let session = open(&queue, frames, None);
        session.set_window(&whole(&session)).expect("open");
        // Last piece first, so segments wait on their input and on the whole
        // mix's statistics.
        for at in (0..frames)
            .step_by(10_000)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            let to = (at + 10_000).min(frames);
            upload(&queue, &session, at, &excerpt(&mix, at, to));
            queue.wake();
        }
        assert!(ready_within(&session, 0, frames, Duration::from_secs(20)));

        for name in ["harmonics", "vocals"] {
            let got = session.read(name, 0, frames).expect("ready").audio;
            let want = want.stem(name).expect("a shipped stem");
            assert_eq!(bits(&got), bits(want), "{name} differs");
        }
        let derived = session.read("drums", 0, frames).expect("ready").audio;
        assert_eq!(
            bits(&derived),
            bits(&want.derived_part(&mix)),
            "drums differs"
        );
    }

    #[test]
    fn nothing_runs_on_a_normalising_model_until_the_mix_is_whole() {
        let _gpu = GPU.lock();
        let frames = 40_000;
        let mix = track(frames, 2);
        let toy = Toy::new(SEGMENT, 0.25);
        let runs = Arc::clone(&toy.runs);
        let queue = start(toy);
        let session = open(&queue, frames, None);
        session.set_window(&whole(&session)).expect("open");
        upload(&queue, &session, 0, &excerpt(&mix, 0, 30_000));
        queue.wake();
        thread::sleep(Duration::from_millis(200));
        assert_eq!(runs.load(Ordering::SeqCst), 0, "ran without the statistics");

        upload(&queue, &session, 30_000, &excerpt(&mix, 30_000, frames));
        queue.wake();
        assert!(ready_within(&session, 0, frames, Duration::from_secs(20)));
    }

    #[test]
    fn with_the_clients_statistics_a_segment_runs_once_its_own_input_is_in() {
        let _gpu = GPU.lock();
        let frames = 400_000;
        let mix = track(frames, 3);
        let queue = start(Toy::new(SEGMENT, 0.25));
        let session = open(&queue, frames, Some(Normalisation::of(&mix)));
        let needle = 200_000;
        session
            .set_window(&WindowRequest {
                regions: vec![Region {
                    at: needle,
                    ahead: 10_000,
                    behind: 0,
                    priority: 0,
                }],
            })
            .expect("open");
        // Only the region around the cue.
        upload(&queue, &session, 190_000, &excerpt(&mix, 190_000, 220_000));
        queue.wake();
        assert!(ready_within(
            &session,
            needle,
            needle + 5_000,
            Duration::from_secs(20)
        ));
        assert!(session.read("vocals", 0, 1_000).is_err());
    }

    #[test]
    fn high_priority_segments_run_between_the_segments_of_a_job() {
        let _gpu = GPU.lock();
        let mut toy = Toy::new(SEGMENT, 0.25);
        toy.delay = Duration::from_millis(20);
        let queue = start(toy);

        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let job = store.create("long".into());
        queue
            .submit(QueuedWork {
                job: Arc::clone(&job),
                // About eighty segments, so well over a second of work.
                mix: track(80 * 3307, 4),
                output: output(StemFormat::Pcm32),
                model_id: "toy-00000000".into(),
                from_api: true,
                file: None,
            })
            .expect("room");
        thread::sleep(Duration::from_millis(100));

        let frames = 20_000;
        let mix = track(frames, 5);
        let session = open(&queue, frames, Some(Normalisation::of(&mix)));
        upload(&queue, &session, 0, &mix);
        session.set_window(&whole(&session)).expect("open");
        queue.wake();

        assert!(ready_within(&session, 0, frames, Duration::from_secs(20)));
        assert!(
            queue.running().is_some(),
            "the session waited for the whole job"
        );
        queue.cancel(&job.id);
    }

    #[test]
    fn a_job_taken_between_the_two_reads_is_listed_once_as_running() {
        let job = |number, running| ClientJob {
            number,
            running,
            progress: Progress::new(Stage::Queued),
            protocol_1: false,
        };
        let listed: Vec<(u64, bool)> =
            running_first(Some(job(1, true)), vec![job(1, false), job(2, false)])
                .into_iter()
                .map(|j| (j.number, j.running))
                .collect();
        assert_eq!(listed, [(1, true), (2, false)]);
    }

    #[test]
    fn client_jobs_list_the_running_one_then_the_queue_and_skip_drops() {
        let _gpu = GPU.lock();
        let mut toy = Toy::new(SEGMENT, 0.25);
        toy.delay = Duration::from_millis(20);
        let queue = start(toy);

        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let submit = |key: &str, from_api: bool| {
            let job = store.create(key.into());
            queue
                .submit(QueuedWork {
                    job: Arc::clone(&job),
                    mix: track(40 * 3307, 8),
                    output: output(StemFormat::Pcm32),
                    model_id: "toy-00000000".into(),
                    from_api,
                    file: None,
                })
                .expect("room");
            job
        };
        let first = submit("first", true);
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while queue.running().is_none() {
            assert!(
                std::time::Instant::now() < deadline,
                "the worker never took the first job"
            );
            thread::sleep(POLL_INTERVAL);
        }
        let dropped = submit("dropped", false);
        let second = submit("second", true);

        let listed: Vec<(u64, bool, Stage)> = queue
            .client_jobs()
            .into_iter()
            .map(|j| (j.number, j.running, j.progress.stage))
            .collect();
        assert_eq!(listed.len(), 2, "the drop is listed: {listed:?}");
        assert_eq!((listed[0].0, listed[0].1), (1, true));
        assert_eq!(listed[1], (2, false, Stage::Queued));
        for job in [&first, &dropped, &second] {
            queue.cancel(&job.id);
        }
    }

    #[test]
    fn turning_the_server_off_cancels_api_jobs_and_keeps_drops() {
        let _gpu = GPU.lock();
        let mut toy = Toy::new(SEGMENT, 0.25);
        toy.delay = Duration::from_millis(20);
        let queue = start(toy);

        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let submit = |key: &str, from_api: bool| {
            let job = store.create(key.into());
            queue
                .submit(QueuedWork {
                    job: Arc::clone(&job),
                    mix: track(40 * 3307, 8),
                    output: output(StemFormat::Pcm32),
                    model_id: "toy-00000000".into(),
                    from_api,
                    file: None,
                })
                .expect("room");
            job
        };
        let running = submit("running", true);
        thread::sleep(Duration::from_millis(100));
        let waiting = submit("waiting", true);
        let dropped = submit("dropped", false);

        assert_eq!(queue.cancel_api_jobs(), 2);
        assert_eq!(waiting.view().progress.stage, Stage::Cancelled);
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while running.view().progress.stage != Stage::Cancelled {
            assert!(
                std::time::Instant::now() < deadline,
                "the running job kept going"
            );
            thread::sleep(Duration::from_millis(10));
        }
        assert_ne!(dropped.view().progress.stage, Stage::Cancelled);
        queue.cancel(&dropped.id);
    }

    #[test]
    fn normal_priority_segments_wait_for_waiting_jobs() {
        let _gpu = GPU.lock();
        let mut toy = Toy::new(SEGMENT, 0.25);
        toy.delay = Duration::from_millis(10);
        let queue = start(toy);

        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let job = store.create("long".into());
        queue
            .submit(QueuedWork {
                job: Arc::clone(&job),
                mix: track(40 * 3307, 6),
                output: output(StemFormat::Pcm32),
                model_id: "toy-00000000".into(),
                from_api: true,
                file: None,
            })
            .expect("room");
        thread::sleep(Duration::from_millis(50));

        let frames = 20_000;
        let mix = track(frames, 7);
        let session = open(&queue, frames, Some(Normalisation::of(&mix)));
        upload(&queue, &session, 0, &mix);
        let mut request = whole(&session);
        request.regions[0].priority = crate::sessions::BEFORE_JOBS;
        session.set_window(&request).expect("open");
        queue.wake();

        assert!(ready_within(&session, 0, frames, Duration::from_secs(20)));
        assert_eq!(
            job.view().progress.stage,
            Stage::Done,
            "a normal-priority segment ran ahead of the job"
        );
    }

    #[test]
    fn a_model_switch_closes_every_session() {
        let _gpu = GPU.lock();
        let queue = start(Toy::new(SEGMENT, 0.25));
        let session = open(&queue, 20_000, None);
        queue
            .install(Box::new(|| {
                Ok(Box::new(Toy::new(SEGMENT, 0.25)) as Box<dyn Separate>)
            }))
            .expect("installing");
        assert!(session.closed().is_some());
        assert!(queue.sessions().is_empty());
    }

    #[test]
    fn a_second_deck_on_the_same_audio_joins_the_first_once_its_upload_completes() {
        let _gpu = GPU.lock();
        let frames = 60_000;
        let mix = track(frames, 8);
        let toy = Toy::new(SEGMENT, 0.25);
        let runs = Arc::clone(&toy.runs);
        let queue = start(toy);

        let a = open_as(&queue, frames, None, StemFormat::Pcm32);
        a.set_window(&whole(&a)).expect("open");
        upload(&queue, &a, 0, &mix);
        queue.wake();
        assert!(ready_within(&a, 0, frames, Duration::from_secs(20)));
        let segments = a.track().spec.grid.len();
        assert_eq!(runs.load(Ordering::SeqCst), segments);

        let b = open_as(&queue, frames, None, StemFormat::Flac);
        upload(&queue, &b, 0, &mix);
        assert!(Arc::ptr_eq(&a.track(), &b.track()), "b did not join a");
        b.set_window(&whole(&b)).expect("open");
        queue.wake();
        assert!(ready_within(&b, 0, frames, Duration::from_secs(1)));
        assert_eq!(runs.load(Ordering::SeqCst), segments, "b separated again");
        assert_eq!(
            a.track().verified().as_deref(),
            Some(crate::sessions::track_id(&mix).as_str())
        );
    }

    #[test]
    fn a_later_session_on_the_same_audio_finds_its_pieces_on_disk() {
        let _gpu = GPU.lock();
        let frames = 30_000;
        let mix = track(frames, 9);
        let id = crate::sessions::track_id(&mix);
        let toy = Toy::new(SEGMENT, 0.25);
        let runs = Arc::clone(&toy.runs);
        let queue = start(toy);
        let pieces = with_pieces(&queue);

        let first = open_as(&queue, frames, None, StemFormat::Flac);
        first.set_window(&whole(&first)).expect("open");
        upload(&queue, &first, 0, &mix);
        queue.wake();
        assert!(ready_within(&first, 0, frames, Duration::from_secs(20)));
        let count = first.track().spec.grid.len();
        let type_name = type_dir(&first.output, first.gain);
        let stems = ["harmonics", "vocals", "drums"];
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let dir = pieces.verified_dir("toy-00000000", &type_name, &id);
        while Pieces::cached(&dir, &stems, "flac", count).len() < count {
            assert!(
                std::time::Instant::now() < deadline,
                "the pieces were never written"
            );
            thread::sleep(POLL_INTERVAL);
        }
        queue.sessions().remove(&first.id, "ended by the client");
        let ran = runs.load(Ordering::SeqCst);

        // A new session on the same audio, once the first has gone.
        let second = open_as(&queue, frames, None, StemFormat::Flac);
        second.set_window(&whole(&second)).expect("open");
        upload(&queue, &second, 0, &mix);
        queue.wake();
        assert_eq!(queue.sessions().cached_for(&second).len(), count);
        thread::sleep(Duration::from_millis(100));
        assert_eq!(
            runs.load(Ordering::SeqCst),
            ran,
            "cached pieces were separated again"
        );
    }

    /// Wait until every piece of `session`'s output is on disk.
    fn stored_within(queue: &Queue, session: &Session, within: Duration) -> bool {
        let deadline = std::time::Instant::now() + within;
        let count = session.track().spec.grid.len();
        while std::time::Instant::now() < deadline {
            if queue.sessions().cached_for(session).len() == count {
                return true;
            }
            thread::sleep(POLL_INTERVAL);
        }
        false
    }

    /// The status, `x-stemd-*` headers and body of a whole-track read.
    fn whole_file(queue: &Queue, session: &Session, name: &str) -> (u16, Vec<String>, Vec<u8>) {
        let frames = session.track().spec.frames;
        parts_of(crate::session_api::encode_range(
            queue.sessions(),
            session,
            name,
            0,
            frames,
        ))
    }

    /// The same, sent from the stored whole file.
    fn sent_whole(queue: &Queue, session: &Session, name: &str) -> (u16, Vec<String>, Vec<u8>) {
        let whole = queue.sessions().whole(session, name).expect("a whole file");
        let response = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime")
            .block_on(crate::session_api::send_whole(
                session.output.format,
                session.gain,
                whole,
            ));
        parts_of(response)
    }

    fn parts_of(
        response: Result<axum::response::Response, crate::api::ApiError>,
    ) -> (u16, Vec<String>, Vec<u8>) {
        let response = match response {
            Ok(response) => response,
            Err(err) => axum::response::IntoResponse::into_response(err),
        };
        let status = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .filter(|(k, _)| k.as_str().starts_with("x-stemd-"))
            .map(|(k, v)| format!("{k}: {}", v.to_str().unwrap_or("")))
            .collect();
        let body = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime")
            .block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
            .expect("the body");
        (status, headers, body.to_vec())
    }

    fn open_at(queue: &Queue, frames: usize, format: StemFormat, rate: OutputRate) -> Arc<Session> {
        let grid = queue
            .segmenting()
            .expect("the toy segments")
            .grid(frames)
            .expect("a grid");
        let mut output = output(format);
        output.rate = rate;
        queue
            .sessions()
            .open(Request {
                output,
                in_format: PcmFormat::F32le,
                track: TrackSpec {
                    frames,
                    channels: 2,
                    sample_rate: stemd_core::toy::RATE,
                    grid,
                    normalises: queue.normalises(),
                    parts: Parts::DEFAULT,
                },
                norm: None,
                peak: None,
                model_id: "toy-00000000".into(),
            })
            .expect("room for a session")
    }

    /// Replace the whole files under `dir` with the pieces copied to `backup`.
    fn restore_pieces(dir: &std::path::Path, backup: &std::path::Path, format: StemFormat) {
        for stem in ["harmonics", "vocals", "drums"] {
            let _ = std::fs::remove_file(crate::whole::file(dir, stem, format.extension()));
            let _ = std::fs::remove_file(crate::whole::meta_file(dir, stem));
            let _ = std::fs::remove_dir_all(dir.join(stem));
            std::fs::create_dir_all(dir.join(stem)).expect("a stem directory");
            for entry in std::fs::read_dir(backup.join(stem)).expect("a backup") {
                let entry = entry.expect("an entry");
                std::fs::copy(entry.path(), dir.join(stem).join(entry.file_name()))
                    .expect("copying a piece");
            }
        }
    }

    /// The whole-track file of a fresh session, one with every piece stored, one
    /// with half of them stored, and one with the stem stored whole.
    fn four_ways(
        format: StemFormat,
        rate: OutputRate,
        seed: u32,
    ) -> [(u16, Vec<String>, Vec<u8>); 4] {
        let frames = 40_000;
        let mix = track(frames, seed);
        let queue = start(Toy::new(SEGMENT, 0.25));
        let pieces = with_held_pieces(&queue);
        let stems = ["harmonics", "vocals", "drums"];

        let fresh = open_at(&queue, frames, format, rate);
        fresh.set_window(&whole(&fresh)).expect("open");
        upload(&queue, &fresh, 0, &mix);
        queue.wake();
        assert!(ready_within(&fresh, 0, frames, Duration::from_secs(20)));
        assert!(stored_within(&queue, &fresh, Duration::from_secs(10)));
        let dir = pieces.verified_dir(
            "toy-00000000",
            &type_dir(&fresh.output, fresh.gain),
            &crate::sessions::track_id(&mix),
        );
        let backup = dir.with_extension("backup");
        for stem in stems {
            std::fs::create_dir_all(backup.join(stem)).expect("a backup");
            for entry in std::fs::read_dir(dir.join(stem)).expect("pieces") {
                let entry = entry.expect("an entry");
                std::fs::copy(entry.path(), backup.join(stem).join(entry.file_name()))
                    .expect("copying a piece");
            }
        }
        let (from, to) = (frames / 3, frames / 3 + 7000);
        let fresh_range = parts_of(crate::session_api::encode_range(
            queue.sessions(),
            &fresh,
            "vocals",
            from,
            to,
        ));
        let fresh_file = whole_file(&queue, &fresh, "vocals");
        assert!(
            !dir.join("vocals").exists(),
            "the pieces stayed beside the whole file"
        );
        queue.sessions().remove(&fresh.id, "ended by the client");

        let whole_session = open_at(&queue, frames, format, rate);
        upload(&queue, &whole_session, 0, &mix);
        let view = queue.view_of(&whole_session);
        assert!(view.complete, "the whole file does not count as stored");
        let sent = sent_whole(&queue, &whole_session, "vocals");
        if rate == OutputRate::default() {
            let cut = parts_of(crate::session_api::encode_range(
                queue.sessions(),
                &whole_session,
                "vocals",
                from,
                to,
            ));
            assert_eq!(
                cut, fresh_range,
                "{format}: a range cut from the whole file"
            );
        }
        assert_eq!(
            whole_file(&queue, &whole_session, "vocals"),
            sent,
            "a whole read decoded from the file differs from the file"
        );
        queue
            .sessions()
            .remove(&whole_session.id, "ended by the client");

        restore_pieces(&dir, &backup, format);
        let stored = open_at(&queue, frames, format, rate);
        upload(&queue, &stored, 0, &mix);
        let view = queue.view_of(&stored);
        assert!(view.ready.is_empty(), "stored pieces were separated");
        assert!(view.complete, "every piece is stored");
        let stored_file = whole_file(&queue, &stored, "vocals");
        queue.sessions().remove(&stored.id, "ended by the client");

        // Drop the second half of the pieces, so that half separates again.
        restore_pieces(&dir, &backup, format);
        let count = fresh.track().spec.grid.len();
        for k in count / 2..count {
            for stem in stems {
                let file = dir
                    .join(stem)
                    .join(format!("{k:04}.{}", format.extension()));
                std::fs::remove_file(&file).expect("a stored piece");
            }
        }
        let partly = open_at(&queue, frames, format, rate);
        upload(&queue, &partly, 0, &mix);
        assert!(
            !partly
                .track()
                .complete_with(&queue.sessions().cached_for(&partly))
        );
        let (status, _, body) = whole_file(&queue, &partly, "vocals");
        assert_eq!(status, 409, "{}", String::from_utf8_lossy(&body));
        partly.set_window(&whole(&partly)).expect("open");
        queue.wake();
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while !partly
            .track()
            .complete_with(&queue.sessions().cached_for(&partly))
        {
            assert!(
                std::time::Instant::now() < deadline,
                "the missing half never ran"
            );
            thread::sleep(POLL_INTERVAL);
        }
        let summary = queue.view_of(&partly);
        assert!(summary.complete);
        assert_ne!(
            summary.ready,
            vec![[0, frames]],
            "the stored half was separated"
        );
        let partly_file = whole_file(&queue, &partly, "vocals");
        [fresh_file, stored_file, partly_file, sent]
    }

    #[test]
    fn a_whole_track_read_is_the_same_file_from_memory_and_from_stored_pieces() {
        let _gpu = GPU.lock();
        for format in [
            StemFormat::Flac,
            StemFormat::Wav,
            StemFormat::Pcm16,
            StemFormat::Pcm32,
        ] {
            let [fresh, stored, partly, whole] = four_ways(format, OutputRate::default(), 11);
            assert_eq!(fresh.0, 200, "{format}");
            assert_eq!(stored, fresh, "{format}: all stored");
            assert_eq!(partly, fresh, "{format}: partly stored");
            assert_eq!(whole, fresh, "{format}: stored whole");
        }
    }

    #[test]
    fn at_another_rate_stored_pieces_differ_by_at_most_one_lsb() {
        let _gpu = GPU.lock();
        let samples = |body: &[u8]| -> Vec<i16> {
            body.chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        for rate in [OutputRate::Hz48000, OutputRate::Hz96000] {
            let [fresh, stored, partly, whole] = four_ways(StemFormat::Pcm16, rate, 13);
            assert_eq!(whole, fresh, "{rate}: stored whole");
            let want = samples(&fresh.2);
            for (label, got) in [("all stored", &stored), ("partly stored", &partly)] {
                assert_eq!(got.1, fresh.1, "{rate}, {label}: headers");
                let got = samples(&got.2);
                assert_eq!(got.len(), want.len(), "{rate}, {label}: length");
                let diffs: Vec<i32> = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
                    .collect();
                // The resampler's rounding where a piece was converted on its own.
                let max = diffs.iter().max().copied().unwrap_or(0);
                let count = diffs.iter().filter(|d| **d > 0).count();
                assert!(max <= 1, "{rate}, {label}: off by {max} LSB");
                assert!(
                    count * 1000 < diffs.len(),
                    "{rate}, {label}: {count} samples differ"
                );
            }
        }
    }

    #[test]
    fn a_piece_deleted_during_a_session_is_separated_again_on_the_next_window() {
        let _gpu = GPU.lock();
        let frames = 30_000;
        let mix = track(frames, 14);
        let queue = start(Toy::new(SEGMENT, 0.25));
        let pieces = with_held_pieces(&queue);

        let first = open_as(&queue, frames, None, StemFormat::Flac);
        first.set_window(&whole(&first)).expect("open");
        upload(&queue, &first, 0, &mix);
        queue.wake();
        assert!(stored_within(&queue, &first, Duration::from_secs(20)));
        let want = stemd_core::encode_stem(
            &first.read("vocals", 0, frames).expect("ready").audio,
            StemFormat::Flac,
            first.gain,
        )
        .expect("encoding");
        queue.sessions().remove(&first.id, "ended by the client");

        let second = open_as(&queue, frames, None, StemFormat::Flac);
        upload(&queue, &second, 0, &mix);
        let dir = pieces.verified_dir(
            "toy-00000000",
            &type_dir(&second.output, second.gain),
            &crate::sessions::track_id(&mix),
        );
        for stem in ["harmonics", "vocals", "drums"] {
            std::fs::remove_file(dir.join(stem).join("0003.flac")).expect("a stored piece");
        }
        let view = queue.view_of(&second);
        assert!(!view.complete);
        assert_eq!(whole_file(&queue, &second, "vocals").0, 409);

        queue
            .sessions()
            .set_window(&second, &whole(&second))
            .expect("open");
        queue.wake();
        assert!(stored_within(&queue, &second, Duration::from_secs(20)));
        assert_eq!(whole_file(&queue, &second, "vocals").2, want);
    }

    #[test]
    fn a_track_with_every_piece_stored_is_stored_whole_once_the_gate_opens() {
        let _gpu = GPU.lock();
        let frames = 30_000;
        let mix = track(frames, 15);
        let queue = start(Toy::new(SEGMENT, 0.25));
        let pieces = with_pieces(&queue);
        let open = Arc::new(AtomicBool::new(false));
        let gate = Arc::clone(&open);
        pieces.set_gate(Box::new(move || gate.load(Ordering::SeqCst)));
        let stems = ["harmonics", "vocals", "drums"];

        let first = open_as(&queue, frames, None, StemFormat::Flac);
        first.set_window(&whole(&first)).expect("open");
        upload(&queue, &first, 0, &mix);
        queue.wake();
        assert!(stored_within(&queue, &first, Duration::from_secs(20)));
        let range = |session: &Session| {
            parts_of(crate::session_api::encode_range(
                queue.sessions(),
                session,
                "vocals",
                frames / 3,
                frames / 3 + 5000,
            ))
        };
        let from_memory = range(&first);
        let dir = pieces.verified_dir(
            "toy-00000000",
            &type_dir(&first.output, first.gain),
            &crate::sessions::track_id(&mix),
        );
        thread::sleep(Duration::from_millis(300));
        assert!(
            stems.iter().all(|s| dir.join(s).is_dir()),
            "stored whole while the gate was shut"
        );
        assert!(pieces.is_consolidating(&dir));

        open.store(true, Ordering::SeqCst);
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while stems
            .iter()
            .any(|s| crate::whole::find(&dir, s, "flac").is_none())
        {
            assert!(std::time::Instant::now() < deadline, "never stored whole");
            thread::sleep(POLL_INTERVAL);
        }
        while pieces.is_consolidating(&dir) {
            thread::sleep(POLL_INTERVAL);
        }
        for stem in stems {
            assert!(!dir.join(stem).exists(), "{stem} kept its pieces");
        }
        queue.sessions().remove(&first.id, "ended by the client");

        let second = open_as(&queue, frames, None, StemFormat::Flac);
        upload(&queue, &second, 0, &mix);
        let view = queue.view_of(&second);
        assert!(view.complete);
        assert!(view.ready.is_empty());
        assert_eq!(
            range(&second),
            from_memory,
            "a range cut from the whole file"
        );
        assert_eq!(
            whole_file(&queue, &second, "vocals"),
            sent_whole(&queue, &second, "vocals")
        );
    }

    /// Wait until every stem of `output` for `mix` is stored whole.
    fn whole_within(
        pieces: &Pieces,
        mix: &Audio,
        output: &Output,
        gain: f32,
    ) -> std::path::PathBuf {
        let dir = pieces.verified_dir(
            "toy-00000000",
            &type_dir(output, gain),
            &crate::sessions::track_id(mix),
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while crate::pieces::stems_of(output)
            .iter()
            .any(|s| crate::whole::find(&dir, s, output.format.extension()).is_none())
            || pieces.is_consolidating(&dir)
        {
            assert!(std::time::Instant::now() < deadline, "never stored whole");
            thread::sleep(POLL_INTERVAL);
        }
        dir
    }

    #[test]
    fn a_lookup_follows_a_track_from_open_to_stored() {
        use crate::separations::{Held, Storage, lookup};
        let _gpu = GPU.lock();
        let frames = 30_000;
        let mix = track(frames, 16);
        let id = crate::sessions::track_id(&mix);
        let queue = start(Toy::new(SEGMENT, 0.25));
        let pieces = with_pieces(&queue);
        let open = Arc::new(AtomicBool::new(false));
        let gate = Arc::clone(&open);
        pieces.set_gate(Box::new(move || gate.load(Ordering::SeqCst)));
        let flac = output(StemFormat::Flac);
        let held = || lookup(Some(&pieces), queue.sessions(), "toy-00000000", &flac, &id);
        assert!(held().is_none());

        let session = open_as(&queue, frames, None, StemFormat::Flac);
        upload(&queue, &session, 0, &mix);
        let now = held().expect("open");
        assert_eq!(now.state, Held::Open);
        assert_eq!(now.frames, frames);
        assert_eq!(now.output.gain, None);

        session.set_window(&whole(&session)).expect("open");
        queue.wake();
        assert!(stored_within(&queue, &session, Duration::from_secs(20)));
        assert_eq!(held().expect("stored").state, Held::Consolidating);

        open.store(true, Ordering::SeqCst);
        let dir = whole_within(&pieces, &mix, &flac, session.gain);
        let now = held().expect("stored");
        assert_eq!(now.state, Held::Stored);
        assert_eq!(now.output.gain, Some(session.gain));
        assert_eq!(
            now.stems
                .iter()
                .map(|s| (s.name, s.sources))
                .collect::<Vec<_>>(),
            [("harmonics", 6), ("vocals", 8), ("drums", 1)]
        );
        for stem in &now.stems {
            assert_eq!(stem.stored, Storage::Whole);
            let len = std::fs::metadata(dir.join(format!("{}.flac", stem.name)))
                .expect("a whole file")
                .len();
            assert_eq!(stem.bytes, Some(len));
        }

        queue.sessions().remove(&session.id, "ended by the client");
        std::fs::remove_file(dir.join("vocals.flac")).expect("a whole file");
        let now = held().expect("partial");
        assert_eq!(now.state, Held::Partial);
        assert_eq!(now.stems[1].stored, Storage::None);
    }

    #[test]
    fn a_track_uploaded_but_never_separated_is_unknown_once_its_session_ends() {
        use crate::separations::lookup;
        let _gpu = GPU.lock();
        let frames = 30_000;
        let mix = track(frames, 23);
        let id = crate::sessions::track_id(&mix);
        let queue = start(Toy::new(SEGMENT, 0.25));
        let pieces = with_held_pieces(&queue);
        let flac = output(StemFormat::Flac);
        let session = open_as(&queue, frames, None, StemFormat::Flac);
        upload(&queue, &session, 0, &mix);
        queue.sessions().remove(&session.id, "ended by the client");
        assert!(lookup(Some(&pieces), queue.sessions(), "toy-00000000", &flac, &id).is_none());
    }

    #[test]
    fn a_track_that_fails_to_be_stored_whole_reads_as_partial_until_the_retry() {
        use crate::separations::{Held, lookup};
        let _gpu = GPU.lock();
        let frames = 30_000;
        let mix = track(frames, 24);
        let id = crate::sessions::track_id(&mix);
        let queue = start(Toy::new(SEGMENT, 0.25));
        let pieces = with_pieces(&queue);
        let open = Arc::new(AtomicBool::new(false));
        let gate = Arc::clone(&open);
        pieces.set_gate(Box::new(move || gate.load(Ordering::SeqCst)));
        let flac = output(StemFormat::Flac);
        let session = open_as(&queue, frames, None, StemFormat::Flac);
        session.set_window(&whole(&session)).expect("open");
        upload(&queue, &session, 0, &mix);
        queue.wake();
        assert!(stored_within(&queue, &session, Duration::from_secs(20)));
        queue.sessions().remove(&session.id, "ended by the client");

        let dir = pieces.verified_dir("toy-00000000", &type_dir(&flac, session.gain), &id);
        let piece = std::fs::read_dir(dir.join("harmonics"))
            .expect("pieces")
            .flatten()
            .next()
            .expect("a piece")
            .path();
        std::fs::write(&piece, b"not flac").expect("writing");
        open.store(true, Ordering::SeqCst);
        while pieces.is_consolidating(&dir) {
            thread::sleep(POLL_INTERVAL);
        }
        assert!(pieces.consolidation_failed(&dir));
        let held = lookup(Some(&pieces), queue.sessions(), "toy-00000000", &flac, &id);
        assert_eq!(held.expect("partial").state, Held::Partial);
        assert!(!pieces.is_consolidating(&dir), "queued again");
    }

    #[test]
    fn a_stored_output_converts_to_another_container_of_the_same_samples() {
        use crate::separations::{Held, derive, lookup};
        let _gpu = GPU.lock();
        let frames = 30_000;
        let queue = start(Toy::new(SEGMENT, 0.25));
        let pieces = with_pieces(&queue);
        let find = |out: &Output, mix: &Audio| {
            lookup(
                Some(&pieces),
                queue.sessions(),
                "toy-00000000",
                out,
                &crate::sessions::track_id(mix),
            )
        };
        let derive_from = |out: &Output, mix: &Audio, source: &Output, gain: f32| {
            let id = crate::sessions::track_id(mix);
            let dir = pieces.verified_dir("toy-00000000", &type_dir(source, gain), &id);
            let source = crate::separations::Found {
                info: Pieces::info(&dir).expect("a track.json"),
                dir,
                output: source.clone(),
                gain,
            };
            let to_gain = find(out, mix)
                .expect("derivable")
                .output
                .gain
                .expect("a gain");
            derive(&pieces, "toy-00000000", &id, &source, out, to_gain).expect("derived")
        };

        // flac to wav: the same 16-bit samples in another container.
        let mix = track(frames, 17);
        let session = open_as(&queue, frames, None, StemFormat::Flac);
        session.set_window(&whole(&session)).expect("open");
        upload(&queue, &session, 0, &mix);
        queue.wake();
        let flac = output(StemFormat::Flac);
        let flac_dir = whole_within(&pieces, &mix, &flac, session.gain);
        let wav = output(StemFormat::Wav);
        let now = find(&wav, &mix).expect("derivable");
        assert_eq!(now.state, Held::Derivable);
        assert_eq!(now.from.expect("a source").format, "flac");
        let wav_dir = derive_from(&wav, &mix, &flac, session.gain).dir;
        assert_eq!(find(&wav, &mix).expect("stored").state, Held::Stored);
        for stem in ["harmonics", "vocals", "drums"] {
            let decoded = |dir: &std::path::Path, format: StemFormat| {
                let path = dir.join(format!("{stem}.{}", format.extension()));
                let bytes = std::fs::read(path).expect("a whole file");
                bits(&crate::pieces::decode(bytes, format, 2, 44_100).expect("decodes"))
            };
            assert_eq!(
                decoded(&wav_dir, StemFormat::Wav),
                decoded(&flac_dir, StemFormat::Flac),
                "{stem}"
            );
        }
        queue.sessions().remove(&session.id, "ended by the client");

        // f32le to flac: a 16-bit read at the gain a session without `peak` uses.
        let mix = track(frames, 18);
        let session = open_as(&queue, frames, None, StemFormat::Pcm32);
        session.set_window(&whole(&session)).expect("open");
        upload(&queue, &session, 0, &mix);
        queue.wake();
        let pcm32 = output(StemFormat::Pcm32);
        whole_within(&pieces, &mix, &pcm32, 1.0);
        let to_gain = gain(&flac, None);
        let want = stemd_core::encode_stem(
            &session.read("vocals", 0, frames).expect("ready").audio,
            StemFormat::Flac,
            to_gain,
        )
        .expect("encoding");
        let flac_dir = derive_from(&flac, &mix, &pcm32, 1.0).dir;
        assert_eq!(
            std::fs::read(flac_dir.join("vocals.flac")).expect("a whole file"),
            want
        );
        assert!(
            find(&output(StemFormat::Pcm32), &track(frames, 19)).is_none(),
            "nothing converts to f32le"
        );
    }

    #[test]
    fn a_view_shows_how_each_stem_is_stored_and_changes_its_rev() {
        use crate::separations::Storage;
        let _gpu = GPU.lock();
        let frames = 30_000;
        let mix = track(frames, 20);
        let queue = start(Toy::new(SEGMENT, 0.25));
        with_held_pieces(&queue);
        let session = open_as(&queue, frames, None, StemFormat::Flac);
        let mut changes = queue.sessions().subscribe();
        changes.borrow_and_update();
        let before = queue.view_of(&session);
        assert_eq!(before.rev, queue.view_of(&session).rev);
        assert_eq!(before.track.id, None);
        assert!(before.stems.iter().all(|s| s.stored == Storage::None));

        session.set_window(&whole(&session)).expect("open");
        upload(&queue, &session, 0, &mix);
        queue.wake();
        assert!(stored_within(&queue, &session, Duration::from_secs(20)));
        assert!(changes.has_changed().expect("a sender"));
        let count = session.track().spec.grid.len();
        let view = queue.view_of(&session);
        assert_ne!(view.rev, before.rev);
        assert_eq!(view.track.id, Some(crate::sessions::track_id(&mix)));
        for stem in &view.stems {
            assert_eq!(stem.stored, Storage::Pieces, "{}", stem.name);
            assert_eq!(stem.pieces, vec![[0, count]]);
        }

        assert_eq!(whole_file(&queue, &session, "vocals").0, 200);
        let view = queue.view_of(&session);
        let vocals = view
            .stems
            .iter()
            .find(|s| s.name == "vocals")
            .expect("vocals");
        assert_eq!(vocals.stored, Storage::Whole);
        assert!(vocals.bytes.is_some_and(|b| b > 0));
        assert!(vocals.pieces.is_empty());
        let harmonics = &view.stems[0];
        assert_eq!(harmonics.stored, Storage::Pieces);
    }

    #[test]
    fn a_session_joins_an_open_track_without_an_upload() {
        let _gpu = GPU.lock();
        let frames = 30_000;
        let mix = track(frames, 21);
        let toy = Toy::new(SEGMENT, 0.25);
        let runs = Arc::clone(&toy.runs);
        let queue = start(toy);
        with_held_pieces(&queue);
        let first = open_as(&queue, frames, None, StemFormat::Flac);
        upload(&queue, &first, 0, &mix);
        let id = crate::sessions::track_id(&mix);
        let track = queue
            .sessions()
            .open_track(&id, Parts::DEFAULT)
            .expect("an open track");

        let second = queue
            .sessions()
            .join(output(StemFormat::Wav), PcmFormat::F32le, None, track)
            .expect("room");
        assert_eq!(second.track().uid, first.track().uid);
        assert_eq!(runs.load(Ordering::SeqCst), 0);
        second.set_window(&whole(&second)).expect("open");
        queue.wake();
        assert!(ready_within(&first, 0, frames, Duration::from_secs(20)));
        assert_eq!(
            runs.load(Ordering::SeqCst),
            second.track().spec.grid.len(),
            "each segment ran once"
        );
        assert!(
            queue
                .sessions()
                .open_track("000000000000000000000000", Parts::DEFAULT)
                .is_none()
        );
    }

    #[test]
    fn a_join_after_the_last_session_left_finds_the_track_gone() {
        let _gpu = GPU.lock();
        let frames = 30_000;
        let mix = track(frames, 22);
        let queue = start(Toy::new(SEGMENT, 0.25));
        with_held_pieces(&queue);
        let first = open(&queue, frames, None);
        upload(&queue, &first, 0, &mix);
        let id = crate::sessions::track_id(&mix);
        let track = queue
            .sessions()
            .open_track(&id, Parts::DEFAULT)
            .expect("an open track");

        assert!(queue.sessions().remove(&first.id, "ended by the client"));
        let joined =
            queue
                .sessions()
                .join(output(StemFormat::Pcm32), PcmFormat::F32le, None, track);
        assert!(
            matches!(joined, Err(crate::sessions::JoinError::Gone)),
            "the join found the track open"
        );
        assert!(queue.sessions().is_empty());
    }

    #[test]
    fn the_segment_time_is_the_median_of_the_last_ones() {
        let mut speed = Speed::default();
        assert_eq!(speed.median(), None);
        speed.record(6.0);
        assert_eq!(speed.median(), Some(6.0), "one segment is enough");
        for _ in 0..SPEED_KEPT / 2 {
            speed.record(0.5);
        }
        assert_eq!(speed.median(), Some(0.5), "a slow outlier does not move it");
        for _ in 0..SPEED_KEPT {
            speed.record(0.4);
        }
        assert_eq!(speed.median(), Some(0.4), "older times left");
    }

    #[test]
    fn session_segments_set_the_segment_time() {
        let _gpu = GPU.lock();
        let frames = 30_000;
        let queue = start(Toy::new(SEGMENT, 0.25));
        let session = open(&queue, frames, None);
        session.set_window(&whole(&session)).expect("open");
        upload(&queue, &session, 0, &track(frames, 22));
        queue.wake();
        assert!(ready_within(&session, 0, frames, Duration::from_secs(20)));
        assert!(queue.view_of(&session).grid.segment_run_seconds.is_some());
    }

    #[test]
    fn a_whole_track_read_before_any_separation_is_refused() {
        let _gpu = GPU.lock();
        let queue = start(Toy::new(SEGMENT, 0.25));
        with_pieces(&queue);
        let frames = 20_000;
        let session = open_at(&queue, frames, StemFormat::Flac, OutputRate::default());
        upload(&queue, &session, 0, &track(frames, 12));
        let (status, _, body) = whole_file(&queue, &session, "vocals");
        assert_eq!(status, 409);
        assert!(
            String::from_utf8_lossy(&body).contains("neither ready nor stored"),
            "{}",
            String::from_utf8_lossy(&body)
        );
    }

    #[test]
    fn a_job_leaves_its_pieces_for_sessions() {
        let _gpu = GPU.lock();
        if crate::disk::memory_pressure() {
            eprintln!("skipped: under memory pressure a job writes no pieces");
            return;
        }
        let queue = start(Toy::new(SEGMENT, 0.25));
        let pieces = with_pieces(&queue);
        let frames = 25_000;
        let mix = track(frames, 10);
        let id = crate::sessions::track_id(&mix);
        let store = crate::jobs::JobStore::new(Duration::from_secs(60));
        let job = store.create("for-pieces".into());
        queue
            .submit(QueuedWork {
                job: Arc::clone(&job),
                mix: mix.clone(),
                output: output(StemFormat::Flac),
                model_id: "toy-00000000".into(),
                from_api: true,
                file: Some(KEY.into()),
            })
            .expect("room");
        let name = type_dir(
            &output(StemFormat::Flac),
            gain(&output(StemFormat::Flac), mix.peak().ok()),
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let dir = pieces.verified_dir("toy-00000000", &name, &id);
        let stems = ["harmonics", "vocals", "drums"];
        // A whole file is published before its pieces are removed.
        let stored = |stem: &&str| {
            crate::whole::find(&dir, stem, "flac").is_some() && !dir.join(stem).exists()
        };
        while !stems.iter().all(stored) {
            assert!(
                std::time::Instant::now() < deadline,
                "the job stored no whole stems"
            );
            thread::sleep(POLL_INTERVAL);
        }
        for stem in ["harmonics", "vocals", "drums"] {
            assert!(
                crate::whole::find(&dir, stem, "flac").is_some(),
                "{stem} is not stored whole"
            );
            assert!(!dir.join(stem).exists(), "{stem} kept its pieces");
        }
        assert_eq!(
            pieces.file_track(KEY),
            Some(id),
            "the file key names the track"
        );
    }

    #[test]
    fn a_session_records_its_file_key_once_the_upload_is_hashed() {
        let _gpu = GPU.lock();
        let frames = 30_000;
        let mix = track(frames, 25);
        let queue = start(Toy::new(SEGMENT, 0.25));
        let pieces = with_held_pieces(&queue);
        let session = open(&queue, frames, None);
        session.set_file(Some(KEY.into()));
        upload(&queue, &session, 0, &excerpt(&mix, 0, frames / 2));
        assert_eq!(pieces.file_track(KEY), None, "recorded before the hash");
        upload(
            &queue,
            &session,
            frames / 2,
            &excerpt(&mix, frames / 2, frames),
        );
        assert_eq!(
            pieces.file_track(KEY),
            Some(crate::sessions::track_id(&mix))
        );
    }
}
