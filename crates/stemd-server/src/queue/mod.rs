//! The job queue and the single separation worker.
//!
//! One worker runs one segment at a time. Whole-track jobs run in FIFO order, with a reported
//! position, backpressure when the queue is full, and cancellation of the running job and the
//! waiting ones.
//!
//! Between segments, the worker takes work in this order:
//!
//! 1. a staged model, when no job is in the middle of a track;
//! 2. a high-priority session segment, also between two segments of a running job;
//! 3. the next waiting job;
//! 4. a normal-priority session segment.
//!
//! The worker builds every separator it runs. See [`BuildSeparator`].

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use anyhow::Result;
use parking_lot::{Condvar, Mutex};
use stemd_core::{Audio, BackendInfo, Progress, Segmenting, Separate, Stage};

use crate::cache::Cache;
use crate::jobs::Job;
use crate::output::Output;
use crate::sessions::{Class, Session, Sessions, Track, View};

pub use crate::models::BuildSeparator;

pub struct QueuedWork {
    pub job: Arc<Job>,
    pub mix: Audio,
    /// The format, rate and part set the client asked for.
    pub output: Output,
    /// The cache id the job was submitted under, which names the model's
    /// directory in the cache.
    pub model_id: String,
    /// True for a job submitted over the API, which turning the server off cancels; false for a
    /// drop.
    pub from_api: bool,
}

#[derive(Debug, thiserror::Error)]
#[error("queue is full ({depth} waiting)")]
pub struct QueueFull {
    pub depth: usize,
}

/// The job the worker has in flight, and the switch that stops it.
///
/// Each job gets a new cancel flag, so a `cancel` that races the end of one job sets the flag of a
/// `Running` the worker has already dropped and cannot cancel the next job.
struct Running {
    id: String,
    from_api: bool,
    /// See [`Pending::number`].
    number: Option<u64>,
    /// The job, for its progress in [`Queue::client_jobs`].
    job: Arc<Job>,
    cancel: Arc<AtomicBool>,
}

/// The number of recent segment times that [`Queue::segment_run_seconds`] takes the median of.
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
    /// The API job's number, counting from 1 since the server started; `None` for a drop.
    number: Option<u64>,
}

/// An API job as the window shows it. It has no id, because a job id contains a prefix of the
/// track's hash.
#[derive(Clone, Debug)]
pub struct ClientJob {
    pub number: u64,
    pub running: bool,
    pub progress: Progress,
    /// True when a protocol 1 client waits on the job.
    pub protocol_1: bool,
}

impl ClientJob {
    /// API job `number` with the current state of `job`.
    fn of(number: u64, running: bool, job: &Job) -> Self {
        Self {
            number,
            running,
            progress: job.progress.lock().clone(),
            protocol_1: job.protocol_1(),
        }
    }
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
    /// Receives the installed model's info, or the reason it did not load. Dropping it without a
    /// send tells the caller that the worker stopped.
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
    /// A separator waiting to be built and to replace the live one, taken between jobs. The worker
    /// owns its separator, so a switch requested during a separation takes effect on the next
    /// track.
    incoming: Mutex<Option<Staged>>,
    /// The info of the separator the worker holds. The worker writes it when it installs a
    /// separator, so it never describes a model the worker does not have.
    ///
    /// `None` only between spawning the worker and its first build; [`Queue::start`] returns only
    /// once it is `Some`.
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

/// How often [`Queue::stop`] checks whether the worker has left the separator.
pub(crate) const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(5);

/// How long [`Queue::stop`] waits for the worker to leave the separator.
///
/// The worker checks for a stop between segments, so the grace covers one segment on the slowest
/// preset on a loaded machine, or a model build already under way when the quit arrives.
pub const STOP_GRACE: std::time::Duration = std::time::Duration::from_secs(10);

impl Queue {
    /// Spawn the worker thread and have it build the first separator.
    ///
    /// Blocks until the build finishes, so a model that does not load is a synchronous error at the
    /// call site. [`BuildSeparator`] explains why the worker allocates the weights.
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
            sessions: Sessions::new(std::time::Duration::from_secs(30), 4, cache),
            speed: Mutex::new(Speed::default()),
            busy: AtomicBool::new(false),
            left: AtomicBool::new(false),
        });

        let (reply, built) = mpsc::sync_channel(1);
        let worker_inner = Arc::clone(&inner);
        let worker = thread::Builder::new()
            .name("stemd-worker".into())
            .spawn(move || {
                // Build the separator first, on this thread; see `BuildSeparator`.
                if let Some(separator) = worker_inner.build_here(build, &reply) {
                    worker_loop(&worker_inner, separator);
                }
                worker_inner.left.store(true, Ordering::SeqCst);
                // Release the queue's state, so a dropped queue frees its cache.
                drop(worker_inner);
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

    /// The info of the separator the worker holds now.
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

    /// The median run time in seconds of the last [`SPEED_KEPT`] segments; `None` until a segment
    /// has run.
    pub fn segment_run_seconds(&self) -> Option<f64> {
        self.inner.speed.lock().median()
    }

    /// What `session`'s client sees now.
    pub fn view_of(&self, session: &Session) -> View {
        self.inner
            .sessions
            .view_of(session, self.segment_run_seconds())
    }

    /// The gate for background consolidation: always open on a GPU, and open on a CPU while the
    /// worker is idle.
    pub fn consolidation_gate(&self) -> crate::cache::Gate {
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

    /// True when the worker runs no job or segment.
    #[cfg(test)]
    pub fn is_idle(&self) -> bool {
        let _pending = self.inner.pending.lock();
        !self.inner.busy.load(Ordering::SeqCst) && self.inner.running.lock().is_none()
    }

    /// Tell the worker that session work may be waiting.
    pub fn wake(&self) {
        // Notify under `pending` so the worker cannot miss the wake between its check and its wait.
        let _parked = self.inner.pending.lock();
        self.inner.ready.notify_all();
    }

    /// Stop the worker: refuse new jobs, cancel the running job, and wait for the worker to leave
    /// the separator. Returns false if the worker was still inside when `grace` ran out.
    // Every exit path must call this and wait for true: ending the process during a forward pass
    // tears down the GPU allocator from a destructor, and nothing catches that abort. The wait is
    // one segment.
    pub fn stop(&self, grace: std::time::Duration) -> bool {
        {
            // Notify under `pending`, as in `Drop`: the worker checks `shutdown` while holding
            // `pending` and only then parks, so a notify sent without the lock could be missed.
            let _parked = self.inner.pending.lock();
            self.inner.shutdown.store(true, Ordering::SeqCst);
        }
        // Release a switch blocked in `install`. Its reply channel sits in `incoming`, which
        // outlives the worker, so the switch thread would otherwise wait for as long as the queue
        // exists.
        if let Some(staged) = self.inner.incoming.lock().take() {
            let _ = staged
                .reply
                .send(Err(anyhow::anyhow!("the server is shutting down")));
        }
        // `pending` is released before taking `running`: the worker takes `running` while holding
        // `pending`, so holding both here in the other order would deadlock.
        if let Some(running) = self.inner.running.lock().as_ref() {
            // The worker logs its own line when it leaves the model.
            tracing::debug!("stopping {} at its next segment", running.id);
            running.cancel.store(true, Ordering::Relaxed);
        }
        self.inner.ready.notify_all();

        let Some(worker) = self.worker.lock().take() else {
            return true;
        };
        // `JoinHandle` has no timed join, and an untimed join would hang the quit when the worker
        // is stuck in the model.
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

    /// Stop a job, whether it is waiting or running. Returns false when the job is neither, because
    /// it finished or was never queued.
    // A waiting job is dropped at once. A running job stops at its next segment boundary, so this
    // returns before the worker is free.
    pub fn cancel(&self, id: &str) -> bool {
        {
            let mut pending = self.inner.pending.lock();
            let before = pending.len();
            pending.retain(|p| p.work.job.id != id);
            if before != pending.len() {
                return true;
            }
        }
        // `pending` is released first: the worker holds `pending` while it takes `running`, so
        // holding both here in the other order would deadlock.
        let running = self.inner.running.lock();
        match running.as_ref() {
            Some(job) if job.id == id => {
                job.cancel.store(true, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }

    /// Cancel every job submitted over the API, waiting or running, and return the number
    /// cancelled. Drops keep their place.
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
        // Lock `pending`, then `running`, never both at once: the worker holds `pending` while it
        // takes `running`. A job taken in between shows up in both lists and keeps its running
        // entry.
        let waiting: Vec<ClientJob> = self
            .inner
            .pending
            .lock()
            .iter()
            .filter_map(|p| {
                p.number
                    .map(|number| ClientJob::of(number, false, &p.work.job))
            })
            .collect();
        let running = self
            .inner
            .running
            .lock()
            .as_ref()
            .and_then(|r| r.number.map(|number| ClientJob::of(number, true, &r.job)));
        running_first(running, waiting)
    }

    /// Have the worker build a new separator and run on it from now on.
    ///
    /// Blocks until the swap has happened, after the running track and the build, so the caller
    /// learns what loaded. The old separator keeps running until the new one is built.
    pub fn install(&self, build: BuildSeparator) -> Result<BackendInfo> {
        let (reply, done) = mpsc::sync_channel(1);
        let superseded = {
            // Notify under `pending`, as in `stop`. Without the lock the notify could be missed,
            // and an idle worker would then wait until the next job arrives, which may never
            // happen.
            let _parked = self.inner.pending.lock();
            // Read `shutdown` under the same lock, so a switch cannot start after a `stop` and wait
            // for a worker that has already left.
            if self.inner.shutdown.load(Ordering::SeqCst) {
                anyhow::bail!("the server is shutting down");
            }
            self.inner.incoming.lock().replace(Staged { build, reply })
        };
        // Reached only when two switches arrive faster than the worker can take one, which
        // `Switcher::request` refuses. Answer the superseded switch, because its caller is blocked
        // on that channel.
        if let Some(staged) = superseded {
            let _ = staged
                .reply
                .send(Err(anyhow::anyhow!("superseded by a later switch")));
        }
        self.inner.ready.notify_one();

        match done.recv() {
            Ok(result) => result,
            // The worker held the sender, so the queue stopped during the switch, on the way out.
            Err(_) => anyhow::bail!("the separation worker stopped before the model could load"),
        }
    }

    /// Progress for a waiting job, with its live queue position filled in.
    ///
    /// The position is computed on read: every waiter's position shifts each time the worker takes
    /// a job, and the queue is short enough that a scan per poll costs less than pushing updates.
    pub fn queued_progress_for(&self, id: &str) -> Option<Progress> {
        let pending = self.inner.pending.lock();
        let position = pending.iter().position(|p| p.work.job.id == id)?;
        Some(queued_progress(position, pending.len()))
    }
}

impl Drop for Queue {
    /// Does nothing once [`Queue::stop`] has run, which is the normal path. It stops a queue
    /// dropped without a `stop`: in every test, and on any early return between `start` and a
    /// running server.
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
    /// Woken with nothing to run, so the worker can drop what closed sessions held.
    Tidy,
}

impl Inner {
    /// Block until there is something to do. `None` means shutdown.
    // A staged model is taken ahead of a waiting job, because a swap is safe only between jobs. It
    // is also taken while idle, so [`Queue::install`] returns on an unused server. A job is
    // registered as running under the same lock that took it out of `pending`.
    fn next(&self) -> Option<Next> {
        let mut pending = self.pending.lock();
        let mut woken = false;
        loop {
            // Check for shutdown before the swap and the job: a model built during teardown would
            // allocate in front of the C runtime's destructors, which is the crash `Queue::stop`
            // prevents.
            if self.shutdown.load(Ordering::SeqCst) {
                return None;
            }
            if let Some(staged) = self.incoming.lock().take() {
                return Some(Next::Install(staged));
            }
            // Set `busy` under `pending`, so `is_idle` never sees a segment taken but not started.
            if let Some((track, k)) = self.sessions.next(Class::BeforeJobs) {
                self.busy.store(true, Ordering::SeqCst);
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
                self.busy.store(true, Ordering::SeqCst);
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

    /// Build a separator on this thread and publish its info. Called only from the worker; see
    /// [`BuildSeparator`].
    ///
    /// Returns `None` when the build failed; the separator the worker already holds stays valid.
    fn build_here(
        &self,
        build: BuildSeparator,
        reply: &mpsc::SyncSender<Result<BackendInfo>>,
    ) -> Option<Box<dyn Separate>> {
        match build() {
            Ok(separator) => {
                let info = separator.info();
                // Publish before replying, so a caller that reads `info` as soon as `install`
                // returns sees the new model.
                *self.info.lock() = Some(info.clone());
                // Open sessions hold output from the old model. Close them before the
                // reply, so `install` returns with them closed.
                self.sessions.close_all("the server switched models");
                *self.speed.lock() = Speed::default();
                *self.segmenting.lock() = separator.segmenting();
                self.normalises
                    .store(separator.normalises(), Ordering::SeqCst);
                // The receiver may be gone if the switch was superseded during the build. Keep the
                // model anyway: it is the newer one, and `info` already names it.
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

mod store;
mod worker;

use worker::worker_loop;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod bench;
