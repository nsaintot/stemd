//! Job queue and the single separation worker.
//!
//! Separation is serialised: one track at a time, FIFO, with a position to report,
//! backpressure when saturated, and cancellation that covers the running job as
//! well as the waiting ones. The separator polls between segments, so cancelling
//! costs one segment.
//!
//! The worker builds every separator it runs. See [`BuildSeparator`].

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

use anyhow::Result;
use parking_lot::{Condvar, Mutex};
use stemd_core::{Audio, BackendInfo, Cancelled, Progress, Separate, Stage, Stems};

use crate::cache::Output;

use crate::cache::Cache;
use crate::jobs::Job;

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
    cancel: Arc<AtomicBool>,
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
    pending: Mutex<VecDeque<QueuedWork>>,
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
            ready: Condvar::new(),
            running: Mutex::new(None),
            max_depth,
            shutdown: AtomicBool::new(false),
            incoming: Mutex::new(None),
            info: Mutex::new(None),
        });

        let (reply, built) = mpsc::sync_channel(1);
        let worker_inner = Arc::clone(&inner);
        let worker = thread::Builder::new()
            .name("stemd-worker".into())
            .spawn(move || {
                // The first thing this thread does, and the reason it is handed
                // a recipe rather than a model.
                let Some(separator) = worker_inner.build_here(build, &reply) else {
                    return;
                };
                worker_loop(&worker_inner, separator, &cache);
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
            // The worker returns the moment the build fails, so joining costs
            // nothing and keeps a refused launch from leaving a thread behind.
            Ok(Err(err)) => {
                let _ = worker.join();
                Err(err)
            }
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
        while !worker.is_finished() {
            if std::time::Instant::now() >= deadline {
                tracing::warn!("the separation worker did not stop within {:.0?}", grace);
                return false;
            }
            thread::sleep(POLL_INTERVAL);
        }
        worker.join().is_ok()
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
        pending.push_back(work);
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
            pending.retain(|w| w.job.id != id);
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

    pub fn depth(&self) -> usize {
        self.inner.pending.lock().len()
    }

    pub fn running(&self) -> Option<String> {
        self.inner.running.lock().as_ref().map(|r| r.id.clone())
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
        let position = pending.iter().position(|w| w.job.id == id)?;
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
            if let Some(work) = pending.pop_front() {
                let cancel = Arc::new(AtomicBool::new(false));
                *self.running.lock() = Some(Running {
                    id: work.job.id.clone(),
                    cancel: Arc::clone(&cancel),
                });
                return Some(Next::Job(Claimed { work, cancel }));
            }
            self.ready.wait(&mut pending);
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

/// What a caught panic said, as far as it can be recovered.
///
/// A payload is `Any`, and the two shapes `panic!` produces are the only ones
/// worth naming. Anything else is a custom payload whose message, if it has
/// one, is not reachable from here.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        return (*text).to_owned();
    }
    if let Some(text) = payload.downcast_ref::<String>() {
        return text.clone();
    }
    "no message".to_owned()
}

/// Take work until asked to stop.
///
/// Nothing a separator does may end this loop. Both callees are caught, so a track
/// that panics fails that track and a build that panics leaves the model already
/// loaded in place. Whether the separator is still sound cannot be known from
/// here, so it is reported at `error`.
fn worker_loop(inner: &Arc<Inner>, mut separator: Box<dyn Separate>, cache: &Cache) {
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
                        tracing::info!("switched to {}", separator.info().model);
                    }
                    // `build_here` has already sent whatever it failed with.
                    Ok(None) => {}
                    Err(payload) => {
                        let what = panic_message(&*payload);
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
                let separating = catch_unwind(AssertUnwindSafe(|| {
                    separate_one(separator.as_mut(), &claimed, cache)
                }));
                if let Err(payload) = separating {
                    let what = panic_message(&*payload);
                    tracing::error!(job = %claimed.work.job.id, "the separation panicked: {what}");
                    // Terminal, so a client polling this job stops waiting for
                    // something that is never going to arrive.
                    claimed
                        .work
                        .job
                        .fail(format!("the separation panicked: {what}"));
                }
                *inner.running.lock() = None;
            }
        }
    }
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

fn separate_one(separator: &mut dyn Separate, claimed: &Claimed, cache: &Cache) {
    let work = &claimed.work;
    let job = &work.job;
    let started = std::time::Instant::now();
    let sink = JobSink {
        job: Arc::clone(job),
        cancel: Arc::clone(&claimed.cancel),
    };

    match separator.separate(&work.mix, &sink) {
        Ok(stems) => publish(job, work, &stems, cache, started.elapsed().as_secs_f64()),
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
    match cache.publish(&job.key, &work.mix, stems, work.output, elapsed) {
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

    /// A separator that records which model "ran", so a swap is observable
    /// without loading 300 MB of weights onto a GPU.
    struct Stub {
        name: &'static str,
        seen: Arc<Mutex<Vec<&'static str>>>,
    }

    impl Separate for Stub {
        fn separate(&mut self, _mix: &Audio, _sink: &dyn ProgressSink) -> anyhow::Result<Stems> {
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
        fn separate(&mut self, _mix: &Audio, _sink: &dyn ProgressSink) -> anyhow::Result<Stems> {
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
        fn separate(&mut self, _mix: &Audio, sink: &dyn ProgressSink) -> anyhow::Result<Stems> {
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
        fn separate(&mut self, _mix: &Audio, sink: &dyn ProgressSink) -> anyhow::Result<Stems> {
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
        fn separate(&mut self, _mix: &Audio, _sink: &dyn ProgressSink) -> anyhow::Result<Stems> {
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
                    dsp: DspMode::default(),
                },
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
