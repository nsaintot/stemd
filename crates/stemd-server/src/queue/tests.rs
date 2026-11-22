use super::store::{store_whole, to_output_rate};
use super::worker::publish;
use super::*;
use crate::cache::Separation;
use crate::ident::track_id;
use crate::testkit::cache;
use std::time::Duration;
use stemd_core::{BackendInfo, DspMode, OutputRate, ProgressSink, StemFormat, Stems};
use stemd_core::{Cancelled, Pass};

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

/// A separator that records which model "ran", so a test can observe a swap without loading 300 MB
/// of weights onto a GPU.
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

/// Wrap an already built separator as the build closure the worker takes. Valid only for stubs,
/// which allocate nothing on any thread.
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

/// Poll until `seen` holds `count` names. `Stub` records under a plain mutex with nothing to wait
/// on, so the deadline turns a swap that never happens into a test failure.
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

fn job(jobs: &crate::jobs::Jobs) -> Arc<Job> {
    job_for(jobs, "test-key")
}

/// A job for `key`. Two jobs in one test need two keys, because `claim` joins a repeated key to the
/// separation already in flight.
fn job_for(jobs: &crate::jobs::Jobs, key: &str) -> Arc<Job> {
    jobs.create(key.into())
}

#[derive(Default)]
struct Segments {
    started: usize,
    cancelled: usize,
}

type Shared = Arc<(Mutex<Segments>, Condvar)>;

/// A separator that, like the real one, works in segments and checks for cancellation only between
/// them, so the queue can be tested without 300 MB of weights on a GPU.
struct Segmented {
    state: Shared,
    /// The number of segments before it finishes on its own. It is long, so a test waiting for the
    /// next job cannot pass by outlasting this one.
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

/// Like [`Segmented`], but it reports progress after each segment, so a test can tell whether
/// progress is published.
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

/// A separator that panics instead of separating, as a panic inside a forward pass would.
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

/// A panic in a separation fails only that track; the worker keeps running and the queue keeps
/// serving jobs.
#[test]
fn a_panic_in_one_separation_does_not_strand_the_jobs_behind_it() {
    let state = shared();
    let queue = start(
        Panics {
            state: Arc::clone(&state),
        },
        cache(u64::MAX, Duration::from_secs(60)),
    );
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));

    let first = job_for(&jobs, "first");
    submit(&queue, Arc::clone(&first));
    assert!(
        wait_until(&state, Duration::from_secs(5), |s| s.started == 1),
        "the worker never picked the first job up"
    );

    // The job is marked failed, so a client polling it learns that it will never finish.
    assert!(
        settled(&first, Duration::from_secs(5)),
        "the job that panicked never reached a terminal state"
    );

    // The queue still runs the next job.
    let second = job_for(&jobs, "second");
    submit(&queue, Arc::clone(&second));
    assert!(
        wait_until(&state, Duration::from_secs(5), |s| s.started == 2),
        "the worker died with the first job and never took the second"
    );
}

/// Poll until the job reaches a terminal stage; false if `within` runs out first.
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

/// Block until the counts satisfy `done`, or until `within` runs out; returns whether they did.
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

/// Enough segments (about 10 s) that no test below can pass by waiting for the job to finish.
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
            cache_id: "test".into(),
            from_api: true,
        })
        .expect("queue has room");
}

/// Quitting during a separation waits for the worker to leave the model, which takes one segment.
#[test]
fn stopping_waits_for_at_most_one_segment() {
    let state = shared();
    let queue = start(
        Segmented {
            state: Arc::clone(&state),
            segments: NEVER_FINISHES,
        },
        cache(u64::MAX, Duration::from_secs(60)),
    );
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    submit(&queue, job(&jobs));
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

/// A switch requested during a track does not stop the track's progress reports. The swap waits for
/// the track, and the window shows the progress bar under the switch dialog.
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
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let job = job(&jobs);
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
        // Stop the queue to let the switch through; otherwise the scope waits for a worker that
        // still has NEVER_FINISHES segments left.
        queue.stop(Duration::from_secs(10));
        let _ = staged.join();
    });
}

/// `Drop` also stops the queue, so the normal path stops it twice. The second stop does nothing: it
/// neither waits nor panics.
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
    assert!(
        began.elapsed() < Duration::from_secs(2),
        "the second stop waited for its grace"
    );
    // `Drop` stops it a third time as the queue goes out of scope here.
}

#[test]
fn a_running_job_stops_between_segments() {
    let state = shared();
    let queue = start(
        Segmented {
            state: Arc::clone(&state),
            segments: NEVER_FINISHES,
        },
        cache(u64::MAX, Duration::from_secs(60)),
    );
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let job = job(&jobs);
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

    // A cancelled job is terminal and carries no error for the client.
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
    // Separation runs one job at a time, so without cancelling the running job, a deck that skips a
    // track would wait for a full separation of the skipped track.
    let state = shared();
    let queue = start(
        Segmented {
            state: Arc::clone(&state),
            segments: NEVER_FINISHES,
        },
        cache(u64::MAX, Duration::from_secs(60)),
    );
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));

    let skipped = job_for(&jobs, "skipped");
    let cued = job_for(&jobs, "cued");
    let (skipped_id, cued_id) = (skipped.id.clone(), cued.id.clone());
    submit(&queue, skipped);
    submit(&queue, cued);

    assert!(
        wait_until(&state, Duration::from_secs(5), |s| s.started == 1),
        "the first track never started"
    );
    // The skipped track must be running, out of `pending`, or the cancel below only drops a waiting
    // job.
    assert_eq!(
        queue.running().as_deref(),
        Some(skipped_id.as_str()),
        "the skipped track should be running, not still queued"
    );
    assert!(queue.cancel(&skipped_id));

    // Well inside the 10 s the skipped track would take, so a broken cancel fails this assertion.
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
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let queue = start(
        Stub {
            name: "first",
            seen: Arc::clone(&seen),
        },
        cache(u64::MAX, Duration::from_secs(60)),
    );

    submit(&queue, job_for(&jobs, "one"));
    assert!(ran(&seen, 1), "the worker never ran the first job");

    // Returns once the swap has happened, on an idle queue as well as a busy
    // one, without waiting for another job.
    queue
        .install(ready(Stub {
            name: "second",
            seen: Arc::clone(&seen),
        }))
        .expect("a stub always builds");
    assert_eq!(queue.info().model, "second");

    submit(&queue, job_for(&jobs, "two"));
    assert!(ran(&seen, 2), "the worker never ran the second job");
    assert_eq!(&*seen.lock(), &["first", "second"]);
}

/// The thread that builds a separator also runs it, on both paths that install one, because MLX's
/// CUDA backend keeps its stream registry in thread-local storage. The test records thread ids, so
/// it needs no GPU.
#[test]
fn the_thread_that_builds_a_model_is_the_thread_that_runs_it() {
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let threads: Threads = Arc::default();

    let (queue, _) = Queue::start(
        traced(&threads),
        8,
        cache(u64::MAX, Duration::from_secs(60)),
    )
    .expect("a stub always builds");
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let happened = |count: usize| {
        while threads.lock().len() < count {
            assert!(
                std::time::Instant::now() < deadline,
                "only {:?} happened",
                threads.lock()
            );
            thread::sleep(POLL_INTERVAL);
        }
    };
    submit(&queue, job_for(&jobs, "one"));
    // A staged model is taken before a waiting job, so install only after the first job ran.
    happened(2);
    queue
        .install(traced(&threads))
        .expect("a stub always builds");
    submit(&queue, job_for(&jobs, "two"));
    happened(4);

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

/// A model that does not load fails `Queue::start`, so `startup::load_model` can fall back to the
/// default preset.
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

/// A switch to a model that does not load keeps the current model, which the window's "Keep current
/// model" button relies on.
#[test]
fn an_install_that_fails_keeps_the_model_that_was_running() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
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
    submit(&queue, job_for(&jobs, "after"));
    assert!(ran(&seen, 1), "the worker stopped running jobs");
    assert_eq!(&*seen.lock(), &["first"]);
}

/// Quitting during a switch releases the switch thread, which waits on a reply channel owned by the
/// queue.
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
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    submit(&queue, job(&jobs));
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

/// When two switches arrive faster than the worker can take one, the first caller still gets an
/// answer. `Switcher::request` refuses this case, but the queue must not leave a caller waiting
/// forever.
#[test]
fn a_superseded_install_answers_its_caller() {
    let state = shared();
    let seen: Arc<Mutex<Vec<&'static str>>> = Arc::default();
    let queue = Arc::new(start(
        Segmented {
            state: Arc::clone(&state),
            segments: NEVER_FINISHES,
        },
        cache(u64::MAX, Duration::from_secs(60)),
    ));
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let job = job(&jobs);
    let id = job.id.clone();
    submit(&queue, job);
    assert!(
        wait_until(&state, Duration::from_secs(5), |s| s.started == 1),
        "the worker never picked the job up"
    );

    // Both switches stay staged because the worker is in the model. Each runs on its own thread,
    // since `install` blocks.
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

    // The first switch returns as soon as the second replaces it, without waiting for the worker.
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

#[test]
fn a_stem_of_nan_fails_the_job() {
    let cache = cache(u64::MAX, Duration::from_secs(60));
    let (mix, mut stems) = crate::testkit::stems(64);
    for (name, audio) in &mut stems.shipped {
        if *name == "vocals" {
            audio.data = audio.data.iter().map(|c| vec![f32::NAN; c.len()]).collect();
        }
    }
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let work = QueuedWork {
        job: jobs.create("diverged".into()),
        mix,
        output: crate::testkit::output(),
        cache_id: "model".into(),
        from_api: true,
    };
    let separation = Separation {
        model_residual_db: -30.0,
        separation_secs: 1.0,
    };
    let err = store_whole(&cache, &work, &stems, 64, separation)
        .expect_err("a stem of NaN must not be stored");
    let message = format!("{err:#}");
    assert!(
        message.contains("vocals") && message.contains("expected a finite number"),
        "the failure has to name the stem: {message}"
    );
    assert_eq!(cache.stats().tracks, 0, "nothing is stored");
}

/// A worker-less job on `mix` at `output`.
fn work_on(store: &crate::jobs::Jobs, mix: &Audio, output: Output) -> QueuedWork {
    QueuedWork {
        job: store.create(format!("{}-{}", output.rate, mix.frames())),
        mix: mix.clone(),
        output,
        cache_id: "model".into(),
        from_api: false,
    }
}

const SEPARATION: Separation = Separation {
    model_residual_db: -30.0,
    separation_secs: 1.0,
};

#[test]
fn a_hit_whose_files_vanished_is_a_miss() {
    let cache = cache(1 << 30, Duration::from_secs(60));
    let (mix, stems) = crate::testkit::stems(4410);
    let store = crate::jobs::Jobs::new(Duration::from_secs(60));
    let work = work_on(&store, &mix, crate::testkit::output());
    store_whole(&cache, &work, &stems, mix.frames(), SEPARATION).expect("stored");
    let id = track_id(&mix);
    let hit = crate::held::cached_entry(&cache, "model", &work.output, &id).expect("a hit");
    std::fs::remove_file(&hit.stem("vocals").unwrap().path).unwrap();
    assert!(crate::held::cached_entry(&cache, "model", &work.output, &id).is_none());
    assert!(!hit.present());
}

#[test]
fn a_fresh_result_over_the_budget_is_kept_until_fetched() {
    let cache = cache(100, Duration::from_secs(60));
    let (mix, stems) = crate::testkit::stems(64);
    let store = crate::jobs::Jobs::new(Duration::from_secs(60));
    let work = work_on(&store, &mix, crate::testkit::output());
    publish(&work.job, &work, &stems, &cache, mix.frames(), 1.0);
    assert_eq!(work.job.progress.lock().stage, Stage::Done);
    let entry = work.job.entry.lock().clone().unwrap();
    cache.clean();
    assert!(entry.present(), "held while unfetched");
    for stem in &entry.stems {
        entry.mark_fetched(&stem.name);
    }
    cache.clean();
    assert!(!entry.dir.exists(), "over the budget once fetched");
}

#[test]
fn a_stored_result_reports_the_rate_it_was_converted_to() {
    let cache = cache(1 << 30, Duration::from_secs(60));
    let (mix, stems) = crate::testkit::stems(44_100);
    let store = crate::jobs::Jobs::new(Duration::from_secs(60));
    for rate in OutputRate::ALL {
        let output = Output {
            rate,
            ..crate::testkit::output()
        };
        let work = work_on(&store, &mix, output);
        let entry = store_whole(&cache, &work, &stems, mix.frames(), SEPARATION).expect("stored");
        assert_eq!(entry.output.rate, rate, "{rate}");
        assert_eq!(
            entry.frames,
            crate::output::output_index(44_100, 44_100, u64::from(rate.hz())),
            "{rate}"
        );
        assert!((entry.duration_secs() - 1.0).abs() < 0.01, "{rate}");
    }
}

fn pcm16_at(rate: OutputRate) -> Output {
    Output {
        format: StemFormat::Pcm16,
        rate,
        derived: false,
        dsp: stemd_core::DspMode::default().into(),
        parts: stemd_core::Parts::DEFAULT,
    }
}

#[test]
fn every_rate_converts_the_parts_then_the_derived_one() {
    let (mix, stems) = crate::testkit::stems(4410);
    for rate in OutputRate::ALL {
        let parts = to_output_rate(&mix, &stems, &pcm16_at(rate)).unwrap();
        let names: Vec<&str> = parts.iter().map(|(n, _)| *n).collect();
        let mut expected = stemd_core::SHIPPED.to_vec();
        expected.push(stemd_core::DERIVED);
        assert_eq!(names, expected, "{rate}");
        for (name, audio) in &parts {
            assert_eq!(audio.sample_rate, rate.hz(), "{name} at the wrong rate");
        }
    }
}

/// A job's own parts ship in [`stemd_core::PART_NAMES`] order, the derived
/// part is what they leave out, and the set still sums to the mix.
#[test]
fn a_parts_list_ships_those_parts_and_rebuilds_the_rest() {
    let (mix, mut stems) = crate::testkit::stems(4410);
    let harmonics = stems.stem("harmonics").unwrap().clone();
    let bass = Audio::new(
        harmonics
            .data
            .iter()
            .map(|c| c.iter().map(|v| v * 0.25).collect())
            .collect(),
        harmonics.sample_rate,
    );
    let other = stemd_core::stems::minus(&harmonics, &bass);
    stems.more = vec![("bass", bass), ("other", other)];

    let output = Output {
        parts: stemd_core::Parts::parse("vocals,other,bass").unwrap(),
        ..pcm16_at(OutputRate::Hz44100)
    };
    let parts = to_output_rate(&mix, &stems, &output).unwrap();
    let names: Vec<&str> = parts.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, ["vocals", "bass", "other", "drums"]);

    let sum = parts.iter().fold(mix.clone(), |rest, (_, audio)| {
        stemd_core::stems::minus(&rest, audio)
    });
    let peak = sum.data.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
    assert!(peak < 1e-5, "the parts miss the mix by {peak}");

    let missing = Output {
        parts: stemd_core::Parts::parse("drums,vocals").unwrap(),
        ..output
    };
    let (_, plain) = crate::testkit::stems(4410);
    assert!(
        to_output_rate(&mix, &plain, &missing).is_err(),
        "no drums estimate to ship"
    );
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
fn a_queued_job_says_how_many_are_ahead() {
    let next = super::queued_progress(0, 1);
    assert_eq!(next.stage, Stage::Queued);
    assert_eq!(next.detail.as_deref(), Some("next"));
    let later = super::queued_progress(2, 3);
    assert_eq!((later.completed, later.total), (2, 3));
    assert_eq!(later.detail.as_deref(), Some("2 ahead"));
}
