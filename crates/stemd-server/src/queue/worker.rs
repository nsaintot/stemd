//! The worker thread: claims work, runs the model, and publishes results.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use stemd_core::{Cancelled, Pass, Progress, Separate, Stage, Stems, whole_track};

use crate::cache::{Cache, Separation};
use crate::jobs::Job;
use crate::sessions::{Class, Track};

use super::{Claimed, Inner, Next, QueuedWork, Staged, store};

/// Take work until asked to stop.
///
/// A separator cannot end this loop: panics in both callees are caught, so a track that panics
/// fails only that track, and a build that panics leaves the loaded model in place. The separator
/// may be unsound after a panic, so the panic is logged at `error`.
pub(super) fn worker_loop(inner: &Arc<Inner>, mut separator: Box<dyn Separate>) {
    // Each open track's partial output, on the device, by track uid. Held here
    // because MLX arrays belong to the thread that made them.
    let mut passes: HashMap<u64, Pass> = HashMap::new();
    while let Some(next) = inner.next() {
        match next {
            Next::Install(staged) => {
                // Destructure the reply out of the caught closure, so a panicking build can still
                // be answered on it. Dropping the reply would tell `install` that the worker died.
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
                let cache = inner.sessions.cache();
                // The time between two calls is one job segment.
                let mut last: Option<std::time::Instant> = None;
                let separating = catch_unwind(AssertUnwindSafe(|| {
                    separate_one(separator.as_mut(), &claimed, cache, &mut |sep| {
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
                    // Mark the job failed, so a client polling it stops waiting.
                    claimed
                        .work
                        .job
                        .fail(format!("the separation panicked: {what}"));
                }
                *inner.running.lock() = None;
                trim_after_job(inner.sessions.cache());
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

/// Release MLX's allocator cache after a job when memory or disk is short, or when the cache holds
/// over half the machine's memory. Otherwise keep it, so the next track in the queue starts warm.
fn trim_after_job(cache: &Cache) {
    let held = stemd_core::memory::cached();
    let ceiling = crate::disk::physical_memory().map_or(8 << 30, |ram| ram / 2);
    let short = crate::disk::memory_pressure() || cache.disk_low();
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
        // Relaxed is enough: the flag orders no other data, and noticing it one segment late is
        // harmless.
        self.cancel.load(Ordering::Relaxed)
    }
}

fn separate_one(
    separator: &mut dyn Separate,
    claimed: &Claimed,
    cache: &Cache,
    between: &mut dyn FnMut(&mut dyn Separate) -> Result<()>,
) {
    let work = &claimed.work;
    let job = &work.job;
    let started = std::time::Instant::now();
    let sink = JobSink {
        job: Arc::clone(job),
        cancel: Arc::clone(&claimed.cancel),
    };

    // Run segment by segment where the separator can, so high-priority session work waits at most
    // one segment.
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
            let stride = segmenting
                .and_then(|s| s.grid(work.mix.frames()).ok())
                .map_or(work.mix.frames().max(1), |grid| {
                    grid.stride().unsigned_abs() as usize
                });
            publish(
                job,
                work,
                &stems,
                cache,
                stride,
                started.elapsed().as_secs_f64(),
            );
        }
        // The client cancelled the job. Logged at info with its cost, so repeated cancels show in
        // the log.
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

/// Store the finished stems in the cache and mark the job done.
pub(super) fn publish(
    job: &Job,
    work: &QueuedWork,
    stems: &Stems,
    cache: &Cache,
    stride: usize,
    elapsed: f64,
) {
    job.set_progress(Progress::new(Stage::Writing));
    // Timed separately, because `elapsed` stops before this runs and `separation_secs` reports only
    // the model's cost.
    let writing = std::time::Instant::now();
    let separation = Separation {
        model_residual_db: stems.model_residual_db,
        separation_secs: elapsed,
    };
    match store::store_whole(cache, work, stems, stride, separation) {
        Ok(entry) => {
            tracing::info!(
                job = %job.id,
                "done in {elapsed:.2}s ({:.1}x realtime, model residual {:.1} dB), \
                 {:.2}s writing {} stems",
                entry.realtime_factor(),
                stems.model_residual_db,
                writing.elapsed().as_secs_f64(),
                entry.stems.len(),
            );
            job.complete(entry, false);
        }
        Err(err) => job.fail(format!("writing stems: {err:#}")),
    }
}
