//! Job store and the result objects it hands back.
//!
//! A job is a handle, not storage: the stems it points at belong to
//! [`crate::cache`], which is the only thing that governs disk. Two jobs can
//! therefore share one separation, and a job nobody collects costs a hash-map
//! entry rather than a gigabyte.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::Serialize;
use stemd_core::{Progress, Stage, StemFormat};

use crate::cache::Entry;

/// One separated stem, as the client sees it.
#[derive(Debug, Clone, Serialize)]
pub struct StemFile {
    pub name: String,
    /// Server-local path. Useful when the client shares the filesystem; remote
    /// clients should fetch `url` instead.
    pub path: std::path::PathBuf,
    /// Relative endpoint that streams the same bytes.
    pub url: String,
    pub bytes: u64,
    /// Scale already applied to this stem so it fits the output format.
    /// Multiply by `1.0 / gain` to restore the original level.
    ///
    /// Per stem rather than shared: a quiet stem quantised against the loudest
    /// stem's peak throws away bits for nothing.
    pub gain: f32,
}

/// What a completed job hands back: the stems, and how to read them.
#[derive(Debug, Clone, Serialize)]
pub struct JobResult {
    pub sample_rate: u32,
    pub channels: usize,
    pub frames: usize,
    pub format: StemFormat,
    pub stems: Vec<StemFile>,
    /// Level of `mix - sum(model sources)`, in dB relative to the mix.
    /// Diagnostic: what the model failed to explain about the track.
    pub model_residual_db: f64,
    /// What the separation cost when it ran, which for a cache hit was some
    /// earlier job.
    pub separation_secs: f64,
    pub realtime_factor: f64,
    /// True when these stems were already on disk and nothing was separated.
    pub cached: bool,
}

impl JobResult {
    /// Describe a cache entry as the result of `job_id`.
    ///
    /// The URLs are per job, so they are built here rather than stored: two
    /// decks sharing one entry each poll their own id.
    pub fn from_entry(entry: &Entry, job_id: &str, cached: bool) -> Self {
        Self {
            sample_rate: entry.sample_rate,
            channels: entry.channels,
            frames: entry.frames,
            format: entry.format,
            stems: entry
                .stems
                .iter()
                .map(|stem| StemFile {
                    name: stem.name.clone(),
                    path: stem.path.clone(),
                    url: format!("/v1/jobs/{job_id}/stems/{}", stem.name),
                    bytes: stem.bytes,
                    gain: stem.gain,
                })
                .collect(),
            model_residual_db: entry.model_residual_db,
            separation_secs: entry.separation_secs,
            realtime_factor: entry.realtime_factor(),
            cached,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct JobView {
    pub id: String,
    pub progress: Progress,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<JobResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub struct Job {
    pub id: String,
    /// Cache entry this job will produce. Also what a second deck matches on to
    /// join a separation already under way instead of starting its own.
    pub key: String,
    pub progress: Mutex<Progress>,
    pub result: Mutex<Option<JobResult>>,
    pub error: Mutex<Option<String>>,
    /// Where this job's stems live, once it has any. Shared with the cache and
    /// with any other job that asked for the same track.
    pub entry: Mutex<Option<Arc<Entry>>>,
    /// One per deck waiting on this job. See [`Job::release`].
    waiters: Mutex<usize>,
    pub created: Instant,
}

impl Job {
    pub fn view(&self) -> JobView {
        JobView {
            id: self.id.clone(),
            progress: self.progress.lock().clone(),
            result: self.result.lock().clone(),
            error: self.error.lock().clone(),
        }
    }

    /// Register another client waiting on this job.
    pub fn join(&self) {
        *self.waiters.lock() += 1;
    }

    /// Drop one waiter, returning how many are left.
    ///
    /// `DELETE` from one client means "I stopped caring", not "cancel this for
    /// everyone": a shared job is only cancelled once the last waiter lets go.
    /// Saturates, so a repeated delete cannot resurrect a job.
    pub fn release(&self) -> usize {
        let mut waiters = self.waiters.lock();
        *waiters = waiters.saturating_sub(1);
        *waiters
    }

    pub fn set_progress(&self, progress: Progress) {
        *self.progress.lock() = progress;
    }

    /// Attach a finished entry and mark the job done.
    ///
    /// The result is published before the stage flips, so a client that sees
    /// `Done` and immediately asks for a stem never races an empty handle.
    pub fn complete(&self, entry: Arc<Entry>, cached: bool) {
        let result = JobResult::from_entry(&entry, &self.id, cached);
        *self.entry.lock() = Some(entry);
        *self.result.lock() = Some(result);
        self.set_progress(Progress::new(Stage::Done));
    }

    /// Mark the job stopped at the client's request.
    ///
    /// Terminal like [`Job::fail`], but with no error attached: nothing went
    /// wrong, so there is nothing for a client to read back. It keeps a job that
    /// outlives its removal from the store from looking like it is still
    /// separating, which would let `claim` join it forever.
    pub fn cancelled(&self) {
        self.set_progress(Progress::new(Stage::Cancelled));
    }

    pub fn fail(&self, message: impl Into<String>) {
        let message = message.into();
        tracing::error!(job = %self.id, "job failed: {message}");
        *self.error.lock() = Some(message);
        self.set_progress(Progress::new(Stage::Failed));
    }
}

/// Jobs indexed both ways: by id for lookups, and by cache key so
/// [`JobStore::claim`] is a lookup rather than a scan over every live job.
#[derive(Default)]
struct Registry {
    by_id: HashMap<String, Arc<Job>>,
    /// Cache key to the id of the job currently representing it.
    by_key: HashMap<String, String>,
}

impl Registry {
    fn insert(&mut self, job: &Arc<Job>) {
        self.by_id.insert(job.id.clone(), Arc::clone(job));
        self.by_key.insert(job.key.clone(), job.id.clone());
    }

    fn remove(&mut self, id: &str) -> bool {
        let Some(job) = self.by_id.remove(id) else {
            return false;
        };
        // Only if this job is still the one representing the key: a later
        // attempt may have taken it over.
        if self.by_key.get(&job.key).is_some_and(|held| held == id) {
            self.by_key.remove(&job.key);
        }
        true
    }

    /// The live job for `key`, if one is still working towards it.
    fn in_flight(&self, key: &str) -> Option<&Arc<Job>> {
        let job = self.by_id.get(self.by_key.get(key)?)?;
        (!job.progress.lock().stage.is_terminal()).then_some(job)
    }
}

pub struct JobStore {
    jobs: Mutex<Registry>,
    /// How long a job that produced no entry, one that failed, stays
    /// readable. Successful jobs live as long as their stems instead.
    retain_failed: Duration,
    /// Distinguishes attempts at the same track. See [`JobStore::claim`].
    attempts: AtomicU64,
}

impl JobStore {
    pub fn new(retain_failed: Duration) -> Self {
        Self {
            jobs: Mutex::new(Registry::default()),
            retain_failed,
            attempts: AtomicU64::new(1),
        }
    }

    #[cfg(test)]
    pub fn create(&self, key: String) -> Arc<Job> {
        self.claim(&key).unwrap_or_else(|existing| existing)
    }

    /// Claim `key`, or join whoever already holds it.
    ///
    /// `Ok` is a fresh job the caller now owns and must drive to a terminal stage;
    /// `Err` is an existing one the caller has been added to as a waiter. Registering
    /// happens under the same lock as the lookup, which makes concurrent submissions
    /// of one track collapse to a single separation.
    pub fn claim(&self, key: &str) -> Result<Arc<Job>, Arc<Job>> {
        let mut jobs = self.jobs.lock();
        if let Some(job) = jobs.in_flight(key) {
            job.join();
            return Err(Arc::clone(job));
        }
        let job = Arc::new(Job {
            id: job_id(key, self.attempts.fetch_add(1, Ordering::Relaxed)),
            key: key.to_owned(),
            progress: Mutex::new(Progress::new(Stage::Queued)),
            result: Mutex::new(None),
            error: Mutex::new(None),
            entry: Mutex::new(None),
            waiters: Mutex::new(1),
            created: Instant::now(),
        });
        jobs.insert(&job);
        Ok(job)
    }

    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.jobs.lock().by_id.get(id).cloned()
    }

    pub fn remove(&self, id: &str) -> bool {
        self.jobs.lock().remove(id)
    }

    pub fn len(&self) -> usize {
        self.jobs.lock().by_id.len()
    }

    /// Drop handles that can no longer serve anyone.
    ///
    /// A job outlives its usefulness the moment the cache reaps its stems, so
    /// the entry's own tombstone decides rather than a second age limit: one
    /// lifecycle, not two that can disagree. Jobs still queued or separating are
    /// never touched, and neither are failures until their message goes stale.
    pub fn prune(&self) -> usize {
        let mut jobs = self.jobs.lock();
        let now = Instant::now();
        let stale: Vec<String> = jobs
            .by_id
            .values()
            .filter(|job| {
                if !job.progress.lock().stage.is_terminal() {
                    return false;
                }
                match job.entry.lock().as_ref() {
                    Some(entry) => entry.evicted(),
                    None => now.duration_since(job.created) > self.retain_failed,
                }
            })
            .map(|job| job.id.clone())
            .collect();

        for id in &stale {
            jobs.remove(id);
        }
        stale.len()
    }
}

/// A job id: the leading characters of the cache key, then the attempt number.
///
/// The same `name-discriminator` shape as `model_id`, and the leading half is what
/// [`crate::cache::short`] prints in the logs, so a log line and a URL match by
/// eye.
///
/// The attempt suffix keeps ids unique. Without it two clients asking for the same
/// track at different times share an id, and a `DELETE` from one would cancel the
/// other's separation.
fn job_id(key: &str, attempt: u64) -> String {
    format!("{}-{attempt}", crate::cache::short(key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::Ingredients;
    use crate::testkit::{cache, publish};
    use stemd_core::PcmFormat;

    /// The budget is one byte so the store counts as full and its uncollected
    /// entry is collectable. What is under test is the handle following the
    /// stems out; the reaping is only how they leave.
    #[test]
    fn a_job_is_pruned_once_its_stems_are_reaped() {
        let cache = cache(1, Duration::from_millis(20));
        let store = JobStore::new(Duration::from_secs(3600));

        let job = store.create("track".into());
        job.complete(publish(&cache, "track", 32, 2.0), false);
        assert_eq!(store.prune(), 0, "the entry is still there");
        assert_eq!(store.len(), 1);

        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(cache.reap().unfetched, 1);
        assert_eq!(store.prune(), 1, "the handle points at nothing now");
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn two_jobs_can_share_one_entry_and_are_pruned_together() {
        // Full, for the reason the test above gives.
        let cache = cache(1, Duration::from_millis(20));
        let store = JobStore::new(Duration::from_secs(3600));

        // A second request for a track already separated gets its own handle
        // onto the same stems. Two handles only ever coexist once the first is
        // terminal, while it is running, `claim` joins it instead.
        let entry = publish(&cache, "doubles", 32, 2.0);
        let first = store.create("doubles".into());
        first.complete(Arc::clone(&entry), false);
        let second = store.create("doubles".into());
        second.complete(entry, true);
        assert_ne!(first.id, second.id);

        let first_url = &first.result.lock().clone().unwrap().stems[0].url;
        assert!(
            first_url.contains(&first.id),
            "urls are per job, not per entry"
        );
        assert!(second.result.lock().clone().unwrap().cached);

        std::thread::sleep(Duration::from_millis(40));
        cache.reap();
        assert_eq!(store.prune(), 2);
    }

    #[test]
    fn a_running_job_is_never_pruned() {
        let store = JobStore::new(Duration::from_millis(1));
        let job = store.create("key".into());
        job.set_progress(Progress::new(Stage::Separating));

        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(store.prune(), 0);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn a_failure_is_kept_readable_then_dropped() {
        let store = JobStore::new(Duration::from_millis(30));
        let job = store.create("key".into());
        job.fail("model exploded");

        assert_eq!(store.prune(), 0, "the client has not read the error yet");
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(store.prune(), 1);
    }

    #[test]
    fn a_cache_hit_reports_the_original_separation_cost() {
        let cache = cache(u64::MAX, Duration::from_secs(3600));
        let store = JobStore::new(Duration::from_secs(3600));
        let entry = publish(&cache, "timing", 32, 2.0);
        let job = store.create("key".into());
        job.complete(entry, true);

        let result = job.result.lock().clone().unwrap();
        assert_eq!(result.separation_secs, 2.0);
        assert!(result.cached);
        // 32 frames at 44100 Hz separated in 2 s: slower than realtime, and the
        // ratio must describe the run that happened, not this instant answer.
        assert!(result.realtime_factor < 1.0);
    }

    #[test]
    fn a_second_deck_joins_a_separation_already_running() {
        let store = JobStore::new(Duration::from_secs(3600));
        let first = store.create("same-track".into());
        first.set_progress(Progress::new(Stage::Separating));

        let Err(joined) = store.claim("same-track") else {
            panic!("second claim should join, not create");
        };
        assert_eq!(joined.id, first.id, "one separation, not two");
        assert_eq!(store.len(), 1, "joining must not create a second job");
    }

    #[test]
    fn concurrent_claims_on_one_key_collapse_to_a_single_job() {
        // The reason `claim` registers under the lock that looks up. With the
        // lookup and the insert apart, every thread here finds nothing in
        // flight and starts its own separation.
        let store = Arc::new(JobStore::new(Duration::from_secs(3600)));
        let start = Arc::new(std::sync::Barrier::new(8));
        let created = Arc::new(Mutex::new(Vec::new()));

        let threads: Vec<_> = (0..8)
            .map(|_| {
                let store = Arc::clone(&store);
                let start = Arc::clone(&start);
                let created = Arc::clone(&created);
                std::thread::spawn(move || {
                    start.wait();
                    if let Ok(job) = store.claim("one-track") {
                        created.lock().push(job.id.clone());
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("claim thread");
        }

        assert_eq!(created.lock().len(), 1, "exactly one claim may create");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn an_id_survives_a_clients_sanitiser() {
        // Job ids reach a client too, in the stem URLs it is handed.
        let id = job_id("a8e29ff15c16d4f2e0b1", 12);
        assert!(crate::ident::is_portable_id(&id), "{id}");
        assert_eq!(
            crate::ident::portable(&id),
            id,
            "a client would rewrite {id}"
        );
    }

    #[test]
    fn an_id_is_the_key_prefix_and_the_attempt() {
        // The same `name-discriminator` shape as the model_id a client stores,
        // and the leading half is what the logs print for this entry.
        let key = "a8e29ff15c16d4f2e0b1"; // longer than SHORT_KEY, as a real one is
        assert_eq!(job_id(key, 3), "a8e29ff15c16-3");
        assert_eq!(
            job_id(key, 3).split('-').next().unwrap(),
            crate::cache::short(key),
            "a log line and a URL must be matchable by eye"
        );
    }

    #[test]
    fn a_second_attempt_at_one_track_gets_its_own_id() {
        // Two ids for one key is what stops the DELETE below from aliasing.
        let store = JobStore::new(Duration::from_secs(3600));
        let first = store.create("same".into());
        first.fail("model exploded");
        let second = store.create("same".into());

        assert_ne!(first.id, second.id);
        assert!(first.id.starts_with("same-") && second.id.starts_with("same-"));
    }

    #[test]
    fn a_finished_client_cannot_cancel_a_later_attempt_at_the_same_track() {
        // The reason ids carry an attempt number. A client that finished and
        // then DELETEs must not release the waiter of, and so cancel, the
        // fresh separation someone else just started for the same track.
        let cache = cache(u64::MAX, Duration::from_secs(3600));
        let store = JobStore::new(Duration::from_secs(3600));

        let finished = store.create("shared-track".into());
        finished.complete(publish(&cache, "shared-track", 32, 2.0), false);

        let retry = store.create("shared-track".into());
        retry.set_progress(Progress::new(Stage::Separating));

        // The late DELETE from the first client, by the id it was given.
        let targeted = store.get(&finished.id).expect("its own handle");
        assert_eq!(targeted.release(), 0);
        store.remove(&finished.id);

        assert_eq!(
            *retry.waiters.lock(),
            1,
            "the running attempt lost a waiter it never had"
        );
        assert!(
            store.get(&retry.id).is_some(),
            "the running attempt was removed by someone else's delete"
        );
    }

    #[test]
    fn claiming_finds_the_live_attempt_after_earlier_ones_are_removed() {
        // Guards the by_key index: a removed job must not leave a pointer that
        // hides the attempt actually in flight.
        let store = JobStore::new(Duration::from_secs(3600));
        let first = store.create("track".into());
        first.fail("boom");
        let second = store.create("track".into());
        second.set_progress(Progress::new(Stage::Separating));

        store.remove(&first.id);
        let Err(joined) = store.claim("track") else {
            panic!("the running attempt should be joined, not replaced");
        };
        assert_eq!(joined.id, second.id);
    }

    #[test]
    fn a_different_track_never_joins() {
        let store = JobStore::new(Duration::from_secs(3600));
        let job = store.create("track-a".into());
        job.set_progress(Progress::new(Stage::Separating));
        assert!(
            store.claim("track-b").is_ok(),
            "a different key gets its own job"
        );
    }

    #[test]
    fn a_finished_job_is_left_to_the_cache() {
        // Once a job is terminal its stems are published, so joining it would
        // hand back a handle when the cache can answer outright.
        let cache = cache(u64::MAX, Duration::from_secs(3600));
        let store = JobStore::new(Duration::from_secs(3600));
        let job = store.create("done-track".into());
        job.complete(publish(&cache, "done-track", 32, 2.0), false);

        assert!(
            store.claim("done-track").is_ok(),
            "a finished job is left to the cache"
        );
    }

    #[test]
    fn a_failed_job_is_not_joined_either() {
        let store = JobStore::new(Duration::from_secs(3600));
        let job = store.create("doomed".into());
        job.fail("model exploded");
        assert!(
            store.claim("doomed").is_ok(),
            "a second client should get a fresh attempt, not someone else's failure"
        );
    }

    #[test]
    fn a_cancelled_job_is_not_joined_either() {
        let store = JobStore::new(Duration::from_secs(3600));
        let job = store.create("skipped".into());
        job.cancelled();
        assert!(
            store.claim("skipped").is_ok(),
            "cueing the track again must start a fresh separation, not join the abandoned one"
        );
    }

    #[test]
    fn one_deck_giving_up_does_not_cancel_the_other() {
        let store = JobStore::new(Duration::from_secs(3600));
        let job = store.create("shared".into());
        job.set_progress(Progress::new(Stage::Separating));
        assert!(store.claim("shared").is_err(), "second client joins");

        assert_eq!(job.release(), 1, "the other deck is still waiting");
        assert_eq!(job.release(), 0, "now nobody is");
    }

    #[test]
    fn releasing_more_than_once_cannot_go_negative() {
        let store = JobStore::new(Duration::from_secs(3600));
        let job = store.create("twice".into());
        assert_eq!(job.release(), 0);
        assert_eq!(job.release(), 0, "a repeated delete must not underflow");
    }

    #[test]
    fn the_ingredients_are_what_the_key_covers() {
        // Guards the field list: adding one to Ingredients without adding it to
        // `key` would leave two different separations sharing an entry.
        let base = Ingredients {
            pcm: b"samples",
            sample_rate: 44100,
            channels: 2,
            in_format: PcmFormat::S16le,
            out_format: StemFormat::Pcm32,
            out_rate: stemd_core::OutputRate::default(),
            include_derived: false,
            dsp: stemd_core::DspMode::default(),
            model: "digest",
        };
        assert_eq!(crate::cache::key(&base).len(), 64);
    }
}
