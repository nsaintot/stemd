//! The table of live jobs and the result objects it hands back.
//!
//! A job is a handle to stems that belong to [`crate::cache`], which alone manages disk space, so
//! two jobs can share one separation.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::Serialize;
use stemd_core::{Progress, Stage};

use crate::cache::Separation;
use crate::output::Output;

/// One separated stem, as the client sees it.
#[derive(Debug, Clone, Serialize)]
pub struct StemFile {
    pub name: String,
    /// The relative URL that streams the stem.
    pub url: String,
    pub bytes: u64,
    /// The scale applied before encoding, one per track; multiply by `1.0 / gain` to restore the
    /// level.
    pub gain: f32,
}

/// What a completed job hands back: the stems, and how to read them.
#[derive(Debug, Clone, Serialize)]
pub struct JobResult {
    pub output: crate::output::OutputView,
    pub channels: usize,
    /// Samples per channel, at the output rate.
    pub frames: usize,
    pub stems: Vec<StemFile>,
    /// The level of `mix - sum(model sources)`, in dB relative to the mix: a diagnostic of what the
    /// model failed to explain. `None` for stems a session stored.
    pub model_residual_db: Option<f64>,
    /// The time the separation took; for a cache hit, the time of the earlier job that made the
    /// stems. 0 for stems a session stored.
    pub separation_secs: f64,
    pub realtime_factor: f64,
    /// True when these stems were already on disk and nothing was separated.
    pub cached: bool,
}

impl JobResult {
    /// Describe a cache entry as the result of `job_id`.
    ///
    /// The URLs are built here because they differ per job: two decks sharing one entry each poll
    /// their own id.
    fn from_entry(entry: &Entry, job_id: &str, cached: bool) -> Self {
        Self {
            output: crate::output::OutputView::of(&entry.output, None),
            channels: entry.channels,
            frames: entry.frames,
            stems: entry
                .stems
                .iter()
                .map(|stem| StemFile {
                    name: stem.name.clone(),
                    url: format!("/v2/jobs/{job_id}/stems/{}", stem.name),
                    bytes: stem.bytes,
                    gain: entry.gain,
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
    /// The track and output this job produces, see [`job_key`]; a second deck joins on it.
    pub key: String,
    pub progress: Mutex<Progress>,
    pub result: Mutex<Option<JobResult>>,
    pub error: Mutex<Option<String>>,
    /// Where this job's stems live in the cache, once it has any.
    pub entry: Mutex<Option<Arc<Entry>>>,
    /// One per deck waiting on this job. See [`Job::release`].
    waiters: Mutex<usize>,
    pub created: Instant,
    /// The mix being separated, at the model's rate.
    pub mix: MixShape,
    /// Set once a protocol 1 client created or joined the job.
    protocol_1: AtomicBool,
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

    /// Record that a protocol 1 client waits on this job.
    pub fn mark_protocol_1(&self) {
        self.protocol_1.store(true, Ordering::Relaxed);
    }

    /// True once a protocol 1 client created or joined the job.
    pub fn protocol_1(&self) -> bool {
        self.protocol_1.load(Ordering::Relaxed)
    }

    /// Register another client waiting on this job.
    pub fn join(&self) {
        *self.waiters.lock() += 1;
    }

    /// Drop one waiter and return the number left.
    ///
    /// A shared job is cancelled only when its last waiter releases it. The count saturates at
    /// zero, so a repeated delete cannot bring a job back.
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
    /// This is terminal like [`Job::fail`], but carries no error. A job that outlives its removal
    /// from the table then no longer looks like it is separating, so `claim` cannot join it.
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

/// A job that a request claimed and has not yet answered or queued. Dropping it fails the job, so
/// a request that ends early does not leave the job `queued` and holding its key.
pub struct Claimed(Option<Arc<Job>>);

impl Claimed {
    pub fn new(job: Arc<Job>) -> Self {
        Self(Some(job))
    }

    /// The job, now in the caller's hands.
    pub fn disarm(mut self) -> Arc<Job> {
        self.0.take().expect("a claimed job")
    }
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if let Some(job) = self.0.take() {
            job.fail("request ended before the job was queued");
        }
    }
}

/// The length and layout of a job's mix, as a lookup reports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MixShape {
    pub frames: usize,
    pub sample_rate: u32,
    pub channels: usize,
}

impl MixShape {
    pub fn of(mix: &stemd_core::Audio) -> Self {
        Self {
            frames: mix.frames(),
            sample_rate: mix.sample_rate,
            channels: mix.channels(),
        }
    }
}

impl From<crate::cache::TrackInfo> for MixShape {
    fn from(info: crate::cache::TrackInfo) -> Self {
        Self {
            frames: info.frames,
            sample_rate: info.sample_rate,
            channels: info.channels,
        }
    }
}

/// Jobs indexed by id, and by cache key for [`Jobs::claim`].
#[derive(Default)]
struct Registry {
    by_id: HashMap<String, Arc<Job>>,
    /// Maps a cache key to the id of the job that currently represents it.
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

pub struct Jobs {
    jobs: Mutex<Registry>,
    /// How long a failed job stays readable, and a finished one after it
    /// finished or after its last stem fetch.
    retain_failed: Duration,
    /// Distinguishes attempts at the same track. See [`Jobs::claim`].
    attempts: AtomicU64,
}

impl Jobs {
    pub fn new(retain_failed: Duration) -> Self {
        Self {
            jobs: Mutex::new(Registry::default()),
            retain_failed,
            attempts: AtomicU64::new(1),
        }
    }

    #[cfg(test)]
    pub fn create(&self, key: String) -> Arc<Job> {
        self.claim(&key, MixShape::default())
            .unwrap_or_else(|existing| existing)
    }

    /// Claim `key`, or join the job that already holds it.
    ///
    /// `Ok` is a new job that the caller owns and must drive to a terminal stage; `Err` is an
    /// existing job that the caller now waits on. Registration happens under the same lock as the
    /// lookup, so concurrent submissions of one track share a single separation.
    pub fn claim(&self, key: &str, mix: MixShape) -> Result<Arc<Job>, Arc<Job>> {
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
            mix,
            protocol_1: AtomicBool::new(false),
        });
        jobs.insert(&job);
        Ok(job)
    }

    /// The job working towards `key`, unless it has ended.
    pub fn running(&self, key: &str) -> Option<Arc<Job>> {
        self.jobs.lock().in_flight(key).cloned()
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

    /// Drop finished jobs unused for `retain_failed`, or whose stems are gone,
    /// and failed ones older than `retain_failed`.
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
                    Some(entry) => {
                        !entry.present() || entry.last_used().elapsed() > self.retain_failed
                    }
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

/// What a job for track `id` at `output` is claimed under: the track id first,
/// so a job id starts with it.
pub fn job_key(id: &str, output: &Output, cache_id: &str) -> String {
    format!(
        "{id}/{}{}/{cache_id}",
        crate::cache::type_prefix(output),
        if output.ships_derived() {
            "/derived"
        } else {
            ""
        }
    )
}

/// The number of key characters shown in a log line.
const SHORT_KEY: usize = 12;

/// The leading `SHORT_KEY` characters of a key, for logging.
pub fn short(key: &str) -> &str {
    &key[..key.len().min(SHORT_KEY)]
}

/// A job id: the leading characters of the job key, then the attempt number.
///
/// It has the same `name-discriminator` shape as `model_id`, and its first part is what [`short`]
/// prints in the logs, so a log line and a URL match by eye.
///
/// The attempt number keeps ids unique: without it, two clients asking for the same track at
/// different times would share an id, and a `DELETE` from one would cancel the other's separation.
fn job_id(key: &str, attempt: u64) -> String {
    format!("{}-{attempt}", short(key))
}

/// One stem file of a job result.
#[derive(Debug, Clone)]
pub struct CachedStem {
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
}

#[derive(Debug)]
struct State {
    /// Stems no client has fetched yet.
    unfetched: HashSet<String>,
    last_used: Instant,
}

/// A job's result: whole stems in one track directory of the cache.
#[derive(Debug)]
pub struct Entry {
    pub dir: PathBuf,
    pub stems: Vec<CachedStem>,
    /// The output the stems are stored as.
    pub output: Output,
    /// The scale applied to every stem before encoding; multiply by `1.0 / gain` to restore the
    /// level.
    pub gain: f32,
    pub channels: usize,
    /// Samples per channel at the output rate.
    pub frames: usize,
    /// The residual level in dB; absent for stems a session stored.
    pub model_residual_db: Option<f64>,
    /// The time the separation took when it ran; 0 for stems a session stored.
    pub separation_secs: f64,
    state: Mutex<State>,
}

impl crate::cache::Hold for Entry {
    fn dir(&self) -> &Path {
        &self.dir
    }

    fn held(&self, unfetched_ttl: Duration) -> bool {
        Entry::held(self, unfetched_ttl)
    }
}

impl Entry {
    /// The stems `output` serves from `dir`, every one stored whole at `gain`.
    pub fn stored(
        dir: &Path,
        output: &Output,
        gain: f32,
        channels: usize,
        separation: Option<Separation>,
    ) -> Option<Self> {
        let extension = output.format.extension();
        let mut stems = Vec::new();
        let mut frames = 0;
        for name in output.served() {
            let whole = crate::whole::find(dir, name, extension)?;
            frames = whole.meta.frames;
            stems.push(CachedStem {
                name: name.to_owned(),
                bytes: std::fs::metadata(&whole.path).ok()?.len(),
                path: whole.path,
            });
        }
        Some(Self {
            dir: dir.to_owned(),
            state: Mutex::new(State {
                unfetched: stems.iter().map(|s| s.name.clone()).collect(),
                last_used: Instant::now(),
            }),
            stems,
            output: output.clone(),
            gain,
            channels,
            frames,
            model_residual_db: separation.map(|s| s.model_residual_db),
            separation_secs: separation.map_or(0.0, |s| s.separation_secs),
        })
    }

    fn last_used(&self) -> Instant {
        self.state.lock().last_used
    }

    /// True while the cache must keep the stems: some stem is unfetched and the
    /// result was made or last fetched less than `ttl` ago.
    pub fn held(&self, ttl: Duration) -> bool {
        let state = self.state.lock();
        !state.unfetched.is_empty() && state.last_used.elapsed() < ttl
    }

    /// Record that `name` was fetched, and restart the clock.
    pub fn mark_fetched(&self, name: &str) {
        let mut state = self.state.lock();
        state.unfetched.remove(name);
        state.last_used = Instant::now();
    }

    pub fn stem(&self, name: &str) -> Option<&CachedStem> {
        self.stems.iter().find(|s| s.name == name)
    }

    /// Whether every stem is still on disk.
    pub fn present(&self) -> bool {
        self.stems.iter().all(|stem| stem.path.exists())
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames as f64 / f64::from(self.output.rate.hz())
    }

    /// How much faster than realtime the separation ran; 0 for stems a session stored.
    pub fn realtime_factor(&self) -> f64 {
        if self.separation_secs > 0.0 {
            self.duration_secs() / self.separation_secs
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{cache, publish};

    #[test]
    fn a_result_names_its_output_and_no_server_path() {
        let cache = cache(1 << 30, Duration::from_secs(60));
        let entry = publish(&cache, "result-shape", 100, 1.0);
        let result = JobResult::from_entry(&entry, "abc-1", false);
        let json = serde_json::to_value(&result).expect("serialises");
        assert_eq!(
            json["output"],
            serde_json::json!({
                "format": "pcm16",
                "sample_rate": 44100,
                "parts": ["harmonics", "vocals"],
                "derived": "drums"
            })
        );
        assert_eq!(json["stems"][0]["url"], "/v2/jobs/abc-1/stems/harmonics");
        assert!(json["stems"][0].get("path").is_none());
        assert!(json.get("sample_rate").is_none());
    }

    #[test]
    fn a_finished_job_is_pruned_once_unused_or_once_its_stems_are_gone() {
        let cache = cache(u64::MAX, Duration::from_secs(3600));
        let jobs = Jobs::new(Duration::from_millis(30));
        let job = jobs.create("track".into());
        job.complete(publish(&cache, "track", 32, 2.0), false);
        assert_eq!(jobs.prune(), 0, "just finished");
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(jobs.prune(), 1, "unused for its retention");

        let jobs = Jobs::new(Duration::from_secs(3600));
        let job = jobs.create("gone".into());
        let entry = publish(&cache, "gone", 32, 2.0);
        std::fs::remove_dir_all(&entry.dir).expect("stems deleted");
        job.complete(entry, false);
        assert_eq!(jobs.prune(), 1, "the handle points at nothing now");
    }

    #[test]
    fn two_jobs_can_share_one_result_with_their_own_urls() {
        let cache = cache(u64::MAX, Duration::from_secs(3600));
        let jobs = Jobs::new(Duration::from_secs(3600));

        // A second request for a track already separated gets its own handle to the same stems. Two
        // handles coexist only once the first is terminal; while it runs, `claim` joins it instead.
        let entry = publish(&cache, "doubles", 32, 2.0);
        let dir = entry.dir.clone();
        let first = jobs.create("doubles".into());
        first.complete(Arc::clone(&entry), false);
        let second = jobs.create("doubles".into());
        second.complete(entry, true);
        assert_ne!(first.id, second.id);

        let first_url = &first.result.lock().clone().unwrap().stems[0].url;
        assert!(first_url.contains(&first.id), "urls are per job");
        assert!(second.result.lock().clone().unwrap().cached);

        std::fs::remove_dir_all(&dir).expect("stems deleted");
        assert_eq!(jobs.prune(), 2, "both handles go with the stems");
    }

    #[test]
    fn a_running_job_is_never_pruned() {
        let jobs = Jobs::new(Duration::from_millis(1));
        let job = jobs.create("key".into());
        job.set_progress(Progress::new(Stage::Separating));

        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(jobs.prune(), 0);
        assert_eq!(jobs.len(), 1);
    }

    #[test]
    fn a_failure_is_kept_readable_then_dropped() {
        let jobs = Jobs::new(Duration::from_millis(30));
        let job = jobs.create("key".into());
        job.fail("model exploded");

        assert_eq!(jobs.prune(), 0, "the client has not read the error yet");
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(jobs.prune(), 1);
    }

    #[test]
    fn a_cache_hit_reports_the_original_separation_cost() {
        let cache = cache(u64::MAX, Duration::from_secs(3600));
        let jobs = Jobs::new(Duration::from_secs(3600));
        let entry = publish(&cache, "timing", 32, 2.0);
        let job = jobs.create("key".into());
        job.complete(entry, true);

        let result = job.result.lock().clone().unwrap();
        assert_eq!(result.separation_secs, 2.0);
        assert!(result.cached);
        // 32 frames at 44100 Hz separated in 2 s is slower than realtime: the ratio describes the
        // original separation, which took 2 s.
        assert!(result.realtime_factor < 1.0);
    }

    #[test]
    fn a_second_deck_joins_a_separation_already_running() {
        let jobs = Jobs::new(Duration::from_secs(3600));
        let first = jobs.create("same-track".into());
        first.set_progress(Progress::new(Stage::Separating));

        let Err(joined) = jobs.claim("same-track", MixShape::default()) else {
            panic!("second claim should join, not create");
        };
        assert_eq!(joined.id, first.id, "one separation, not two");
        assert_eq!(jobs.len(), 1, "joining must not create a second job");
    }

    #[test]
    fn concurrent_claims_on_one_key_collapse_to_a_single_job() {
        // `claim` registers under the lookup's lock; with the lookup and the insert apart, every
        // thread here would find nothing in flight and start its own separation.
        let jobs = Arc::new(Jobs::new(Duration::from_secs(3600)));
        let start = Arc::new(std::sync::Barrier::new(8));
        let created = Arc::new(Mutex::new(Vec::new()));

        let threads: Vec<_> = (0..8)
            .map(|_| {
                let jobs = Arc::clone(&jobs);
                let start = Arc::clone(&start);
                let created = Arc::clone(&created);
                std::thread::spawn(move || {
                    start.wait();
                    if let Ok(job) = jobs.claim("one-track", MixShape::default()) {
                        created.lock().push(job.id.clone());
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("claim thread");
        }

        assert_eq!(created.lock().len(), 1, "exactly one claim may create");
        assert_eq!(jobs.len(), 1);
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
            short(key),
            "a log line and a URL must be matchable by eye"
        );
    }

    #[test]
    fn a_second_attempt_at_one_track_gets_its_own_id() {
        // Two ids for one key keep the DELETE below from hitting the wrong job.
        let jobs = Jobs::new(Duration::from_secs(3600));
        let first = jobs.create("same".into());
        first.fail("model exploded");
        let second = jobs.create("same".into());

        assert_ne!(first.id, second.id);
        assert!(first.id.starts_with("same-") && second.id.starts_with("same-"));
    }

    #[test]
    fn a_finished_client_cannot_cancel_a_later_attempt_at_the_same_track() {
        // Ids carry an attempt number so that a client that finished and then sends DELETE does not
        // release, and so cancel, a new separation that another client started for the same track.
        let cache = cache(u64::MAX, Duration::from_secs(3600));
        let jobs = Jobs::new(Duration::from_secs(3600));

        let finished = jobs.create("shared-track".into());
        finished.complete(publish(&cache, "shared-track", 32, 2.0), false);

        let retry = jobs.create("shared-track".into());
        retry.set_progress(Progress::new(Stage::Separating));

        // The late DELETE from the first client, by the id it was given.
        let targeted = jobs.get(&finished.id).expect("its own handle");
        assert_eq!(targeted.release(), 0);
        jobs.remove(&finished.id);

        assert_eq!(
            *retry.waiters.lock(),
            1,
            "the running attempt lost a waiter it never had"
        );
        assert!(
            jobs.get(&retry.id).is_some(),
            "the running attempt was removed by someone else's delete"
        );
    }

    #[test]
    fn claiming_finds_the_live_attempt_after_earlier_ones_are_removed() {
        // Guards the by_key index: a removed job must not leave a pointer that
        // hides the attempt actually in flight.
        let jobs = Jobs::new(Duration::from_secs(3600));
        let first = jobs.create("track".into());
        first.fail("boom");
        let second = jobs.create("track".into());
        second.set_progress(Progress::new(Stage::Separating));

        jobs.remove(&first.id);
        let Err(joined) = jobs.claim("track", MixShape::default()) else {
            panic!("the running attempt should be joined, not replaced");
        };
        assert_eq!(joined.id, second.id);
    }

    #[test]
    fn a_different_track_never_joins() {
        let jobs = Jobs::new(Duration::from_secs(3600));
        let job = jobs.create("track-a".into());
        job.set_progress(Progress::new(Stage::Separating));
        assert!(
            jobs.claim("track-b", MixShape::default()).is_ok(),
            "a different key gets its own job"
        );
    }

    #[test]
    fn a_finished_job_is_left_to_the_cache() {
        // Once a job is terminal its stems are published, so joining it would
        // hand back a handle when the cache can answer outright.
        let cache = cache(u64::MAX, Duration::from_secs(3600));
        let jobs = Jobs::new(Duration::from_secs(3600));
        let job = jobs.create("done-track".into());
        job.complete(publish(&cache, "done-track", 32, 2.0), false);

        assert!(
            jobs.claim("done-track", MixShape::default()).is_ok(),
            "a finished job is left to the cache"
        );
    }

    #[test]
    fn a_failed_job_is_not_joined_either() {
        let jobs = Jobs::new(Duration::from_secs(3600));
        let job = jobs.create("doomed".into());
        job.fail("model exploded");
        assert!(
            jobs.claim("doomed", MixShape::default()).is_ok(),
            "a second client should get a fresh attempt, not someone else's failure"
        );
    }

    #[test]
    fn a_claim_dropped_before_it_is_queued_fails_and_frees_its_key() {
        let jobs = Jobs::new(Duration::from_secs(3600));
        let Ok(kept) = jobs.claim("kept", MixShape::default()) else {
            panic!("a fresh key is claimed");
        };
        let kept = Claimed::new(kept).disarm();
        assert_eq!(kept.view().progress.stage, Stage::Queued);

        let Ok(dropped) = jobs.claim("dropped", MixShape::default()) else {
            panic!("a fresh key is claimed");
        };
        drop(Claimed::new(Arc::clone(&dropped)));
        let view = dropped.view();
        assert_eq!(view.progress.stage, Stage::Failed);
        assert!(view.error.is_some());
        assert!(jobs.claim("dropped", MixShape::default()).is_ok());
    }

    #[test]
    fn a_cancelled_job_is_not_joined_either() {
        let jobs = Jobs::new(Duration::from_secs(3600));
        let job = jobs.create("skipped".into());
        job.cancelled();
        assert!(
            jobs.claim("skipped", MixShape::default()).is_ok(),
            "cueing the track again must start a fresh separation, not join the abandoned one"
        );
    }

    #[test]
    fn one_deck_giving_up_does_not_cancel_the_other() {
        let jobs = Jobs::new(Duration::from_secs(3600));
        let job = jobs.create("shared".into());
        job.set_progress(Progress::new(Stage::Separating));
        assert!(
            jobs.claim("shared", MixShape::default()).is_err(),
            "second client joins"
        );

        assert_eq!(job.release(), 1, "the other deck is still waiting");
        assert_eq!(job.release(), 0, "now nobody is");
    }

    #[test]
    fn releasing_more_than_once_cannot_go_negative() {
        let jobs = Jobs::new(Duration::from_secs(3600));
        let job = jobs.create("twice".into());
        assert_eq!(job.release(), 0);
        assert_eq!(job.release(), 0, "a repeated delete must not underflow");
    }

    #[test]
    fn a_job_key_starts_with_the_track_and_tells_outputs_apart() {
        let output = crate::testkit::output();
        let key = job_key("a8e29ff15c16d4f2e0b1aaaa", &output, "model-1");
        assert!(key.starts_with("a8e29ff15c16"), "{key}");
        let derived = Output {
            derived: true,
            ..output.clone()
        };
        assert_ne!(
            key,
            job_key("a8e29ff15c16d4f2e0b1aaaa", &derived, "model-1")
        );
        let parts = Output {
            parts: stemd_core::Parts::parse("vocals,bass,other").unwrap(),
            ..output.clone()
        };
        assert_ne!(key, job_key("a8e29ff15c16d4f2e0b1aaaa", &parts, "model-1"));
        assert_ne!(key, job_key("a8e29ff15c16d4f2e0b1aaaa", &output, "model-2"));
    }

    #[test]
    fn a_partial_fetch_restarts_the_hold() {
        let ttl = std::time::Duration::from_millis(200);
        let cache = crate::testkit::cache(u64::MAX, ttl);
        let entry = crate::testkit::publish(&cache, "half", 64, 1.0);
        std::thread::sleep(std::time::Duration::from_millis(150));
        entry.mark_fetched(&entry.stems[0].name);
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(entry.held(ttl), "the fetch restarted the clock");
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!entry.held(ttl), "past the ttl");
    }
}
