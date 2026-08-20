//! Content-addressed store for separated stems.
//!
//! An entry is keyed by the uploaded samples and by everything else that changes
//! what comes back, so a hit is the bytes a miss would have produced.
//!
//! `max_bytes` bounds the disk. Two rules apply, and neither runs while the store
//! has room:
//!
//! 1. Past half full, an entry nobody pulled and has not touched for
//!    `unfetched_ttl` goes.
//! 2. Over the cap, least-recently-used, until it fits.
//!
//! The entry directory holds the stems; a job only borrows it.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use stemd_core::{Audio, DERIVED, DspMode, OutputRate, PcmFormat, StemFormat, Stems, resample};

/// Invalidates every stored entry when bumped by hand. Covers a change to which
/// stems ship, or to how one is encoded or scaled on the way out. The model is in
/// the key by digest, so re-tracing one invalidates its own entries without
/// touching this.
///
/// 2: FLAC streams declare a fixed block size; earlier entries decode short.
const EPOCH: u32 = 2;

/// What a job asks for on the way out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Output {
    pub format: StemFormat,
    pub rate: OutputRate,
    /// Ship the derived part as well, instead of leaving the client to rebuild
    /// it from the mix. See [`to_output_rate`].
    pub derived: bool,
    /// Filter the conversion to `rate` runs through.
    pub dsp: DspMode,
}

impl Output {
    /// How many parts this asks for, for a log line.
    pub const fn parts(self) -> usize {
        stemd_core::SHIPPED.len() + if self.derived { 1 } else { 0 }
    }
}

/// Everything that decides what a separation returns.
///
/// A struct rather than positional arguments: leaving one out is not a compile
/// error, it is a cache that serves the wrong audio.
#[derive(Clone, Copy)]
pub struct Ingredients<'a> {
    /// The uploaded bytes, hashed as received, which keeps the key independent
    /// of how samples are represented in memory.
    pub pcm: &'a [u8],
    pub sample_rate: u32,
    pub channels: usize,
    pub in_format: PcmFormat,
    pub out_format: StemFormat,
    /// Rate the stems are converted to on the way out. In the key because a hit
    /// must never hand back audio at a rate the caller did not ask for.
    pub out_rate: OutputRate,
    /// Whether the derived part ships. In the key for the same reason: it
    /// changes which files an entry holds.
    pub include_derived: bool,
    /// Filter the conversion to `out_rate` ran through. In the key because two
    /// modes at one rate are two different files.
    pub dsp: DspMode,
    /// Digest of the model artefact where one is pinned, else its name.
    pub model: &'a str,
}

/// Hash the ingredients into an entry name.
///
/// About 0.3 GB/s, so a five-minute s16le upload takes roughly 160 ms: kept off
/// the async runtime. Fields are length-prefixed so different splits cannot
/// collide.
pub fn key(ingredients: &Ingredients<'_>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(ingredients.pcm);
    for field in [
        EPOCH.to_string(),
        ingredients.sample_rate.to_string(),
        ingredients.channels.to_string(),
        ingredients.in_format.to_string(),
        ingredients.out_format.to_string(),
        ingredients.out_rate.to_string(),
        ingredients.include_derived.to_string(),
        ingredients.dsp.to_string(),
        ingredients.model.to_owned(),
    ] {
        hasher.update(u32::try_from(field.len()).unwrap_or(u32::MAX).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Sweeps per TTL. More than one so an abandoned separation cannot sit for
/// twice as long as configured; the bounds keep a very short or very long TTL
/// from turning that into a busy loop or an hour of silence.
const SWEEPS_PER_TTL: u32 = 5;
const MIN_SWEEP: Duration = Duration::from_secs(5);
const MAX_SWEEP: Duration = Duration::from_secs(60);

/// Characters of a key worth putting in a log line. Enough to tell entries
/// apart by eye without wrapping the message.
const SHORT_KEY: usize = 12;

/// The leading `SHORT_KEY` characters of a key, for logging.
pub fn short(key: &str) -> &str {
    &key[..key.len().min(SHORT_KEY)]
}

/// One stem file inside an entry.
#[derive(Debug, Clone)]
pub struct CachedStem {
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
    /// Scale already applied so the samples fit the output format. The client
    /// multiplies by `1.0 / gain` to restore the original level.
    pub gain: f32,
}

#[derive(Debug)]
struct State {
    /// Stems no client has pulled yet. Empty means the entry did its job.
    unfetched: HashSet<String>,
    last_used: Instant,
}

/// One separation's output, owned by the cache and borrowed by jobs.
#[derive(Debug)]
pub struct Entry {
    pub key: String,
    pub dir: PathBuf,
    pub stems: Vec<CachedStem>,
    pub sample_rate: u32,
    pub channels: usize,
    pub frames: usize,
    pub format: StemFormat,
    pub model_residual_db: f64,
    /// What the separation cost when it actually ran. Preserved across hits, so
    /// a client can tell a cheap answer from an expensive one.
    pub separation_secs: f64,
    pub bytes: u64,
    /// Set when the cache deletes this entry's directory, so a job holding the
    /// last reference can tell that its stems are gone.
    evicted: AtomicBool,
    state: Mutex<State>,
}

impl Entry {
    /// True once every shipped stem has been pulled at least once.
    ///
    /// All of them, not any: a client that fetched one stem and died did not get
    /// a usable result, and rule 1 should still collect it.
    pub fn consumed(&self) -> bool {
        self.state.lock().unfetched.is_empty()
    }

    pub fn last_used(&self) -> Instant {
        self.state.lock().last_used
    }

    pub fn touch(&self) {
        self.state.lock().last_used = Instant::now();
    }

    pub fn evicted(&self) -> bool {
        self.evicted.load(Ordering::SeqCst)
    }

    /// Record that `name` was pulled, and restart the idle clock.
    ///
    /// Touching here is what stops rule 1 from collecting an entry between a
    /// client's first and second stem fetch.
    pub fn mark_fetched(&self, name: &str) {
        let mut state = self.state.lock();
        state.unfetched.remove(name);
        state.last_used = Instant::now();
    }

    pub fn stem(&self, name: &str) -> Option<&CachedStem> {
        self.stems.iter().find(|s| s.name == name)
    }

    /// Whether every stem this entry promises is still readable.
    fn present(&self) -> bool {
        self.stems.iter().all(|stem| stem.path.exists())
    }

    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.frames as f64 / f64::from(self.sample_rate)
    }

    /// How much faster than realtime the separation ran. The single
    /// definition; the worker's log line and the API response both read it.
    pub fn realtime_factor(&self) -> f64 {
        if self.separation_secs > 0.0 {
            self.duration_secs() / self.separation_secs
        } else {
            f64::INFINITY
        }
    }
}

/// Bytes held by every entry in the store.
fn total_bytes(entries: &HashMap<String, Arc<Entry>>) -> u64 {
    entries.values().map(|e| e.bytes).sum()
}

/// What one sweep collected.
#[derive(Debug, Default, Clone, Copy)]
pub struct Reaped {
    /// Entries deleted by rule 1: over budget, separated, never pulled.
    pub unfetched: usize,
    /// Entries deleted by rule 2: the store was over budget.
    pub rotated: usize,
    pub bytes: u64,
}

impl Reaped {
    pub fn any(self) -> bool {
        self.unfetched > 0 || self.rotated > 0
    }
}

pub struct Cache {
    root: PathBuf,
    entries: Mutex<HashMap<String, Arc<Entry>>>,
    max_bytes: u64,
    unfetched_ttl: Duration,
    /// Makes staging directory names unique. The worker is serialised today, so
    /// two publishes cannot overlap, this keeps that from being load-bearing.
    seq: AtomicU64,
}

/// Where entries live when no directory is configured.
///
/// ```text
/// macOS    ~/Library/Caches/stemd
/// Linux    ~/.cache/stemd              ($XDG_CACHE_HOME)
/// Windows  %LOCALAPPDATA%\stemd\cache
/// ```
///
/// The cache root, not the data root: every byte here is derived. Model artefacts
/// live under [`crate::models::support_dir`], outside the startup clear below.
///
/// Windows takes an extra component: `FOLDERID_LocalAppData` is what both
/// `dirs::cache_dir` and `dirs::data_local_dir` return there.
pub fn default_dir() -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .context("no cache directory for this platform")?
        .join("stemd");
    Ok(if cfg!(windows) {
        dir.join("cache")
    } else {
        dir
    })
}

impl Cache {
    /// Prepare `root`, discarding whatever a previous run left behind.
    ///
    /// Whether an entry was ever pulled is not recorded on disk and rule 1 turns on
    /// it, so start empty. Also disposes of any `.part` directory left by a kill
    /// mid-write. The contents go, not the directory: `root` defaults to the whole
    /// cache directory.
    pub fn new(root: PathBuf, max_bytes: u64, unfetched_ttl: Duration) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        clear_contents(&root).with_context(|| format!("clearing {}", root.display()))?;
        Ok(Arc::new(Self {
            root,
            entries: Mutex::new(HashMap::new()),
            max_bytes,
            unfetched_ttl,
            seq: AtomicU64::new(0),
        }))
    }

    /// Fetch an entry, marking it as used. `None` on a miss.
    ///
    /// An entry counts as a hit only while its files are there. This map decides what
    /// exists and the bytes are read later by another request, so the two diverge
    /// when something outside this process empties the root, such as a second
    /// instance starting on the same directory.
    pub fn get(&self, key: &str) -> Option<Arc<Entry>> {
        let entry = {
            //  Touched under the lock the sweep takes, so a hit and the decision to collect
            //  cannot interleave.
            let entries = self.entries.lock();
            let entry = entries.get(key).cloned()?;
            entry.touch();
            entry
        };

        if !entry.present() {
            tracing::warn!(
                "{} is in the index but not on disk; separating it again",
                short(&entry.key)
            );
            let mut entries = self.entries.lock();
            // Only if it is still the one that went missing: a publish may have
            // replaced it under this key in the meantime.
            if entries
                .get(key)
                .is_some_and(|held| Arc::ptr_eq(held, &entry))
            {
                entries.remove(key);
                entry.evicted.store(true, Ordering::SeqCst);
            }
            return None;
        }
        Some(entry)
    }

    /// Delete every entry and its stems. Returns the tracks and bytes freed.
    ///
    /// A download already reading its file finishes from the open descriptor; one
    /// that has not opened yet gets a 410.
    pub fn clear(&self) -> (usize, u64) {
        let mut entries = self.entries.lock();
        let tracks = entries.len();
        let freed = entries.values().map(|entry| entry.bytes).sum();
        for entry in entries.values() {
            self.delete(entry);
        }
        entries.clear();
        (tracks, freed)
    }

    /// One consistent snapshot, rather than three locks that can disagree.
    pub fn stats(&self) -> Stats {
        let entries = self.entries.lock();
        Stats {
            tracks: entries.len(),
            bytes: entries.values().map(|e| e.bytes).sum(),
            max_bytes: self.max_bytes,
        }
    }

    /// How often both rules are applied.
    ///
    /// A fifth of the deadline they enforce, derived rather than constant so a short
    /// `--unfetched-ttl` keeps its meaning.
    pub fn sweep_interval(&self) -> Duration {
        (self.unfetched_ttl / SWEEPS_PER_TTL).clamp(MIN_SWEEP, MAX_SWEEP)
    }

    /// Write the shipped stems and make them visible under `key`.
    pub fn publish(
        &self,
        key: &str,
        mix: &Audio,
        stems: &Stems,
        output: Output,
        separation_secs: f64,
    ) -> Result<Arc<Entry>> {
        // Converted before encoding, so an entry holds exactly the bytes a
        // client is handed and a hit costs no further work.
        let shipped = to_output_rate(mix, stems, output)?;

        let dir = self.root.join(key);
        let staging = self.staging_dir(key)?;
        let files = write_stems(&staging, &dir, &shipped, output.format)?;
        install(&staging, &dir)?;

        let entry = Arc::new(Entry {
            key: key.to_owned(),
            bytes: files.iter().map(|f| f.bytes).sum(),
            stems: files,
            dir,
            // The geometry a client receives: the stems' after conversion,
            // not the mix's that produced them.
            sample_rate: output.rate.hz(),
            channels: mix.channels(),
            frames: shipped.first().map_or(0, |(_, audio)| audio.frames()),
            format: output.format,
            model_residual_db: stems.model_residual_db,
            separation_secs,
            evicted: AtomicBool::new(false),
            state: Mutex::new(State {
                unfetched: shipped.iter().map(|(name, _)| (*name).to_owned()).collect(),
                last_used: Instant::now(),
            }),
        });

        let mut entries = self.entries.lock();
        entries.insert(key.to_owned(), Arc::clone(&entry));
        // Apply the cap now rather than waiting for the sweep: a burst of long
        // tracks would otherwise sit over budget for a whole interval.
        let (rotated, freed) = self.enforce_cap(&mut entries);
        if rotated > 0 {
            tracing::info!(
                "cache over budget, rotated {rotated} entries ({:.0} MB)",
                freed as f64 / 1e6
            );
        }
        Ok(entry)
    }

    /// A fresh directory beside the final name.
    ///
    /// Written to and then renamed, so a kill mid-write cannot leave behind a
    /// directory that looks like a complete entry.
    fn staging_dir(&self, key: &str) -> Result<PathBuf> {
        let staging = self.root.join(format!(
            "{key}.{}.part",
            self.seq.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&staging)
            .with_context(|| format!("creating {}", staging.display()))?;
        Ok(staging)
    }

    /// Apply both rules. Returns what it collected. Nothing is deleted while the
    /// store fits.
    pub fn reap(&self) -> Reaped {
        let mut entries = self.entries.lock();
        let now = Instant::now();
        let mut reaped = Reaped::default();

        // Rule 1: space is starting to matter, so give up the separations nobody
        // came back for before rule 2 has to touch any that were wanted.
        if self.under_pressure(&entries) {
            entries.retain(|_, entry| {
                let idle = now.duration_since(entry.last_used());
                if entry.consumed() || idle <= self.unfetched_ttl {
                    return true;
                }
                tracing::debug!(
                    "reaping {} ({:.0} MB): separated {:.0}s ago, never pulled, \
                     and the store is filling up",
                    short(&entry.key),
                    entry.bytes as f64 / 1e6,
                    idle.as_secs_f64()
                );
                reaped.unfetched += 1;
                reaped.bytes += entry.bytes;
                self.delete(entry);
                false
            });
        }

        // Rule 2: still over budget.
        let (rotated, freed) = self.enforce_cap(&mut entries);
        reaped.rotated = rotated;
        reaped.bytes += freed;
        reaped
    }

    /// Is the store full enough that an uncollected entry is worth its space?
    ///
    /// Half the budget. [`Cache::publish`] enforces the cap on its way out and leaves
    /// the store under it, so waiting for the cap would make rule 1 nearly
    /// unreachable.
    fn under_pressure(&self, entries: &HashMap<String, Arc<Entry>>) -> bool {
        total_bytes(entries) > self.max_bytes / 2
    }

    /// Drop least-recently-used entries until the store fits.
    fn enforce_cap(&self, entries: &mut HashMap<String, Arc<Entry>>) -> (usize, u64) {
        let mut total = total_bytes(entries);
        if total <= self.max_bytes {
            return (0, 0);
        }

        let mut order: Vec<(Instant, String)> = entries
            .values()
            .map(|e| (e.last_used(), e.key.clone()))
            .collect();
        order.sort_by_key(|(used, _)| *used);
        // Never rotate the most recent entry: it is the one a job is most
        // likely still holding, and a single track larger than the whole budget
        // would otherwise be separated and discarded in the same breath.
        order.pop();

        let mut count = 0;
        let mut freed = 0;
        for (_, key) in order {
            if total <= self.max_bytes {
                break;
            }
            if let Some(entry) = entries.remove(&key) {
                total -= entry.bytes;
                freed += entry.bytes;
                count += 1;
                self.delete(&entry);
            }
        }
        (count, freed)
    }

    /// Remove an entry's directory and mark it evicted.
    ///
    /// Safe while a client is streaming one of its stems: the handler opens the file
    /// first, and an unlinked file stays readable through an open descriptor. A fetch
    /// that has not opened yet gets a 410.
    fn delete(&self, entry: &Entry) {
        entry.evicted.store(true, Ordering::SeqCst);
        if let Err(err) = std::fs::remove_dir_all(&entry.dir) {
            tracing::warn!("could not remove {}: {err}", entry.dir.display());
        }
    }
}

/// Delete everything inside `dir`, leaving the directory itself in place.
fn clear_contents(dir: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            std::fs::remove_dir_all(&path)?;
        } else {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(())
}

/// The parts an entry stores, converted to the requested output rate.
///
/// [`stemd_core::SHIPPED`] always; the derived part only when asked for. It is
/// computed here at the native rate, where `mix - harmonics - vocals` is exact and
/// carries the model's residual, then put through the same filter as the others.
/// Resampling is linear, so the set still sums to the resampled mix. The model's
/// own `drums` source is missing the residual and would not sum.
///
/// Separation runs at the model's rate and this is the only place the output rate
/// is applied, so the cached bytes and the rate an entry reports cannot disagree.
fn to_output_rate(
    mix: &Audio,
    stems: &Stems,
    output: Output,
) -> Result<Vec<(&'static str, Audio)>> {
    let derived = output.derived.then(|| stems.derived_part(mix));
    let parts: Vec<(&'static str, &Audio)> = stems
        .shipped
        .iter()
        .map(|(name, audio)| (*name, audio))
        .chain(derived.iter().map(|audio| (DERIVED, audio)))
        .collect();

    //  Concurrent for every rate rather than by case: the 48 kHz path does not
    //  oversubscribe the way nesting suggests, the 96 kHz one is memory-bound and
    //  gains nothing however it is arranged, and at the model's own rate this is a
    //  clone. See `is_converting_three_stems_at_once_worth_it`.
    std::thread::scope(|scope| {
        let running: Vec<_> = parts
            .iter()
            .map(|(name, audio)| scope.spawn(|| one_rate(name, audio, output)))
            .collect();
        running
            .into_iter()
            .map(|thread| {
                thread
                    .join()
                    .unwrap_or_else(|p| std::panic::resume_unwind(p))
            })
            .collect()
    })
}

fn one_rate(name: &'static str, audio: &Audio, output: Output) -> Result<(&'static str, Audio)> {
    let began = std::time::Instant::now();
    let converted = resample::to_rate_with(audio, output.rate.hz(), output.dsp)
        .with_context(|| format!("converting {name} to {} Hz", output.rate))?;
    //  Timed because which of conversion and encode dominates depends on the format
    //  asked for.
    tracing::debug!(
        "{name}: {} -> {} Hz in {:.2?}",
        audio.sample_rate,
        output.rate,
        began.elapsed()
    );
    Ok((name, converted))
}

/// Encode each shipped stem into `staging`, described by its eventual path under
/// `dir`.
///
/// One thread per stem, at most three, so the stage costs the longest encode
/// rather than the sum. `scope` lets each thread borrow its stem instead of
/// copying it.
fn write_stems(
    staging: &Path,
    dir: &Path,
    shipped: &[(&'static str, Audio)],
    format: StemFormat,
) -> Result<Vec<CachedStem>> {
    std::thread::scope(|scope| {
        let running: Vec<_> = shipped
            .iter()
            .map(|(name, audio)| scope.spawn(|| one_stem(staging, dir, name, audio, format)))
            .collect();
        running
            .into_iter()
            // A panic in an encoder is re-raised here rather than turned into an
            // error: it is a bug, and the caller's `?` would file it under
            // "could not write the stems".
            .map(|thread| {
                thread
                    .join()
                    .unwrap_or_else(|p| std::panic::resume_unwind(p))
            })
            .collect()
    })
}

fn one_stem(
    staging: &Path,
    dir: &Path,
    name: &'static str,
    audio: &Audio,
    format: StemFormat,
) -> Result<CachedStem> {
    //  One traversal for both the peak the transfer gain needs and the non-finite
    //  check. The stem is named here so the failure says which part diverged.
    let peak = audio
        .peak()
        .with_context(|| format!("the separated {name} is not finite"))?;
    let gain = transfer_gain(peak, format);
    if gain < 1.0 {
        tracing::debug!("{name} peaks at {peak:.3}, scaling by {gain:.4}");
    }

    let file = format!("{name}.{}", format.extension());
    let began = std::time::Instant::now();
    let bytes = stemd_core::encode_stem(audio, format, gain)?;
    let encoded = began.elapsed();

    let began = std::time::Instant::now();
    std::fs::write(staging.join(&file), &bytes)
        .with_context(|| format!("writing {} into {}", file, staging.display()))?;
    // Fractional, because integer megabytes render every stem under one as
    // "0 MB", which is most of them at 320 kbps.
    tracing::debug!(
        "{file}: encoded {:.1} MB in {:.2?}, wrote it in {:.2?}",
        bytes.len() as f64 / 1_048_576.0,
        encoded,
        began.elapsed()
    );

    Ok(CachedStem {
        name: name.to_owned(),
        path: dir.join(&file),
        bytes: bytes.len() as u64,
        gain,
    })
}

/// Scale that keeps a stem inside the output format's range.
///
/// A separated stem is not bounded by the mix: the model over-subtracts in places,
/// so one peaking past full scale is scaled to fit. `f32le` has no ceiling and
/// always ships at unity. Takes the peak rather than the buffer, which
/// [`one_stem`] has already computed.
const fn transfer_gain(peak: f32, format: StemFormat) -> f32 {
    if format.has_headroom() || peak <= 1.0 {
        1.0
    } else {
        1.0 / peak
    }
}

/// Move a staged directory into place, replacing whatever was there.
fn install(staging: &Path, dir: &Path) -> Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir).with_context(|| format!("replacing {}", dir.display()))?;
    }
    std::fs::rename(staging, dir)
        .with_context(|| format!("publishing {} as {}", staging.display(), dir.display()))
}

/// A consistent view of what the cache is holding.
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub tracks: usize,
    pub bytes: u64,
    pub max_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testkit::{cache, publish};

    /// A 200 must be followed by stems the client can actually download.
    ///
    /// The TTL is zero so every published entry is collectable at once; a nonzero one
    /// would make the collision depend on scheduling. The sweeper yields rather than
    /// spinning.
    #[test]
    fn a_hit_is_never_handed_out_after_its_stems_are_gone() {
        // A budget of one byte, so the store is always past the mark where
        // uncollected entries are collectable. With `u64::MAX` the sweeper now
        // correctly finds a store with room and deletes nothing, and there is no
        // race left to run.
        let cache = cache(1, Duration::ZERO);
        let stop = Arc::new(AtomicBool::new(false));

        let sweeper = {
            let cache = Arc::clone(&cache);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut collected = 0;
                while !stop.load(Ordering::SeqCst) {
                    collected += cache.reap().unfetched;
                    std::thread::yield_now();
                }
                collected
            })
        };

        for i in 0..300 {
            let entry = publish(&cache, &format!("track-{i}"), 64, 1.0);
            if let Some(hit) = cache.get(&entry.key) {
                // Present, or tombstoned: never neither. `delete` raises
                // `evicted` before it unlinks, so an entry whose stems have gone
                // always says so. That is the guarantee a job depends on:
                // whatever it is handed is either usable or admits it is gone.
                assert!(
                    hit.present() || hit.evicted(),
                    "handed out {} with its stems deleted and no tombstone",
                    short(&hit.key)
                );
            }
        }

        stop.store(true, Ordering::SeqCst);
        let collected = sweeper.join().unwrap();
        // Without this the test passes just as happily when the sweeper never
        // ran, and the assertion above stops guarding anything.
        assert!(collected > 0, "the sweeper collected nothing to race with");
    }

    /// A diverged model must fail the job, not fill the cache with silence.
    ///
    /// NaN peaks at 0.0 and quantises to 0, so the entry would be a well-formed set of
    /// stems that decode to nothing, cached under a key asserting they are what a
    /// fresh separation would produce. The error names the stem, since only one half
    /// diverged.
    #[test]
    fn a_stem_of_nan_fails_the_job_instead_of_being_published() {
        let cache = cache(u64::MAX, Duration::from_secs(60));
        let (mix, mut parts) = crate::testkit::stems(64);
        // Only the vocals, so the assertion below is about *which* stem is
        // named rather than about there being a name at all.
        for (name, audio) in &mut parts.shipped {
            if *name == "vocals" {
                audio.data = audio.data.iter().map(|c| vec![f32::NAN; c.len()]).collect();
            }
        }

        let err = cache
            .publish(
                "diverged",
                &mix,
                &parts,
                Output {
                    format: StemFormat::Pcm16,
                    rate: OutputRate::default(),
                    derived: false,
                    dsp: DspMode::default(),
                },
                1.0,
            )
            .expect_err("a stem of NaN must not publish");

        let message = format!("{err:#}");
        assert!(
            message.contains("vocals") && message.contains("not a finite"),
            "the failure has to name the stem: {message}"
        );
        //  Nothing indexed, so no later request can hit a half-written entry. The `.part`
        //  staging directory survives, and `Cache::new` clears those at startup.
        assert!(cache.get("diverged").is_none());
        assert_eq!(cache.stats().tracks, 0);
    }

    #[test]
    fn clearing_takes_the_index_and_the_stems_together() {
        let cache = cache(u64::MAX, Duration::from_secs(60));
        let first = publish(&cache, "one", 64, 1.0);
        let second = publish(&cache, "two", 32, 1.0);

        let (tracks, freed) = cache.clear();
        assert_eq!(tracks, 2);
        assert_eq!(freed, first.bytes + second.bytes);
        assert_eq!(cache.stats().tracks, 0);
        assert!(!first.dir.exists() && !second.dir.exists());
        assert!(
            first.evicted() && second.evicted(),
            "a job still holding one has to be able to tell"
        );
        assert!(cache.get(&first.key).is_none());
    }

    #[test]
    fn an_entry_whose_files_vanished_is_a_miss_not_a_broken_hit() {
        let cache = cache(u64::MAX, Duration::from_secs(60));
        let entry = publish(&cache, "wiped", 64, 1.0);
        assert!(cache.get(&entry.key).is_some());

        // What a second server does to the root on startup, before it finds
        // out the port is taken.
        std::fs::remove_dir_all(&entry.dir).unwrap();

        assert!(
            cache.get(&entry.key).is_none(),
            "a hit here promises stems that would 410 on download"
        );
        assert!(entry.evicted(), "the job holding it must be told");
        assert_eq!(cache.stats().tracks, 0, "the phantom must leave the index");
    }

    /// An expired TTL alone is not reason enough to collect: it threw away real
    /// separations out of a store nowhere near its budget.
    #[test]
    fn an_entry_nobody_pulls_survives_while_there_is_room_for_it() {
        let cache = cache(u64::MAX, Duration::from_millis(30));
        let entry = publish(&cache, "abandoned", 64, 1.0);
        assert!(entry.dir.exists());

        std::thread::sleep(Duration::from_millis(50));

        assert_eq!(cache.reap().unfetched, 0, "the store is nowhere near full");
        assert!(!entry.evicted());
        assert!(entry.dir.exists(), "the stems are still worth having");
    }

    /// But it is still the first thing to give up once room starts to matter.
    ///
    /// An entry is 512 B, so a budget of 1500 holds both with room to spare: under the
    /// cap, so rule 2 stays out of it, and over half, so rule 1 is awake.
    #[test]
    fn an_entry_nobody_pulls_is_the_first_to_go_once_room_matters() {
        let cache = cache(1500, Duration::from_millis(30));
        let wanted = publish(&cache, "wanted", 64, 1.0);
        for stem in &wanted.stems {
            wanted.mark_fetched(&stem.name);
        }
        let abandoned = publish(&cache, "abandoned", 64, 1.0);

        std::thread::sleep(Duration::from_millis(50));

        let reaped = cache.reap();
        assert_eq!(
            reaped.unfetched, 1,
            "the abandoned entry pays for the space"
        );
        assert!(abandoned.evicted(), "the job holding it must be told");
        assert!(!abandoned.dir.exists(), "the stems must go with the entry");
        assert!(!wanted.evicted(), "the entry somebody collected stays");
    }

    #[test]
    fn an_entry_whose_stems_were_all_pulled_survives() {
        let cache = cache(u64::MAX, Duration::from_millis(30));
        let entry = publish(&cache, "wanted", 64, 1.0);
        for stem in &entry.stems {
            entry.mark_fetched(&stem.name);
        }
        assert!(entry.consumed());

        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(cache.reap().unfetched, 0);
        assert!(!entry.evicted());
    }

    #[test]
    fn a_partial_fetch_restarts_the_idle_clock() {
        let cache = cache(u64::MAX, Duration::from_millis(60));
        let entry = publish(&cache, "half", 64, 1.0);

        // A deck that pulls one stem and is midway through the second must not
        // have the entry deleted out from under it.
        std::thread::sleep(Duration::from_millis(40));
        entry.mark_fetched(&entry.stems[0].name);
        std::thread::sleep(Duration::from_millis(40));

        assert!(!entry.consumed(), "one stem is still outstanding");
        assert_eq!(cache.reap().unfetched, 0, "the fetch reset the clock");
        assert!(!entry.evicted());
    }

    #[test]
    fn the_cap_rotates_the_least_recently_used_first() {
        // Two stems of 64 frames x 2 ch x 2 bytes = 512 B per entry, so a
        // budget of 1100 holds exactly two of the three.
        let cache = cache(1100, Duration::from_secs(3600));
        let old = publish(&cache, "oldest", 64, 1.0);
        let mid = publish(&cache, "middle", 64, 1.0);
        for entry in [&old, &mid] {
            for stem in &entry.stems {
                entry.mark_fetched(&stem.name);
            }
        }
        // Re-use the older one so recency, not insertion order, decides.
        std::thread::sleep(Duration::from_millis(2));
        old.touch();
        std::thread::sleep(Duration::from_millis(2));

        let third = publish(&cache, "newest", 64, 1.0);
        assert_eq!(old.bytes, 512, "the budget above assumes this size");
        assert!(cache.stats().bytes <= 1100, "publish must apply the cap");
        assert!(mid.evicted(), "least recently used goes first");
        assert!(!old.evicted());
        assert!(!third.evicted());
        assert!(!mid.dir.exists());
        assert!(third.dir.exists());
    }

    #[test]
    fn the_newest_entry_survives_a_budget_it_alone_exceeds() {
        let cache = cache(16, Duration::from_secs(3600));
        let entry = publish(&cache, "huge", 512, 1.0);
        assert!(
            !entry.evicted(),
            "separating and immediately discarding is worse than being over budget"
        );
        assert!(entry.dir.exists());
    }

    fn output(rate: OutputRate, derived: bool) -> Output {
        Output {
            format: StemFormat::Pcm16,
            rate,
            derived,
            dsp: DspMode::default(),
        }
    }

    /// Two stems by default, at every rate. The rate does not decide what
    /// ships: only the client's request does.
    #[test]
    fn what_ships_is_the_request_not_the_rate() {
        let (mix, stems) = crate::testkit::stems(4410);
        for rate in OutputRate::ALL {
            let parts = to_output_rate(&mix, &stems, output(rate, false)).unwrap();
            let names: Vec<&str> = parts.iter().map(|(n, _)| *n).collect();
            assert_eq!(names, stemd_core::SHIPPED, "{rate} changed what ships");

            let parts = to_output_rate(&mix, &stems, output(rate, true)).unwrap();
            let names: Vec<&str> = parts.iter().map(|(n, _)| *n).collect();
            assert_eq!(
                names.len(),
                stemd_core::SHIPPED.len() + 1,
                "{rate}: {names:?}"
            );
            assert_eq!(names.last(), Some(&DERIVED), "{rate}: {names:?}");
            for (name, audio) in &parts {
                assert_eq!(audio.sample_rate, rate.hz(), "{name} at the wrong rate");
            }
        }
    }

    /// Asking for the derived part must not be answerable from an entry that
    /// does not hold it: the request changes the files, so it changes the key.
    #[test]
    fn the_derived_request_is_its_own_entry() {
        let base = Ingredients {
            pcm: b"same samples",
            sample_rate: 44100,
            channels: 2,
            in_format: PcmFormat::S16le,
            out_format: StemFormat::Pcm16,
            out_rate: OutputRate::Hz44100,
            include_derived: false,
            dsp: DspMode::General,
            model: "digest",
        };
        assert_ne!(
            key(&base),
            key(&Ingredients {
                include_derived: true,
                ..base
            })
        );
    }

    /// The rate has to survive into what is stored, not just into the key:
    /// an entry reports its own geometry, and a client trusts it.
    #[test]
    fn a_published_entry_reports_the_rate_it_was_converted_to() {
        let cache = cache(u64::MAX, Duration::from_secs(60));
        let (mix, stems) = crate::testkit::stems(44_100);
        assert_eq!(mix.sample_rate, 44_100);

        for rate in OutputRate::ALL {
            let entry = cache
                .publish(
                    &format!("track-{rate}"),
                    &mix,
                    &stems,
                    output(rate, false),
                    1.0,
                )
                .expect("publish");

            assert_eq!(entry.sample_rate, rate.hz(), "{rate} misreported");
            // One second in, one second out, whatever the rate.
            assert!(
                (entry.duration_secs() - 1.0).abs() < 0.01,
                "{rate}: {:.3}s",
                entry.duration_secs()
            );
            // Frames follow the rate, so the bytes on disk do too.
            let expected = rate.hz() as f64 / 44_100.0;
            let ratio = entry.frames as f64 / 44_100.0;
            assert!((ratio - expected).abs() < 0.01, "{rate}: {ratio:.3}");
        }
    }

    #[test]
    fn the_key_separates_runs_that_would_return_different_bytes() {
        let base = Ingredients {
            pcm: b"the same samples",
            sample_rate: 44100,
            channels: 2,
            in_format: PcmFormat::S16le,
            out_format: StemFormat::Pcm16,
            out_rate: OutputRate::Hz44100,
            include_derived: false,
            dsp: DspMode::General,
            model: "digest",
        };
        let reference = key(&base);

        assert_eq!(
            reference,
            key(&Ingredients { ..base }),
            "the key must be stable"
        );
        assert_ne!(
            reference,
            key(&Ingredients {
                pcm: b"other samples!!!",
                ..base
            })
        );
        assert_ne!(
            reference,
            key(&Ingredients {
                sample_rate: 48000,
                ..base
            })
        );
        assert_ne!(
            reference,
            key(&Ingredients {
                channels: 1,
                ..base
            })
        );
        assert_ne!(
            reference,
            key(&Ingredients {
                in_format: PcmFormat::F32le,
                ..base
            })
        );
        assert_ne!(
            reference,
            key(&Ingredients {
                out_format: StemFormat::Pcm32,
                ..base
            })
        );
        assert_ne!(
            reference,
            key(&Ingredients {
                model: "another",
                ..base
            })
        );
        // Two filters at one rate are two different sets of samples, so a hit on
        // one must never answer for the other.
        assert_ne!(
            reference,
            key(&Ingredients {
                dsp: DspMode::Mode1,
                ..base
            })
        );
        // The one that would be least visible if it were missing: same bytes,
        // same model, same container: different audio out.
        for rate in OutputRate::ALL {
            let keyed = key(&Ingredients {
                out_rate: rate,
                ..base
            });
            assert_eq!(
                keyed == reference,
                rate == OutputRate::Hz44100,
                "{rate} Hz must have its own entry"
            );
        }
    }
}

#[cfg(test)]
mod resample_bench {
    use stemd_core::{Audio, resample};

    fn stem(secs: f64) -> Audio {
        let n = (secs * 44100.0) as usize;
        Audio::new(
            (0..2)
                .map(|c| {
                    (0..n)
                        .map(|i| ((i as f32 * 0.01) + c as f32).sin() * 0.3)
                        .collect()
                })
                .collect(),
            44_100,
        )
    }

    /// Is converting three stems at once faster than one at a time? Not obviously:
    /// the conversion can be memory-bound, and the 44.1 to 48 kHz path threads
    /// internally already.
    #[test]
    #[ignore]
    fn is_converting_three_stems_at_once_worth_it() {
        let stems: Vec<Audio> = (0..3).map(|_| stem(120.0)).collect();
        for target in [48_000u32, 96_000] {
            for round in 0..3 {
                let serial = std::time::Instant::now();
                for s in &stems {
                    std::hint::black_box(resample::to_rate(s, target).unwrap());
                }
                let serial = serial.elapsed();

                let parallel = std::time::Instant::now();
                std::thread::scope(|scope| {
                    let running: Vec<_> = stems
                        .iter()
                        .map(|s| scope.spawn(move || resample::to_rate(s, target).unwrap()))
                        .collect();
                    for t in running {
                        std::hint::black_box(t.join().unwrap());
                    }
                });
                let parallel = parallel.elapsed();
                println!(
                    "  {target} Hz round {round}: serial {serial:>8.2?}  parallel {parallel:>8.2?}  \
                     ({:.2}x)",
                    serial.as_secs_f64() / parallel.as_secs_f64()
                );
            }
        }
    }
}
