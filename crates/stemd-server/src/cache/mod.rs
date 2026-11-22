//! The cache of separated stems on disk, each piece encoded and ready to send as it is.
//!
//! ```text
//! <root>/<model_id>/<type>/<track>/<stem>/<k>.<ext>
//! <root>/<model_id>/<type>/<track>/<stem>.<ext>       the whole stem, see [`crate::whole`]
//! <root>/<model_id>/<type>/<track>/track.json         the length and piece grid, see [`TrackInfo`]
//! ```
//!
//! | Level   | Example                                        |
//! |---------|------------------------------------------------|
//! | `model_id` | `htdemucs-22c7f2c9`, as in `/v2/health`     |
//! | `type`  | `flac-96000-dsp1-harmonics.vocals-g1`; `-dsp1` or `-fir<id>` names the filter |
//! | `track` | the 24-hex track id, see [`crate::ident::track_id`] |
//! | `k`     | piece index, four digits                       |
//!
//! Sessions, jobs and drops share this one cache. A piece is written only after the track's whole
//! upload has been hashed, so the track id in the path is always the server's own. The root
//! survives restarts, since each path names everything its bytes depend on.
//!
//! Once every piece of a track directory is stored, a background thread stores each stem as one
//! whole file and deletes its pieces; ranges are then read from the whole file. A job stores its
//! stems whole at once.
//!
//! A track directory is whole when every stem in it is stored whole, and partial otherwise. A
//! `.used` file in it records its last use. [`Rules`] decide which tracks are deleted, and skip a
//! track in use: one that an open session plays, or a job result that was not fully fetched within
//! [`Rules::unfetched_ttl`].
//!
//! | Rule | |
//! | --- | --- |
//! | partial, unused for `partial_max_age` | deleted |
//! | whole, unused for `whole_max_age` | deleted |
//! | over the budget | partial tracks deleted oldest first, then whole tracks by last use |
//!
//! The budget is `max_bytes`, or less when free disk space would drop below [`crate::disk::FLOOR`].
//! The age rules run at start and every [`CLEAN_EVERY`]; the budget is also applied after each
//! write.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::output::{Output, output_index, output_span, piece_span};

/// The file whose modification time is a track directory's last use.
const USED: &str = ".used";
/// The directory under the root that holds one file per file key, containing its track id.
const FILES: &str = ".files";
/// The file in a track directory that a job writes its [`Separation`] to.
const SEPARATION: &str = ".separation.json";
/// The lock file in the root, which emptying the cache leaves in place.
pub const LOCK_FILE: &str = "stemd.lock";

/// How often the age rules run.
pub const CLEAN_EVERY: Duration = Duration::from_secs(3600);

/// What the cache deletes, and when.
#[derive(Debug, Clone, Copy)]
pub struct Rules {
    pub max_bytes: u64,
    pub partial_max_age: Duration,
    pub whole_max_age: Duration,
    /// How long a job result stays in use after it finished or after its last
    /// stem fetch, until every stem has been fetched.
    pub unfetched_ttl: Duration,
}

impl Rules {
    /// A budget of `max_bytes` and the default ages.
    #[cfg(test)]
    pub fn sized(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            partial_max_age: Duration::from_secs(24 * 3600),
            whole_max_age: Duration::from_secs(30 * 86400),
            unfetched_ttl: Duration::from_secs(300),
        }
    }
}

/// What a job's separation cost, kept beside its stems.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Separation {
    pub model_residual_db: f64,
    pub separation_secs: f64,
}

/// A job result that keeps its track directory from deletion while held.
pub trait Hold: Send + Sync {
    /// The track directory.
    fn dir(&self) -> &Path;
    /// True while the directory must stay, given how long an unfetched result is kept.
    fn held(&self, unfetched_ttl: Duration) -> bool;
}

/// The cache's size, for `/v2/health` and the window.
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub tracks: usize,
    pub bytes: u64,
    pub max_bytes: u64,
}

/// Returns the track ids that open sessions play, which eviction skips.
pub type InUse = Box<dyn Fn() -> HashSet<String> + Send + Sync>;

/// True when background consolidation may use the CPU now.
pub type Gate = Box<dyn Fn() -> bool + Send + Sync>;

/// Reads the free bytes on the cache's volume, `None` when unknown.
pub type FreeProbe = Box<dyn Fn() -> Option<u64> + Send + Sync>;

/// The file in each track directory that holds its [`TrackInfo`].
const INFO: &str = "track.json";

/// A track's length and piece grid, at the model's rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackInfo {
    pub frames: usize,
    pub sample_rate: u32,
    pub channels: usize,
    /// Samples per piece; the last piece may be shorter.
    pub stride: usize,
    pub pieces: usize,
}

impl TrackInfo {
    /// The samples that piece `k` covers.
    pub fn piece(&self, k: usize) -> std::ops::Range<usize> {
        piece_span(k, self.stride, self.frames)
    }

    /// The whole file of `stem` in the track directory `dir` of `output`, when that file covers the
    /// whole track.
    pub fn find_whole(
        &self,
        dir: &Path,
        stem: &str,
        output: &Output,
    ) -> Option<crate::whole::Whole> {
        crate::whole::find_track(
            dir,
            stem,
            output.format.extension(),
            self.frames_at(output.rate.hz()),
        )
    }

    /// The track's length at `rate`.
    pub fn frames_at(&self, rate: u32) -> usize {
        output_index(self.frames, u64::from(self.sample_rate), u64::from(rate))
    }

    /// Each piece's length at `rate`.
    pub fn lengths(&self, rate: u32) -> Vec<usize> {
        let (model, out) = (u64::from(self.sample_rate), u64::from(rate));
        (0..self.pieces)
            .map(|k| output_span(&self.piece(k), model, out).len())
            .collect()
    }
}

pub struct Cache {
    root: PathBuf,
    rules: Mutex<Rules>,
    bytes: AtomicU64,
    /// Job results, in use until fetched; see [`Hold::held`].
    holds: Mutex<Vec<Weak<dyn Hold>>>,
    /// Track directories that a job is storing its stems in.
    writing: Mutex<HashSet<PathBuf>>,
    /// The last track count, and when it was taken.
    counted: Mutex<Option<(Instant, usize)>>,
    /// Held while a path is chosen and written, and while a track's pieces move,
    /// so no write lands in a directory that has just moved.
    lock: Mutex<()>,
    in_use: Mutex<Option<InUse>>,
    gate: Mutex<Option<Gate>>,
    /// Called after a stem is stored whole in the background.
    on_change: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    /// Track directories queued or being stored whole.
    consolidating: Mutex<HashSet<PathBuf>>,
    /// Tasks for the single consolidation thread, and whether that thread is running.
    tasks: Mutex<(VecDeque<Consolidation>, bool)>,
    /// Track directories that failed to be stored whole, and when.
    failed: Mutex<HashMap<PathBuf, Instant>>,
    /// One lock per track directory that a conversion is writing.
    converting: Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>,
    /// Numbers staged files so two writers never share one.
    staged: AtomicU64,
    /// Set once eviction has warned that in-use tracks keep the cache over budget.
    warned_kept: AtomicBool,
    /// Reads the free bytes on the cache's volume; [`crate::disk::free_bytes`] unless a test sets
    /// one.
    free: Mutex<Option<FreeProbe>>,
    /// Test directories deleted when the cache is dropped.
    #[cfg(test)]
    scratch: Mutex<Vec<crate::scratch::Dir>>,
}

impl Cache {
    /// Read the free space through `probe` from now on.
    #[cfg(test)]
    pub fn set_free_probe(&self, probe: FreeProbe) {
        *self.free.lock() = Some(probe);
    }

    /// Bytes free on the cache's volume, or `None` when they cannot be read.
    pub fn free_bytes(&self) -> Option<u64> {
        match self.free.lock().as_ref() {
            Some(probe) => probe(),
            None => crate::disk::free_bytes(&self.root),
        }
    }

    /// True when the cache's volume has less than [`crate::disk::FLOOR`] free.
    pub fn disk_low(&self) -> bool {
        self.free_bytes()
            .is_some_and(|free| free < crate::disk::FLOOR)
    }

    /// Fail unless the cache's volume keeps [`crate::disk::FLOOR`] free.
    pub fn ensure_room(&self) -> Result<(), crate::disk::DiskFull> {
        crate::disk::check_room(self.free_bytes(), 0)
    }

    /// Delete `dir` when the cache is dropped.
    #[cfg(test)]
    pub fn keep(&self, dir: crate::scratch::Dir) {
        self.scratch.lock().push(dir);
    }

    /// Open `root` with a budget of `max_bytes` and the default ages.
    #[cfg(test)]
    pub fn open(root: PathBuf, max_bytes: u64) -> Result<Arc<Self>> {
        Self::open_with(root, Rules::sized(max_bytes))
    }

    /// Open `root`, delete leftover `.part` files and the stems of stemd 0.2.0
    /// and earlier, and apply `rules`.
    pub fn open_with(root: PathBuf, rules: Rules) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        crate::migrate::remove_old_layout(&root);
        let mut bytes = 0;
        for (path, size, _) in files(&root) {
            if path.extension().is_some_and(|e| e == "part") {
                let _ = std::fs::remove_file(&path);
            } else {
                bytes += size;
            }
        }
        let cache = Arc::new(Self {
            root,
            rules: Mutex::new(rules),
            bytes: AtomicU64::new(bytes),
            holds: Mutex::new(Vec::new()),
            writing: Mutex::new(HashSet::new()),
            counted: Mutex::new(None),
            lock: Mutex::new(()),
            in_use: Mutex::new(None),
            gate: Mutex::new(None),
            on_change: Mutex::new(None),
            consolidating: Mutex::new(HashSet::new()),
            tasks: Mutex::new((VecDeque::new(), false)),
            failed: Mutex::new(HashMap::new()),
            converting: Mutex::new(HashMap::new()),
            staged: AtomicU64::new(0),
            warned_kept: AtomicBool::new(false),
            free: Mutex::new(None),
            #[cfg(test)]
            scratch: Mutex::new(Vec::new()),
        });
        cache.clean();
        Ok(cache)
    }

    /// Change the byte cap and the whole-track age, and apply them. Returns the
    /// tracks and bytes deleted.
    pub fn set_limits(&self, max_bytes: u64, whole_max_age: Duration) -> (usize, u64) {
        {
            let mut rules = self.rules.lock();
            rules.max_bytes = max_bytes;
            rules.whole_max_age = whole_max_age;
        }
        self.clean()
    }

    fn rules(&self) -> Rules {
        *self.rules.lock()
    }

    /// Subtract `n` from the byte count, stopping at zero.
    fn sub_bytes(&self, n: u64) {
        let _ = self
            .bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |b| {
                Some(b.saturating_sub(n))
            });
    }

    /// Keep `entry`'s track directory while [`Hold::held`] says so.
    pub fn hold<H: Hold + 'static>(&self, entry: &Arc<H>) {
        let _held = self.lock.lock();
        let mut holds = self.holds.lock();
        holds.retain(|held| held.strong_count() > 0);
        let weak: Weak<H> = Arc::downgrade(entry);
        holds.push(weak);
    }

    /// Keep `track_dir` from deletion while the guard lives, as a job stores its stems there.
    pub fn writing(&self, track_dir: &Path) -> Writing<'_> {
        let _held = self.lock.lock();
        self.writing.lock().insert(track_dir.to_owned());
        Writing {
            cache: self,
            dir: track_dir.to_owned(),
        }
    }

    /// Record what a job's separation cost in `track_dir`.
    pub fn write_separation(&self, track_dir: &Path, separation: &Separation) -> Result<()> {
        let bytes = serde_json::to_vec(separation)?;
        self.publish_counted(&track_dir.join(SEPARATION), &bytes)
    }

    /// The [`Separation`] recorded by the job that stored `track_dir`, if a job stored it.
    pub fn separation(track_dir: &Path) -> Option<Separation> {
        let bytes = std::fs::read(track_dir.join(SEPARATION)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Set how eviction learns which tracks are in use.
    pub fn set_in_use(&self, in_use: InUse) {
        *self.in_use.lock() = Some(in_use);
    }

    /// Set the gate that decides when background consolidation may run. Without a gate, it runs at
    /// once.
    pub fn set_gate(&self, gate: Gate) {
        *self.gate.lock() = Some(gate);
    }

    /// Set what runs after a stem is stored whole in the background.
    pub fn set_on_change(&self, on_change: Box<dyn Fn() + Send + Sync>) {
        *self.on_change.lock() = Some(on_change);
    }

    /// The pieces of `stem` below `count` stored under `track_dir`.
    pub fn stem_pieces(track_dir: &Path, stem: &str, extension: &str, count: usize) -> Vec<usize> {
        (0..count)
            .filter(|&k| track_dir.join(stem).join(file(k, extension)).is_file())
            .collect()
    }

    /// The [`TrackInfo`] of `track_dir`, when it was written.
    pub fn info(track_dir: &Path) -> Option<TrackInfo> {
        let bytes = std::fs::read(track_dir.join(INFO)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// Write `info` into `track_dir` unless it is there already.
    pub fn write_info(&self, track_dir: &Path, info: &TrackInfo) -> Result<()> {
        if Self::info(track_dir).as_ref() == Some(info) {
            return Ok(());
        }
        let bytes = serde_json::to_vec(info)?;
        std::fs::create_dir_all(track_dir)
            .with_context(|| format!("creating {}", track_dir.display()))?;
        self.publish_counted(&track_dir.join(INFO), &bytes)
    }

    /// The type directories under `model_id`.
    pub fn types(&self, model_id: &str) -> Vec<String> {
        subdirs(&self.root.join(model_id))
            .iter()
            .filter_map(|dir| dir.file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .collect()
    }

    /// Mark a track directory used now.
    pub fn touch(track_dir: &Path) {
        if let Ok(f) = std::fs::File::options()
            .create(true)
            .append(true)
            .open(track_dir.join(USED))
        {
            let _ = f.set_modified(SystemTime::now());
        }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// The directory of a verified track's pieces of one type.
    pub fn verified_dir(&self, model_id: &str, type_dir: &str, track: &str) -> PathBuf {
        self.root.join(model_id).join(type_dir).join(track)
    }

    /// The file for piece `k` of `stem` under `track_dir`, when it exists.
    pub fn lookup(track_dir: &Path, stem: &str, k: usize, extension: &str) -> Option<PathBuf> {
        let path = track_dir.join(stem).join(file(k, extension));
        path.is_file().then(|| {
            Self::touch(track_dir);
            path
        })
    }

    /// The piece indices below `count` stored under `dir` for every one of `stems`, as a
    /// piece file or in a whole file.
    pub fn cached(dir: &Path, stems: &[&str], extension: &str, count: usize) -> Vec<usize> {
        if !dir.is_dir() {
            return Vec::new();
        }
        Self::touch(dir);
        let whole: Vec<bool> = stems
            .iter()
            .map(|stem| crate::whole::find(dir, stem, extension).is_some())
            .collect();
        (0..count)
            .filter(|&k| {
                stems.iter().zip(&whole).all(|(stem, &whole)| {
                    whole || dir.join(stem).join(file(k, extension)).is_file()
                })
            })
            .collect()
    }

    /// Write piece `k` of `stem` under `track_dir`, unless it or the whole stem
    /// is there already.
    pub fn write_verified(
        &self,
        track_dir: &Path,
        stem: &str,
        k: usize,
        extension: &str,
        bytes: &[u8],
    ) -> Result<()> {
        let held = self.lock.lock();
        let dir = track_dir.join(stem);
        if dir.join(file(k, extension)).is_file()
            || crate::whole::file(track_dir, stem, extension).is_file()
        {
            return Ok(());
        }
        self.write_in(&dir, k, extension, bytes)?;
        Self::touch(track_dir);
        drop(held);
        self.evict();
        Ok(())
    }

    /// The cache root.
    #[cfg(test)]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Write a piece file.
    fn write_in(&self, dir: &Path, k: usize, extension: &str, bytes: &[u8]) -> Result<()> {
        self.ensure_room()?;
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        self.publish_counted(&dir.join(file(k, extension)), bytes)
    }
}

impl Cache {
    /// Write `bytes` to a new `.part` file next to `path`, and return the `.part` file's path.
    fn stage(&self, path: &Path, bytes: &[u8]) -> Result<PathBuf> {
        let n = self.staged.fetch_add(1, Ordering::Relaxed);
        let mut staged = path.as_os_str().to_owned();
        staged.push(format!(".{n}.part"));
        let staged = PathBuf::from(staged);
        if let Err(err) = std::fs::write(&staged, bytes) {
            let _ = std::fs::remove_file(&staged);
            return Err(err).with_context(|| format!("writing {}", staged.display()));
        }
        Ok(staged)
    }

    /// Publish `path` and count the change in its size.
    fn publish_counted(&self, path: &Path, bytes: &[u8]) -> Result<()> {
        let replaced = std::fs::metadata(path).map_or(0, |m| m.len());
        self.publish_file(path, bytes)?;
        self.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        self.sub_bytes(replaced);
        Ok(())
    }

    /// Write `path` through a staged file and a rename, so a reader never sees half of it.
    fn publish_file(&self, path: &Path, bytes: &[u8]) -> Result<()> {
        let staged = self.stage(path, bytes)?;
        std::fs::rename(&staged, path).map_err(|err| {
            let _ = std::fs::remove_file(&staged);
            anyhow::Error::new(err).context(format!("publishing {}", path.display()))
        })
    }
}

/// Keeps a track directory from deletion while a job writes it.
pub struct Writing<'a> {
    cache: &'a Cache,
    dir: PathBuf,
}

impl Drop for Writing<'_> {
    fn drop(&mut self) {
        self.cache.writing.lock().remove(&self.dir);
    }
}

/// The directories directly under `dir`.
fn subdirs(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default()
}

/// Every file under `dir`, with its size and when it was last used.
fn files(dir: &Path) -> Vec<(PathBuf, u64, SystemTime)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(path);
            } else {
                let used = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                out.push((path, meta.len(), used));
            }
        }
    }
    out
}

mod consolidate;
mod evict;
mod keys;
mod layout;

pub use consolidate::Consolidation;
use layout::file;
pub use layout::{default_dir, lock, type_dir, type_prefix};

#[cfg(test)]
mod tests;
