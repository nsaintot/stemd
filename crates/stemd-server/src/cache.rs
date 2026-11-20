//! Separated pieces on disk, each one encoded and ready to send as it is.
//!
//! ```text
//! <root>/<model_id>/<type>/<track>/<stem>/<k>.<ext>
//! <root>/<model_id>/<type>/<track>/<stem>.<ext>       the whole stem, see [`crate::whole`]
//! <root>/<model_id>/<type>/<track>/track.json         length and grid, see [`TrackInfo`]
//! ```
//!
//! | Level   | Example                                        |
//! |---------|------------------------------------------------|
//! | `model_id` | `htdemucs-22c7f2c9`, as in `/v2/health`     |
//! | `type`  | `flac-96000-dsp1-harmonics.vocals-g1`; `-dsp1` or `-fir<id>` names the filter |
//! | `track` | the 24-hex track id, see [`crate::sessions::track_id`] |
//! | `k`     | piece index, four digits                       |
//!
//! The one cache of separated stems, for sessions, jobs and drops alike. A
//! piece is written once its track's whole upload has been hashed, so the track
//! id in the path is always the server's own. The root survives restarts, since
//! a path names everything its bytes depend on. Once a stem is stored whole,
//! its pieces are deleted, and ranges are read from the whole file. Once
//! every piece of a track directory is stored, a background thread stores each
//! stem whole. A job stores its stems whole at once.
//!
//! A track directory is whole when every stem in it is stored whole, else
//! partial. A `.used` file in it records its last use. [`Rules`] decide what
//! goes, skipping a track in use: one an open session plays, or a job result
//! not fully fetched within [`Rules::unfetched_ttl`].
//!
//! | Rule | |
//! | --- | --- |
//! | partial, unused for `partial_max_age` | deleted |
//! | whole, unused for `whole_max_age` | deleted |
//! | over the budget | partial tracks, oldest first; then whole tracks, least recently used |
//!
//! The budget is `max_bytes`, or less when the disk is short of
//! [`crate::disk::FLOOR`]. The age rules run at start and every
//! [`CLEAN_EVERY`]; the budget also after each write.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result, ensure};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use stemd_core::{Audio, StemFormat};

use crate::jobs::Entry;
use crate::output::Output;
use crate::sessions::output_index;

/// The directory name for one output at one gain.
pub fn type_dir(output: &Output, gain: f32) -> String {
    format!("{}{gain}", type_prefix(output))
}

/// The directory name for one output, up to its gain.
pub fn type_prefix(output: &Output) -> String {
    let dsp = output.dsp.dir_suffix();
    let parts: Vec<&str> = output.parts.names().collect();
    format!(
        "{}-{}{dsp}-{}-g",
        output.format,
        output.rate,
        parts.join(".")
    )
}

/// The file whose modification time is a track directory's last use.
const USED: &str = ".used";
/// Directory under the root holding one file per file key, naming its track id.
const FILES: &str = ".files";
/// What a file key's file holds once two uploads under it hashed differently.
const AMBIGUOUS: &str = "ambiguous";
/// The file in a track directory a job writes its [`Separation`] to.
const SEPARATION: &str = ".separation.json";
/// The lock file in the root, which emptying the cache keeps.
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

/// The cache's size, for `/v2/health` and the window.
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub tracks: usize,
    pub bytes: u64,
    pub max_bytes: u64,
}

/// The track ids open sessions play, which eviction skips.
pub type InUse = Box<dyn Fn() -> HashSet<String> + Send + Sync>;

/// True when background consolidation may use the CPU now.
pub type Gate = Box<dyn Fn() -> bool + Send + Sync>;

/// The file in each track directory that holds its [`TrackInfo`].
const INFO: &str = "track.json";

/// How often a consolidation waiting on its [`Gate`] checks again.
const GATE_POLL: std::time::Duration = std::time::Duration::from_millis(250);
/// How long a track directory that failed to be stored whole waits before another try.
const RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(600);

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
    /// Samples piece `k` covers.
    pub fn piece(&self, k: usize) -> std::ops::Range<usize> {
        (k * self.stride).min(self.frames)..((k + 1) * self.stride).min(self.frames)
    }

    /// The track's length at `rate`.
    pub fn frames_at(&self, rate: u32) -> usize {
        output_index(self.frames, u64::from(self.sample_rate), u64::from(rate))
    }

    /// Each piece's length at `rate`.
    pub fn lengths(&self, rate: u32) -> Vec<usize> {
        let (model, out) = (u64::from(self.sample_rate), u64::from(rate));
        (0..self.pieces)
            .map(|k| {
                let piece = self.piece(k);
                output_index(piece.end, model, out) - output_index(piece.start, model, out)
            })
            .collect()
    }
}

/// A track directory of one output, to store whole stem by stem.
#[derive(Debug, Clone)]
pub struct Consolidation {
    pub dir: PathBuf,
    pub format: StemFormat,
    /// The output rate.
    pub rate: u32,
    pub info: TrackInfo,
    pub stems: Vec<&'static str>,
}

#[cfg(test)]
impl Consolidation {
    /// A one-piece track of `frames` samples at 44.1 kHz in `dir`, stored as `format`.
    pub fn at_44100(dir: PathBuf, format: StemFormat, frames: usize) -> Self {
        let info = TrackInfo {
            frames,
            sample_rate: 44_100,
            channels: 2,
            stride: frames,
            pieces: 1,
        };
        Self {
            dir,
            format,
            rate: 44_100,
            info,
            stems: Vec::new(),
        }
    }
}

/// A file name for piece `k`.
fn file(k: usize, extension: &str) -> String {
    format!("{k:04}.{extension}")
}

pub struct Cache {
    root: PathBuf,
    rules: Mutex<Rules>,
    bytes: AtomicU64,
    /// Job results, in use until fetched; see [`Entry::held`].
    holds: Mutex<Vec<Weak<Entry>>>,
    /// Track directories a job is storing its stems in.
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
    /// Tasks for the one consolidation thread, and whether it runs.
    tasks: Mutex<(VecDeque<Consolidation>, bool)>,
    /// Track directories that failed to be stored whole, and when.
    failed: Mutex<HashMap<PathBuf, Instant>>,
    /// One lock per track directory a conversion is writing.
    converting: Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>,
    /// Numbers staged files so two writers never share one.
    staged: AtomicU64,
}

impl Cache {
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

    /// Keep `entry`'s track directory while [`Entry::held`] says so.
    pub fn hold(&self, entry: &Arc<Entry>) {
        let _held = self.lock.lock();
        let mut holds = self.holds.lock();
        holds.retain(|held| held.strong_count() > 0);
        holds.push(Arc::downgrade(entry));
    }

    /// Keep `track_dir` while the guard lives, as a job stores its stems there.
    pub fn writing(&self, track_dir: &Path) -> Writing<'_> {
        let _held = self.lock.lock();
        self.writing.lock().insert(track_dir.to_owned());
        Writing {
            cache: self,
            dir: track_dir.to_owned(),
        }
    }

    /// The track directories jobs are writing or their results hold now.
    fn held_dirs(&self) -> HashSet<PathBuf> {
        let mut dirs = self.writing.lock().clone();
        let mut holds = self.holds.lock();
        holds.retain(|held| held.strong_count() > 0);
        dirs.extend(
            holds
                .iter()
                .filter_map(Weak::upgrade)
                .filter(|entry| entry.held(self.rules().unfetched_ttl))
                .map(|entry| entry.dir.clone()),
        );
        dirs
    }

    /// Record what a job's separation cost in `track_dir`.
    pub fn write_separation(&self, track_dir: &Path, separation: &Separation) -> Result<()> {
        let bytes = serde_json::to_vec(separation)?;
        self.publish_file(&track_dir.join(SEPARATION), &bytes)?;
        self.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// What the job that stored `track_dir` recorded, if a job did.
    pub fn separation(track_dir: &Path) -> Option<Separation> {
        let bytes = std::fs::read(track_dir.join(SEPARATION)).ok()?;
        serde_json::from_slice(&bytes).ok()
    }

    /// The number of track directories, counted at most once a second, the
    /// bytes held and the cap.
    pub fn stats(&self) -> Stats {
        let mut counted = self.counted.lock();
        let tracks = match *counted {
            Some((at, tracks)) if at.elapsed() < Duration::from_secs(1) => tracks,
            _ => {
                let tracks = count_tracks(&self.root);
                *counted = Some((Instant::now(), tracks));
                tracks
            }
        };
        Stats {
            tracks,
            bytes: self.bytes(),
            max_bytes: self.rules().max_bytes,
        }
    }

    /// Delete every track not in use and every file key. Returns the tracks and
    /// bytes freed.
    pub fn clear(&self) -> (usize, u64) {
        let _held = self.lock.lock();
        let in_use = self.in_use();
        let (mut tracks, mut freed) = (0, 0);
        for track in track_dirs(&self.root) {
            if !in_use.blocks(&track) && std::fs::remove_dir_all(&track.dir).is_ok() {
                tracks += 1;
                freed += track.bytes;
            }
        }
        let _ = std::fs::remove_dir_all(self.root.join(FILES));
        self.recount();
        (tracks, freed)
    }

    /// Delete everything in the root but the lock file.
    pub fn wipe(&self) {
        let _held = self.lock.lock();
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.file_name().is_some_and(|name| name == LOCK_FILE) {
                    continue;
                }
                let _ = if path.is_dir() {
                    std::fs::remove_dir_all(&path)
                } else {
                    std::fs::remove_file(&path)
                };
            }
        }
        self.recount();
    }

    /// Set the byte count from the disk, and count the tracks again on the next [`Self::stats`].
    fn recount(&self) {
        let bytes = files(&self.root).iter().map(|(_, size, _)| size).sum();
        self.bytes.store(bytes, Ordering::Relaxed);
        *self.counted.lock() = None;
    }

    /// What may not be deleted now.
    fn in_use(&self) -> InUseNow {
        InUseNow {
            tracks: self.in_use.lock().as_ref().map(|f| f()).unwrap_or_default(),
            dirs: self.held_dirs(),
        }
    }

    /// Apply the age rules, then the budget. Returns the tracks and bytes
    /// deleted.
    pub fn clean(&self) -> (usize, u64) {
        let rules = self.rules();
        let now = SystemTime::now();
        let held = self.lock.lock();
        let in_use = self.in_use();
        let (mut count, mut freed) = (0usize, 0u64);
        for track in track_dirs(&self.root) {
            let max_age = if track.whole {
                rules.whole_max_age
            } else {
                rules.partial_max_age
            };
            let unused = now.duration_since(track.used).unwrap_or_default();
            if unused > max_age
                && !in_use.blocks(&track)
                && std::fs::remove_dir_all(&track.dir).is_ok()
            {
                count += 1;
                freed += track.bytes;
            }
        }
        self.bytes
            .fetch_sub(freed.min(self.bytes()), Ordering::Relaxed);
        drop(held);
        if count > 0 {
            *self.counted.lock() = None;
            tracing::info!(
                "cache: deleted {count} tracks unused past their age ({:.0} MB)",
                freed as f64 / 1e6
            );
        }
        let (evicted, evicted_bytes) = self.evict();
        (count + evicted, freed + evicted_bytes)
    }

    /// Set how eviction learns which tracks are in use.
    pub fn set_in_use(&self, in_use: InUse) {
        *self.in_use.lock() = Some(in_use);
    }

    /// Set when background consolidation may run. Without one it runs at once.
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

    fn gate_open(&self) -> bool {
        self.gate.lock().as_ref().is_none_or(|open| open())
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
        self.publish_file(&track_dir.join(INFO), &bytes)?;
        self.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// True while `track_dir` is queued or being stored whole.
    #[cfg(test)]
    pub fn is_consolidating(&self, track_dir: &Path) -> bool {
        self.consolidating.lock().contains(track_dir)
    }

    /// True when every stem of `task` is stored and one is still in pieces.
    pub fn needs_consolidation(task: &Consolidation) -> bool {
        let extension = task.format.extension();
        let mut whole = 0;
        for stem in &task.stems {
            if crate::whole::find(&task.dir, stem, extension).is_some() {
                whole += 1;
            } else if !(0..task.info.pieces)
                .all(|k| task.dir.join(stem).join(file(k, extension)).is_file())
            {
                return false;
            }
        }
        whole < task.stems.len()
    }

    /// True when `track_dir` failed to be stored whole less than [`RETRY_AFTER`] ago.
    pub fn consolidation_failed(&self, track_dir: &Path) -> bool {
        let mut failed = self.failed.lock();
        failed.retain(|_, at| at.elapsed() < RETRY_AFTER);
        failed.contains_key(track_dir)
    }

    /// Queue every stem of `task` to be stored whole; false when it failed recently.
    pub fn consolidate_later(self: &Arc<Self>, task: Consolidation) -> bool {
        if self.consolidation_failed(&task.dir) {
            return false;
        }
        if !self.consolidating.lock().insert(task.dir.clone()) {
            return true;
        }
        let mut tasks = self.tasks.lock();
        tasks.0.push_back(task);
        if !tasks.1 {
            tasks.1 = true;
            let cache = Arc::clone(self);
            std::thread::spawn(move || cache.consolidate_queued());
        }
        true
    }

    /// Run queued consolidations one at a time until the queue is empty.
    fn consolidate_queued(&self) {
        loop {
            let task = {
                let mut tasks = self.tasks.lock();
                match tasks.0.pop_front() {
                    Some(task) => task,
                    None => {
                        tasks.1 = false;
                        return;
                    }
                }
            };
            if let Err(err) = self.consolidate_task(&task) {
                tracing::warn!("storing {} whole: {err:#}", task.dir.display());
                self.failed.lock().insert(task.dir.clone(), Instant::now());
            }
            self.consolidating.lock().remove(&task.dir);
        }
    }

    /// Store every stem of `task` whole, waiting for the gate before each.
    fn consolidate_task(&self, task: &Consolidation) -> Result<()> {
        for stem in &task.stems {
            while !self.gate_open() {
                std::thread::sleep(GATE_POLL);
            }
            ensure!(
                !crate::disk::low(&self.root),
                "the disk is nearly full, so {stem} stays in pieces"
            );
            self.consolidate_pieces(task, stem)
                .with_context(|| format!("storing {stem} whole"))?;
            if let Some(on_change) = self.on_change.lock().as_ref() {
                on_change();
            }
        }
        Ok(())
    }

    /// The lock a conversion holds while it writes into `track_dir`.
    pub fn converting(&self, track_dir: &Path) -> Arc<Mutex<()>> {
        let mut converting = self.converting.lock();
        converting.retain(|_, lock| Arc::strong_count(lock) > 1);
        Arc::clone(converting.entry(track_dir.to_owned()).or_default())
    }

    /// Store `stem` as one file built from its pieces, and delete the pieces.
    fn consolidate_pieces(&self, task: &Consolidation, stem: &str) -> Result<()> {
        let extension = task.format.extension();
        if crate::whole::find(&task.dir, stem, extension).is_some() {
            return Ok(());
        }
        let channels = task.info.channels;
        let mut data = vec![Vec::new(); channels];
        for (k, len) in task.info.lengths(task.rate).into_iter().enumerate() {
            let path = task.dir.join(stem).join(file(k, extension));
            let bytes =
                std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            let audio = decode(bytes, task.format, channels, task.rate)
                .with_context(|| format!("decoding {}", path.display()))?;
            ensure!(
                audio.frames() == len,
                "{} holds {} samples, not {len}",
                path.display(),
                audio.frames()
            );
            for (dst, src) in data.iter_mut().zip(audio.data) {
                dst.extend(src);
            }
        }
        let audio = Audio::new(data, task.rate);
        let bytes = stemd_core::encode_stem(&audio, task.format, 1.0)?;
        let clipped =
            (!task.format.has_headroom()).then(|| crate::whole::clipped(&audio, 1.0, true));
        self.consolidate(task, stem, &bytes, audio.frames(), clipped)?;
        tracing::info!("stored {stem} whole in {}", task.dir.display());
        Ok(())
    }

    /// Record that an upload sent with `key` hashed to `track`; a key of two tracks is ambiguous.
    pub fn record_file(&self, key: &str, track: &str) -> Result<()> {
        let path = self.root.join(FILES).join(key);
        let _held = self.lock.lock();
        let value = match std::fs::read_to_string(&path) {
            Ok(known) if known == track || known == AMBIGUOUS => return Ok(()),
            Ok(_) => {
                tracing::info!("file key {key} names two tracks; lookups by it are refused");
                AMBIGUOUS
            }
            Err(_) => track,
        };
        std::fs::create_dir_all(path.parent().expect("a parent"))
            .with_context(|| format!("creating {}", path.display()))?;
        self.publish_file(&path, value.as_bytes())
    }

    /// The track id recorded for `key`, unless the key is unknown or ambiguous.
    pub fn file_track(&self, key: &str) -> Option<String> {
        let known = std::fs::read_to_string(self.root.join(FILES).join(key)).ok()?;
        (known != AMBIGUOUS).then_some(known)
    }

    /// The type directories under `model_id`.
    pub fn types(&self, model_id: &str) -> Vec<String> {
        std::fs::read_dir(self.root.join(model_id))
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default()
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

    /// Where a verified track's pieces of one type live.
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

    /// Store `bytes`, the whole of `stem`, in `task`'s directory and delete the stem's pieces.
    pub fn consolidate(
        &self,
        task: &Consolidation,
        stem: &str,
        bytes: &[u8],
        frames: usize,
        clipped: Option<usize>,
    ) -> Result<()> {
        let (track_dir, format) = (task.dir.as_path(), task.format);
        let extension = format.extension();
        let expected = task.info.frames_at(task.rate);
        ensure!(
            frames == expected,
            "{stem} holds {frames} samples, not the track's {expected}"
        );
        if crate::whole::find(track_dir, stem, extension).is_some() {
            return Ok(());
        }
        // Described and written to staged files outside the lock, which separation also takes.
        let meta = crate::whole::describe(bytes, format, frames, clipped)?;
        let meta = serde_json::to_vec(&meta)?;
        crate::disk::ensure_room(&self.root)?;
        std::fs::create_dir_all(track_dir)
            .with_context(|| format!("creating {}", track_dir.display()))?;
        let file_path = crate::whole::file(track_dir, stem, extension);
        let meta_path = crate::whole::meta_file(track_dir, stem);
        let staged_file = self.stage(&file_path, bytes)?;
        let staged_meta = match self.stage(&meta_path, &meta) {
            Ok(path) => path,
            Err(err) => {
                let _ = std::fs::remove_file(&staged_file);
                return Err(err);
            }
        };
        let held = self.lock.lock();
        if crate::whole::find(track_dir, stem, extension).is_some() {
            let _ = std::fs::remove_file(&staged_file);
            let _ = std::fs::remove_file(&staged_meta);
            return Ok(());
        }
        // The description last: a whole file counts once both are in place.
        std::fs::rename(&staged_file, &file_path)
            .with_context(|| format!("publishing {}", file_path.display()))?;
        std::fs::rename(&staged_meta, &meta_path)
            .with_context(|| format!("publishing {}", meta_path.display()))?;
        let pieces_dir = track_dir.join(stem);
        let freed: u64 = files(&pieces_dir).iter().map(|(_, size, _)| size).sum();
        if pieces_dir.is_dir() {
            std::fs::remove_dir_all(&pieces_dir)
                .with_context(|| format!("removing {}", pieces_dir.display()))?;
        }
        let added = (bytes.len() + meta.len()) as u64;
        self.bytes.fetch_add(added, Ordering::Relaxed);
        self.bytes
            .fetch_sub(freed.min(self.bytes()), Ordering::Relaxed);
        Self::touch(track_dir);
        drop(held);
        self.evict();
        Ok(())
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

    /// The directory the pieces live in.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Write a piece file.
    fn write_in(&self, dir: &Path, k: usize, extension: &str, bytes: &[u8]) -> Result<()> {
        crate::disk::ensure_room(&self.root)?;
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        self.publish_file(&dir.join(file(k, extension)), bytes)?;
        self.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Delete track directories until the cache is under budget: partial ones
    /// first, oldest first, then whole ones, least recently used first.
    /// Returns the tracks and bytes deleted.
    pub fn evict(&self) -> (usize, u64) {
        let budget = crate::disk::budget(self.rules().max_bytes, self.bytes(), &self.root);
        if self.bytes() <= budget {
            return (0, 0);
        }
        let _held = self.lock.lock();
        let in_use = self.in_use();
        let mut tracks = track_dirs(&self.root);
        tracks.sort_by_key(|t| (t.whole, t.used));
        let target = budget / 10 * 9;
        let (mut freed, mut kept, mut deleted) = (0u64, 0usize, 0usize);
        for track in tracks {
            if self.bytes().saturating_sub(freed) <= target {
                break;
            }
            if in_use.blocks(&track) {
                kept += 1;
                continue;
            }
            if std::fs::remove_dir_all(&track.dir).is_ok() {
                freed += track.bytes;
                deleted += 1;
            }
        }
        self.bytes
            .fetch_sub(freed.min(self.bytes()), Ordering::Relaxed);
        if deleted > 0 {
            *self.counted.lock() = None;
        }
        tracing::info!("cache over budget, deleted {:.0} MB", freed as f64 / 1e6);
        if self.bytes() > target && kept > 0 {
            tracing::warn!("cache still over budget: {kept} tracks in use were kept");
        }
        (deleted, freed)
    }
}

impl Cache {
    /// Write `bytes` to a `.part` file of its own beside `path`, and return it.
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

/// The tracks open sessions play and the directories job results hold.
struct InUseNow {
    tracks: HashSet<String>,
    dirs: HashSet<PathBuf>,
}

impl InUseNow {
    fn blocks(&self, track: &TrackDir) -> bool {
        self.tracks.contains(&track.id) || self.dirs.contains(&track.dir)
    }
}

/// One track's stems of one type.
struct TrackDir {
    dir: PathBuf,
    id: String,
    bytes: u64,
    used: SystemTime,
    /// Some stem is stored and every one is whole: no piece directory is left.
    whole: bool,
}

/// How many `<model_id>/<type>/<track>` directories `root` holds.
fn count_tracks(root: &Path) -> usize {
    let subdirs = |dir: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
                    .map(|e| e.path())
                    .collect()
            })
            .unwrap_or_default()
    };
    subdirs(root)
        .iter()
        .filter(|model| model.file_name().is_none_or(|n| n != FILES))
        .flat_map(|model| subdirs(model))
        .map(|kind| subdirs(&kind).len())
        .sum()
}

/// Every `<model_id>/<type>/<track>` directory under `root`. A directory without
/// a `.used` file counts as used when its newest file was written.
fn track_dirs(root: &Path) -> Vec<TrackDir> {
    let subdirs = |dir: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
                    .map(|e| e.path())
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut out = Vec::new();
    for model in subdirs(root) {
        if model.file_name().is_some_and(|n| n == FILES) {
            continue;
        }
        for kind in subdirs(&model) {
            for dir in subdirs(&kind) {
                let all = files(&dir);
                let stem_file = |path: &Path| {
                    path.file_name()
                        .is_some_and(|n| ![INFO, USED, SEPARATION].contains(&&*n.to_string_lossy()))
                };
                let whole = subdirs(&dir).is_empty() && all.iter().any(|(p, _, _)| stem_file(p));
                let marker = std::fs::metadata(dir.join(USED)).and_then(|m| m.modified());
                let used = marker.unwrap_or_else(|_| {
                    all.iter()
                        .map(|(_, _, t)| *t)
                        .max()
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                });
                out.push(TrackDir {
                    id: dir
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    bytes: all.iter().map(|(_, size, _)| size).sum(),
                    used,
                    whole,
                    dir,
                });
            }
        }
    }
    out
}

/// Decode a stored piece to the samples `encode_stem(_, format, 1.0)` turns back
/// into the same bytes: 16-bit samples as `q / 32767`, `f32le` as stored.
pub fn decode(bytes: Vec<u8>, format: StemFormat, channels: usize, rate: u32) -> Result<Audio> {
    let planar = |samples: Vec<f32>| -> Result<Audio> {
        anyhow::ensure!(
            channels > 0 && samples.len().is_multiple_of(channels),
            "{} samples do not divide into {channels} channels",
            samples.len()
        );
        let mut data = vec![Vec::with_capacity(samples.len() / channels); channels];
        for (i, v) in samples.into_iter().enumerate() {
            data[i % channels].push(v);
        }
        Ok(Audio::new(data, rate))
    };
    let s16 = |raw: &[u8]| -> Vec<f32> {
        raw.chunks_exact(2)
            .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / S16_SCALE)
            .collect()
    };
    match format {
        StemFormat::Pcm32 => planar(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ),
        StemFormat::Pcm16 => planar(s16(&bytes)),
        // The 44-byte header `encode_stem` writes.
        StemFormat::Wav => planar(s16(bytes
            .get(44..)
            .context("a wav piece without its header")?)),
        StemFormat::Flac => {
            // Symphonia reads `q` as `q / 32768`, which is exact in f32.
            let decoded = stemd_audio::decode_flac(bytes)?;
            anyhow::ensure!(
                decoded.data.len() == channels,
                "a flac piece with {} channels, not {channels}",
                decoded.data.len()
            );
            let data = decoded
                .data
                .into_iter()
                .map(|c| {
                    c.into_iter()
                        .map(|v| (v * 32768.0).round() / S16_SCALE)
                        .collect()
                })
                .collect();
            Ok(Audio::new(data, rate))
        }
        StemFormat::Mp3 => anyhow::bail!("mp3 is never stored as pieces"),
    }
}

/// The multiplier `encode_stem` quantises 16-bit samples with.
const S16_SCALE: f32 = 32767.0;

/// Every part a piece directory holds: the parts, and the derived one.
///
/// Written whether or not a session asked for the derived part, so one type
/// directory serves every session of that type.
pub fn stems_of(output: &Output) -> Vec<&'static str> {
    output.parts.names().chain(output.parts.derived()).collect()
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

/// Where the cache lives when no directory is configured.
///
/// ```text
/// macOS    ~/Library/Caches/stemd
/// Linux    ~/.cache/stemd              ($XDG_CACHE_HOME)
/// Windows  %LOCALAPPDATA%\stemd\cache
/// ```
///
/// The cache root, not the data root: every byte here is derived. Model artefacts
/// live under [`crate::models::support_dir`].
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

/// The cache root `args` names, or [`default_dir`].
pub fn root(args: &crate::cli::Args) -> Result<PathBuf> {
    match args.cache_dir.clone() {
        Some(dir) => Ok(dir),
        None => default_dir(),
    }
}

/// Lock `root` for this process, or fail if another stemd holds it, whatever
/// port that one serves on. The system releases the lock when the process ends.
pub fn lock(root: &Path) -> Result<std::fs::File> {
    std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
    let path = root.join(LOCK_FILE);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => anyhow::bail!(
            "another stemd is already running with its cache in {}",
            root.display()
        ),
        Err(std::fs::TryLockError::Error(err)) => {
            Err(err).with_context(|| format!("locking {}", path.display()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stemd_core::{OutputRate, Parts};

    #[test]
    fn a_second_lock_on_the_same_root_fails() {
        let root = std::env::temp_dir().join(format!("stemd-lock-test-{}", std::process::id()));
        let held = lock(&root).unwrap();
        let err = lock(&root).unwrap_err();
        assert!(err.to_string().contains("already running"), "{err:#}");
        assert!(root.join(LOCK_FILE).exists());
        drop(held);
        lock(&root).unwrap();
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_file_key_seen_with_two_tracks_names_neither() {
        let root = std::env::temp_dir().join(format!("stemd-files-test-{}", std::process::id()));
        let cache = Cache::open(root.clone(), 1 << 20).expect("open");
        let (key, a, b) = ("aa".repeat(12), "11".repeat(12), "22".repeat(12));
        assert_eq!(cache.file_track(&key), None);
        cache.record_file(&key, &a).expect("recorded");
        cache.record_file(&key, &a).expect("recorded again");
        assert_eq!(cache.file_track(&key), Some(a.clone()));
        cache.record_file(&key, &b).expect("recorded");
        assert_eq!(cache.file_track(&key), None, "ambiguous");
        cache.record_file(&key, &a).expect("recorded");
        assert_eq!(cache.file_track(&key), None, "stays ambiguous");
        let reopened = Cache::open(root.clone(), 1 << 20).expect("reopen");
        assert_eq!(reopened.file_track(&key), None);
        assert!(
            track_dirs(&root).is_empty(),
            "aliases are not track directories"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn type_dir_names_every_setting_that_changes_the_bytes() {
        let mut output = Output {
            format: StemFormat::Flac,
            rate: OutputRate::default(),
            derived: false,
            dsp: stemd_core::DspMode::General.into(),
            parts: Parts::DEFAULT,
        };
        assert_eq!(
            type_dir(&output, std::f32::consts::FRAC_1_SQRT_2),
            "flac-44100-harmonics.vocals-g0.70710677"
        );
        output.dsp = stemd_core::DspMode::Mode1.into();
        output.rate = OutputRate::Hz96000;
        assert_eq!(
            type_dir(&output, 1.0),
            "flac-96000-dsp1-harmonics.vocals-g1"
        );
        output.derived = true;
        assert_eq!(
            type_dir(&output, 1.0),
            "flac-96000-dsp1-harmonics.vocals-g1",
            "the derived part is one more file, not another type"
        );

        let body = crate::filters::encode(44_100, 96_000, &[1.0; 320]);
        let id = crate::filters::id_of(&body);
        output.dsp = crate::filters::Dsp::Fir(std::sync::Arc::new(crate::filters::Fir {
            id: id.clone(),
            poly: crate::filters::parse(&body).unwrap(),
        }));
        assert_eq!(
            type_dir(&output, 1.0),
            format!("flac-96000-fir{id}-harmonics.vocals-g1")
        );
    }

    fn root() -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "stemd-pieces-unit-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn a_reopened_store_keeps_its_pieces_and_drops_partial_writes() {
        let root = root();
        let cache = Cache::open(root.clone(), 1 << 20).expect("open");
        let dir = cache.verified_dir("model", "type", "track");
        cache
            .write_verified(&dir, "vocals", 0, "flac", &[1; 100])
            .expect("write");
        std::fs::write(dir.join("vocals").join("0001.flac.part"), [0; 50]).expect("partial");
        drop(cache);

        let reopened = Cache::open(root, 1 << 20).expect("reopen");
        assert_eq!(reopened.bytes(), 100);
        assert!(Cache::lookup(&dir, "vocals", 0, "flac").is_some());
        assert!(!dir.join("vocals").join("0001.flac.part").exists());
    }

    /// Two pieces of 300 bytes for each of `tracks`, oldest first.
    fn fill(cache: &Cache, tracks: &[&str]) {
        for track in tracks {
            let dir = cache.verified_dir("model", "type", track);
            for stem in ["vocals", "harmonics"] {
                cache
                    .write_verified(&dir, stem, 0, "flac", &[0; 300])
                    .expect("write");
            }
            // Distinct modification times.
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    fn present(cache: &Cache, track: &str) -> Vec<&'static str> {
        ["vocals", "harmonics"]
            .into_iter()
            .filter(|stem| {
                cache
                    .verified_dir("model", "type", track)
                    .join(stem)
                    .join(file(0, "flac"))
                    .is_file()
            })
            .collect()
    }

    #[test]
    fn over_the_budget_the_least_recently_used_track_is_deleted() {
        let cache = Cache::open(root(), 1300).expect("open");
        fill(&cache, &["a", "b"]);
        // Reading `a` makes `b` the least recently used.
        assert!(
            Cache::lookup(
                &cache.verified_dir("model", "type", "a"),
                "vocals",
                0,
                "flac"
            )
            .is_some()
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
        fill(&cache, &["c"]);
        assert!(cache.bytes() <= 1300, "{} bytes kept", cache.bytes());
        assert_eq!(present(&cache, "b"), Vec::<&str>::new());
        assert_eq!(present(&cache, "a"), vec!["vocals", "harmonics"]);
        assert_eq!(present(&cache, "c"), vec!["vocals", "harmonics"]);
    }

    #[test]
    fn eviction_skips_the_tracks_open_sessions_play() {
        let cache = Cache::open(root(), 1300).expect("open");
        cache.set_in_use(Box::new(|| HashSet::from(["a".to_owned()])));
        fill(&cache, &["a", "b", "c"]);
        assert_eq!(present(&cache, "a"), vec!["vocals", "harmonics"]);
        assert_eq!(present(&cache, "b"), Vec::<&str>::new());
    }

    /// A track whose two stems are stored whole, 300 bytes each.
    fn store_whole(cache: &Cache, track: &str) {
        let task = Consolidation::at_44100(
            cache.verified_dir("model", "type", track),
            StemFormat::Pcm16,
            150,
        );
        for stem in ["vocals", "harmonics"] {
            cache
                .consolidate(&task, stem, &[0; 300], 150, Some(0))
                .expect("store");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    fn exists(cache: &Cache, track: &str) -> bool {
        cache.verified_dir("model", "type", track).is_dir()
    }

    fn rules(max_bytes: u64, partial: u64, whole: u64) -> Rules {
        Rules {
            max_bytes,
            partial_max_age: std::time::Duration::from_millis(partial),
            whole_max_age: std::time::Duration::from_millis(whole),
            unfetched_ttl: std::time::Duration::from_secs(60),
        }
    }

    #[test]
    fn a_track_unused_past_its_age_is_deleted_partial_and_whole_alike() {
        let cache = Cache::open_with(root(), rules(1 << 20, 30, 3_600_000)).expect("open");
        fill(&cache, &["partial"]);
        store_whole(&cache, "whole");
        std::thread::sleep(std::time::Duration::from_millis(40));
        cache.clean();
        assert!(!exists(&cache, "partial"), "a partial track past a day");
        assert!(exists(&cache, "whole"), "a whole track inside 30 days");

        let cache = Cache::open_with(root(), rules(1 << 20, 3_600_000, 30)).expect("open");
        store_whole(&cache, "whole");
        std::thread::sleep(std::time::Duration::from_millis(40));
        cache.clean();
        assert!(!exists(&cache, "whole"));
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn over_the_budget_partial_tracks_go_before_whole_ones() {
        let cache = Cache::open(root(), 1 << 20).expect("open");
        store_whole(&cache, "old-whole");
        fill(&cache, &["new-partial"]);
        store_whole(&cache, "newest-whole");
        // Just over: deleting the partial track is enough.
        let budget = cache.bytes() - 100;
        let cache = Cache::open(cache.root().to_owned(), budget).expect("reopen");
        assert!(!exists(&cache, "new-partial"), "partial goes first");
        assert!(exists(&cache, "old-whole"));
        assert!(exists(&cache, "newest-whole"));
    }

    #[test]
    fn a_job_result_is_kept_until_every_stem_is_fetched() {
        let cache = Cache::open(root(), 1300).expect("open");
        store_whole(&cache, "held");
        let dir = cache.verified_dir("model", "type", "held");
        let output = Output {
            format: StemFormat::Pcm16,
            rate: OutputRate::default(),
            derived: false,
            dsp: stemd_core::DspMode::General.into(),
            parts: Parts::parse("vocals,harmonics").unwrap(),
        };
        let entry = Arc::new(Entry::stored(&dir, &output, 1.0, 2, None).expect("whole"));
        cache.hold(&entry);
        store_whole(&cache, "next");
        assert!(exists(&cache, "held"), "kept while unfetched");
        entry.mark_fetched("harmonics");
        entry.mark_fetched("vocals");
        store_whole(&cache, "last");
        assert!(!exists(&cache, "held"), "fetched, so it goes first");
    }

    #[test]
    fn a_reopened_cache_drops_the_old_job_entries_and_keeps_tracks() {
        let root = root();
        let cache = Cache::open(root.clone(), 1 << 20).expect("open");
        store_whole(&cache, "kept");
        let old = root.join("ab".repeat(32));
        std::fs::create_dir_all(&old).expect("an old entry");
        std::fs::write(old.join("vocals.flac"), [0; 10]).expect("an old stem");
        std::fs::write(root.join(LOCK_FILE), []).expect("a lock file");
        drop(cache);
        let cache = Cache::open(root.clone(), 1 << 20).expect("reopen");
        assert!(!old.exists());
        assert!(exists(&cache, "kept"));
        assert_eq!(cache.stats().tracks, 1);

        cache.wipe();
        assert!(!exists(&cache, "kept"));
        assert!(root.join(LOCK_FILE).exists(), "the lock stays");
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn emptying_the_cache_keeps_the_tracks_sessions_play() {
        let cache = Cache::open(root(), 1 << 20).expect("open");
        cache.set_in_use(Box::new(|| HashSet::from(["playing".to_owned()])));
        fill(&cache, &["playing", "idle"]);
        assert_eq!(cache.clear().0, 1);
        assert!(exists(&cache, "playing"));
        assert!(!exists(&cache, "idle"));
    }

    #[test]
    fn emptying_the_cache_frees_its_bytes_and_file_keys_and_keeps_held_results() {
        let cache = Cache::open(root(), 1 << 20).expect("open");
        store_whole(&cache, "held");
        store_whole(&cache, "idle");
        cache.record_file("key", "idle").expect("recorded");
        let output = Output {
            format: StemFormat::Pcm16,
            rate: OutputRate::default(),
            derived: false,
            dsp: stemd_core::DspMode::General.into(),
            parts: Parts::parse("vocals,harmonics").unwrap(),
        };
        let dir = cache.verified_dir("model", "type", "held");
        let entry = Arc::new(Entry::stored(&dir, &output, 1.0, 2, None).expect("whole"));
        cache.hold(&entry);
        let before = cache.bytes();
        let (tracks, freed) = cache.clear();
        assert_eq!(tracks, 1);
        assert!(exists(&cache, "held"));
        assert!(!exists(&cache, "idle"));
        assert_eq!(cache.bytes(), before - freed);
        assert_eq!(cache.file_track("key"), None);
    }

    #[test]
    fn the_track_count_drops_as_soon_as_the_cache_is_emptied() {
        let cache = Cache::open(root(), 1 << 20).expect("open");
        fill(&cache, &["a", "b"]);
        assert_eq!(cache.stats().tracks, 2);
        cache.clear();
        assert_eq!(cache.stats().tracks, 0);
        fill(&cache, &["c"]);
        cache.wipe();
        assert_eq!(cache.stats().tracks, 0);
    }

    #[test]
    fn a_track_without_a_used_file_counts_from_its_newest_piece() {
        let root = root();
        let cache = Cache::open(root.clone(), 1 << 20).expect("open");
        fill(&cache, &["a", "b"]);
        for track in ["a", "b"] {
            std::fs::remove_file(cache.verified_dir("model", "type", track).join(USED))
                .expect("a used file");
        }
        drop(cache);
        let cache = Cache::open(root, 1300).expect("reopen");
        fill(&cache, &["c"]);
        assert_eq!(present(&cache, "a"), Vec::<&str>::new());
        assert_eq!(present(&cache, "b"), vec!["vocals", "harmonics"]);
    }
}
