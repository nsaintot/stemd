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
//! A piece is written once its track's whole upload has been hashed, so the
//! track id in the path is always the server's own. The root survives restarts,
//! since a piece's path names everything its bytes depend on. Past the budget,
//! the least recently used track directories go first, all their stems together,
//! skipping tracks an open session plays. A `.used` file in each track directory
//! records its last use. Once a stem is stored whole, its pieces are deleted,
//! and ranges are read from the whole file. Once every piece of a track
//! directory is stored, a background thread stores each stem whole.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime};

use anyhow::{Context, Result, ensure};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use stemd_core::{Audio, StemFormat};

use crate::cache::Output;
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

/// A file name for piece `k`.
fn file(k: usize, extension: &str) -> String {
    format!("{k:04}.{extension}")
}

pub struct Pieces {
    root: PathBuf,
    max_bytes: u64,
    bytes: AtomicU64,
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

impl Pieces {
    /// Open `root`, deleting leftover `.part` files.
    pub fn open(root: PathBuf, max_bytes: u64) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        let mut bytes = 0;
        for (path, size, _) in files(&root) {
            if path.extension().is_some_and(|e| e == "part") {
                let _ = std::fs::remove_file(&path);
            } else {
                bytes += size;
            }
        }
        Ok(Arc::new(Self {
            root,
            max_bytes,
            bytes: AtomicU64::new(bytes),
            lock: Mutex::new(()),
            in_use: Mutex::new(None),
            gate: Mutex::new(None),
            on_change: Mutex::new(None),
            consolidating: Mutex::new(HashSet::new()),
            tasks: Mutex::new((VecDeque::new(), false)),
            failed: Mutex::new(HashMap::new()),
            converting: Mutex::new(HashMap::new()),
            staged: AtomicU64::new(0),
        }))
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
            let pieces = Arc::clone(self);
            std::thread::spawn(move || pieces.consolidate_queued());
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
        // Stored samples do not record clipping, so one at full scale counts as clipped.
        let clipped = (!task.format.has_headroom()).then(|| {
            audio
                .data
                .iter()
                .flatten()
                .filter(|v| v.abs() >= 1.0)
                .count()
        });
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

    /// Delete the least recently used track directories until the store is under
    /// budget: its cap, or less when the disk is short of [`crate::disk::FLOOR`].
    pub fn evict(&self) {
        let budget = crate::disk::budget(self.max_bytes, self.bytes(), &self.root);
        if self.bytes() <= budget {
            return;
        }
        let in_use = self.in_use.lock().as_ref().map(|f| f()).unwrap_or_default();
        let _held = self.lock.lock();
        let mut tracks = track_dirs(&self.root);
        tracks.sort_by_key(|t| t.used);
        let target = budget / 10 * 9;
        let (mut freed, mut kept) = (0u64, 0usize);
        for track in tracks {
            if self.bytes().saturating_sub(freed) <= target {
                break;
            }
            if in_use.contains(&track.id) {
                kept += 1;
                continue;
            }
            if std::fs::remove_dir_all(&track.dir).is_ok() {
                freed += track.bytes;
            }
        }
        self.bytes
            .fetch_sub(freed.min(self.bytes()), Ordering::Relaxed);
        tracing::info!("pieces over budget, evicted {:.0} MB", freed as f64 / 1e6);
        if self.bytes() > target && kept > 0 {
            tracing::warn!("pieces still over budget: {kept} tracks in use were kept");
        }
    }
}

impl Pieces {
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

/// One track's pieces of one type.
struct TrackDir {
    dir: PathBuf,
    id: String,
    bytes: u64,
    used: SystemTime,
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
        for kind in subdirs(&model) {
            for dir in subdirs(&kind) {
                let all = files(&dir);
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

#[cfg(test)]
mod tests {
    use super::*;
    use stemd_core::{OutputRate, Parts};

    #[test]
    fn a_file_key_seen_with_two_tracks_names_neither() {
        let root = std::env::temp_dir().join(format!("stemd-files-test-{}", std::process::id()));
        let pieces = Pieces::open(root.clone(), 1 << 20).expect("open");
        let (key, a, b) = ("aa".repeat(12), "11".repeat(12), "22".repeat(12));
        assert_eq!(pieces.file_track(&key), None);
        pieces.record_file(&key, &a).expect("recorded");
        pieces.record_file(&key, &a).expect("recorded again");
        assert_eq!(pieces.file_track(&key), Some(a.clone()));
        pieces.record_file(&key, &b).expect("recorded");
        assert_eq!(pieces.file_track(&key), None, "ambiguous");
        pieces.record_file(&key, &a).expect("recorded");
        assert_eq!(pieces.file_track(&key), None, "stays ambiguous");
        let reopened = Pieces::open(root.clone(), 1 << 20).expect("reopen");
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
        let pieces = Pieces::open(root.clone(), 1 << 20).expect("open");
        let dir = pieces.verified_dir("model", "type", "track");
        pieces
            .write_verified(&dir, "vocals", 0, "flac", &[1; 100])
            .expect("write");
        std::fs::write(dir.join("vocals").join("0001.flac.part"), [0; 50]).expect("partial");
        drop(pieces);

        let reopened = Pieces::open(root, 1 << 20).expect("reopen");
        assert_eq!(reopened.bytes(), 100);
        assert!(Pieces::lookup(&dir, "vocals", 0, "flac").is_some());
        assert!(!dir.join("vocals").join("0001.flac.part").exists());
    }

    /// Two pieces of 300 bytes for each of `tracks`, oldest first.
    fn fill(pieces: &Pieces, tracks: &[&str]) {
        for track in tracks {
            let dir = pieces.verified_dir("model", "type", track);
            for stem in ["vocals", "harmonics"] {
                pieces
                    .write_verified(&dir, stem, 0, "flac", &[0; 300])
                    .expect("write");
            }
            // Distinct modification times.
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    fn present(pieces: &Pieces, track: &str) -> Vec<&'static str> {
        ["vocals", "harmonics"]
            .into_iter()
            .filter(|stem| {
                pieces
                    .verified_dir("model", "type", track)
                    .join(stem)
                    .join(file(0, "flac"))
                    .is_file()
            })
            .collect()
    }

    #[test]
    fn over_the_budget_the_least_recently_used_track_is_deleted() {
        let pieces = Pieces::open(root(), 1300).expect("open");
        fill(&pieces, &["a", "b"]);
        // Reading `a` makes `b` the least recently used.
        assert!(
            Pieces::lookup(
                &pieces.verified_dir("model", "type", "a"),
                "vocals",
                0,
                "flac"
            )
            .is_some()
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
        fill(&pieces, &["c"]);
        assert!(pieces.bytes() <= 1300, "{} bytes kept", pieces.bytes());
        assert_eq!(present(&pieces, "b"), Vec::<&str>::new());
        assert_eq!(present(&pieces, "a"), vec!["vocals", "harmonics"]);
        assert_eq!(present(&pieces, "c"), vec!["vocals", "harmonics"]);
    }

    #[test]
    fn eviction_skips_the_tracks_open_sessions_play() {
        let pieces = Pieces::open(root(), 1300).expect("open");
        pieces.set_in_use(Box::new(|| HashSet::from(["a".to_owned()])));
        fill(&pieces, &["a", "b", "c"]);
        assert_eq!(present(&pieces, "a"), vec!["vocals", "harmonics"]);
        assert_eq!(present(&pieces, "b"), Vec::<&str>::new());
    }

    #[test]
    fn a_track_without_a_used_file_counts_from_its_newest_piece() {
        let root = root();
        let pieces = Pieces::open(root.clone(), 1 << 20).expect("open");
        fill(&pieces, &["a", "b"]);
        for track in ["a", "b"] {
            std::fs::remove_file(pieces.verified_dir("model", "type", track).join(USED))
                .expect("a used file");
        }
        drop(pieces);
        let pieces = Pieces::open(root, 1300).expect("reopen");
        fill(&pieces, &["c"]);
        assert_eq!(present(&pieces, "a"), Vec::<&str>::new());
        assert_eq!(present(&pieces, "b"), vec!["vocals", "harmonics"]);
    }
}
