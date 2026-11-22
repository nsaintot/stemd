//! Storing a track's stems whole once every piece is stored, and deleting the pieces.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use parking_lot::Mutex;
use stemd_core::{Audio, StemFormat};

use super::{Cache, TrackInfo, file, files};
use crate::output::Output;

/// How often a consolidation waiting on its [`Gate`] checks again.
const GATE_POLL: std::time::Duration = std::time::Duration::from_millis(250);
/// How long a track directory that failed to be stored whole waits before another try.
const RETRY_AFTER: std::time::Duration = std::time::Duration::from_secs(600);

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

impl Consolidation {
    /// The track directory `dir` of `output`, holding the track `info` describes.
    pub fn new(dir: PathBuf, output: &Output, info: TrackInfo) -> Self {
        Self {
            dir,
            format: output.format,
            rate: output.rate.hz(),
            info,
            stems: output.stored_parts(),
        }
    }
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

impl Cache {
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
    pub(super) fn consolidate_queued(&self) {
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
    pub(super) fn consolidate_task(&self, task: &Consolidation) -> Result<()> {
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
    pub(super) fn consolidate_pieces(&self, task: &Consolidation, stem: &str) -> Result<()> {
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
            let audio = crate::whole::decode(bytes, task.format, channels, task.rate)
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
        self.sub_bytes(freed);
        Self::touch(track_dir);
        drop(held);
        self.evict();
        Ok(())
    }

    pub(super) fn gate_open(&self) -> bool {
        self.gate.lock().as_ref().is_none_or(|open| open())
    }
}
