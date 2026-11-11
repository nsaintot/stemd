//! Separated pieces on disk, each one encoded and ready to send as it is.
//!
//! ```text
//! <root>/<model_id>/<type>/<track>/<stem>/<k>.<ext>
//! ```
//!
//! | Level   | Example                                        |
//! |---------|------------------------------------------------|
//! | `model_id` | `htdemucs-22c7f2c9`, as in `/v1/health`     |
//! | `type`  | `flac-44100-harmonics.vocals-g0.70710677`, plus `-dsp1` when not 0 |
//! | `track` | the 24-hex track id, see [`crate::sessions::track_id`] |
//! | `k`     | piece index, four digits                       |
//!
//! A piece is written once its track's whole upload has been hashed, so the
//! track id in the path is always the server's own. The root survives restarts,
//! since a piece's path names everything its bytes depend on. Past the budget,
//! the least recently used files go first.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use stemd_core::DspMode;

use crate::cache::Output;
use crate::sessions::{RESAMPLE_CONTEXT, Sessions, Track};

/// The directory name for one output at one gain.
pub fn type_dir(output: &Output, gain: f32) -> String {
    let dsp = if output.dsp == DspMode::General {
        String::new()
    } else {
        format!("-dsp{}", output.dsp)
    };
    let parts: Vec<&str> = output.parts.names().collect();
    format!(
        "{}-{}{dsp}-{}-g{gain}",
        output.format,
        output.rate,
        parts.join(".")
    )
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
        }))
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Where a verified track's pieces of one type live.
    pub fn verified_dir(&self, model_id: &str, type_dir: &str, track: &str) -> PathBuf {
        self.root.join(model_id).join(type_dir).join(track)
    }

    /// The file for piece `k` of `stem`, when it exists.
    pub fn lookup(
        &self,
        model_id: &str,
        type_dir: &str,
        track: &str,
        stem: &str,
        k: usize,
        extension: &str,
    ) -> Option<PathBuf> {
        let path = self
            .verified_dir(model_id, type_dir, track)
            .join(stem)
            .join(file(k, extension));
        path.is_file().then(|| {
            // Marks it recently used for eviction.
            if let Ok(f) = std::fs::File::options().append(true).open(&path) {
                let _ = f.set_modified(SystemTime::now());
            }
            path
        })
    }

    /// The piece indices below `count` stored for every one of `stems`.
    pub fn cached(
        &self,
        model_id: &str,
        type_dir: &str,
        track: &str,
        stems: &[&str],
        extension: &str,
        count: usize,
    ) -> Vec<usize> {
        let dir = self.verified_dir(model_id, type_dir, track);
        if !dir.is_dir() {
            return Vec::new();
        }
        (0..count)
            .filter(|&k| {
                stems
                    .iter()
                    .all(|stem| dir.join(stem).join(file(k, extension)).is_file())
            })
            .collect()
    }

    /// Write piece `k` of `stem` under `track_dir`, unless it is there already.
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
        if dir.join(file(k, extension)).is_file() {
            return Ok(());
        }
        self.write_in(&dir, k, extension, bytes)?;
        drop(held);
        self.evict();
        Ok(())
    }

    /// The directory the pieces live in.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Write a piece file through a `.part` and a rename, so a reader never
    /// sees half of one.
    fn write_in(&self, dir: &Path, k: usize, extension: &str, bytes: &[u8]) -> Result<()> {
        crate::disk::ensure_room(&self.root)?;
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(file(k, extension));
        let partial = path.with_extension(format!("{extension}.part"));
        if let Err(err) = std::fs::write(&partial, bytes) {
            let _ = std::fs::remove_file(&partial);
            return Err(err).with_context(|| format!("writing {}", partial.display()));
        }
        std::fs::rename(&partial, &path)
            .with_context(|| format!("publishing {}", path.display()))?;
        self.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Encode and write the pieces near `touched` that are complete in
    /// `track`, for every output its sessions ask for.
    ///
    /// Does nothing until the track's whole upload has been hashed.
    pub fn publish(&self, sessions: &Sessions, track: &Track, touched: std::ops::Range<usize>) {
        let Some(id) = track.verified() else {
            return;
        };
        // Pieces are a cache, so a nearly full disk skips them and the job goes on.
        if crate::disk::low(&self.root) {
            tracing::debug!(track = %id, "disk nearly full, not writing pieces");
            return;
        }
        let stride = track.spec.grid.stride().unsigned_abs() as usize;
        let count = track.spec.grid.len();
        let first = touched.start.saturating_sub(RESAMPLE_CONTEXT) / stride;
        let last = (touched.end + RESAMPLE_CONTEXT).div_ceil(stride).min(count);

        let mut outputs: Vec<(String, Output, f32)> = Vec::new();
        for session in sessions.of_track(track) {
            let name = type_dir(&session.output, session.gain);
            if !outputs.iter().any(|(n, _, _)| *n == name) {
                let mut output = session.output;
                output.derived = true;
                outputs.push((name, output, session.gain));
            }
        }
        for (name, output, gain) in outputs {
            let converts = output.rate.hz() != track.spec.sample_rate;
            let dir = self.verified_dir(&track.model_id, &name, &id);
            for k in first..last {
                if !track.piece_writable(k, &name, converts) {
                    continue;
                }
                let piece = track.piece(k);
                let written = stems_of(&output).into_iter().try_for_each(|stem| {
                    let excerpt = track.read(&output, stem, piece.start, piece.end)?;
                    let bytes = stemd_core::encode_stem(&excerpt.audio, output.format, gain)?;
                    self.write_verified(&dir, stem, k, output.format.extension(), &bytes)
                });
                match written {
                    Ok(()) => track.mark_written(k, &name),
                    Err(err) => tracing::warn!(track = %id, "piece {k}: {err:#}"),
                }
            }
        }
    }

    /// Delete the least recently used files until the store is under budget:
    /// its cap, or less when the disk is short of [`crate::disk::FLOOR`].
    pub fn evict(&self) {
        let budget = crate::disk::budget(self.max_bytes, self.bytes(), &self.root);
        if self.bytes() <= budget {
            return;
        }
        let _held = self.lock.lock();
        let mut all = files(&self.root);
        all.sort_by_key(|(_, _, used)| *used);
        let target = budget / 10 * 9;
        let mut freed = 0u64;
        for (path, size, _) in all {
            if self.bytes().saturating_sub(freed) <= target {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                freed += size;
            }
        }
        self.bytes
            .fetch_sub(freed.min(self.bytes()), Ordering::Relaxed);
        tracing::info!("pieces over budget, evicted {:.0} MB", freed as f64 / 1e6);
    }
}

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
    use stemd_core::{OutputRate, Parts, StemFormat};

    #[test]
    fn type_dir_names_every_setting_that_changes_the_bytes() {
        let mut output = Output {
            format: StemFormat::Flac,
            rate: OutputRate::default(),
            derived: false,
            dsp: DspMode::General,
            parts: Parts::DEFAULT,
        };
        assert_eq!(
            type_dir(&output, std::f32::consts::FRAC_1_SQRT_2),
            "flac-44100-harmonics.vocals-g0.70710677"
        );
        output.dsp = DspMode::Mode1;
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
        assert!(
            reopened
                .lookup("model", "type", "track", "vocals", 0, "flac")
                .is_some()
        );
        assert!(!dir.join("vocals").join("0001.flac.part").exists());
    }

    #[test]
    fn past_the_budget_the_least_recently_used_pieces_go_first() {
        let pieces = Pieces::open(root(), 1000).expect("open");
        let dir = pieces.verified_dir("model", "type", "track");
        for k in 0..4 {
            pieces
                .write_verified(&dir, "vocals", k, "flac", &[0; 300])
                .expect("write");
            // Distinct modification times, oldest first.
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(pieces.bytes() <= 1000, "{} bytes kept", pieces.bytes());
        let kept: Vec<usize> = (0..4)
            .filter(|&k| {
                pieces
                    .lookup("model", "type", "track", "vocals", k, "flac")
                    .is_some()
            })
            .collect();
        assert!(kept.contains(&3), "the newest piece was evicted: {kept:?}");
        assert!(!kept.contains(&0), "the oldest piece was kept: {kept:?}");
    }
}
