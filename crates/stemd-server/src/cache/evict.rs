//! Applies the cache's age rules and byte budget, and empties the cache.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Weak;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime};

use super::{Cache, FILES, INFO, LOCK_FILE, SEPARATION, Stats, USED, files, subdirs};

impl Cache {
    /// The track directories that jobs are writing or that job results hold now.
    pub(super) fn held_dirs(&self) -> HashSet<PathBuf> {
        let mut dirs = self.writing.lock().clone();
        let mut holds = self.holds.lock();
        holds.retain(|held| held.strong_count() > 0);
        dirs.extend(
            holds
                .iter()
                .filter_map(Weak::upgrade)
                .filter(|entry| entry.held(self.rules().unfetched_ttl))
                .map(|entry| entry.dir().to_owned()),
        );
        dirs
    }

    /// The track count (recounted at most once a second), the bytes held and the byte cap.
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

    /// Delete everything in the root except the lock file.
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
    pub(super) fn recount(&self) {
        let bytes = files(&self.root).iter().map(|(_, size, _)| size).sum();
        self.bytes.store(bytes, Ordering::Relaxed);
        *self.counted.lock() = None;
    }

    /// The tracks and directories in use now, which deletion must skip.
    pub(super) fn in_use(&self) -> InUseNow {
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
        self.sub_bytes(freed);
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
        self.sub_bytes(freed);
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

/// The tracks that open sessions play and the directories that job results hold.
pub(super) struct InUseNow {
    tracks: HashSet<String>,
    dirs: HashSet<PathBuf>,
}

impl InUseNow {
    pub(super) fn blocks(&self, track: &TrackDir) -> bool {
        self.tracks.contains(&track.id) || self.dirs.contains(&track.dir)
    }
}

/// One track's stems of one type.
pub(super) struct TrackDir {
    dir: PathBuf,
    id: String,
    bytes: u64,
    used: SystemTime,
    /// At least one stem is stored and every stored stem is whole, so no piece directory is left.
    whole: bool,
}

/// How many `<model_id>/<type>/<track>` directories `root` holds.
pub(super) fn count_tracks(root: &Path) -> usize {
    subdirs(root)
        .iter()
        .filter(|model| model.file_name().is_none_or(|n| n != FILES))
        .flat_map(|model| subdirs(model))
        .map(|kind| subdirs(&kind).len())
        .sum()
}

/// Every `<model_id>/<type>/<track>` directory under `root`. A directory without a `.used` file
/// counts as last used when its newest file was written.
pub(super) fn track_dirs(root: &Path) -> Vec<TrackDir> {
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
