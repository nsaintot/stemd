//! Locating, verifying and fetching the model artefacts.
//!
//! The bundle ships without weights: they are ~170 MB, they are not source, and
//! `models/` is gitignored, so a fresh clone would otherwise build a binary with
//! no model to load and no way to make one. A pinned URL fixes both.
//!
//! # Where they land
//!
//! Not next to the executable. On macOS the bundle's contents are covered by the
//! code signature and writing into them invalidates it; on Windows and Linux the
//! install directory is not writable by the user who runs the program. So they
//! land in the per-user data directory: see [`support_dir`] for where that is
//! on each platform.
//!
//! # Layering
//!
//! ```text
//! preset     the catalogue: which artefacts exist and their pinned digests
//! integrity  hashing what is on disk against those digests
//! download   fetching what is missing
//! ```

mod download;
mod integrity;
mod preset;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use download::fetch;
pub use integrity::verify;
pub use preset::{DEFAULT_PRESET, ModelSource, Preset, RETIRED, WEIGHTS_EXTENSION};

// One pinned file at a time, and the type describing one. Only `cuda` wants
// either: everything else fetches a whole `ModelSource`. Gated rather than
// always exported, so the platform that cannot install CUDA does not carry two
// names nothing there can use.
#[cfg(windows)]
pub use download::fetch_one as fetch_file;
#[cfg(windows)]
pub use preset::RemoteFile;

/// Bytes hashed or copied per read. Large enough that a 675 MB artefact is not
/// resident, small enough to stay in cache.
const READ_CHUNK: usize = 1 << 20;

/// The application's data directory, created if absent.
///
/// ```text
/// macOS    ~/Library/Application Support/stemd
/// Linux    ~/.local/share/stemd             ($XDG_DATA_HOME)
/// Windows  %LOCALAPPDATA%\stemd\data
/// ```
///
/// Everything the server keeps that a user would not want swept: the model
/// artefacts and the settings file. Separated stems are the other half, and they
/// live under the cache root instead, because losing them costs a separation and
/// nothing else.
///
/// Local rather than roaming on Windows: `dirs::data_local_dir`, not
/// `dirs::data_dir`. A roaming profile is copied to and from the domain
/// controller at every logon, and 170 MB of weights that any machine can
/// re-download is precisely what must not be in one. The extra `data` component
/// is the other half of the split described on [`crate::cache::default_dir`]:
/// local app data is also where the cache lands, and the cache root is emptied
/// at every start.
pub fn support_dir() -> Result<PathBuf> {
    let dir = dirs::data_local_dir()
        .context("no data directory for this platform")?
        .join("stemd");
    let dir = if cfg!(windows) { dir.join("data") } else { dir };
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// `support_dir()/models`, created if absent.
pub fn cache_dir() -> Result<PathBuf> {
    let dir = support_dir()?.join("models");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// Where a preset's artefact is, if it is anywhere.
///
/// Mirrors the startup search order, so the window and the command line cannot
/// disagree about whether a switch needs the network.
pub fn locate(dirs: &[PathBuf], preset: Preset) -> Option<PathBuf> {
    dirs.iter()
        .find(|d| is_complete(d, preset.source()))
        .cloned()
}

/// True when every file of `source` is present in `dir`.
///
/// Presence only: see [`mismatched`] for whether the contents are right.
pub fn is_complete(dir: &Path, source: &ModelSource) -> bool {
    source.files.iter().all(|f| dir.join(f.name).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completeness_needs_every_file() {
        let dir = std::env::temp_dir().join(format!("stemd-models-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        // Named from the preset rather than spelled out, so retiring or
        // renaming one moves this test with it instead of breaking it.
        let source = Preset::Fast.source();

        assert!(!is_complete(&dir, source));
        for file in source.files {
            std::fs::write(dir.join(file.name), b"x").unwrap();
        }
        assert!(is_complete(&dir, source));

        // A directory holding another preset's artefact is not this one's.
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        for file in Preset::Balanced.source().files {
            std::fs::write(dir.join(file.name), b"x").unwrap();
        }
        assert!(!is_complete(&dir, source));

        std::fs::remove_dir_all(&dir).ok();
    }
}
