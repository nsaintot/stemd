//! Locates, verifies and downloads the model artefacts.
//!
//! The bundle ships without weights; each artefact is downloaded from a pinned URL and checked
//! against its digest.
//!
//! # Where they land
//!
//! The weights go in the per-user data directory (see [`support_dir`]), never next to the
//! executable. On macOS the code signature covers the bundle's contents, and writing into them
//! invalidates it; on Windows and Linux the user who runs the program cannot write to the install
//! directory.
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
mod load;
mod preset;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub use download::fetch;
pub use integrity::verify;
pub use load::{BuildSeparator, builder};
pub use preset::{DEFAULT_PRESET, ModelSource, Preset, RETIRED, WEIGHTS_EXTENSION};

// The download function for one pinned file, and the type that describes the file. Only `cuda` uses
// them; everything else downloads a whole `ModelSource`. They are exported only on Windows, where
// CUDA can be installed.
#[cfg(windows)]
pub use download::fetch_one;
/// Used by the updater to download Setup.
#[cfg(windows)]
pub use download::stream_to_file;
#[cfg(windows)]
pub use preset::RemoteFile;

/// The bytes hashed or copied per read: a 675 MB artefact is streamed in chunks small enough to
/// stay in the CPU cache.
const READ_CHUNK: usize = 1 << 20;

/// The application's data directory, created if absent.
///
/// ```text
/// macOS    ~/Library/Application Support/stemd
/// Linux    ~/.local/share/stemd             ($XDG_DATA_HOME)
/// Windows  %LOCALAPPDATA%\stemd\data
/// ```
///
/// It holds what a user would not want cleaned up: the model artefacts and the settings file.
/// Separated stems live under the cache root instead, since losing them costs only a separation.
///
/// On Windows this uses `dirs::data_local_dir`, the local profile: a roaming profile is copied to
/// and from the domain controller at every logon, and weights that any machine can download again
/// do not belong there. The extra `data` component keeps it apart from the cache, which also lives
/// in local app data (see [`crate::cache::default_dir`]).
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

/// The directory that holds a preset's artefact, if any.
///
/// It follows the startup search order, so the window and the command line agree on whether a
/// switch needs the network.
pub fn locate(dirs: &[PathBuf], preset: Preset) -> Option<PathBuf> {
    dirs.iter()
        .find(|d| is_complete(d, preset.source()))
        .cloned()
}

/// True when every file of `source` is present in `dir`.
///
/// Presence only: see [`integrity::mismatched`] for whether the contents are right.
pub fn is_complete(dir: &Path, source: &ModelSource) -> bool {
    source.files.iter().all(|f| dir.join(f.name).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completeness_needs_every_file() {
        let dir = crate::scratch::Dir::new("models");
        // Take the name from the preset, so retiring or renaming a preset updates this test.
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
