//! Directory and file names under the cache root, and where the root is.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::LOCK_FILE;
use crate::output::Output;

/// The directory name for one output at one gain.
pub fn type_dir(output: &Output, gain: f32) -> String {
    format!("{}{gain}", type_prefix(output))
}

/// The directory name for one output, without the gain value at its end.
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

/// A file name for piece `k`.
pub(super) fn file(k: usize, extension: &str) -> String {
    format!("{k:04}.{extension}")
}

/// Where the cache lives when no directory is configured.
///
/// ```text
/// macOS    ~/Library/Caches/stemd
/// Linux    ~/.cache/stemd              ($XDG_CACHE_HOME)
/// Windows  %LOCALAPPDATA%\stemd\cache
/// ```
///
/// Everything under this root is derived data; model artefacts live under
/// [`crate::models::support_dir`].
///
/// On Windows, `dirs::cache_dir` and `dirs::data_local_dir` both return `FOLDERID_LocalAppData`, so
/// the path adds a `cache` component.
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

/// Lock `root` for this process, or fail if another stemd process holds it, on any port. The system
/// releases the lock when the process ends.
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
