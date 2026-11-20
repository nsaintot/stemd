//! What stemd 0.2.0 left behind: its cache layout and its flag names.
//!
//! Removed together with protocol 1, along with the `--pieces-max-gb` alias of
//! `--cache-max-gb` in cli.rs.

use std::path::Path;

/// The 0.2.0 name of `--cache-dir`.
pub const PIECES_DIR: &str = "pieces-dir";

/// Delete what stemd 0.2.0 and earlier kept in the root: one directory per
/// job, named by 64 hex characters, and their staging directories.
pub fn remove_old_layout(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let key = name.split('.').next().unwrap_or_default();
        if key.len() == 64
            && key.bytes().all(|b| b.is_ascii_hexdigit())
            && entry.file_type().is_ok_and(|t| t.is_dir())
        {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Delete `stemd-pieces` under the platform's cache directory, where stemd
/// 0.2.0 kept session pieces, on a background thread.
pub fn remove_old_pieces_dir() {
    let Some(old) = dirs::cache_dir().map(|dir| dir.join("stemd-pieces")) else {
        return;
    };
    if !old.is_dir() {
        return;
    }
    std::thread::spawn(move || match std::fs::remove_dir_all(&old) {
        Ok(()) => tracing::info!("deleted stemd 0.2.0's piece cache in {}", old.display()),
        Err(err) => tracing::warn!("could not delete {}: {err}", old.display()),
    });
}
