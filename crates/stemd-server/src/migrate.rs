//! What stemd 0.2.0 left behind: its cache layout and its flag names.
//!
//! Removed together with protocol 1.

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
