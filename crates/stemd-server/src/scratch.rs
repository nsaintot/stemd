//! Fresh directories for tests.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

/// An empty directory under the system temp dir, unique to this process and call.
pub fn dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "stemd-{tag}-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}
