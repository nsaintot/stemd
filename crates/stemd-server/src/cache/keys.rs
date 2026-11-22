//! Client file keys, each naming the track id an upload under it hashed to.

use anyhow::{Context, Result};

use super::{Cache, FILES};

/// What a file key's file holds once two uploads under it hashed differently.
const AMBIGUOUS: &str = "ambiguous";

impl Cache {
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
}
