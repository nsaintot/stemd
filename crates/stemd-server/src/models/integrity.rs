//! Checking what is on disk against the pinned digests.
//!
//! A truncated or replaced artefact is otherwise discovered at load, as a failure
//! naming a tensor whose shape disagreed, which says nothing about the file being
//! wrong. Hashing the larger artefact costs about a second, against a process that
//! then spends most of a minute on one track.

use std::io::{BufReader, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use super::{READ_CHUNK, download, preset::Preset};

/// A writer that hashes everything on its way to `inner`.
///
/// Lets the download be verified as it is written, rather than by reading the
/// file back afterwards.
pub(super) struct Hashing<W> {
    inner: W,
    hasher: Sha256,
    written: u64,
}

impl<W: Write> Hashing<W> {
    pub(super) fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
            written: 0,
        }
    }

    /// The digest of everything written, and how many bytes that was.
    pub(super) fn finish(self) -> (String, u64) {
        (format!("{:x}", self.hasher.finalize()), self.written)
    }
}

impl<W: Write> Write for Hashing<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// SHA-256 of a file, streamed so a 675 MB artefact is not read into memory.
pub fn digest_of(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hashing = Hashing::new(std::io::sink());
    std::io::copy(
        &mut BufReader::with_capacity(READ_CHUNK, file),
        &mut hashing,
    )
    .with_context(|| format!("reading {}", path.display()))?;
    Ok(hashing.finish().0)
}

/// Names of files in `dir` whose contents do not match the pinned digest.
pub fn mismatched(dir: &Path, preset: Preset) -> Vec<&'static str> {
    preset
        .source()
        .files
        .iter()
        .filter(|f| {
            let path = dir.join(f.name);
            match digest_of(&path) {
                Ok(actual) => actual != f.sha256,
                // Unreadable counts as wrong: either way it cannot be loaded.
                Err(err) => {
                    tracing::warn!("could not hash {}: {err:#}", path.display());
                    true
                }
            }
        })
        .map(|f| f.name)
        .collect()
}

/// Check `dir` against the pinned digests, repairing the cache if it can.
///
/// A file we downloaded ourselves that no longer hashes correctly is corrupt,
/// so it is deleted and refetched. Outside the cache the mismatch is only
/// reported: a developer's own re-trace legitimately differs from the published
/// artefact, and failing on that would make local work impossible.
pub fn verify(dir: &Path, cache: &Path, preset: Preset, offline: bool) -> Result<()> {
    let bad = mismatched(dir, preset);
    if bad.is_empty() {
        return Ok(());
    }
    let what = describe(dir, &bad);

    if dir != cache {
        // Not a warning: outside the cache a mismatch is the expected state,
        // and a warning about the normal case is one nobody reads.
        tracing::info!(
            "in {}, {what} — using it anyway, since a local trace is expected \
             to differ from the published artefact",
            dir.display()
        );
        return Ok(());
    }

    // Refuse before touching anything: with no way to replace the file,
    // deleting it only destroys what someone might want to look at.
    if offline {
        bail!("in the model cache, {what} — and --offline forbids refetching");
    }

    tracing::warn!("in the model cache, {what}; refetching");
    for name in &bad {
        let _ = std::fs::remove_file(dir.join(name));
    }
    download::fetch(cache, preset.source(), |_, _, _| {})?;

    let still_bad = mismatched(dir, preset);
    if !still_bad.is_empty() {
        bail!(
            "{} still fails verification after refetching — the release may \
             have been replaced since this build pinned it",
            still_bad.join(", ")
        );
    }
    Ok(())
}

/// Absent and corrupt both need replacing, but they send you looking in very
/// different places, so say which.
fn describe(dir: &Path, bad: &[&'static str]) -> String {
    let (absent, corrupt): (Vec<&str>, Vec<&str>) =
        bad.iter().copied().partition(|n| !dir.join(n).exists());
    match (absent.as_slice(), corrupt.as_slice()) {
        ([], c) => format!("{} does not match its pinned digest", c.join(", ")),
        (a, []) => format!("{} is missing", a.join(", ")),
        (a, c) => format!(
            "{} is missing and {} does not match its pinned digest",
            a.join(", "),
            c.join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("stemd-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn digest_matches_a_known_value() {
        let dir = scratch("digest");
        let path = dir.join("abc");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            digest_of(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wrong_contents_are_reported_even_though_the_names_are_right() {
        let dir = scratch("bad");
        for f in Preset::Balanced.source().files {
            std::fs::write(dir.join(f.name), b"not the model").unwrap();
        }
        // is_complete only looks at filenames, so it is happy.
        assert!(super::super::is_complete(&dir, Preset::Balanced.source()));
        // Hashing is what actually catches it.
        let files = Preset::Balanced.source().files.len();
        assert_eq!(mismatched(&dir, Preset::Balanced).len(), files);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_file_counts_as_mismatched_rather_than_panicking() {
        let dir = scratch("gone");
        let files = Preset::Fast.source().files.len();
        assert_eq!(mismatched(&dir, Preset::Fast).len(), files);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn absent_and_corrupt_are_described_separately() {
        let dir = scratch("describe");
        std::fs::write(dir.join("there"), b"x").unwrap();
        assert!(describe(&dir, &["gone"]).contains("is missing"));
        assert!(describe(&dir, &["there"]).contains("does not match"));
        let both = describe(&dir, &["gone", "there"]);
        assert!(both.contains("is missing") && both.contains("does not match"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
