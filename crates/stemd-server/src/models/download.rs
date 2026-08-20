//! Fetching missing artefacts.
//!
//! Each file is written to a temporary sibling and renamed into place, so an
//! interrupted run cannot leave a half-written model that looks complete. The
//! digest is computed as the bytes are written and checked before the rename, with
//! no escape hatch for an unpinned digest.

use std::io::{BufWriter, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};

use super::READ_CHUNK;
use super::integrity::Hashing;
use super::preset::{ModelSource, RemoteFile};

/// Bytes between progress callbacks. Often enough to drive a progress bar,
/// rarely enough not to spam a log.
const PROGRESS_STEP: u64 = 8 << 20;

/// Download whatever is missing into `dir`. Files already present are left
/// alone, this is a first-run path, not a repair one.
///
/// `on_progress` is called with `(file_name, bytes_done, bytes_total)`.
pub fn fetch(
    dir: &Path,
    source: &ModelSource,
    mut on_progress: impl FnMut(&str, u64, u64),
) -> Result<()> {
    for file in source.files {
        if dir.join(file.name).is_file() {
            continue;
        }
        fetch_one(dir, file, &mut on_progress)?;
    }
    Ok(())
}

/// Fetch one file, verify it against its pinned digest, and publish it.
///
/// Public because weights are not the only pinned thing this fetches: the CUDA
/// runtime arrives the same way. Unlike [`fetch`] this does not skip a file
/// already present, since its caller may have consumed and deleted the last one.
pub fn fetch_one(
    dir: &Path,
    file: &RemoteFile,
    on_progress: &mut impl FnMut(&str, u64, u64),
) -> Result<()> {
    tracing::info!("fetching {} ({:.0} MB)", file.name, file.bytes as f64 / 1e6);

    let response = ureq::get(file.url)
        .call()
        .map_err(|e| anyhow::anyhow!("GET {}: {e}", file.url))?;
    let total = content_length(&response).unwrap_or(file.bytes);

    let partial = dir.join(format!("{}.part", file.name));
    let (digest, written) = stream_to_file(response.into_body().into_reader(), &partial, |done| {
        on_progress(file.name, done, total);
    })?;
    on_progress(file.name, written, total);

    if digest != file.sha256 {
        let _ = std::fs::remove_file(&partial);
        bail!(
            "{} failed its checksum (expected {}, got {digest}) — the download \
             was corrupted or the release was replaced",
            file.name,
            file.sha256
        );
    }

    let dest = dir.join(file.name);
    std::fs::rename(&partial, &dest).with_context(|| format!("publishing {}", dest.display()))?;
    tracing::info!("{} ready", file.name);
    Ok(())
}

/// Copy `reader` into `path`, returning the digest and length of what landed.
fn stream_to_file(
    reader: impl Read,
    path: &Path,
    on_bytes: impl FnMut(u64),
) -> Result<(String, u64)> {
    let file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut sink = Hashing::new(BufWriter::with_capacity(READ_CHUNK, file));
    let mut source = Reporting {
        inner: reader,
        on_bytes,
        done: 0,
        reported: 0,
    };
    std::io::copy(&mut source, &mut sink).with_context(|| format!("writing {}", path.display()))?;
    Ok(sink.finish())
}

fn content_length(response: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    response
        .headers()
        .get("content-length")?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

/// A reader that reports its running byte count every [`PROGRESS_STEP`].
struct Reporting<R, F> {
    inner: R,
    on_bytes: F,
    done: u64,
    reported: u64,
}

impl<R: Read, F: FnMut(u64)> Read for Reporting<R, F> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.done += n as u64;
        if self.done - self.reported >= PROGRESS_STEP {
            (self.on_bytes)(self.done);
            self.reported = self.done;
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_hashes_and_counts_what_it_wrote() {
        let dir = std::env::temp_dir().join(format!("stemd-stream-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out");

        let (digest, written) = stream_to_file(&b"abc"[..], &path, |_| {}).unwrap();
        assert_eq!(written, 3);
        assert_eq!(
            digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "the digest must cover the bytes as written"
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"abc");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn progress_is_reported_at_most_once_per_step() {
        let dir = std::env::temp_dir().join(format!("stemd-progress-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let payload = vec![0u8; (PROGRESS_STEP * 2) as usize + 1];
        let mut seen = Vec::new();
        stream_to_file(&payload[..], &dir.join("out"), |done| seen.push(done)).unwrap();

        assert_eq!(
            seen.len(),
            2,
            "one callback per step, not per read: {seen:?}"
        );
        assert!(seen[0] >= PROGRESS_STEP && seen[1] >= PROGRESS_STEP * 2);

        std::fs::remove_dir_all(&dir).ok();
    }
}
