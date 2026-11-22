//! Conversion filters uploaded by clients, stored by content hash.
//!
//! A client that rebuilds `drums = mix - harmonics - vocals` from a mix it
//! resampled itself uploads its own resampler here, then names it on a job with
//! `dsp_filter=<id>`, so the stems go through the same taps as its mix.
//!
//! # File format
//!
//! Little-endian. The id is the lowercase hex SHA-256 of the whole file.
//!
//! | Offset | Size | Field |
//! |--------|------|-------|
//! | 0  | 4 | magic `SFIR` |
//! | 4  | 4 | version, `u32`, `1` |
//! | 8  | 4 | input rate, `u32` |
//! | 12 | 4 | output rate, `u32`, one of the offered output rates |
//! | 16 | 4 × n | prototype taps, `f32`; see [`stemd_core::Polyphase`] |
//!
//! # Storage
//!
//! `<dir>/<id>.fir`, kept across restarts. At most [`MAX_FILTERS`]; past that the
//! least recently used goes, where using means uploading it or a job naming it.
//! A file's modification time records its last use. A job holds its filter, so
//! evicting one never breaks a job already accepted.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use stemd_core::{Audio, DspMode, OutputRate, Polyphase, resample};

/// Filters kept at once.
pub const MAX_FILTERS: usize = 64;

const MAGIC: &[u8; 4] = b"SFIR";
const VERSION: u32 = 1;
const HEADER: usize = 16;

/// Largest file accepted: the header and [`Polyphase::MAX_TAPS`] taps.
pub const MAX_FILE_BYTES: usize = HEADER + 4 * Polyphase::MAX_TAPS;

const EXTENSION: &str = "fir";

/// An uploaded filter and its id.
#[derive(Debug)]
pub struct Fir {
    pub id: String,
    pub poly: Polyphase,
}

/// What a filter looks like over the API.
#[derive(Debug, Serialize)]
pub struct Summary {
    pub id: String,
    pub in_rate: u32,
    pub out_rate: u32,
    pub taps: usize,
    pub branch_taps: usize,
}

impl Fir {
    pub fn summary(&self) -> Summary {
        Summary {
            id: self.id.clone(),
            in_rate: self.poly.in_rate(),
            out_rate: self.poly.out_rate(),
            taps: self.poly.taps(),
            branch_taps: self.poly.branch_taps(),
        }
    }
}

/// The filter a job's conversion to its output rate runs through.
#[derive(Debug, Clone)]
pub enum Dsp {
    /// A numbered mode built into the server.
    Mode(DspMode),
    /// A filter a client uploaded.
    Fir(Arc<Fir>),
}

impl Default for Dsp {
    fn default() -> Self {
        Self::Mode(DspMode::General)
    }
}

impl PartialEq for Dsp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Mode(a), Self::Mode(b)) => a == b,
            (Self::Fir(a), Self::Fir(b)) => a.id == b.id,
            _ => false,
        }
    }
}

impl Eq for Dsp {}

impl From<DspMode> for Dsp {
    fn from(mode: DspMode) -> Self {
        Self::Mode(mode)
    }
}

/// The cache key's spelling: the mode number, or `fir:<id>`.
impl fmt::Display for Dsp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mode(mode) => write!(f, "{mode}"),
            Self::Fir(fir) => write!(f, "fir:{}", fir.id),
        }
    }
}

impl Dsp {
    /// The rate pair this filter converts, or `None` when it converts any.
    pub fn only_pair(&self) -> Option<(u32, u32)> {
        match self {
            Self::Mode(mode) => mode.only_pair(),
            Self::Fir(fir) => Some((fir.poly.in_rate(), fir.poly.out_rate())),
        }
    }

    pub fn is_general(&self) -> bool {
        *self == Self::Mode(DspMode::General)
    }

    /// The suffix a piece directory takes: empty, `-dsp1` or `-fir<id>`.
    pub fn dir_suffix(&self) -> String {
        match self {
            Self::Mode(DspMode::General) => String::new(),
            Self::Mode(mode) => format!("-dsp{mode}"),
            Self::Fir(fir) => format!("-fir{}", fir.id),
        }
    }

    /// Convert `audio` to `target` Hz, or return it unchanged when it is there.
    pub fn convert(&self, audio: &Audio, target: u32) -> Result<Audio> {
        match self {
            Self::Mode(mode) => resample::to_rate_with(audio, target, *mode),
            Self::Fir(fir) => {
                if audio.sample_rate == target || audio.frames() == 0 {
                    return Ok(audio.clone());
                }
                anyhow::ensure!(
                    target == fir.poly.out_rate(),
                    "filter {} converts to {} Hz, not {target}",
                    fir.id,
                    fir.poly.out_rate()
                );
                fir.poly.resample(audio)
            }
        }
    }
}

/// Whether `id` is 64 lowercase hex digits.
pub fn is_id(id: &str) -> bool {
    id.len() == 64 && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The id of a filter file: the lowercase hex SHA-256 of its bytes.
pub fn id_of(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Read a filter file. The error names what is wrong with it.
pub fn parse(bytes: &[u8]) -> Result<Polyphase, String> {
    if bytes.len() > MAX_FILE_BYTES {
        return Err(format!(
            "{} bytes is over the limit of {MAX_FILE_BYTES}",
            bytes.len()
        ));
    }
    if bytes.len() < HEADER || &bytes[..4] != MAGIC {
        return Err("not a filter file: it must start with SFIR".into());
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"));
    let version = word(4);
    if version != VERSION {
        return Err(format!(
            "version {version} is not supported; expected {VERSION}"
        ));
    }
    let (in_rate, out_rate) = (word(8), word(12));
    if OutputRate::from_hz(out_rate).is_none() {
        return Err(format!(
            "output rate {out_rate} is not offered; expected one of {}",
            OutputRate::ALL
                .iter()
                .map(|r| r.hz().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let body = &bytes[HEADER..];
    if !body.len().is_multiple_of(4) {
        return Err(format!("{} tap bytes is not whole f32s", body.len()));
    }
    let taps: Vec<f32> = body
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Polyphase::new(in_rate, out_rate, &taps).map_err(|e| format!("{e:#}"))
}

/// Encode a filter file.
#[cfg(test)]
pub fn encode(in_rate: u32, out_rate: u32, taps: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + 4 * taps.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&in_rate.to_le_bytes());
    out.extend_from_slice(&out_rate.to_le_bytes());
    for t in taps {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out
}

#[derive(Debug, thiserror::Error)]
pub enum PutError {
    #[error("{0:?} is not a filter id: expected 64 lowercase hex digits")]
    BadId(String),
    #[error("the body hashes to {computed}, not {claimed}")]
    Mismatch { claimed: String, computed: String },
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] anyhow::Error),
}

struct Stored {
    fir: Arc<Fir>,
    used: SystemTime,
}

pub struct Filters {
    dir: PathBuf,
    cap: usize,
    stored: Mutex<HashMap<String, Stored>>,
}

impl Filters {
    /// Load the filters in `dir`, deleting invalid files and those past `cap`.
    pub fn open(dir: PathBuf, cap: usize) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let mut stored = HashMap::new();
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?
        {
            let path = entry?.path();
            match load(&path) {
                Some((id, fir, used)) => {
                    stored.insert(id, Stored { fir, used });
                }
                None => {
                    tracing::warn!("removing {}: not a valid filter", path.display());
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        let filters = Arc::new(Self {
            dir,
            cap,
            stored: Mutex::new(stored),
        });
        filters.evict(None);
        Ok(filters)
    }

    pub fn len(&self) -> usize {
        self.stored.lock().len()
    }

    /// The filter `id`, marked as used.
    pub fn get(&self, id: &str) -> Option<Arc<Fir>> {
        let fir = {
            let mut stored = self.stored.lock();
            let entry = stored.get_mut(id)?;
            entry.used = SystemTime::now();
            Arc::clone(&entry.fir)
        };
        self.touch(id);
        Some(fir)
    }

    /// The filter `id` without marking it as used.
    pub fn peek(&self, id: &str) -> Option<Arc<Fir>> {
        self.stored.lock().get(id).map(|s| Arc::clone(&s.fir))
    }

    /// Store `body` as filter `id`. `true` when it was not stored already.
    pub fn put(&self, id: &str, body: &[u8]) -> Result<(Arc<Fir>, bool), PutError> {
        if !is_id(id) {
            return Err(PutError::BadId(id.to_owned()));
        }
        let computed = id_of(body);
        if computed != id {
            return Err(PutError::Mismatch {
                claimed: id.to_owned(),
                computed,
            });
        }
        if let Some(fir) = self.get(id) {
            return Ok((fir, false));
        }
        let poly = parse(body).map_err(PutError::Invalid)?;

        let path = self.path(id);
        let part = path.with_extension("part");
        std::fs::write(&part, body).with_context(|| format!("writing {}", part.display()))?;
        std::fs::rename(&part, &path)
            .with_context(|| format!("moving {} into place", part.display()))?;

        let fir = Arc::new(Fir {
            id: id.to_owned(),
            poly,
        });
        let created = self
            .stored
            .lock()
            .insert(
                id.to_owned(),
                Stored {
                    fir: Arc::clone(&fir),
                    used: SystemTime::now(),
                },
            )
            .is_none();
        self.evict(Some(id));
        Ok((fir, created))
    }

    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.{EXTENSION}"))
    }

    /// Record a use on disk, so the order survives a restart.
    fn touch(&self, id: &str) {
        let path = self.path(id);
        let touched = std::fs::File::options()
            .write(true)
            .open(&path)
            .and_then(|f| f.set_modified(SystemTime::now()));
        if let Err(err) = touched {
            tracing::debug!("could not touch {}: {err}", path.display());
        }
    }

    /// Remove the least recently used filters past the cap, never `keep`.
    fn evict(&self, keep: Option<&str>) {
        let gone: Vec<String> = {
            let mut stored = self.stored.lock();
            let mut gone = Vec::new();
            while stored.len() > self.cap {
                let Some(oldest) = stored
                    .iter()
                    .filter(|(id, _)| Some(id.as_str()) != keep)
                    .min_by_key(|(_, s)| s.used)
                    .map(|(id, _)| id.clone())
                else {
                    break;
                };
                stored.remove(&oldest);
                gone.push(oldest);
            }
            gone
        };
        for id in gone {
            tracing::info!("filter {} evicted: over {} filters", &id[..12], self.cap);
            let _ = std::fs::remove_file(self.path(&id));
        }
    }
}

/// A stored filter file, if its name is its hash and it parses.
fn load(path: &Path) -> Option<(String, Arc<Fir>, SystemTime)> {
    if path.extension()? != EXTENSION {
        return None;
    }
    let id = path.file_stem()?.to_str()?.to_owned();
    if !is_id(&id) {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    if id_of(&bytes) != id {
        return None;
    }
    let poly = parse(&bytes).ok()?;
    let used = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    Some((id.clone(), Arc::new(Fir { id, poly }), used))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn dir() -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "stemd-filters-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// A valid 44.1 to 96 kHz file, distinct for each `seed`.
    fn file(seed: u32) -> Vec<u8> {
        let mut taps = vec![0.0f32; 320 * 4];
        taps[0] = 1.0 + seed as f32;
        encode(44_100, 96_000, &taps)
    }

    #[test]
    fn a_stored_filter_survives_a_restart() {
        let dir = dir();
        let body = file(0);
        let id = id_of(&body);
        {
            let filters = Filters::open(dir.clone(), 4).unwrap();
            let (fir, created) = filters.put(&id, &body).unwrap();
            assert!(created);
            assert_eq!(fir.poly.taps(), 1280);
            let (_, again) = filters.put(&id, &body).unwrap();
            assert!(!again, "a second upload is a no-op");
        }
        let filters = Filters::open(dir, 4).unwrap();
        assert_eq!(filters.get(&id).unwrap().id, id);
    }

    #[test]
    fn the_id_must_be_the_hash_of_the_body() {
        let filters = Filters::open(dir(), 4).unwrap();
        let body = file(0);
        let other = id_of(&file(1));
        assert!(matches!(
            filters.put(&other, &body),
            Err(PutError::Mismatch { .. })
        ));
        assert!(matches!(
            filters.put("../../etc/passwd", &body),
            Err(PutError::BadId(_))
        ));
        assert!(matches!(
            filters.put(&id_of(&body).to_uppercase(), &body),
            Err(PutError::BadId(_))
        ));
        assert_eq!(filters.len(), 0);
    }

    #[test]
    fn a_malformed_file_is_refused_with_the_reason() {
        let filters = Filters::open(dir(), 4).unwrap();
        let cases: [(Vec<u8>, &str); 4] = [
            (b"nope".to_vec(), "SFIR"),
            (encode(44_100, 96_000, &[0.0; 319]), "multiple of 320"),
            (encode(44_100, 22_050, &[0.0; 320]), "not offered"),
            (
                {
                    let mut b = file(0);
                    b[4] = 2;
                    b
                },
                "version 2",
            ),
        ];
        for (body, says) in cases {
            let err = filters.put(&id_of(&body), &body).unwrap_err().to_string();
            assert!(err.contains(says), "{err:?} should mention {says:?}");
        }
    }

    #[test]
    fn past_the_cap_the_least_recently_used_goes() {
        let dir = dir();
        let filters = Filters::open(dir.clone(), 2).unwrap();
        let ids: Vec<String> = (0..3).map(|s| id_of(&file(s))).collect();
        filters.put(&ids[0], &file(0)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        filters.put(&ids[1], &file(1)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        // Using the first makes the second the oldest.
        filters.get(&ids[0]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        filters.put(&ids[2], &file(2)).unwrap();

        assert!(filters.peek(&ids[0]).is_some());
        assert!(filters.peek(&ids[1]).is_none());
        assert!(filters.peek(&ids[2]).is_some());
        assert!(!dir.join(format!("{}.fir", ids[1])).exists());

        // The order is on disk, so a restart with a smaller cap keeps the newest.
        drop(filters);
        let filters = Filters::open(dir, 1).unwrap();
        assert!(filters.peek(&ids[2]).is_some());
        assert_eq!(filters.len(), 1);
    }

    #[test]
    fn a_file_that_is_not_its_hash_is_dropped_on_open() {
        let dir = dir();
        std::fs::create_dir_all(&dir).unwrap();
        let body = file(0);
        let wrong = id_of(&file(1));
        std::fs::write(dir.join(format!("{wrong}.fir")), &body).unwrap();
        std::fs::write(dir.join("stray.txt"), b"x").unwrap();
        let filters = Filters::open(dir.clone(), 4).unwrap();
        assert_eq!(filters.len(), 0);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    }

    #[test]
    fn mode_1_uploaded_as_a_filter_converts_the_same() {
        let proto = include_bytes!("../../stemd-core/data/mode1_proto_f32.bin");
        let taps: Vec<f32> = proto
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let body = encode(44_100, 96_000, &taps);
        let filters = Filters::open(dir(), 4).unwrap();
        let (fir, _) = filters.put(&id_of(&body), &body).unwrap();

        let plane: Vec<f32> = (0..4410)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 100.0)
            .collect();
        let audio = Audio::new(vec![plane.clone(), plane], 44_100);
        let a = Dsp::Fir(fir).convert(&audio, 96_000).unwrap();
        let b = Dsp::Mode(DspMode::Mode1).convert(&audio, 96_000).unwrap();
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn the_spellings_name_the_filter() {
        let body = file(0);
        let id = id_of(&body);
        let fir = Arc::new(Fir {
            id: id.clone(),
            poly: parse(&body).unwrap(),
        });
        assert_eq!(Dsp::default().to_string(), "0");
        assert_eq!(Dsp::Mode(DspMode::Mode1).to_string(), "1");
        assert_eq!(Dsp::Fir(Arc::clone(&fir)).to_string(), format!("fir:{id}"));
        assert_eq!(Dsp::default().dir_suffix(), "");
        assert_eq!(Dsp::Mode(DspMode::Mode1).dir_suffix(), "-dsp1");
        assert_eq!(Dsp::Fir(fir).dir_suffix(), format!("-fir{id}"));
    }
}
