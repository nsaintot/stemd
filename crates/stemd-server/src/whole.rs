//! One stem of a whole track, stored as a single file beside its track's pieces.
//!
//! ```text
//! <track dir>/<stem>.<ext>    the bytes a whole-track read sends
//! <track dir>/<stem>.json     frame count, clip count and, for FLAC, the frame index
//! ```
//!
//! A range is cut from the file without decoding the rest: by byte offset for
//! `wav`, `s16le` and `f32le`, and by frame for FLAC, whose blocks have a fixed size.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use stemd_core::{Audio, StemFormat};

/// Bytes before the samples of a `wav` file `encode_stem` writes.
const WAV_HEADER: usize = 44;

/// What a whole file holds, written beside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Meta {
    /// Samples per channel, at the output rate.
    pub frames: usize,
    /// Samples clipped to fit 16 bits, absent for `f32le`.
    pub clipped: Option<usize>,
    /// FLAC only: bytes before the first frame.
    #[serde(default)]
    pub header: usize,
    /// FLAC only: samples per frame.
    #[serde(default)]
    pub block: usize,
    /// FLAC only: where each frame starts.
    #[serde(default)]
    pub offsets: Vec<u64>,
}

/// A stored whole file and its description.
#[derive(Debug, Clone)]
pub struct Whole {
    pub path: PathBuf,
    pub meta: Meta,
}

pub fn file(track_dir: &Path, stem: &str, extension: &str) -> PathBuf {
    track_dir.join(format!("{stem}.{extension}"))
}

pub fn meta_file(track_dir: &Path, stem: &str) -> PathBuf {
    track_dir.join(format!("{stem}.json"))
}

/// The whole file of `stem` under `track_dir`, when it exists with a description that fits it.
pub fn find(track_dir: &Path, stem: &str, extension: &str) -> Option<Whole> {
    let path = file(track_dir, stem, extension);
    let meta: Meta =
        serde_json::from_slice(&std::fs::read(meta_file(track_dir, stem)).ok()?).ok()?;
    let size = std::fs::metadata(&path).ok().filter(|m| m.is_file())?.len();
    let fits = extension != StemFormat::Flac.extension() || meta.indexes(size);
    fits.then_some(Whole { path, meta })
}

impl Meta {
    /// True when the FLAC index covers `frames` in order inside a file of `size` bytes.
    fn indexes(&self, size: u64) -> bool {
        let header = self.header as u64;
        self.block > 0
            && self.offsets.len() == self.frames.div_ceil(self.block)
            && header <= size
            && self.offsets.first().is_none_or(|&first| header <= first)
            && self.offsets.windows(2).all(|w| w[0] < w[1])
            && self.offsets.last().is_none_or(|&last| last < size)
    }
}

/// The whole file of `stem` under `track_dir`, when it holds `frames` samples.
pub fn find_track(track_dir: &Path, stem: &str, extension: &str, frames: usize) -> Option<Whole> {
    find(track_dir, stem, extension).filter(|whole| whole.meta.frames == frames)
}

/// Describe `bytes`, an encoded stem of `frames` samples.
pub fn describe(
    bytes: &[u8],
    format: StemFormat,
    frames: usize,
    clipped: Option<usize>,
) -> Result<Meta> {
    let mut meta = Meta {
        frames,
        clipped,
        header: 0,
        block: 0,
        offsets: Vec::new(),
    };
    if format == StemFormat::Flac {
        let index = flac_index(bytes)?;
        ensure!(
            index.offsets.len() == frames.div_ceil(index.block),
            "{} flac frames of {} for {frames} samples",
            index.offsets.len(),
            index.block
        );
        meta.header = index.header;
        meta.block = index.block;
        meta.offsets = index.offsets;
    }
    Ok(meta)
}

/// Samples `from..to` of `whole`, at the output rate and the transfer scale.
pub fn read(
    whole: &Whole,
    format: StemFormat,
    channels: usize,
    rate: u32,
    from: usize,
    to: usize,
) -> Result<Audio> {
    ensure!(
        from < to && to <= whole.meta.frames,
        "{from}..{to} is not a range of this {}-sample stem",
        whole.meta.frames
    );
    let mut f = std::fs::File::open(&whole.path)
        .with_context(|| format!("opening {}", whole.path.display()))?;
    let mut take = |at: u64, len: usize| -> Result<Vec<u8>> {
        let mut buf = vec![0; len];
        f.seek(SeekFrom::Start(at))?;
        f.read_exact(&mut buf)?;
        Ok(buf)
    };
    match format {
        StemFormat::Flac => {
            let meta = &whole.meta;
            let (a, b) = (from / meta.block, to.div_ceil(meta.block));
            let start = meta.offsets[a];
            let end = match meta.offsets.get(b) {
                Some(&end) => end,
                None => std::fs::metadata(&whole.path)?.len(),
            };
            let mut bytes = take(0, meta.header)?;
            unknown_length(&mut bytes)?;
            bytes.extend(take(start, (end - start) as usize)?);
            let audio = crate::pieces::decode(bytes, format, channels, rate)?;
            let skip = from - a * meta.block;
            ensure!(
                audio.frames() >= skip + (to - from),
                "flac frames {a}..{b} decoded to {} samples",
                audio.frames()
            );
            Ok(crate::sessions::excerpt(&audio, skip, skip + (to - from)))
        }
        StemFormat::Wav | StemFormat::Pcm16 | StemFormat::Pcm32 => {
            let (header, size, raw) = match format {
                StemFormat::Wav => (WAV_HEADER, 2, StemFormat::Pcm16),
                StemFormat::Pcm16 => (0, 2, StemFormat::Pcm16),
                _ => (0, 4, StemFormat::Pcm32),
            };
            let per_frame = channels * size;
            let bytes = take((header + from * per_frame) as u64, (to - from) * per_frame)?;
            crate::pieces::decode(bytes, raw, channels, rate)
        }
        StemFormat::Mp3 => bail!("mp3 is never stored"),
    }
}

/// A FLAC stream's layout: where frames start, and how many samples each holds.
#[derive(Debug)]
struct FlacIndex {
    header: usize,
    block: usize,
    offsets: Vec<u64>,
}

/// Index a fixed-block-size FLAC stream by reading its frame headers.
fn flac_index(bytes: &[u8]) -> Result<FlacIndex> {
    ensure!(bytes.starts_with(b"fLaC"), "not a flac stream");
    let mut at = 4;
    let mut block = 0;
    loop {
        let head = bytes
            .get(at..at + 4)
            .context("flac metadata runs past the end")?;
        let len = u32::from_be_bytes([0, head[1], head[2], head[3]]) as usize;
        if head[0] & 0x7f == 0 {
            let info = bytes.get(at + 4..at + 8).context("a short STREAMINFO")?;
            block = usize::from(u16::from_be_bytes([info[0], info[1]]));
            ensure!(
                info[0..2] == info[2..4] && block > 0,
                "flac blocks are not one fixed size"
            );
        }
        at += 4 + len;
        if head[0] & 0x80 != 0 {
            break;
        }
    }
    ensure!(block > 0, "flac stream without STREAMINFO");

    let header = at;
    let mut offsets: Vec<u64> = Vec::new();
    while at < bytes.len() {
        // A header counts only where the previous frame's CRC-16 ends, so a
        // header-shaped run of bytes inside a frame is skipped.
        if frame_number(bytes, at) == Some(offsets.len() as u64)
            && offsets
                .last()
                .is_none_or(|&start| frame_crc_holds(bytes, start as usize, at))
        {
            offsets.push(at as u64);
            // A frame header alone is at least 6 bytes.
            at += 6;
        } else {
            at += 1;
        }
    }
    ensure!(!offsets.is_empty(), "flac stream without frames");
    let last = *offsets.last().expect("not empty") as usize;
    ensure!(
        frame_crc_holds(bytes, last, bytes.len()),
        "the last flac frame fails its CRC-16"
    );
    Ok(FlacIndex {
        header,
        block,
        offsets,
    })
}

/// The frame number of a fixed-block-size frame header at `at`, if one is there
/// and its CRC-8 holds.
fn frame_number(bytes: &[u8], at: usize) -> Option<u64> {
    let b = bytes.get(at..)?;
    if b.len() < 6 || b[0] != 0xff || b[1] != 0xf8 || b[2] >> 4 == 0 || b[2] & 0x0f == 0x0f {
        return None;
    }
    if b[3] & 1 != 0 {
        return None;
    }
    // The frame number, UTF-8 coded.
    let first = b[4];
    let extra = first.leading_ones() as usize;
    let (mut number, mut p) = match extra {
        0 => (u64::from(first), 5),
        2..=7 => (u64::from(first & (0x7f >> extra)), 5),
        _ => return None,
    };
    if extra > 0 {
        for i in 0..extra - 1 {
            let c = *b.get(5 + i)?;
            if c & 0xc0 != 0x80 {
                return None;
            }
            number = number << 6 | u64::from(c & 0x3f);
        }
        p += extra - 1;
    }
    p += match b[2] >> 4 {
        6 => 1,
        7 => 2,
        _ => 0,
    };
    p += match b[2] & 0x0f {
        12 => 1,
        13 | 14 => 2,
        _ => 0,
    };
    (crc8(b.get(..p)?) == *b.get(p)?).then_some(number)
}

/// True when the frame at `from..to` ends with the CRC-16 of the bytes before it.
fn frame_crc_holds(bytes: &[u8], from: usize, to: usize) -> bool {
    if to < from + 8 || to > bytes.len() {
        return false;
    }
    let stored = u16::from_be_bytes([bytes[to - 2], bytes[to - 1]]);
    crc16(&bytes[from..to - 2]) == stored
}

/// CRC-16 with polynomial 0x8005, as FLAC frames end with.
fn crc16(bytes: &[u8]) -> u16 {
    static TABLE: std::sync::OnceLock<[u16; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u16; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            let mut crc = (i as u16) << 8;
            for _ in 0..8 {
                crc = if crc & 0x8000 != 0 {
                    crc << 1 ^ 0x8005
                } else {
                    crc << 1
                };
            }
            *entry = crc;
        }
        table
    });
    bytes.iter().fold(0u16, |crc, &byte| {
        crc << 8 ^ table[usize::from((crc >> 8) as u8 ^ byte)]
    })
}

/// CRC-8 with polynomial 0x07, as FLAC frame headers use.
fn crc8(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0u8, |mut crc, &byte| {
        crc ^= byte;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                crc << 1 ^ 0x07
            } else {
                crc << 1
            };
        }
        crc
    })
}

/// Clear STREAMINFO's sample count and MD5, so a decoder reads a cut stream to its end.
fn unknown_length(header: &mut [u8]) -> Result<()> {
    // "fLaC", the block header, then STREAMINFO: the count is the low 36 bits
    // of bytes 10..18, and the MD5 follows.
    let info = header.get_mut(8..42).context("a short STREAMINFO")?;
    info[13] &= 0xf0;
    info[14..34].fill(0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(frames: usize, seed: u32) -> Audio {
        let mut state = seed;
        let data = (0..2)
            .map(|_| {
                (0..frames)
                    .map(|_| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.8
                    })
                    .collect()
            })
            .collect();
        Audio::new(data, 96_000)
    }

    fn bits(audio: &Audio) -> Vec<u32> {
        audio.data.iter().flatten().map(|v| v.to_bits()).collect()
    }

    fn stored(dir: &Path, format: StemFormat, audio: &Audio) -> Whole {
        let bytes = stemd_core::encode_stem(audio, format, 1.0).expect("encoding");
        let meta = describe(&bytes, format, audio.frames(), None).expect("describing");
        let path = dir.join(format!("stem-{format}"));
        std::fs::write(&path, &bytes).expect("writing");
        Whole { path, meta }
    }

    #[test]
    fn a_range_cut_from_a_whole_file_equals_the_same_range_of_the_whole_decode() {
        let dir = std::env::temp_dir().join(format!("stemd-whole-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a directory");
        for frames in [1, 4096, 4097, 50_001] {
            let audio = noise(frames, frames as u32);
            for format in [
                StemFormat::Flac,
                StemFormat::Wav,
                StemFormat::Pcm16,
                StemFormat::Pcm32,
            ] {
                let whole = stored(&dir, format, &audio);
                let all = crate::pieces::decode(
                    std::fs::read(&whole.path).expect("reading"),
                    format,
                    2,
                    96_000,
                )
                .expect("decoding");
                assert_eq!(all.frames(), frames, "{format}");
                for (from, to) in [
                    (0, frames),
                    (frames / 3, frames - frames / 4),
                    (frames - 1, frames),
                ] {
                    if from >= to {
                        continue;
                    }
                    let got = read(&whole, format, 2, 96_000, from, to).expect("a range");
                    let want = crate::sessions::excerpt(&all, from, to);
                    assert_eq!(bits(&got), bits(&want), "{format}, {frames}: {from}..{to}");
                }
            }
        }
    }

    #[test]
    fn a_flac_description_that_does_not_fit_its_file_is_not_found() {
        let dir = std::env::temp_dir().join(format!("stemd-whole-find-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a directory");
        let audio = noise(20_000, 9);
        let bytes = stemd_core::encode_stem(&audio, StemFormat::Flac, 1.0).expect("encoding");
        std::fs::write(file(&dir, "vocals", "flac"), &bytes).expect("writing");
        let meta = describe(&bytes, StemFormat::Flac, audio.frames(), None).expect("describing");
        assert!(meta.offsets.len() > 2, "too few frames for the test");
        let describe_as = |meta: &Meta| {
            std::fs::write(meta_file(&dir, "vocals"), serde_json::to_vec(meta).unwrap())
                .expect("writing");
        };
        describe_as(&meta);
        assert!(find(&dir, "vocals", "flac").is_some());

        let mut past_the_end = meta.clone();
        *past_the_end.offsets.last_mut().unwrap() = bytes.len() as u64;
        let mut out_of_order = meta.clone();
        out_of_order.offsets.swap(1, 2);
        for (what, bad) in [
            (
                "a short index",
                Meta {
                    offsets: meta.offsets[..1].to_vec(),
                    ..meta.clone()
                },
            ),
            (
                "no block size",
                Meta {
                    block: 0,
                    ..meta.clone()
                },
            ),
            (
                "a header past the first frame",
                Meta {
                    header: meta.offsets[0] as usize + 1,
                    ..meta.clone()
                },
            ),
            ("a frame past the end", past_the_end),
            ("frames out of order", out_of_order),
        ] {
            describe_as(&bad);
            assert!(find(&dir, "vocals", "flac").is_none(), "{what}");
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_header_shaped_run_inside_a_frame_is_not_indexed() {
        let bytes =
            stemd_core::encode_stem(&noise(40_000, 5), StemFormat::Flac, 1.0).expect("encoding");
        let want = flac_index(&bytes).expect("an index").offsets;
        let (start, next) = (want[3] as usize, want[4] as usize);
        assert!(next - start > 64, "frames too short for the test");
        // Plant frame 4's header inside frame 3, and keep frame 3's CRC-16 valid.
        let mut planted = bytes.clone();
        let fake = start + 20;
        planted.copy_within(next..next + 16, fake);
        let crc = crc16(&planted[start..next - 2]);
        planted[next - 2..next].copy_from_slice(&crc.to_be_bytes());
        assert_eq!(
            frame_number(&planted, fake),
            Some(4),
            "the planted bytes parse as frame 4's header"
        );
        assert_eq!(flac_index(&planted).expect("an index").offsets, want);
    }

    #[test]
    fn a_frame_header_needs_its_crc() {
        let bytes =
            stemd_core::encode_stem(&noise(10_000, 3), StemFormat::Flac, 1.0).expect("encoding");
        let index = flac_index(&bytes).expect("an index");
        let at = index.offsets[1] as usize;
        assert_eq!(frame_number(&bytes, at), Some(1));
        let mut broken = bytes.clone();
        broken[at + 4] ^= 0x01;
        assert_eq!(frame_number(&broken, at), None);
    }
}
