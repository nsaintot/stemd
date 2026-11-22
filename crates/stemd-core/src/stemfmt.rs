//! How a separated stem is written for transfer.
//!
//! It differs from [`PcmFormat`], which describes raw samples and is what an upload is parsed as.
//! `flac` is a container, so it is never a valid upload format.
//!
//! Four of the five formats are lossless; `mp3` is lossy. A perceptual codec hides its noise under
//! a masking threshold computed for the signal as encoded, and a stem's gain is changed afterwards,
//! which unmasks that noise. Lossy parts also do not sum, and the client's rebuild of the third
//! part needs them to. A client may still choose `mp3`.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::pcm::{Audio, PcmError, PcmFormat};
use crate::resample::OutputRate;

/// The bits per sample in a WAV stem: 16, as [`stemd_audio::wire::FLAC_BITS`], since the
/// quantisation floor is already below the model's own error. A 16-bit WAV also opens everywhere.
const WAV_BITS: u16 = 16;

/// How hard LAME searches when quantising: its own recommended setting.
///
/// At a constant 320 kbps there are enough bits that a slower search gains nothing: `q=0` runs at
/// 18x realtime against 47x here and differs by -47 dB, below the coding noise both add. The
/// benchmarks are the `#[ignore]` tests below.
const MP3_QUALITY: mp3lame_encoder::Quality = mp3lame_encoder::Quality::NearBest;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StemFormat {
    /// FLAC with 16-bit samples. It is lossless, so it decodes to exactly what `pcm16` would carry,
    /// in roughly half the bytes.
    Flac,
    /// A 16-bit WAV: the same samples as FLAC, with a RIFF header and no compression. Use it when
    /// the next program to open the file is fussy.
    Wav,
    /// MP3 at 320 kbps, the only lossy option; the module docs say when that matters.
    Mp3,
    /// Raw interleaved 16-bit LE: the same samples with no container and no decode step.
    Pcm16,
    /// Raw interleaved 32-bit float LE: twice the bytes, no ceiling at full scale, and the only
    /// format that gives an exact reconstruction null with no quantisation anywhere.
    Pcm32,
}

impl StemFormat {
    pub const ALL: [Self; 5] = [Self::Flac, Self::Wav, Self::Mp3, Self::Pcm16, Self::Pcm32];

    /// The raw sample encoding, when the payload is samples and needs no decoding.
    #[cfg(test)]
    pub const fn as_pcm(self) -> Option<PcmFormat> {
        match self {
            Self::Pcm16 => Some(PcmFormat::S16le),
            Self::Pcm32 => Some(PcmFormat::F32le),
            Self::Flac | Self::Wav | Self::Mp3 => None,
        }
    }

    /// Whether the decoded output is exactly the input.
    ///
    /// The derived part depends on this: parts that are not lossless do not sum back to the mix.
    #[cfg(test)]
    pub const fn is_lossless(self) -> bool {
        !matches!(self, Self::Mp3)
    }

    /// Whether samples above full scale survive. Only the float format has no
    /// ceiling; everything else needs a stem that peaks over 1.0 scaled to fit.
    pub const fn has_headroom(self) -> bool {
        matches!(self, Self::Pcm32)
    }

    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Pcm16 | Self::Pcm32 => "application/octet-stream",
            Self::Flac => "audio/flac",
            Self::Wav => "audio/wav",
            Self::Mp3 => "audio/mpeg",
        }
    }

    /// The file extension used inside a cache entry.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Pcm16 | Self::Pcm32 => "pcm",
            Self::Flac => "flac",
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
        }
    }

    /// Whether this format can carry `rate`.
    // Only MP3 cannot carry every rate. MPEG-1 Layer III has three sample rates, 32, 44.1 and 48
    // kHz, and 96 kHz is not among them at any bitrate. LAME does not refuse it: it turns on its
    // own resampler and encodes 48 kHz.
    pub const fn carries(self, rate: OutputRate) -> bool {
        match self {
            Self::Mp3 => matches!(
                rate,
                OutputRate::Hz24000 | OutputRate::Hz44100 | OutputRate::Hz48000
            ),
            _ => true,
        }
    }

    /// The rates this format can carry, for a menu to offer.
    pub fn rates(self) -> impl Iterator<Item = OutputRate> {
        OutputRate::ALL
            .into_iter()
            .filter(move |r| self.carries(*r))
    }

    /// The label the window shows. `Display` gives the name the API takes, which does not always
    /// read best in a menu.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Flac => "FLAC",
            Self::Wav => "WAV",
            Self::Mp3 => "MP3",
            Self::Pcm16 => "PCM 16-bit",
            Self::Pcm32 => "PCM 32-bit float",
        }
    }
}

impl fmt::Display for StemFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Flac => "flac",
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Pcm16 => "pcm16",
            Self::Pcm32 => "pcm32",
        })
    }
}

impl FromStr for StemFormat {
    type Err = PcmError;

    /// The `s16le` and `f32le` spellings are accepted as aliases.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "flac" => Ok(Self::Flac),
            "wav" => Ok(Self::Wav),
            "mp3" => Ok(Self::Mp3),
            "pcm16" | "s16le" | "s16" => Ok(Self::Pcm16),
            "pcm32" | "f32le" | "f32" => Ok(Self::Pcm32),
            other => Err(PcmError::UnknownFormat(other.to_owned())),
        }
    }
}

/// Encode a stem for transfer, applying `gain` on the way out.
pub fn encode(audio: &Audio, format: StemFormat, gain: f32) -> anyhow::Result<Vec<u8>> {
    match format {
        StemFormat::Pcm16 => Ok(audio.to_interleaved_scaled(PcmFormat::S16le, gain)),
        StemFormat::Pcm32 => Ok(audio.to_interleaved_scaled(PcmFormat::F32le, gain)),
        StemFormat::Flac => to_flac(audio, gain),
        StemFormat::Wav => Ok(to_wav(audio, gain)),
        StemFormat::Mp3 => to_mp3(audio, gain),
    }
}

/// A 16-bit WAV: a 44-byte canonical RIFF header, then the same interleaved
/// samples `pcm16` would have carried.
fn to_wav(audio: &Audio, gain: f32) -> Vec<u8> {
    let samples = audio.to_interleaved_scaled(PcmFormat::S16le, gain);
    let channels = audio.channels() as u16;
    let block_align = channels * WAV_BITS / 8;
    let byte_rate = audio.sample_rate * u32::from(block_align);

    let mut out = Vec::with_capacity(44 + samples.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + samples.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // format tag: integer PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&audio.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&WAV_BITS.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    out.extend_from_slice(&samples);
    out
}

/// MP3 at a constant 320 kbps, encoded from the float samples.
fn to_mp3(audio: &Audio, gain: f32) -> anyhow::Result<Vec<u8>> {
    to_mp3_at(audio, gain, MP3_QUALITY)
}

fn to_mp3_at(
    audio: &Audio,
    gain: f32,
    quality: mp3lame_encoder::Quality,
) -> anyhow::Result<Vec<u8>> {
    // Each encoder stage reports a different error type, so this is a generic function: a closure
    // would bind to the first type it saw.
    fn fail<E: std::fmt::Debug>(what: &'static str) -> impl Fn(E) -> anyhow::Error {
        move |e| anyhow::anyhow!("mp3 {what}: {e:?}")
    }

    use mp3lame_encoder::{Bitrate, Builder, DualPcm, FlushNoGap, MonoPcm};

    let channels = audio.channels();
    anyhow::ensure!(channels > 0, "cannot encode zero channels to mp3");
    anyhow::ensure!(
        channels <= 2,
        "mp3: expected at most two channels, got {channels}"
    );

    // Check here as well as where the pair is chosen, because the cost lands here: LAME accepts a
    // rate it has no mode for and resamples to one it has, taking eighteen seconds per stem to
    // produce a file at a rate nobody asked for.
    anyhow::ensure!(
        OutputRate::from_hz(audio.sample_rate).is_none_or(|r| StemFormat::Mp3.carries(r)),
        "mp3 has no {} Hz mode; 44100 and 48000 are the ones above 24000",
        audio.sample_rate
    );

    let mut builder = Builder::new().ok_or_else(|| anyhow::anyhow!("mp3 encoder unavailable"))?;
    builder
        .set_num_channels(channels as u8)
        .map_err(fail("channels"))?;
    builder
        .set_sample_rate(audio.sample_rate)
        .map_err(fail("sample rate"))?;
    builder
        .set_brate(Bitrate::Kbps320)
        .map_err(fail("bitrate"))?;
    builder.set_quality(quality).map_err(fail("quality"))?;
    let mut encoder = builder.build().map_err(fail("build"))?;

    let scaled: Vec<Vec<f32>> = audio
        .data
        .iter()
        .map(|ch| ch.iter().map(|s| (s * gain).clamp(-1.0, 1.0)).collect())
        .collect();

    let mut out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(audio.frames()));
    if channels == 1 {
        encoder
            .encode_to_vec(MonoPcm(&scaled[0]), &mut out)
            .map_err(fail("encode"))?;
    } else {
        encoder
            .encode_to_vec(
                DualPcm {
                    left: &scaled[0],
                    right: &scaled[1],
                },
                &mut out,
            )
            .map_err(fail("encode"))?;
    }
    encoder
        .flush_to_vec::<FlushNoGap>(&mut out)
        .map_err(fail("flush"))?;
    Ok(out)
}

/// FLAC-encode at [`stemd_audio::wire::FLAC_BITS`], quantised as `s16le` is.
fn to_flac(audio: &Audio, gain: f32) -> anyhow::Result<Vec<u8>> {
    // One below the crate's default of 10: 13% faster encoding for 0.7% more bytes, measured by
    // `encode_cost`. Order 4 costs 9.5% more size, and no LPC costs 25%. Encoding dominates the
    // writing stage and is already multithreaded, so this is the only setting left to tune.
    let config = stemd_audio::wire::flac_config(Some(8))?;
    to_flac_with(audio, gain, &config)
}

/// The body of [`to_flac`], with the encoder configuration supplied, so `encode_cost` can vary it.
fn to_flac_with(
    audio: &Audio,
    gain: f32,
    config: &flacenc::error::Verified<flacenc::config::Encoder>,
) -> anyhow::Result<Vec<u8>> {
    stemd_audio::wire::encode_flac(&audio.data, audio.sample_rate, gain, config)
}

#[cfg(test)]
mod tests {
    /// MPEG-1 Layer III has three sample rates, and 96 kHz is not one of them. Every other format
    /// here is a container over samples and carries any rate.
    #[test]
    fn only_mp3_turns_a_rate_down() {
        use super::{OutputRate, StemFormat};
        for format in StemFormat::ALL {
            for rate in OutputRate::ALL {
                let expected = format != StemFormat::Mp3 || rate != OutputRate::Hz96000;
                assert_eq!(format.carries(rate), expected, "{format} at {}", rate.hz());
            }
        }
        assert_eq!(StemFormat::Mp3.rates().count(), 3);
        assert_eq!(StemFormat::Flac.rates().count(), OutputRate::ALL.len());
    }

    /// The MP3 encoder refuses 96 kHz even when the rate selection is bypassed, since LAME would
    /// accept the rate and resample.
    #[test]
    fn the_mp3_encoder_refuses_a_rate_it_has_no_mode_for() {
        use super::{StemFormat, encode};
        use crate::pcm::Audio;
        let audio = Audio::new(vec![vec![0.0; 4800], vec![0.0; 4800]], 96_000);
        let err = encode(&audio, StemFormat::Mp3, 1.0).unwrap_err();
        assert!(
            format!("{err:#}").contains("96000"),
            "unhelpful error: {err:#}"
        );
        // The same samples at a rate LAME has a mode for still encode.
        let audio = Audio::new(vec![vec![0.0; 4800], vec![0.0; 4800]], 48_000);
        assert!(!encode(&audio, StemFormat::Mp3, 1.0).unwrap().is_empty());
    }

    use super::*;

    fn tone(frames: usize) -> Audio {
        let mut left = Vec::with_capacity(frames);
        let mut right = Vec::with_capacity(frames);
        for i in 0..frames {
            let t = i as f32 / 44100.0;
            left.push((t * 440.0 * std::f32::consts::TAU).sin() * 0.6);
            right.push((t * 277.0 * std::f32::consts::TAU).sin() * 0.4);
        }
        Audio::new(vec![left, right], 44100)
    }

    /// `(min_block_size, max_block_size)` from STREAMINFO, which sits at a fixed
    /// offset: `fLaC`, a four-byte metadata header, then the block itself.
    fn declared_block_sizes(bytes: &[u8]) -> (u16, u16) {
        (
            u16::from_be_bytes([bytes[8], bytes[9]]),
            u16::from_be_bytes([bytes[10], bytes[11]]),
        )
    }

    /// Whether the first audio frame is coded variable-block-size, read from the
    /// blocking-strategy bit that follows the 14-bit frame sync.
    fn frames_are_variable_blocked(bytes: &[u8]) -> bool {
        let mut at = 4;
        loop {
            let last = bytes[at] & 0x80 != 0;
            let len = u32::from_be_bytes([0, bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize;
            at += 4 + len;
            if last {
                break;
            }
        }
        assert_eq!(bytes[at], 0xff, "expected a frame sync at {at}");
        bytes[at + 1] & 0x01 != 0
    }

    fn decode_flac(bytes: &[u8]) -> Vec<i16> {
        let mut reader =
            claxon::FlacReader::new(std::io::Cursor::new(bytes)).expect("a valid flac stream");
        reader
            .samples()
            .map(|s| s.expect("sample") as i16)
            .collect()
    }

    #[test]
    fn flac_is_smaller_than_the_pcm_it_replaces() {
        let audio = tone(44100);
        let pcm = encode(&audio, StemFormat::Pcm16, 1.0).unwrap();
        let flac = encode(&audio, StemFormat::Flac, 1.0).unwrap();
        assert!(
            flac.len() < pcm.len(),
            "flac {} not smaller than pcm {}",
            flac.len(),
            pcm.len()
        );
        assert_eq!(&flac[..4], b"fLaC", "must be a real FLAC stream");
    }

    #[test]
    fn flac_carries_exactly_the_s16_samples() {
        // Only the container differs. If these diverged, a client rebuilding the third part would
        // get a different answer depending on the format it asked for.
        let audio = tone(4096);
        let pcm = encode(&audio, StemFormat::Pcm16, 1.0).unwrap();
        let expected: Vec<i16> = pcm
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(
            decode_flac(&encode(&audio, StemFormat::Flac, 1.0).unwrap()),
            expected
        );
    }

    #[test]
    fn gain_is_applied_before_encoding() {
        let audio = tone(2048);
        let unity = decode_flac(&encode(&audio, StemFormat::Flac, 1.0).unwrap());
        let halved = decode_flac(&encode(&audio, StemFormat::Flac, 0.5).unwrap());
        let peak_unity = unity.iter().map(|v| v.abs() as i32).max().unwrap();
        let peak_halved = halved.iter().map(|v| v.abs() as i32).max().unwrap();
        assert!(
            (peak_halved * 2 - peak_unity).abs() <= 2,
            "gain not applied: {peak_unity} vs {peak_halved}"
        );
    }

    #[test]
    fn a_partial_final_block_still_declares_a_fixed_block_size() {
        // Use a length that is not a whole number of blocks: that makes the encoder's declared
        // minimum disagree with the frames it wrote. A decoder that sees `min != max` locates
        // frames by sample number and gives up partway through.
        let flac = encode(&tone(4096 * 3 + 777), StemFormat::Flac, 1.0).unwrap();
        let (min, max) = declared_block_sizes(&flac);
        assert_eq!(min, max, "declared variable block size: {min} != {max}");
        assert!(
            !frames_are_variable_blocked(&flac),
            "frames are variable-blocked, so STREAMINFO must not claim fixed"
        );
    }

    #[test]
    fn a_partial_final_block_decodes_to_every_sample() {
        let frames = 4096 * 3 + 777;
        let audio = tone(frames);
        assert_eq!(
            decode_flac(&encode(&audio, StemFormat::Flac, 1.0).unwrap()).len(),
            frames * 2,
            "decoder stopped short of the end"
        );
    }

    /// The 16-bit formats agree sample for sample. A client rebuilding the derived part by
    /// subtraction must get the same answer whichever format it asked for, and "wav" equals "flac
    /// without the compression" only if this holds.
    #[test]
    fn every_sixteen_bit_format_carries_the_same_samples() {
        let audio = tone(4096);
        let raw = encode(&audio, StemFormat::Pcm16, 1.0).unwrap();
        let expected: Vec<i16> = raw
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();

        assert_eq!(
            decode_flac(&encode(&audio, StemFormat::Flac, 1.0).unwrap()),
            expected
        );

        let wav = encode(&audio, StemFormat::Wav, 1.0).unwrap();
        let from_wav: Vec<i16> = wav[44..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(from_wav, expected);
    }

    /// A WAV header that every reader opens: the canonical 44 bytes, with the sizes and the
    /// geometry filled in.
    #[test]
    fn the_wav_header_describes_the_samples_that_follow() {
        let audio = tone(1000);
        let wav = encode(&audio, StemFormat::Wav, 1.0).unwrap();
        let u32_at = |at: usize| u32::from_le_bytes(wav[at..at + 4].try_into().unwrap());
        let u16_at = |at: usize| u16::from_le_bytes(wav[at..at + 2].try_into().unwrap());

        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u16_at(20), 1, "integer PCM");
        assert_eq!(u16_at(22), 2, "channels");
        assert_eq!(u32_at(24), 44100, "sample rate");
        assert_eq!(u16_at(34), 16, "bits per sample");
        assert_eq!(u32_at(28), 44100 * 4, "byte rate");
        assert_eq!(u16_at(32), 4, "block align");

        // The two sizes are what a reader trusts to find the end of the audio.
        let data = 1000 * 2 * 2;
        assert_eq!(u32_at(40) as usize, data, "data chunk size");
        assert_eq!(u32_at(4) as usize, 36 + data, "riff size");
        assert_eq!(wav.len(), 44 + data);
    }

    #[test]
    fn mp3_is_a_real_stream_and_smaller_than_the_samples() {
        let audio = tone(44100);
        let mp3 = encode(&audio, StemFormat::Mp3, 1.0).unwrap();
        let raw = encode(&audio, StemFormat::Pcm16, 1.0).unwrap();

        assert!(!mp3.is_empty(), "the encoder produced nothing");
        assert!(mp3.len() < raw.len(), "{} !< {}", mp3.len(), raw.len());
        // Either an ID3 tag or a frame sync: LAME writes one of the two first.
        let starts_framed = mp3.starts_with(b"ID3") || (mp3[0] == 0xFF && mp3[1] & 0xE0 == 0xE0);
        assert!(starts_framed, "not an mp3 stream: {:02X?}", &mp3[..4]);
    }

    /// The lossless property that the module docs rely on, checked directly.
    #[test]
    fn only_mp3_is_lossy_and_only_the_float_format_has_headroom() {
        for format in StemFormat::ALL {
            assert_eq!(
                format.is_lossless(),
                format != StemFormat::Mp3,
                "{format} lossless?"
            );
            assert_eq!(
                format.has_headroom(),
                format == StemFormat::Pcm32,
                "{format} headroom?"
            );
        }
    }

    /// FNV-1a over `bytes`, to pin encoder output in a test.
    fn fnv(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        })
    }

    #[test]
    fn each_format_encodes_to_the_recorded_bytes() {
        let audio = tone(4096 * 2 + 333);
        let got: Vec<u64> = [
            StemFormat::Flac,
            StemFormat::Wav,
            StemFormat::Pcm16,
            StemFormat::Pcm32,
        ]
        .into_iter()
        .map(|format| fnv(&encode(&audio, format, 0.8).expect("encoding")))
        .collect();
        assert_eq!(
            got,
            [
                6_879_416_745_213_006_112,
                17_670_478_727_201_919_131,
                6_547_095_000_092_806_396,
                7_324_643_346_458_908_958
            ]
        );
    }

    #[test]
    fn every_format_name_parses_back_to_itself() {
        for format in StemFormat::ALL {
            assert_eq!(format.to_string().parse::<StemFormat>().unwrap(), format);
        }
        for (spelling, expected) in [
            ("s16le", StemFormat::Pcm16),
            ("s16", StemFormat::Pcm16),
            ("f32le", StemFormat::Pcm32),
            ("f32", StemFormat::Pcm32),
        ] {
            assert_eq!(spelling.parse::<StemFormat>().unwrap(), expected);
        }
        assert!("opus".parse::<StemFormat>().is_err());
    }

    #[test]
    fn flac_is_never_an_upload_format() {
        assert!("flac".parse::<StemFormat>().is_ok());
        assert!("flac".parse::<PcmFormat>().is_err());
        assert_eq!(StemFormat::Flac.as_pcm(), None);
    }
}

#[cfg(test)]
mod encode_cost {
    use super::*;
    use crate::pcm::Audio;

    /// What the encoder configuration costs, on a real stem.
    // Re-run it after changing the encoder: the result depends on the material, the machine and the
    // crate version.
    //
    //     STEM_BENCH=/path/to/a/96k/stem.flac \
    //     cargo test --release -p stemd-core encode_cost -- --nocapture
    //
    // An 8:06 stem at 96 kHz, flacenc 0.5.1:
    //
    // | lpc_order      | encode  | size    |
    // |----------------|---------|---------|
    // | 10 (crate)     | 1.02 s  | 41.0 MB |
    // | 8 (shipped)    | 0.94 s  | 41.3 MB |
    // | 4              | 0.92 s  | 44.9 MB |
    // | 2              | 0.77 s  | 51.2 MB |
    // | none           | 0.47 s  | 51.3 MB |
    //
    // Order 2 is worse than turning LPC off on both counts. The encoder is already multithreaded:
    // the same stem takes 5.08 s single-threaded, and two stems take 2.20 s concurrently against
    // 2.38 s in sequence.
    #[test]
    fn encode_cost() {
        use flacenc::error::Verify;

        let Ok(path) = std::env::var("STEM_BENCH") else {
            eprintln!("STEM_BENCH unset, skipping");
            return;
        };
        let mut r = claxon::FlacReader::open(&path).unwrap();
        let rate = r.streaminfo().sample_rate;
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for (i, s) in r.samples().map(Result::unwrap).enumerate() {
            let v = s as f32 / 32768.0;
            if i % 2 == 0 {
                left.push(v)
            } else {
                right.push(v)
            }
        }
        let audio = Audio::new(vec![left, right], rate);
        eprintln!("source: {} frames @{rate} Hz", audio.frames());

        // Discard the first encode: it pays for faulting in a ~300 MB interleave buffer and
        // starting the encoder's threads, about 25%, which would be charged to the row that ran
        // first.
        let _ = encode(&audio, StemFormat::Flac, 1.0).unwrap();

        // Time the shipping configuration first, so the report shows what stemd costs as well as
        // what the alternatives would.
        let began = std::time::Instant::now();
        let shipped = encode(&audio, StemFormat::Flac, 1.0).unwrap();
        eprintln!(
            "{:18} {:>8.2?}  {:>5.1} MB   <- to_flac()",
            "SHIPPING",
            began.elapsed(),
            shipped.len() as f64 / 1_048_576.0
        );

        let base = flacenc::config::Encoder::default();
        let mut cases = vec![("crate default".to_owned(), base.clone())];
        for order in [8usize, 4, 2] {
            let mut c = base.clone();
            c.subframe_coding.qlpc.lpc_order = order;
            cases.push((format!("lpc_order={order}"), c));
        }
        let mut c = base.clone();
        c.subframe_coding.use_lpc = false;
        cases.push(("no_lpc".to_owned(), c));
        let mut c = base;
        c.multithread = false;
        cases.push(("single_threaded".to_owned(), c));

        for (name, cfg) in cases {
            let cfg = cfg.into_verified().unwrap();
            let began = std::time::Instant::now();
            let bytes = to_flac_with(&audio, 1.0, &cfg).unwrap();
            eprintln!(
                "{name:18} {:>8.2?}  {:>5.1} MB",
                began.elapsed(),
                bytes.len() as f64 / 1_048_576.0
            );
        }
    }
}

#[cfg(test)]
mod mp3_bench {
    use crate::pcm::Audio;
    use crate::stemfmt::{StemFormat, encode};

    /// A signal closer to music than a tone or noise: a moving harmonic stack, transients on a
    /// beat, and a little noise for air.
    fn musical(secs: f64, rate: u32) -> Audio {
        let n = (secs * f64::from(rate)) as usize;
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 8_388_608.0 - 1.0
        };
        let period = f64::from(rate) / 2.0; // 120 bpm
        let channels: Vec<Vec<f32>> = (0..2)
            .map(|c| {
                (0..n)
                    .map(|i| {
                        let t = i as f64 / f64::from(rate);
                        // A chord that slides, so the encoder cannot settle.
                        let root = 110.0 * (1.0 + 0.02 * (t * 0.3).sin());
                        let tone: f64 = (1..=8)
                            .map(|h| {
                                (std::f64::consts::TAU * root * f64::from(h) * t).sin()
                                    / f64::from(h)
                            })
                            .sum();
                        // A transient every half second: broadband, and where a
                        // codec's pre-echo would show up.
                        let phase = (i as f64 % period) / period;
                        let hit = (-40.0 * phase).exp();
                        0.30 * tone as f32
                            + 0.35 * hit as f32 * rand()
                            + 0.02 * rand()
                            + 0.03 * c as f32
                    })
                    .collect()
            })
            .collect();
        Audio::new(channels, rate)
    }

    use mp3lame_encoder::Quality;

    /// The levels to sweep, named as LAME names them. `MP3_QUALITY` is `NearBest`.
    fn quality_levels() -> [(&'static str, Quality); 6] {
        [
            ("Best", Quality::Best),
            ("SecondBest", Quality::SecondBest),
            ("NearBest", Quality::NearBest),
            ("VeryNice", Quality::VeryNice),
            ("Good", Quality::Good),
            ("Ok", Quality::Ok),
        ]
    }

    /// Decode an mp3 back to interleaved samples.
    fn decode_mp3(bytes: &[u8]) -> Vec<f32> {
        use symphonia::core::codecs::DecoderOptions;
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;
        use symphonia::core::probe::Hint;

        let source = std::io::Cursor::new(bytes.to_vec());
        let mss = MediaSourceStream::new(Box::new(source), Default::default());
        let mut hint = Hint::new();
        hint.with_extension("mp3");
        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .expect("probe");
        let mut format = probed.format;
        let track = format.default_track().expect("track").clone();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .expect("decoder");

        let mut out = Vec::new();
        while let Ok(packet) = format.next_packet() {
            let Ok(buf) = decoder.decode(&packet) else {
                continue;
            };
            let mut sample_buf = symphonia::core::audio::SampleBuffer::<f32>::new(
                buf.capacity() as u64,
                *buf.spec(),
            );
            sample_buf.copy_interleaved_ref(buf);
            out.extend_from_slice(sample_buf.samples());
        }
        out
    }

    fn rms_db(samples: &[f32]) -> f64 {
        if samples.is_empty() {
            return f64::NEG_INFINITY;
        }
        let sum: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
        20.0 * (sum / samples.len() as f64).sqrt().log10()
    }

    /// What lowering the quality setting costs. Each setting gets two numbers: how far its output
    /// sits from `q=0`'s, and the coding noise both add against the input. A distance well below
    /// that coding noise is inaudible.
    #[test]
    #[ignore]
    fn what_does_the_quality_knob_cost() {
        let audio = musical(20.0, 44_100);
        let source: Vec<f32> = (0..audio.frames())
            .flat_map(|i| [audio.data[0][i], audio.data[1][i]])
            .collect();

        let reference = decode_mp3(&super::to_mp3_at(&audio, 1.0, Quality::Best).unwrap());
        // LAME's encoder delay, found by search: the offset at which the decoded output best
        // matches the input.
        let lag = (0..3000)
            .min_by(|a, b| {
                let err = |off: usize| -> f64 {
                    (0..20_000)
                        .map(|i| {
                            let d = f64::from(reference[off * 2 + i]) - f64::from(source[i]);
                            d * d
                        })
                        .sum()
                };
                err(*a).partial_cmp(&err(*b)).unwrap()
            })
            .unwrap();

        let noise = |decoded: &[f32]| -> f64 {
            let n = (source.len() - lag * 2).min(decoded.len() - lag * 2);
            let diff: Vec<f32> = (0..n).map(|i| decoded[lag * 2 + i] - source[i]).collect();
            rms_db(&diff)
        };

        println!(
            "source {:.1} dBFS, encoder delay {lag} frames",
            rms_db(&source)
        );
        println!("  {:>10}   vs Best   coding noise vs source", "quality");
        for (name, q) in quality_levels() {
            let decoded = decode_mp3(&super::to_mp3_at(&audio, 1.0, q).unwrap());
            let n = decoded.len().min(reference.len());
            let delta: Vec<f32> = (0..n).map(|i| decoded[i] - reference[i]).collect();
            println!(
                "  {name:>10}  {:>8.1} dB   {:>8.1} dB",
                rms_db(&delta),
                noise(&decoded)
            );
        }
    }

    #[test]
    #[ignore]
    fn how_long_does_mp3_take() {
        let audio = musical(60.0, 44_100);
        println!("60 s of stereo 44.1 kHz material");
        for (name, q) in quality_levels() {
            let began = std::time::Instant::now();
            let bytes = super::to_mp3_at(&audio, 1.0, q).unwrap();
            let took = began.elapsed();
            println!(
                "  {name:>10}: {:>8.2?}  ({:.1} MB, {:.1}x realtime)",
                took,
                bytes.len() as f64 / 1e6,
                60.0 / took.as_secs_f64()
            );
        }
        // For comparison, time FLAC.
        let began = std::time::Instant::now();
        let bytes = encode(&audio, StemFormat::Flac, 1.0).unwrap();
        println!(
            "  flac: {:>7.2?}  ({:.1} MB)",
            began.elapsed(),
            bytes.len() as f64 / 1e6
        );
    }
}
