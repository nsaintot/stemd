//! PCM buffers on the wire and in memory.
//!
//! The wire format is interleaved; everything internal is planar `[channel][sample]`
//! so the STFT and the model see contiguous per-channel slices.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Divisor turning an `i16` into `[-1.0, 1.0)`. The full negative magnitude, so
/// `i16::MIN` maps to exactly `-1.0` and no input can exceed the range.
const S16_TO_FLOAT: f32 = 32768.0;

/// Multiplier turning `[-1.0, 1.0]` back into an `i16`. One less than the
/// divisor above, so `+1.0` lands on `i16::MAX` rather than wrapping.
const FLOAT_TO_S16: f32 = 32767.0;

/// Quantise one sample to 16-bit, clamping first: a separated stem is not
/// bounded by the mix and can sit outside [-1.0, 1.0].
///
/// Shared with the FLAC encoder, so a client gets the same integers whichever
/// container it asked for.
pub(crate) fn quantise_s16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * FLOAT_TO_S16).round() as i16
}

/// Sample encoding used on the wire and for stem output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PcmFormat {
    /// 16-bit signed little-endian. The default for stem output: the separation
    /// residual sits far above a -96 dB noise floor, so float32 doubles the
    /// transfer for nothing.
    S16le,
    /// 32-bit float little-endian.
    F32le,
}

impl PcmFormat {
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::S16le => 2,
            Self::F32le => 4,
        }
    }
}

impl fmt::Display for PcmFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::S16le => "s16le",
            Self::F32le => "f32le",
        })
    }
}

impl FromStr for PcmFormat {
    type Err = PcmError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "s16le" | "s16" => Ok(Self::S16le),
            "f32le" | "f32" => Ok(Self::F32le),
            other => Err(PcmError::UnknownFormat(other.to_owned())),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PcmError {
    #[error("unknown pcm format {0:?}, expected s16le or f32le")]
    UnknownFormat(String),
    #[error("payload of {len} bytes is not a whole number of {channels}-channel frames")]
    Ragged { len: usize, channels: usize },
    /// Channels of unequal length. Every reader takes the frame count from the
    /// first channel, so this is a panic waiting for whichever one gets there.
    #[error("channel {channel} has {found} samples where channel 0 has {frames}")]
    RaggedChannels {
        channel: usize,
        found: usize,
        frames: usize,
    },
    #[error("expected {expected} channels, got {got}")]
    ChannelCount { expected: usize, got: usize },
    /// A sample that is NaN or infinite. See [`Audio::peak`] for why this is an
    /// error rather than something to clamp away.
    #[error("sample {frame} of channel {channel} is {value}, not a finite number")]
    NotFinite {
        channel: usize,
        frame: usize,
        value: f32,
    },
}

/// Planar float audio: `data[channel][sample]`, nominally in [-1.0, 1.0].
#[derive(Debug, Clone)]
pub struct Audio {
    pub data: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

impl Audio {
    /// For data whose geometry the caller already knows to be rectangular,
    /// which is every internal producer: they build each channel from the same
    /// frame count in the same loop.
    ///
    /// The assertion is not idle. `frames()` is the first channel's length and
    /// the rest are indexed with it, so a shorter one is a panic rather than a
    /// wrong answer, and it lands in whichever of the model, the resampler or
    /// the encoder reaches it first. Untrusted data goes through
    /// [`Self::checked`] instead.
    pub fn new(data: Vec<Vec<f32>>, sample_rate: u32) -> Self {
        debug_assert!(
            data.iter().all(|c| c.len() == data.first().map_or(0, Vec::len)),
            "Audio::new was handed ragged channels: {:?}",
            data.iter().map(Vec::len).collect::<Vec<_>>()
        );
        Self { data, sample_rate }
    }

    /// The same, for data that came from outside: a decoder, a file, a client.
    ///
    /// Refused rather than trimmed to the shortest channel. A decode that
    /// disagrees with itself about how long the track is has already gone wrong,
    /// and quietly dropping the difference would hand the model a track missing
    /// a piece nobody chose to lose.
    pub fn checked(data: Vec<Vec<f32>>, sample_rate: u32) -> Result<Self, PcmError> {
        let frames = data.first().map_or(0, Vec::len);
        if let Some((channel, found)) = data
            .iter()
            .enumerate()
            .map(|(c, samples)| (c, samples.len()))
            .find(|(_, len)| *len != frames)
        {
            return Err(PcmError::RaggedChannels {
                channel,
                found,
                frames,
            });
        }
        Ok(Self { data, sample_rate })
    }

    pub fn channels(&self) -> usize {
        self.data.len()
    }

    pub fn frames(&self) -> usize {
        self.data.first().map_or(0, Vec::len)
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / f64::from(self.sample_rate)
    }

    /// Allocate silence with the same geometry.
    pub fn silence_like(&self) -> Self {
        Self {
            data: self.data.iter().map(|c| vec![0.0; c.len()]).collect(),
            sample_rate: self.sample_rate,
        }
    }

    /// Decode an interleaved wire payload.
    pub fn from_interleaved(
        bytes: &[u8],
        format: PcmFormat,
        channels: usize,
        sample_rate: u32,
    ) -> Result<Self, PcmError> {
        let stride = format.bytes_per_sample() * channels;
        if channels == 0 || !bytes.len().is_multiple_of(stride) {
            return Err(PcmError::Ragged {
                len: bytes.len(),
                channels,
            });
        }
        let frames = bytes.len() / stride;
        let mut data = vec![Vec::with_capacity(frames); channels];

        match format {
            PcmFormat::S16le => {
                for frame in bytes.chunks_exact(stride) {
                    for (ch, sample) in frame.chunks_exact(2).enumerate() {
                        let v = i16::from_le_bytes([sample[0], sample[1]]);
                        data[ch].push(f32::from(v) / S16_TO_FLOAT);
                    }
                }
            }
            PcmFormat::F32le => {
                for frame in bytes.chunks_exact(stride) {
                    for (ch, sample) in frame.chunks_exact(4).enumerate() {
                        data[ch].push(f32::from_le_bytes([
                            sample[0], sample[1], sample[2], sample[3],
                        ]));
                    }
                }
            }
        }
        Ok(Self { data, sample_rate })
    }

    /// Peak absolute sample value, or [`PcmError::NotFinite`] if any sample is NaN or
    /// infinite.
    ///
    /// The check rides on this traversal because it is already the pass that reads
    /// every sample of a finished stem.
    ///
    /// An error rather than a clamp, because the failure is silent in both
    /// directions. `f32::max` returns the non-NaN operand, so a stem that is entirely
    /// NaN peaks at `0.0` and quantises to digital silence, which nothing downstream
    /// would question. One sample of `inf` peaks the whole stem at `inf`, and the
    /// transfer gain `1.0 / inf` then scales every other sample to zero.
    pub fn peak(&self) -> Result<f32, PcmError> {
        let mut peak = 0.0f32;
        for (channel, samples) in self.data.iter().enumerate() {
            for (frame, &value) in samples.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PcmError::NotFinite {
                        channel,
                        frame,
                        value,
                    });
                }
                peak = peak.max(value.abs());
            }
        }
        Ok(peak)
    }

    /// Encode to an interleaved wire payload, scaling by `gain` first.
    ///
    /// One shared gain across all stems rather than a per-stem clamp: stems are not
    /// individually bounded by the mix, and clamping each would destroy the exact-sum
    /// property. Scaling is linear, so the sum survives; the client is handed `gain`
    /// to undo.
    pub fn to_interleaved_scaled(&self, format: PcmFormat, gain: f32) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(self.frames() * self.channels() * format.bytes_per_sample());
        for i in 0..self.frames() {
            for ch in &self.data {
                let v = ch[i] * gain;
                match format {
                    PcmFormat::S16le => out.extend_from_slice(&quantise_s16(v).to_le_bytes()),
                    PcmFormat::F32le => out.extend_from_slice(&v.to_le_bytes()),
                }
            }
        }
        out
    }

    /// Encode to an interleaved wire payload at unity gain.
    pub fn to_interleaved(&self, format: PcmFormat) -> Vec<u8> {
        self.to_interleaved_scaled(format, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rectangular data is what almost everything is, and it must not become an
    /// error on the way in.
    #[test]
    fn checked_accepts_rectangular_channels() {
        let audio = Audio::checked(vec![vec![0.0, 0.5], vec![0.25, -0.25]], 44100)
            .expect("equal channels are not ragged");
        assert_eq!(audio.frames(), 2);
        assert_eq!(audio.channels(), 2);
        // Nothing is a shape either, and the model refuses it by frame count
        // rather than here.
        assert!(Audio::checked(Vec::new(), 44100).is_ok());
        assert!(Audio::checked(vec![Vec::new(), Vec::new()], 44100).is_ok());
    }

    /// The shape behind the first crash report against a release: channel 0 with
    /// samples in it and channel 1 empty. `frames()` reads channel 0, so every
    /// reader downstream indexes channel 1 at 0 and panics with "the len is 0
    /// but the index is 0". It has to be an error here, where the file can be
    /// named, not a panic three crates later.
    #[test]
    fn checked_refuses_ragged_channels() {
        let err = Audio::checked(vec![vec![0.1; 4096], Vec::new()], 44100)
            .expect_err("an empty second channel is ragged");
        assert!(matches!(
            err,
            PcmError::RaggedChannels {
                channel: 1,
                found: 0,
                frames: 4096
            }
        ));
        // The message names both lengths, because "ragged" alone does not tell
        // anyone which channel came up short.
        let text = err.to_string();
        assert!(text.contains("channel 1"), "{text}");
        assert!(text.contains("4096"), "{text}");

        // A near miss is refused on the same terms as an empty one.
        assert!(Audio::checked(vec![vec![0.1; 100], vec![0.1; 99]], 44100).is_err());
    }

    /// The reader that turned raggedness into a panic, held here so the
    /// guarantee `checked` provides is the one the encoders rely on.
    #[test]
    fn a_rectangular_buffer_interleaves_every_frame() {
        let audio = Audio::checked(vec![vec![0.0, 0.5], vec![0.25, -0.25]], 44100).unwrap();
        let bytes = audio.to_interleaved(PcmFormat::F32le);
        assert_eq!(bytes.len(), 2 * 2 * 4);
    }

    #[test]
    fn interleaved_round_trip_f32() {
        let audio = Audio::new(vec![vec![0.0, 0.5, -0.5], vec![0.25, -0.25, 1.0]], 44100);
        let bytes = audio.to_interleaved(PcmFormat::F32le);
        let back = Audio::from_interleaved(&bytes, PcmFormat::F32le, 2, 44100).unwrap();
        assert_eq!(back.data, audio.data);
    }

    #[test]
    fn interleaved_round_trip_s16_is_lossy_but_close() {
        let audio = Audio::new(vec![vec![0.0, 0.5, -0.5]], 44100);
        let bytes = audio.to_interleaved(PcmFormat::S16le);
        let back = Audio::from_interleaved(&bytes, PcmFormat::S16le, 1, 44100).unwrap();
        for (a, b) in audio.data[0].iter().zip(&back.data[0]) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn ragged_payload_is_rejected() {
        let err = Audio::from_interleaved(&[0u8; 5], PcmFormat::F32le, 2, 44100);
        assert!(err.is_err());
    }

    #[test]
    fn peak_is_the_largest_magnitude_either_way() {
        let audio = Audio::new(vec![vec![0.1, -0.9], vec![0.5, 0.2]], 44100);
        assert!((audio.peak().expect("finite") - 0.9).abs() < 1e-6);
        assert_eq!(Audio::new(vec![vec![]], 44100).peak().expect("finite"), 0.0);
    }

    /// The failure this guard exists for, stated as the arithmetic that hides
    /// it: fold `f32::max` over NaN and the answer is 0.0, because `max`
    /// returns the operand that is not NaN. A stem of nothing but NaN therefore
    /// *looks* like a stem of silence, and a silent stem is not a fault here.
    #[test]
    fn an_all_nan_stem_would_otherwise_pass_for_silence() {
        let nan: Vec<f32> = vec![f32::NAN; 8];
        let unguarded = nan.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        assert_eq!(unguarded, 0.0, "the premise of the guard no longer holds");
        // And every one of those samples quantises to the middle of the scale,
        // so what got published was digital silence rather than an error.
        assert_eq!(quantise_s16(f32::NAN), 0);

        let err = Audio::new(vec![nan], 44100)
            .peak()
            .expect_err("must not pass");
        assert!(matches!(err, PcmError::NotFinite { channel: 0, .. }));
    }

    /// Infinity fails the opposite way and just as quietly: it peaks the stem
    /// at `inf`, and a transfer gain of `1.0 / inf` scales everything else to
    /// zero. Both are caught by asking for finite rather than for not-NaN.
    #[test]
    fn one_infinite_sample_is_caught_and_located() {
        let audio = Audio::new(
            vec![vec![0.1, 0.2, 0.3], vec![0.4, f32::NEG_INFINITY, 0.6]],
            44100,
        );
        let err = audio.peak().expect_err("must not pass");
        let PcmError::NotFinite {
            channel,
            frame,
            value,
        } = err
        else {
            panic!("wrong variant: {err}");
        };
        // Located, not merely detected, which sample diverged is the first
        // thing anyone debugging a model asks.
        assert_eq!((channel, frame), (1, 1));
        assert_eq!(value, f32::NEG_INFINITY);
    }
}
