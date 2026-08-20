//! Writing the assembled parts to disk, and the levels reported alongside them.
//!
//! The FLAC encoder mirrors the server's rather than sharing it. The block-size
//! correction below is the part that must not drift, and
//! `flac_declares_a_fixed_block_size` holds both copies to it.

use std::path::Path;

use anyhow::Result;

/// Planar float samples, `data[channel][sample]`.
pub type Planar = Vec<Vec<f32>>;

/// One part written out, and where it came from.
pub struct Part {
    /// "downloaded" or "rebuilt here", which side of the wire produced it.
    pub origin: &'static str,
    pub name: String,
    pub samples: Planar,
}

/// How the parts are written.
#[derive(Clone, Copy)]
pub enum Encoding {
    /// 16-bit FLAC, scaled by a shared gain so the files stay comparable.
    Flac { gain: f32 },
    /// Unscaled float32 wav, for a DAW that wants the samples untouched.
    WavF32,
}

impl Encoding {
    /// Pick an encoding, scaling to fit 16 bits when that is where it is going.
    ///
    /// One shared scale across every part, so they can be auditioned against each
    /// other. Downloaded stems are re-encoded rather than written as they arrived
    /// because each carries its own transfer gain.
    pub fn choose(parts: &[Part], wav_f32: bool) -> Self {
        if wav_f32 {
            return Self::WavF32;
        }
        let peak = parts
            .iter()
            .flat_map(|part| part.samples.iter())
            .flat_map(|ch| ch.iter())
            .fold(0.0f32, |acc, v| acc.max(v.abs()));
        Self::Flac {
            gain: if peak <= 1.0 { 1.0 } else { 1.0 / peak },
        }
    }

    const fn extension(self) -> &'static str {
        match self {
            Self::Flac { .. } => "flac",
            Self::WavF32 => "wav",
        }
    }
}

/// Write one part, returning the filename it landed under.
pub fn write(dir: &Path, part: &Part, sample_rate: u32, encoding: Encoding) -> Result<String> {
    let file = format!("{}.{}", part.name, encoding.extension());
    let path = dir.join(&file);
    match encoding {
        Encoding::Flac { gain } => {
            std::fs::write(path, encode_flac(&part.samples, sample_rate, gain)?)?;
        }
        Encoding::WavF32 => write_wav_f32(&path, &part.samples, sample_rate)?,
    }
    Ok(file)
}

/// Encode planar float to 16-bit FLAC.
fn encode_flac(planar: &[Vec<f32>], sample_rate: u32, gain: f32) -> Result<Vec<u8>> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let channels = planar.len();
    anyhow::ensure!(channels > 0, "cannot encode zero channels to flac");

    let mut interleaved = Vec::with_capacity(planar[0].len() * channels);
    for i in 0..planar[0].len() {
        for ch in planar {
            interleaved.push(i32::from(quantise_s16(ch[i] * gain)));
        }
    }

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|e| anyhow::anyhow!("flac encoder config: {e:?}"))?;
    let source =
        flacenc::source::MemSource::from_samples(&interleaved, channels, 16, sample_rate as usize);
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| anyhow::anyhow!("flac encode: {e}"))?;

    // The encoder folds the trailing partial block into STREAMINFO's minimum,
    // which declares the stream variable-block-size while its frames are written
    // fixed. Decoders that believe it locate frames by sample number and stop
    // early.
    stream
        .stream_info_mut()
        .set_block_sizes(config.block_size, config.block_size)
        .map_err(|e| anyhow::anyhow!("flac block sizes: {e:?}"))?;

    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| anyhow::anyhow!("flac serialise: {e}"))?;
    Ok(sink.into_inner())
}

fn quantise_s16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

fn write_wav_f32(path: &Path, planar: &[Vec<f32>], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for i in 0..planar[0].len() {
        for ch in planar {
            writer.write_sample(ch[i])?;
        }
    }
    writer.finalize()?;
    Ok(())
}

/// `a - b`, channel-wise, tolerating a shorter `b`.
pub fn subtract(a: &[Vec<f32>], b: &[Vec<f32>]) -> Planar {
    a.iter()
        .enumerate()
        .map(|(c, ch)| {
            ch.iter()
                .enumerate()
                .map(|(i, v)| v - at(b, c, i))
                .collect()
        })
        .collect()
}

fn at(planar: &[Vec<f32>], c: usize, i: usize) -> f32 {
    planar
        .get(c)
        .and_then(|ch| ch.get(i))
        .copied()
        .unwrap_or(0.0)
}

pub fn rms_dbfs(planar: &[Vec<f32>]) -> f64 {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for ch in planar {
        for v in ch {
            sum += f64::from(*v) * f64::from(*v);
            n += 1;
        }
    }
    if n == 0 || sum == 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * (sum / n as f64).sqrt().log10()
}

/// Level of `mix - sum(parts)` relative to `mix`, in dB.
///
/// With the third part derived by subtraction this is exact arithmetic, so
/// anything above roughly -120 dB means a part went missing rather than that the
/// model separated badly. Model quality is a different number entirely.
pub fn reconstruction_null_db(mix: &[Vec<f32>], parts: &[Part]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (c, mix_ch) in mix.iter().enumerate() {
        for (i, m) in mix_ch.iter().enumerate() {
            let sum: f32 = parts.iter().map(|part| at(&part.samples, c, i)).sum();
            num += f64::from(m - sum).powi(2);
            den += f64::from(*m).powi(2);
        }
    }
    if den == 0.0 || num == 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * (num / den).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deliberately not a whole number of blocks: the trailing partial block is
    /// what makes an encoder mis-declare the stream.
    fn tone(frames: usize) -> Planar {
        (0..2)
            .map(|c| {
                (0..frames)
                    .map(|i| {
                        let t = i as f32 / 44100.0;
                        (t * (220.0 + 111.0 * c as f32) * std::f32::consts::TAU).sin() * 0.5
                    })
                    .collect()
            })
            .collect()
    }

    fn part(name: &str, samples: Planar) -> Part {
        Part {
            origin: "test",
            name: name.to_owned(),
            samples,
        }
    }

    #[test]
    fn flac_declares_a_fixed_block_size() {
        let flac = encode_flac(&tone(4096 * 3 + 777), 44100, 1.0).unwrap();
        // STREAMINFO sits at a fixed offset: `fLaC`, a four-byte metadata
        // header, then min and max block size as two big-endian u16s.
        let min = u16::from_be_bytes([flac[8], flac[9]]);
        let max = u16::from_be_bytes([flac[10], flac[11]]);
        assert_eq!(min, max, "declared variable block size: {min} != {max}");
    }

    #[test]
    fn a_written_stem_reads_back_whole() {
        let frames = 4096 * 3 + 777;
        let flac = encode_flac(&tone(frames), 44100, 1.0).unwrap();
        let back = stemd_audio::decode_flac(flac).expect("decoding a stem this client wrote");
        assert_eq!(back.frames(), frames, "decoder stopped short of the end");
    }

    #[test]
    fn subtraction_tolerates_a_shorter_operand() {
        let a = vec![vec![1.0, 1.0, 1.0]];
        let b = vec![vec![0.25]];
        assert_eq!(subtract(&a, &b), vec![vec![0.75, 1.0, 1.0]]);
    }

    #[test]
    fn parts_that_sum_to_the_mix_null_out() {
        let mix = vec![vec![1.0, 0.5, -0.25]];
        let parts = vec![
            part("a", vec![vec![0.75, 0.25, -0.25]]),
            part("b", vec![vec![0.25, 0.25, 0.0]]),
        ];
        assert!(
            reconstruction_null_db(&mix, &parts) < -100.0,
            "an exact reconstruction must null"
        );
    }

    #[test]
    fn a_missing_part_shows_up_as_a_poor_null() {
        let mix = vec![vec![1.0, 0.5, -0.25]];
        let parts = vec![part("a", vec![vec![0.75, 0.25, -0.25]])];
        assert!(reconstruction_null_db(&mix, &parts) > -20.0);
    }

    #[test]
    fn the_shared_gain_brings_a_hot_part_back_under_full_scale() {
        // A separated stem can peak above full scale, which would clip in 16-bit.
        let parts = vec![part("hot", vec![vec![1.5, -0.5]])];
        let Encoding::Flac { gain } = Encoding::choose(&parts, false) else {
            panic!("expected flac");
        };
        assert!((gain - 1.0 / 1.5).abs() < 1e-6);
        assert!(1.5 * gain <= 1.0);
    }

    #[test]
    fn a_part_already_within_range_is_not_touched() {
        let parts = vec![part("quiet", vec![vec![0.5, -0.25]])];
        let Encoding::Flac { gain } = Encoding::choose(&parts, false) else {
            panic!("expected flac");
        };
        assert_eq!(gain, 1.0, "scaling what already fits only loses bits");
    }

    #[test]
    fn float_output_is_never_scaled() {
        let parts = vec![part("hot", vec![vec![1.5]])];
        assert!(
            matches!(Encoding::choose(&parts, true), Encoding::WavF32),
            "f32 has no ceiling to scale into"
        );
    }
}
