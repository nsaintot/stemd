//! Stem topology, and the player-side maths that consumes it.
//!
//! The server produces exactly [`SHIPPED`]. A player drives three faders and
//! reconstructs the third part itself:
//!
//! ```text
//! drums = mix - harmonics - vocals
//! ```
//!
//! so the parts sum to the mix by construction, for every sample, without relying
//! on the model's own sources summing to its input. See
//! [`Stems::model_residual_db`].
//!
//! Evaluating the faders never materialises that third part. Substituting
//! `S_drums` into `out = sum(g_q * S_q)` collapses to
//!
//! ```text
//! out = g_drums * mix + (g_harmonics - g_drums) * harmonics
//!                     + (g_vocals    - g_drums) * vocals
//! ```
//!
//! so a player mixes against buffers it already holds. At unity both stem
//! coefficients are exactly zero and the result is the mix, bit for bit.
//!
//! **Where the residual lands is the preset's business, not the player's.**
//! `Fast` ships `harmonics` as the model's `bass + other`, so the leftover from
//! four sources that do not quite sum arrives on the rebuilt drums. `Balanced`
//! and `Quality` build `harmonics` as `mix - vocals - drums`, so the leftover is
//! inside `harmonics`.
//!
//! None of that reaches the player. The server's subtraction and the client's are
//! inverses: `mix - harmonics - vocals` returns whatever the server called drums.

use crate::pcm::Audio;

/// The stems the server produces and transfers.
///
/// `harmonics` is the model's `bass` and `other` summed: the split is not one a
/// DJ reaches for, and merging halves what crosses the wire.
pub const SHIPPED: [&str; 2] = ["harmonics", "vocals"];

/// Source order used by the four-stem models (demucs, Open-Unmix).
pub const FOUR_STEM_SOURCES: [&str; 4] = ["drums", "bass", "other", "vocals"];

/// Which model sources make up each shipped stem. The model's `drums` is in
/// neither, which is what makes it the part a player rebuilds for itself.
pub const STEM_SOURCES: [(&str, &[&str]); 2] =
    [("harmonics", &["bass", "other"]), ("vocals", &["vocals"])];

/// The three things a player puts a fader on.
///
/// Player-side only. The server produces [`SHIPPED`] and has no notion of a
/// third part.
pub const PARTS: [&str; 3] = ["drums", "harmonics", "vocals"];

/// The part a player reconstructs rather than receiving. Player-side only, and
/// fixed. See the module docs for why it is `drums`.
pub const DERIVED: &str = "drums";

/// Fader positions, one per [`PARTS`] entry. `1.0` is unity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartGains {
    pub drums: f32,
    pub harmonics: f32,
    pub vocals: f32,
}

impl PartGains {
    pub const UNITY: Self = Self {
        drums: 1.0,
        harmonics: 1.0,
        vocals: 1.0,
    };

    pub fn get(self, part: &str) -> f32 {
        match part {
            "drums" => self.drums,
            "harmonics" => self.harmonics,
            "vocals" => self.vocals,
            _ => 0.0,
        }
    }

    /// True when this is exactly unity, so mixing can return the mix untouched.
    ///
    /// Exact equality is deliberate: the guarantee is bit-for-bit identity, so
    /// a UI must snap its faders to a literal `1.0` rather than land near it.
    pub fn is_unity(self) -> bool {
        self == Self::UNITY
    }
}

impl Default for PartGains {
    fn default() -> Self {
        Self::UNITY
    }
}

#[derive(Debug, Clone)]
pub struct Stems {
    /// The stems in [`SHIPPED`] order. These are all the server produces.
    pub shipped: Vec<(&'static str, Audio)>,
    /// Level of the model's own reconstruction error `mix - sum(sources)`, relative to
    /// the mix, in dB.
    ///
    /// Diagnostic, not a correctness gate: the three parts still sum exactly. It only
    /// means anything where the model's sources are supposed to sum to the mix and
    /// miss. An arrangement that builds one part by subtraction has no such redundancy
    /// and collapses to float noise.
    pub model_residual_db: f64,
}

impl Stems {
    pub fn stem(&self, name: &str) -> Option<&Audio> {
        self.shipped
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, audio)| audio)
    }

    /// The derived part as the player reconstructs it: `mix - shipped`.
    ///
    /// This is the model's own version plus the residual.
    pub fn derived_part(&self, mix: &Audio) -> Audio {
        let mut out = mix.clone();
        for (_, stem) in &self.shipped {
            for (c, ch) in out.data.iter_mut().enumerate() {
                for (i, sample) in ch.iter_mut().enumerate() {
                    *sample -= at(stem, c, i);
                }
            }
        }
        out
    }

    /// Apply the three faders.
    ///
    /// Unity short-circuits to a copy of the mix: the bit-exactness guarantee
    /// made explicit, so a later change to the arithmetic cannot break it.
    pub fn mix(&self, mix: &Audio, gains: PartGains) -> Audio {
        if gains.is_unity() {
            return mix.clone();
        }
        let base = gains.get(DERIVED);
        let terms: Vec<(f32, &Audio)> = self
            .shipped
            .iter()
            .map(|(name, audio)| (gains.get(name) - base, audio))
            .collect();

        let mut out = mix.clone();
        for (c, ch) in out.data.iter_mut().enumerate() {
            for (i, sample) in ch.iter_mut().enumerate() {
                let mut acc = base * *sample;
                for (k, stem) in &terms {
                    acc += k * at(stem, c, i);
                }
                *sample = acc;
            }
        }
        out
    }

    /// Render one part as the player would hear it soloed.
    pub fn part(&self, name: &str, mix: &Audio) -> Option<Audio> {
        if name == DERIVED {
            return Some(self.derived_part(mix));
        }
        self.stem(name).cloned()
    }
}

/// Sample `c`,`i` of an audio buffer, or silence past its end.
fn at(audio: &Audio, c: usize, i: usize) -> f32 {
    audio
        .data
        .get(c)
        .and_then(|ch| ch.get(i))
        .copied()
        .unwrap_or(0.0)
}

/// Level of `mix - sum(sources)` relative to `mix`, in dB.
pub fn unexplained_db(mix: &Audio, sources: &[Audio]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (c, mix_ch) in mix.data.iter().enumerate() {
        for (i, m) in mix_ch.iter().enumerate() {
            let mut sum = 0.0f32;
            for src in sources {
                sum += at(src, c, i);
            }
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

    fn audio(v: &[f32]) -> Audio {
        Audio::new(vec![v.to_vec(), v.to_vec()], 44100)
    }

    /// A mix whose parts deliberately do NOT sum to it, so the residual is real
    /// and lands on [`DERIVED`].
    fn fixture() -> (Audio, Stems) {
        let mix = audio(&[1.0, 0.5, -0.25, 0.125]);
        // The model's own drums is in `flat` but never shipped: it is what the
        // residual is measured against, exactly as the backend does it.
        let model_drums = audio(&[0.30, 0.10, -0.05, 0.20]);
        let shipped = vec![
            ("harmonics", audio(&[0.25, 0.30, 0.12, -0.09])),
            ("vocals", audio(&[0.40, 0.05, -0.15, 0.00])),
        ];
        let mut flat: Vec<Audio> = shipped.iter().map(|(_, a)| a.clone()).collect();
        flat.push(model_drums);
        let stems = Stems {
            model_residual_db: unexplained_db(&mix, &flat),
            shipped,
        };
        (mix, stems)
    }

    #[test]
    fn the_three_parts_sum_to_the_mix() {
        let (mix, stems) = fixture();
        let reconstructed = stems.derived_part(&mix);
        for c in 0..mix.channels() {
            for i in 0..mix.frames() {
                let sum: f32 = stems.shipped.iter().map(|(_, a)| a.data[c][i]).sum::<f32>()
                    + reconstructed.data[c][i];
                assert!(
                    (sum - mix.data[c][i]).abs() < 1e-6,
                    "ch{c} sample{i}: {sum} vs {}",
                    mix.data[c][i]
                );
            }
        }
    }

    #[test]
    fn unity_returns_the_mix_bit_for_bit() {
        let (mix, stems) = fixture();
        let out = stems.mix(&mix, PartGains::UNITY);
        assert_eq!(out.data, mix.data, "unity altered a sample");
    }

    #[test]
    fn a_shipped_fader_at_zero_removes_exactly_that_stem() {
        let (mix, stems) = fixture();
        let gains = PartGains {
            vocals: 0.0,
            ..PartGains::UNITY
        };
        let out = stems.mix(&mix, gains);
        let vocals = stems.stem("vocals").unwrap();
        for c in 0..mix.channels() {
            for i in 0..mix.frames() {
                let want = mix.data[c][i] - vocals.data[c][i];
                assert!((out.data[c][i] - want).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn killing_the_derived_fader_removes_the_residual_too() {
        let (mix, stems) = fixture();
        let gains = PartGains {
            drums: 0.0,
            ..PartGains::UNITY
        };
        let out = stems.mix(&mix, gains);
        // Only the three shipped stems remain: no leftover rides along.
        for c in 0..mix.channels() {
            for i in 0..mix.frames() {
                let want: f32 = stems.shipped.iter().map(|(_, a)| a.data[c][i]).sum();
                assert!((out.data[c][i] - want).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn soloing_a_shipped_part_is_the_model_output_untouched() {
        let (mix, stems) = fixture();
        let gains = PartGains {
            drums: 0.0,
            harmonics: 1.0,
            vocals: 0.0,
        };
        let out = stems.mix(&mix, gains);
        let harmonics = stems.stem("harmonics").unwrap();
        for c in 0..mix.channels() {
            for i in 0..mix.frames() {
                assert!(
                    (out.data[c][i] - harmonics.data[c][i]).abs() < 1e-6,
                    "a shipped solo must not pick up the residual"
                );
            }
        }
    }

    #[test]
    fn near_unity_is_not_unity() {
        let gains = PartGains {
            vocals: 0.9997,
            ..PartGains::UNITY
        };
        assert!(!gains.is_unity(), "only an exact 1.0 may short-circuit");
    }

    #[test]
    fn unexplained_is_zero_when_sources_sum_to_the_mix() {
        let mix = audio(&[1.0, 0.5]);
        let half = audio(&[0.5, 0.25]);
        assert!(unexplained_db(&mix, &[half.clone(), half]) < -100.0);
    }
}
