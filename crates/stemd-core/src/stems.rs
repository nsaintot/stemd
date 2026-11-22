//! The stem layout, and the player-side maths that uses it.
//!
//! The server produces exactly [`SHIPPED`]. A player drives three faders and rebuilds the third
//! part itself:
//!
//! ```text
//! drums = mix - harmonics - vocals
//! ```
//!
//! so the parts sum to the mix by construction, for every sample, whether or not the model's own
//! sources sum to its input. See [`Stems::model_residual_db`].
//!
//! Evaluating the faders never computes that third part. Substituting `S_drums` into `out = sum(g_q
//! * S_q)` gives
//!
//! ```text
//! out = g_drums * mix + (g_harmonics - g_drums) * harmonics
//!                     + (g_vocals    - g_drums) * vocals
//! ```
//!
//! so a player mixes buffers it already holds. At unity both stem coefficients are exactly zero and
//! the result is the mix, bit for bit.
//!
//! **The preset decides where the residual lands.** `Fast` ships `harmonics` as the model's `bass +
//! other`, so the leftover from four sources that do not quite sum arrives on the rebuilt drums.
//! `Balanced` and `Quality` build `harmonics` as `mix - vocals - drums`, so the leftover is inside
//! `harmonics`.
//!
//! The player is unaffected. The server's subtraction and the client's are inverses: `mix -
//! harmonics - vocals` returns whatever the server called drums.

use crate::pcm::Audio;

/// The stems the server produces and transfers.
///
/// `harmonics` is the model's `bass` and `other` summed: DJs rarely want that split, and merging
/// halves the transfer.
pub const SHIPPED: [&str; 2] = ["harmonics", "vocals"];

/// The source order of the four-stem models (demucs, Open-Unmix).
pub(crate) const FOUR_STEM_SOURCES: [&str; 4] = ["drums", "bass", "other", "vocals"];

/// The model sources that make up each shipped stem. The model's `drums` is in neither, so it is
/// the part a player rebuilds.
pub(crate) const STEM_SOURCES: [(&str, &[&str]); 2] =
    [("harmonics", &["bass", "other"]), ("vocals", &["vocals"])];

/// The three parts a player has faders for.
///
/// Player-side only: the server produces [`SHIPPED`] and has no third part.
pub const PARTS: [&str; 3] = ["drums", "harmonics", "vocals"];

/// Every part a job can ask for, in the order they ship. `harmonics` is `bass` and
/// `other` together.
pub const PART_NAMES: [&str; 5] = ["harmonics", "vocals", "drums", "bass", "other"];

/// `a - b`, sample by sample.
pub fn minus(a: &Audio, b: &Audio) -> Audio {
    let data = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p - q).collect())
        .collect();
    Audio::new(data, a.sample_rate)
}

/// `mix` minus every part in `parts`.
pub fn remainder<'a>(mix: &Audio, parts: impl IntoIterator<Item = &'a Audio>) -> Audio {
    let mut out = mix.clone();
    for part in parts {
        for (c, ch) in out.data.iter_mut().enumerate() {
            for (i, sample) in ch.iter_mut().enumerate() {
                *sample -= at(part, c, i);
            }
        }
    }
    out
}

/// Every part name, for a separator that can produce all of them.
pub(crate) fn all_parts() -> Vec<String> {
    PART_NAMES.iter().map(|p| (*p).to_owned()).collect()
}

/// The model sources a part is made of.
pub fn sources_of(part: &str) -> Option<&'static [&'static str]> {
    Some(match part {
        "harmonics" => &["bass", "other"],
        "vocals" => &["vocals"],
        "drums" => &["drums"],
        "bass" => &["bass"],
        "other" => &["other"],
        _ => return None,
    })
}

/// Any set of [`PART_NAMES`], unchecked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PartSet(u8);

impl PartSet {
    fn bit(part: &str) -> Option<u8> {
        PART_NAMES.iter().position(|p| *p == part).map(|i| 1 << i)
    }

    /// Parse a comma-separated list, refusing unknown names.
    pub fn parse(list: &str) -> Result<Self, String> {
        let mut set = Self::default();
        for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            set.0 |= Self::bit(name)
                .ok_or_else(|| format!("unknown part {name:?}; expected some of {PART_NAMES:?}"))?;
        }
        Ok(set)
    }

    pub fn has(self, part: &str) -> bool {
        Self::bit(part).is_some_and(|b| self.0 & b != 0)
    }

    #[must_use]
    pub fn with(self, part: &str) -> Self {
        Self(self.0 | Self::bit(part).unwrap_or(0))
    }

    #[must_use]
    pub fn without(self, part: &str) -> Self {
        Self(self.0 & !Self::bit(part).unwrap_or(0))
    }

    /// The parts, in [`PART_NAMES`] order.
    pub fn names(self) -> impl Iterator<Item = &'static str> {
        PART_NAMES.into_iter().filter(move |p| self.has(p))
    }

    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for PartSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.names().collect::<Vec<_>>().join(","))
    }
}

/// The parts a job ships.
///
/// The sources they cover must not overlap, and what they leave out must be one
/// part or nothing: that part is the one a player rebuilds as `mix` minus the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Parts(PartSet);

impl Parts {
    /// `harmonics` and `vocals`, with `drums` rebuilt: protocol 1.
    pub const DEFAULT: Self = Self(PartSet(0b00011));

    /// [`Self::DEFAULT`] and its derived part.
    pub const DEFAULT_WITH_DERIVED: PartSet = PartSet(0b00111);

    /// Parse a comma-separated list such as `bass,other,vocals`.
    pub fn parse(list: &str) -> Result<Self, String> {
        Self::try_from(PartSet::parse(list)?)
    }

    pub fn names(self) -> impl Iterator<Item = &'static str> {
        self.0.names()
    }

    pub fn len(self) -> usize {
        self.0.len()
    }

    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(self, part: &str) -> bool {
        self.0.has(part)
    }

    /// The model sources that none of the parts cover.
    pub(crate) fn left_out(self) -> Vec<&'static str> {
        FOUR_STEM_SOURCES
            .into_iter()
            .filter(|source| {
                !self
                    .names()
                    .any(|part| sources_of(part).is_some_and(|s| s.contains(source)))
            })
            .collect()
    }

    /// The part a player rebuilds: whatever is left out, or nothing if all is shipped.
    pub fn derived(self) -> Option<&'static str> {
        let left = self.left_out();
        PART_NAMES.into_iter().find(|part| {
            sources_of(part)
                .is_some_and(|s| s.len() == left.len() && s.iter().all(|x| left.contains(x)))
        })
    }
}

impl TryFrom<PartSet> for Parts {
    type Error = String;

    fn try_from(set: PartSet) -> Result<Self, String> {
        if set.is_empty() {
            return Err("ask for at least one part".into());
        }
        let mut seen: Vec<&str> = Vec::new();
        for part in set.names() {
            for source in sources_of(part).expect("a known part") {
                if seen.contains(source) {
                    return Err(format!("{source} is asked for twice"));
                }
                seen.push(source);
            }
        }
        let parts = Self(set);
        if !parts.left_out().is_empty() && parts.derived().is_none() {
            return Err(format!(
                "{set} leaves {:?} out, which is not one part; ask for all but one",
                parts.left_out()
            ));
        }
        Ok(parts)
    }
}

impl std::fmt::Display for Parts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// The part a player rebuilds from the others. Player-side only and fixed; the module docs explain
/// why it is `drums`.
pub const DERIVED: &str = "drums";

/// The fader positions, one per [`PARTS`] entry. `1.0` is unity.
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

    /// True when every gain is exactly `1.0`, so mixing can return the mix untouched. A UI must
    /// snap its faders to exactly `1.0` to get the untouched mix.
    pub(crate) fn is_unity(self) -> bool {
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
    /// The level of the model's own reconstruction error `mix - sum(sources)`, relative to the mix,
    /// in dB.
    // A diagnostic only: the three parts still sum exactly. It is meaningful only where the
    // model's sources should sum to the mix and miss. An arrangement that builds one part by
    // subtraction has no such redundancy, and the value collapses to float noise.
    pub model_residual_db: f64,
    /// Parts beyond [`SHIPPED`] that this separation estimated.
    pub more: Vec<(&'static str, Audio)>,
}

impl Stems {
    pub fn stem(&self, name: &str) -> Option<&Audio> {
        self.shipped
            .iter()
            .chain(&self.more)
            .find(|(n, _)| *n == name)
            .map(|(_, audio)| audio)
    }

    /// The derived part as the player reconstructs it: `mix - shipped`.
    ///
    /// This is the model's own version plus the residual.
    pub fn derived_part(&self, mix: &Audio) -> Audio {
        remainder(mix, self.shipped.iter().map(|(_, stem)| stem))
    }

    /// Apply the three faders. Unity returns a copy of the mix.
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

/// The level of `mix - sum(sources)` relative to `mix`, in dB.
pub(crate) fn unexplained_db<'a>(mix: &Audio, sources: impl IntoIterator<Item = &'a Audio>) -> f64 {
    let sources: Vec<&Audio> = sources.into_iter().collect();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (c, mix_ch) in mix.data.iter().enumerate() {
        for (i, m) in mix_ch.iter().enumerate() {
            let mut sum = 0.0f32;
            for src in &sources {
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

    /// A mix whose parts deliberately do not sum to it, so the residual is real
    /// and lands on [`DERIVED`].
    fn fixture() -> (Audio, Stems) {
        let mix = audio(&[1.0, 0.5, -0.25, 0.125]);
        // The model's own drums are in `flat` but never shipped: the residual is measured against
        // them, as the backend does.
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
            more: Vec::new(),
        };
        (mix, stems)
    }

    #[test]
    fn the_default_parts_are_protocol_1() {
        assert_eq!(Parts::DEFAULT.names().collect::<Vec<_>>(), SHIPPED);
        assert_eq!(Parts::DEFAULT.derived(), Some(DERIVED));
        assert_eq!(Parts::parse("vocals,harmonics"), Ok(Parts::DEFAULT));
    }

    #[test]
    fn a_request_names_what_is_rebuilt() {
        let derived = |list: &str| Parts::parse(list).map(Parts::derived);
        assert_eq!(derived("bass,other,vocals"), Ok(Some("drums")));
        assert_eq!(derived("drums,vocals"), Ok(Some("harmonics")));
        assert_eq!(derived("drums,bass,other,vocals"), Ok(None));
        assert!(derived("harmonics,bass").is_err(), "bass twice");
        assert!(derived("vocals").is_err(), "three sources left out");
        assert!(derived("piano").is_err());
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
    fn a_gain_near_unity_is_not_unity() {
        let gains = PartGains {
            vocals: 0.9997,
            ..PartGains::UNITY
        };
        assert!(!gains.is_unity(), "only an exact 1.0 may short-circuit");
    }

    #[test]
    fn the_unexplained_level_is_zero_when_the_sources_sum_to_the_mix() {
        let mix = audio(&[1.0, 0.5]);
        let half = audio(&[0.5, 0.25]);
        assert!(unexplained_db(&mix, &[half.clone(), half]) < -100.0);
    }
}
