//! What a backend does around the model, whichever runtime runs the weights.
//!
//! demucs normalises the whole track by the mean and standard deviation of its
//! mono mixdown before the model sees it, and undoes that afterwards: skipping it
//! measurably changes the output, so it is part of the model's contract even
//! though it sits outside the graph. And the four sources a model produces have to
//! be folded into the two stems that cross the wire.

use anyhow::{Context, Result};

use crate::pcm::Audio;

/// Added to the mixture's standard deviation before inverting it, so silence
/// does not divide by zero.
pub const STD_FLOOR: f32 = 1e-8;

/// The loudness normalisation demucs applies to the whole track and undoes at
/// the end.
pub struct Normalisation {
    mean: f32,
    std: f32,
    inv_std: f32,
}

impl Normalisation {
    pub fn of(mix: &Audio) -> Self {
        let (mean, std) = mono_mean_std(mix);
        Self {
            mean,
            std,
            inv_std: 1.0 / (std + STD_FLOOR),
        }
    }

    pub fn apply(&self, v: f32) -> f32 {
        (v - self.mean) * self.inv_std
    }

    pub fn restore(&self, v: f32) -> f32 {
        v * self.std + self.mean
    }
}

/// Fold the model's sources into the stems that ship.
///
/// Summing `bass` and `other` before the wire rather than after is what keeps
/// the transfer at two stems. `names` is the source order the model produces,
/// which is what makes this independent of any one backend's manifest.
pub fn fold_into_shipped(
    sources: &[Audio],
    names: &[String],
) -> Result<Vec<(&'static str, Audio)>> {
    crate::stems::STEM_SOURCES
        .iter()
        .map(|(stem, members)| {
            let mut sum: Option<Audio> = None;
            for member in *members {
                let audio = source(sources, names, member)?;
                match &mut sum {
                    None => sum = Some(audio.clone()),
                    Some(sum) => add_into(sum, audio),
                }
            }
            Ok((*stem, sum.context("stem has no member sources")?))
        })
        .collect()
}

/// The output plane for one model source, by name.
fn source<'a>(sources: &'a [Audio], names: &[String], name: &str) -> Result<&'a Audio> {
    let index = names
        .iter()
        .position(|s| s == name)
        .with_context(|| format!("model has no source {name:?}"))?;
    sources
        .get(index)
        .with_context(|| format!("source {name:?} missing from output"))
}

/// Add `src` into `dst` sample by sample.
fn add_into(dst: &mut Audio, src: &Audio) {
    for (c, ch) in dst.data.iter_mut().enumerate() {
        for (i, sample) in ch.iter_mut().enumerate() {
            *sample += src.data[c][i];
        }
    }
}

/// The mono mixdown of one frame, as demucs computes it.
fn mono_at(mix: &Audio, i: usize) -> f64 {
    let sum: f64 = mix.data.iter().map(|ch| f64::from(ch[i])).sum();
    sum / mix.channels().max(1) as f64
}

/// Mean and standard deviation of the mono mixdown, as demucs computes them.
fn mono_mean_std(mix: &Audio) -> (f32, f32) {
    let frames = mix.frames();
    if frames == 0 {
        return (0.0, 1.0);
    }
    // Two passes rather than one buffered mixdown: the mixdown of a ten-minute
    // track is hundreds of megabytes, and the sums are cheap beside a forward.
    let mean = (0..frames).map(|i| mono_at(mix, i)).sum::<f64>() / frames as f64;
    let var = (0..frames)
        .map(|i| (mono_at(mix, i) - mean).powi(2))
        .sum::<f64>()
        / frames as f64;
    (mean as f32, var.sqrt().max(f64::from(STD_FLOOR)) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_stats_match_a_hand_computation() {
        let a = Audio::new(vec![vec![1.0, -1.0], vec![1.0, -1.0]], 44100);
        let (mean, std) = mono_mean_std(&a);
        assert!(mean.abs() < 1e-6);
        assert!((std - 1.0).abs() < 1e-6);
    }

    /// Normalising and restoring has to be a round trip, or the backend hands
    /// back audio at the wrong level and nothing downstream would notice.
    #[test]
    fn normalisation_round_trips() {
        let a = Audio::new(vec![vec![0.5, -0.25, 0.75], vec![0.1, 0.2, -0.3]], 44100);
        let norm = Normalisation::of(&a);
        for &v in &[0.5f32, -0.25, 0.0, 1.0] {
            assert!(
                (norm.restore(norm.apply(v)) - v).abs() < 1e-5,
                "{v} did not survive"
            );
        }
    }

    /// `harmonics` is `bass + other` summed, and `vocals` is passed through. A
    /// backend whose source order differs would otherwise fold the wrong planes
    /// together and produce two plausible, wrong stems.
    #[test]
    fn folding_sums_bass_and_other_and_leaves_vocals_alone() {
        let plane = |v: f32| Audio::new(vec![vec![v; 4], vec![v; 4]], 44100);
        // drums, bass, other, vocals
        let sources = [plane(1.0), plane(2.0), plane(4.0), plane(8.0)];
        let names: Vec<String> = crate::stems::FOUR_STEM_SOURCES
            .iter()
            .map(|s| (*s).into())
            .collect();

        let folded = fold_into_shipped(&sources, &names).expect("folding");
        let of = |want: &str| {
            folded
                .iter()
                .find(|(stem, _)| *stem == want)
                .map(|(_, a)| a.data[0][0])
                .expect("stem is present")
        };
        assert!((of("harmonics") - 6.0).abs() < 1e-6, "bass + other");
        assert!((of("vocals") - 8.0).abs() < 1e-6, "vocals alone");
    }

    /// A model that does not produce a source a stem needs has to say which
    /// one, because the two ways this fails look identical from the outside: a
    /// source the model never had, and a source it has a name for but no plane.
    #[test]
    fn a_missing_source_names_itself() {
        let plane = Audio::new(vec![vec![0.0; 2]], 44100);

        let named: Vec<String> = ["drums", "bass", "vocals"]
            .iter()
            .map(|s| (*s).into())
            .collect();
        let err = fold_into_shipped(&[plane.clone(), plane.clone(), plane.clone()], &named)
            .expect_err("the model has no `other`");
        assert!(format!("{err}").contains("no source \"other\""), "{err}");

        let all: Vec<String> = crate::stems::FOUR_STEM_SOURCES
            .iter()
            .map(|s| (*s).into())
            .collect();
        let err = fold_into_shipped(&[plane], &all).expect_err("only one plane came back");
        assert!(format!("{err}").contains("missing from output"), "{err}");
    }
}
