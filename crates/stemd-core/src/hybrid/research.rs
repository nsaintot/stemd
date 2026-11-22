//! The RoFormer-and-demucs arrangement and the chained halves, built with the `research` feature.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use stemd_mlx::Precision;
use stemd_mlx::apply::over_track;
use stemd_mlx::roformer::BsRoformer;
use stemd_mlx::weights::Weights;

use super::{
    DRUMS, Half, HybridConfig, HybridSeparator, check_geometry, open, remainder_stems, to_planes,
};
use crate::mixture::Normalisation;
use crate::pcm::Audio;
use crate::progress::{Cancelled, Progress, ProgressSink, Stage};
use crate::stems::{Parts, Stems};

/// The progress units for one separation: thousandths of the bar, since the two halves differ
/// greatly in cost.
const UNITS: u32 = 1000;

/// The share of [`UNITS`] that belongs to the vocals half.
///
/// On a 120 s track that takes 69.1 s, BS-RoFormer takes about 62.6 s and the single demucs model
/// about 6.5 s. Splitting by chunk count would make the bar move far from the elapsed time.
const VOCALS_UNITS: u32 = 900;

/// Where a half's chunk count lands on the overall bar.
fn within(base: u32, span: u32, done: usize, total: usize) -> Progress {
    let fraction = if total == 0 {
        0.0
    } else {
        done as f32 / total as f32
    };
    let completed = base + (fraction * span as f32) as u32;
    Progress::counted(Stage::Separating, completed.min(UNITS), UNITS)
}

/// `[C, T]` planes as one `[1, C, T]` array, optionally normalised.
pub(crate) fn to_array(mix: &Audio, norm: Option<&Normalisation>) -> Result<Array> {
    let frames = mix.frames();
    let mut flat = Vec::with_capacity(mix.channels() * frames);
    for channel in &mix.data {
        match norm {
            Some(n) => flat.extend(channel.iter().map(|&v| n.apply(v))),
            None => flat.extend(channel.iter().copied()),
        }
    }
    let shape = [1, i32::try_from(mix.channels())?, i32::try_from(frames)?];
    Ok(Array::from_slice(&flat, &shape))
}

impl Half {
    fn roformer(weights: &Weights, precision: Precision) -> Result<Self> {
        // The artefact decides the variant: a BS-RoFormer and a BS PolarFormer share one
        // architecture at different widths, and only the tensors in the file tell them apart.
        let config = stemd_mlx::roformer::Config {
            precision,
            ..stemd_mlx::roformer::Config::of(weights)
        };
        Ok(Self {
            model: Box::new(BsRoformer::load(weights, config)?),
            source: 0,
            normalised: false,
        })
    }
}

impl HybridSeparator {
    /// BS-RoFormer for the vocals, a demucs bag's drums specialist for the rest.
    pub fn roformer_and_demucs(
        dir: &Path,
        vocals: &str,
        drums: &str,
        config: HybridConfig,
    ) -> Result<Self> {
        let (v, d) = (config.vocals_precision, config.drums_precision);
        Ok(Self::assemble(
            Half::roformer(&open(dir, vocals, v)?, v)
                .with_context(|| format!("loading {vocals} as a BS-RoFormer"))?,
            Half::demucs(&open(dir, drums, d)?, DRUMS, d)
                .with_context(|| format!("loading {drums}'s drums specialist"))?,
            format!("{vocals}+{drums}"),
            config,
        ))
    }

    /// Both halves over the whole track, one after the other.
    ///
    /// Used for a cascade, whose drums half needs the vocals over its whole window
    /// first, and for halves on different grids.
    pub(super) fn separate_in_halves(
        &mut self,
        mix: &Audio,
        parts: Parts,
        sink: &dyn ProgressSink,
    ) -> Result<Stems> {
        check_geometry(mix, self.channels, self.sample_rate)?;
        sink.update(Progress::new(Stage::Analysing));

        let frames = mix.frames();
        if frames == 0 {
            bail!("nothing to separate: the mix has no frames");
        }

        // Both halves report onto one bar, each into its own span. The spans are equal when both
        // halves are demucs, and uneven when one is a RoFormer, which does nine times the work.
        let vocals_units = if self.halves_cost_the_same() {
            UNITS / 2
        } else {
            VOCALS_UNITS
        };
        sink.update(within(0, 0, 0, 1));

        // The normalisation follows the audio a half receives: under `cascade` the second half gets
        // the instrumental, whose mono statistics differ from the mixture's.
        let run = |half: &Half, source: &Audio, base: u32, span: u32| -> Result<Vec<Vec<f32>>> {
            let norm = Normalisation::of(source);
            let input = to_array(source, half.normalised.then_some(&norm))?;
            let out = over_track(
                half.model.as_ref(),
                &input,
                self.config.overlap,
                &mut |done, total| {
                    // Refusing here stops `over_track` at this chunk. Do not set a
                    // flag and read it afterwards: the vocals half is most of the
                    // run, so the cancel would wait for the whole model.
                    if sink.cancelled() {
                        return Err(Cancelled.into());
                    }
                    sink.update(within(base, span, done, total));
                    Ok(())
                },
            )?;
            to_planes(
                &out.index((0, half.source)),
                self.channels,
                half.normalised.then_some(&norm),
                frames,
            )
        };

        let vocals = run(&self.vocals, mix, 0, vocals_units)?;

        // The drums half sees either the mixture or the audio the vocals half left. The binding is
        // declared outside the `if`, so the common path passes `mix` by reference and copies no
        // six-minute track.
        let instrumental;
        let for_drums: &Audio = if self.config.cascade {
            instrumental = Audio::new(
                (0..self.channels)
                    .map(|c| (0..frames).map(|i| mix.data[c][i] - vocals[c][i]).collect())
                    .collect(),
                self.sample_rate,
            );
            &instrumental
        } else {
            mix
        };

        // The bass specialist, when asked for, shares the drums half's span of the
        // bar: both are one demucs model over the track.
        let bass_half = self
            .bass
            .as_ref()
            .filter(|_| parts.contains("bass") || parts.contains("other"));
        let drums_units = match bass_half {
            Some(_) => (UNITS - vocals_units) / 2,
            None => UNITS - vocals_units,
        };
        let drums = run(&self.drums, for_drums, vocals_units, drums_units)?;
        let bass = bass_half
            .map(|half| {
                let base = vocals_units + drums_units;
                run(half, mix, base, UNITS - base)
            })
            .transpose()?;
        sink.update(within(UNITS, 0, 0, 1));

        sink.update(Progress::new(Stage::Reconstructing));
        Ok(remainder_stems(mix, drums, vocals, bass))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bar advances through both halves and never goes backwards.
    #[test]
    fn progress_advances_through_both_halves() {
        let mut seen = Vec::new();
        for done in 0..=20 {
            seen.push(within(0, VOCALS_UNITS, done, 20).fraction);
        }
        for done in 0..=4 {
            seen.push(within(VOCALS_UNITS, UNITS - VOCALS_UNITS, done, 4).fraction);
        }
        seen.push(within(UNITS, 0, 0, 1).fraction);

        assert!(
            seen.windows(2).all(|w| w[1] >= w[0]),
            "progress went backwards: {seen:?}"
        );
        assert!(
            seen.first().unwrap() < seen.last().unwrap(),
            "progress never moved: {seen:?}"
        );
        // Nothing may pass the end of the stage, or a later stage would appear
        // to go backwards.
        let ceiling = Progress::counted(Stage::Separating, 1, 1).fraction;
        assert!(
            seen.iter().all(|f| *f <= ceiling),
            "progress ran past the end of its stage: {seen:?}"
        );
    }

    /// The vocals half owns most of the bar, because it owns most of the time.
    #[test]
    fn the_bar_is_weighted_the_way_the_work_is() {
        let handover = within(VOCALS_UNITS, 0, 0, 1).fraction;
        let floor = Progress::new(Stage::Separating).fraction;
        let ceiling = Progress::counted(Stage::Separating, 1, 1).fraction;
        let share = (handover - floor) / (ceiling - floor);
        assert!(
            (share - 0.9).abs() < 0.01,
            "the vocals half takes ~90% of the time but {share:.2} of the bar"
        );
    }
}
