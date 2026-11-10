//! Two models, one stem each, and the third derived.
//!
//! BS-RoFormer separates vocals better than demucs does, and separates nothing
//! else. demucs splits the rest. So this runs both and takes each stem from
//! whichever is better at it:
//!
//! ```text
//! vocals    = BS-RoFormer(mix)
//! drums     = htdemucs_ft's drums specialist(mix - vocals)
//! harmonics = mix - vocals - drums          (the remainder, and what ships)
//! ```
//!
//! Three things about that arrangement are load-bearing.
//!
//! **The halves are chained, not parallel.** The drums model is handed what the
//! vocals model left behind rather than the track. It cannot change the vocals,
//! which are the same forward pass over the same audio either way, so all it
//! moves is where the line between drums and harmonics falls. It is worth exactly
//! what the model at the front of it is worth, which is why
//! [`Self::demucs_specialists`] does not do it: chaining a demucs vocals
//! specialist into a demucs drums specialist measured worse on every track tried.
//!
//! **`harmonics` is the remainder, not demucs's `bass + other`.** The player
//! derives `drums = mix - harmonics - vocals`, which with this arrangement
//! returns the drums estimate untouched. Shipping `bass + other` instead makes
//! that subtraction evaluate to demucs's drums plus the two models' disagreement
//! about the vocals, which is the error RoFormer was brought in to remove,
//! reappearing on the drums fader.
//!
//! **Each half is one model, not a bag.** `htdemucs_ft` is four fine-tuned
//! checkpoints combined by an identity matrix, so its drums come from `model_0`
//! alone and its vocals from `model_3` alone. Only the ones a half needs are
//! loaded, and only those run. That is why `Balanced` is built this way too: two
//! models with harmonics as the remainder is half the work and measurably better
//! than running all four. See docs/evaluation.md.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use stemd_mlx::Precision;
use stemd_mlx::apply::{Chunked, DEFAULT_OVERLAP, over_track};
use stemd_mlx::htdemucs::HtDemucs;
use stemd_mlx::roformer::BsRoformer;
use stemd_mlx::weights::Weights;

use crate::backend::{BackendInfo, Separate};
use crate::mixture::Normalisation;
use crate::pcm::Audio;
use crate::progress::{Cancelled, Progress, ProgressSink, Stage};
use crate::stems::{Parts, Stems, minus};

/// Progress is reported in thousandths rather than in chunks, because the two
/// halves are nowhere near the same size.
const UNITS: u32 = 1000;

/// How much of that belongs to the vocals half.
///
/// Measured: of 69.1 s on a 120 s track, BS-RoFormer is about 62.6 s and the
/// single demucs model about 6.5 s. Counting chunks instead would be close for the
/// count and badly wrong for the time.
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

pub struct HybridConfig {
    pub overlap: f32,
    /// Precision of the vocals half. See [`Precision::preferred`](crate::Precision::preferred).
    pub vocals_precision: Precision,
    /// Precision of the drums half.
    pub drums_precision: Precision,
    /// Hand the drums half the instrumental rather than the mixture.
    ///
    /// The two halves are independent by default: both see the track, and the only
    /// thing joining them is the subtraction at the end. Chaining cannot touch the
    /// vocals, which are the same forward pass either way; all it moves is where the
    /// line between drums and harmonics falls. Off by default.
    pub cascade: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            overlap: DEFAULT_OVERLAP,
            vocals_precision: Precision::F16,
            drums_precision: Precision::F16,
            cascade: false,
        }
    }
}

pub(crate) fn open(dir: &Path, name: &str, precision: Precision) -> Result<Weights> {
    let path = dir.join(format!("{name}.safetensors"));
    Weights::load(&path)?.cast(precision)
}

/// One model, and which of its outputs this arrangement wants.
struct Half {
    model: Box<dyn Chunked + Send>,
    /// The source to take. A fine-tuned demucs checkpoint still produces all
    /// four and only one of them is the one it was tuned for; BS-RoFormer
    /// produces one.
    source: i32,
    /// Whether the model expects the track-level normalisation.
    ///
    /// demucs does: it normalises by the statistics of the mono mixdown and undoes
    /// that after. BS-RoFormer does not, having been trained on raw audio.
    normalised: bool,
}

impl Half {
    fn demucs(weights: &Weights, index: i32, precision: Precision) -> Result<Self> {
        let config = stemd_mlx::htdemucs::Config {
            precision,
            ..stemd_mlx::htdemucs::Config::default()
        };
        Ok(Self {
            model: Box::new(HtDemucs::load(weights, &format!("model_{index}"), config)?),
            source: index,
            normalised: true,
        })
    }

    fn roformer(weights: &Weights, precision: Precision) -> Result<Self> {
        // Which variant is the artefact's business, not the preset's: a
        // BS-RoFormer and a BS PolarFormer are the same architecture at
        // different widths and only the tensors say which is in the file.
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

pub struct HybridSeparator {
    vocals: Half,
    drums: Half,
    /// A bass specialist, run only for a job that asks for `bass` or `other`.
    bass: Option<Half>,
    model: String,
    sample_rate: u32,
    channels: usize,
    config: HybridConfig,
}

/// Index of a source in demucs's `drums, bass, other, vocals` order.
const DRUMS: i32 = 0;
const BASS: i32 = 1;
const VOCALS: i32 = 3;

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

    /// Both halves from one demucs bag: its vocals specialist and its drums one.
    pub fn demucs_specialists(dir: &Path, artefact: &str, config: HybridConfig) -> Result<Self> {
        // Both halves come out of one file, so one cast serves both. They are
        // the same family and therefore want the same precision anyway; the
        // caller is trusted to have asked for that, and the vocals value wins
        // if it somehow did not.
        let precision = config.vocals_precision;
        let weights = open(dir, artefact, precision)?;
        let mut separator = Self::assemble(
            Half::demucs(&weights, VOCALS, precision)
                .with_context(|| format!("loading {artefact}'s vocals specialist"))?,
            Half::demucs(&weights, DRUMS, precision)
                .with_context(|| format!("loading {artefact}'s drums specialist"))?,
            artefact.to_owned(),
            config,
        );
        separator.bass = Some(
            Half::demucs(&weights, BASS, precision)
                .with_context(|| format!("loading {artefact}'s bass specialist"))?,
        );
        Ok(separator)
    }

    fn assemble(vocals: Half, drums: Half, model: String, config: HybridConfig) -> Self {
        let architecture = stemd_mlx::htdemucs::Config::default();
        Self {
            vocals,
            drums,
            bass: None,
            model,
            sample_rate: architecture.sample_rate.unsigned_abs(),
            channels: architecture.audio_channels.unsigned_abs() as usize,
            config,
        }
    }
}

pub(crate) fn check_geometry(mix: &Audio, channels: usize, sample_rate: u32) -> Result<()> {
    if mix.channels() != channels {
        bail!(
            "these models expect {channels} channels, got {}",
            mix.channels()
        );
    }
    if mix.sample_rate != sample_rate {
        bail!(
            "these models expect {sample_rate} Hz, got {} Hz",
            mix.sample_rate
        );
    }
    Ok(())
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

/// One `[C, T]` slice back to planes, optionally denormalised.
pub(crate) fn to_planes(
    out: &Array,
    channels: usize,
    norm: Option<&Normalisation>,
    frames: usize,
) -> Result<Vec<Vec<f32>>> {
    mlx_rs::transforms::eval([out])?;
    let full = out.as_dtype(mlx_rs::Dtype::Float32)?;
    mlx_rs::transforms::eval([&full])?;
    let flat = full
        .try_as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("the separated stem is not contiguous f32: {e:?}"))?;
    Ok((0..channels)
        .map(|c| {
            let plane = &flat[c * frames..(c + 1) * frames];
            match norm {
                Some(n) => plane.iter().map(|&v| n.restore(v)).collect(),
                None => plane.to_vec(),
            }
        })
        .collect())
}

/// The stems from drums and vocals estimates, with harmonics the remainder.
pub(crate) fn remainder_stems(
    mix: &Audio,
    drums: Vec<Vec<f32>>,
    vocals: Vec<Vec<f32>>,
    bass: Option<Vec<Vec<f32>>>,
) -> Stems {
    let rate = mix.sample_rate;
    let harmonics: Vec<Vec<f32>> = (0..mix.channels())
        .map(|c| {
            (0..mix.frames())
                .map(|i| mix.data[c][i] - vocals[c][i] - drums[c][i])
                .collect()
        })
        .collect();
    let (drums, harmonics, vocals) = (
        Audio::new(drums, rate),
        Audio::new(harmonics, rate),
        Audio::new(vocals, rate),
    );
    // Constructed to sum, so this is float noise rather than a measurement.
    let model_residual_db = crate::stems::unexplained_db(mix, [&drums, &harmonics, &vocals]);

    let mut more = Vec::new();
    if let Some(bass) = bass {
        let bass = Audio::new(bass, rate);
        // `other` is the remainder less the bass, so the four sum to the mix.
        more.push(("other", minus(&harmonics, &bass)));
        more.push(("bass", bass));
    }
    more.push(("drums", drums));
    Stems {
        model_residual_db,
        shipped: vec![("harmonics", harmonics), ("vocals", vocals)],
        more,
    }
}

impl HybridSeparator {
    /// True when neither half is a RoFormer, so the bar can be split evenly.
    fn halves_cost_the_same(&self) -> bool {
        self.vocals.model.chunk() == self.drums.model.chunk()
    }
}

impl Separate for HybridSeparator {
    fn separate(&mut self, mix: &Audio, parts: Parts, sink: &dyn ProgressSink) -> Result<Stems> {
        check_geometry(mix, self.channels, self.sample_rate)?;
        sink.update(Progress::new(Stage::Analysing));

        let frames = mix.frames();
        if frames == 0 {
            bail!("nothing to separate: the mix has no frames");
        }

        // Both halves report onto one bar, each into its own span of it. The
        // spans are equal when both halves are demucs and lopsided when one is
        // a RoFormer, which is nine times the work.
        let vocals_units = if self.halves_cost_the_same() {
            UNITS / 2
        } else {
            VOCALS_UNITS
        };
        sink.update(within(0, 0, 0, 1));

        // The normalisation belongs to the audio a half is given, not to the
        // track: under `cascade` the second half is handed the instrumental,
        // whose mono statistics are not the mixture's.
        let run = |half: &Half, source: &Audio, base: u32, span: u32| -> Result<Vec<Vec<f32>>> {
            let norm = Normalisation::of(source);
            let input = to_array(source, half.normalised.then_some(&norm))?;
            let out = over_track(
                half.model.as_ref(),
                &input,
                self.config.overlap,
                &mut |done, total| {
                    // Refusing here stops `over_track` at this chunk. Setting a
                    // flag and reading it afterwards is the version that looks
                    // right and freezes: the vocals half is nine tenths of this
                    // tier, so a cancel in its first second was answered four
                    // minutes later, after the whole model had run anyway.
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

        // Either the drums half sees the mixture, or it sees what the vocals
        // half left behind. Held outside the `if` so the common path passes
        // `mix` by reference rather than copying a six-minute track to say the
        // same thing.
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

    fn info(&self) -> BackendInfo {
        BackendInfo {
            // The runtime, not the arrangement: this is the same MLX as the
            // single-model path, and a client switching on the value should
            // not have to learn a second name for it. Which models ran is
            // `model`'s business.
            backend: "mlx".into(),
            model: self.model.clone(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            stems: crate::stems::SHIPPED.iter().map(|s| (*s).into()).collect(),
            // Asked, for the same reason as the single-model path.
            device: stemd_mlx::Accelerator::detect().as_str().into(),
            parts: if self.bass.is_some() {
                crate::stems::all_parts()
            } else {
                ["harmonics", "vocals", "drums"].map(String::from).to_vec()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stems::FOUR_STEM_SOURCES;

    /// The two indices name the sources they claim to.
    ///
    /// [`Half::demucs`] uses one number for which checkpoint to load and which of its
    /// four outputs to keep, because that is what a fine-tuned set is. Swapping them
    /// would load the wrong specialist and take the wrong output, and nothing
    /// downstream could see it: the shapes agree and the parts still sum to the mix.
    #[test]
    fn the_specialists_are_the_sources_they_are_named_for() {
        assert_eq!(FOUR_STEM_SOURCES[DRUMS.unsigned_abs() as usize], "drums");
        assert_eq!(FOUR_STEM_SOURCES[BASS.unsigned_abs() as usize], "bass");
        assert_eq!(FOUR_STEM_SOURCES[VOCALS.unsigned_abs() as usize], "vocals");
    }

    /// The bar advances through both halves and never goes backwards. They used to
    /// report 0, 1 and 2 of 2 and nothing in between, so a four-minute separation sat
    /// at the Separating stage's floor for almost all of it.
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
