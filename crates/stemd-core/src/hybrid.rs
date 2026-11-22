//! Two models, one stem each, with the third stem derived.
//!
//! BS-RoFormer separates vocals better than demucs, and separates nothing else; demucs splits the
//! rest. This separator runs both and takes each stem from the model that does it better:
//!
//! ```text
//! vocals    = BS-RoFormer(mix)
//! drums     = htdemucs_ft's drums specialist(mix - vocals)
//! harmonics = mix - vocals - drums          (the remainder, and what ships)
//! ```
//!
//! Three properties of this arrangement matter.
//!
//! **The halves are chained.** The drums model receives the audio the vocals model left behind.
//! Chaining cannot change the vocals, which come from the same forward pass over the same audio
//! either way; it only moves the line between drums and harmonics. Its benefit depends on the
//! vocals model in front, so [`HybridSeparator::demucs_specialists`] does not chain: a demucs
//! vocals specialist chained into a demucs drums specialist separates worse.
//!
//! **`harmonics` is the remainder.** It is never demucs's `bass + other`. The player derives `drums
//! = mix - harmonics - vocals`, which with this arrangement returns the drums estimate unchanged.
//! Shipping `bass + other` would make that subtraction give demucs's drums plus the two models'
//! disagreement about the vocals, so the error that RoFormer is there to remove would reappear on
//! the drums fader.
//!
//! **Each half is one model.** `htdemucs_ft` is four fine-tuned checkpoints combined by an identity
//! matrix, so its drums come from `model_0` alone and its vocals from `model_3` alone. Only the
//! checkpoints a half needs are loaded and run. `Balanced` is built the same way: two models with
//! harmonics as the remainder do half the work and measure better than all four. See
//! docs/evaluation.md.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use stemd_mlx::Precision;
use stemd_mlx::apply::{Accumulator, Chunked, DEFAULT_OVERLAP};
use stemd_mlx::htdemucs::HtDemucs;
use stemd_mlx::weights::Weights;

use crate::backend::{BackendInfo, Separate};
use crate::mixture::Normalisation;
use crate::pass::{Pass, Segmenting, check_range, chunk, source_planes, whole_track};
use crate::pcm::Audio;
use crate::progress::ProgressSink;
use crate::stems::{Parts, Stems, minus};

pub struct HybridConfig {
    pub overlap: f32,
    /// The precision of the vocals half. See [`Precision::preferred`](crate::Precision::preferred).
    pub vocals_precision: Precision,
    /// The precision of the drums half.
    pub drums_precision: Precision,
    /// Give the drums half the instrumental, the mix minus the vocals, as input. Off by default.
    // By default both halves see the track, and only the final subtraction joins them.
    #[cfg(feature = "research")]
    pub cascade: bool,
}

impl HybridConfig {
    /// Independent halves at `overlap` and the two precisions.
    pub const fn new(
        overlap: f32,
        vocals_precision: Precision,
        drums_precision: Precision,
    ) -> Self {
        Self {
            overlap,
            vocals_precision,
            drums_precision,
            #[cfg(feature = "research")]
            cascade: false,
        }
    }
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self::new(DEFAULT_OVERLAP, Precision::F16, Precision::F16)
    }
}

pub(crate) fn open(dir: &Path, name: &str, precision: Precision) -> Result<Weights> {
    let path = dir.join(format!("{name}.safetensors"));
    Weights::load(&path)?.cast(precision)
}

#[cfg(feature = "research")]
mod research;

/// One model, and which of its outputs this arrangement wants.
struct Half {
    model: Box<dyn Chunked + Send>,
    /// The source to keep. A fine-tuned demucs checkpoint produces all four sources but is tuned
    /// for only one; BS-RoFormer produces one.
    source: i32,
    /// Whether the model expects the track-level normalisation.
    ///
    /// demucs does: it normalises by the statistics of the mono mixdown and undoes that afterwards.
    /// BS-RoFormer, trained on raw audio, does not.
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

/// The index of each source in demucs's `drums, bass, other, vocals` order.
const DRUMS: i32 = 0;
const BASS: i32 = 1;
const VOCALS: i32 = 3;

impl HybridSeparator {
    /// Both halves from one demucs bag: its vocals specialist and its drums one.
    pub fn demucs_specialists(dir: &Path, artefact: &str, config: HybridConfig) -> Result<Self> {
        // Both halves come from one file, so one cast serves both. They are the same family and
        // want the same precision; the caller is expected to ask for that, and the vocals precision
        // wins if it did not.
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

/// Convert one `[C, T]` slice to planes, optionally denormalised.
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
    // The stems sum to the mix by construction, so this is float noise and measures nothing.
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

    /// True when the drums half is handed the instrumental. See [`HybridConfig::cascade`].
    #[cfg(feature = "research")]
    const fn cascades(&self) -> bool {
        self.config.cascade
    }

    /// Always false: chaining the halves is a research arrangement.
    #[cfg(not(feature = "research"))]
    const fn cascades(&self) -> bool {
        false
    }

    /// True when the halves can run segment by segment: one grid, and no half
    /// waits on another's output.
    fn segmentable(&self) -> bool {
        !self.cascades()
            && self.halves_cost_the_same()
            && self
                .bass
                .as_ref()
                .is_none_or(|b| b.model.chunk() == self.vocals.model.chunk())
    }

    /// The halves a job for `parts` runs, in [`Pass`] order.
    fn halves_for(&self, parts: Parts) -> Vec<&Half> {
        let bass = self
            .bass
            .as_ref()
            .filter(|_| parts.contains("bass") || parts.contains("other"));
        [Some(&self.vocals), Some(&self.drums), bass]
            .into_iter()
            .flatten()
            .collect()
    }
}

impl Separate for HybridSeparator {
    fn normalises(&self) -> bool {
        self.vocals.normalised
            || self.drums.normalised
            || self.bass.as_ref().is_some_and(|b| b.normalised)
    }

    fn segmenting(&self) -> Option<Segmenting> {
        self.segmentable().then(|| Segmenting {
            segment: self.vocals.model.chunk(),
            overlap: self.config.overlap,
        })
    }

    fn begin(&self, frames: usize, parts: Parts, norm: Normalisation) -> Result<Pass> {
        let Some(segmenting) = self.segmenting() else {
            bail!(
                "{} separates whole tracks only: its halves cannot run segment by segment",
                self.model
            );
        };
        let grid = segmenting.grid(frames)?;
        let mut pass = Pass::new(grid, parts, norm);
        for _ in self.halves_for(parts) {
            // One source per half: the one it is the specialist for.
            pass.halves
                .push(Accumulator::new(grid, 1, 1, i32::try_from(self.channels)?)?);
        }
        Ok(pass)
    }

    fn run(&mut self, pass: &mut Pass, window: &Audio, k: usize) -> Result<()> {
        check_geometry(window, self.channels, self.sample_rate)?;
        let halves = self.halves_for(pass.parts);
        for (i, half) in halves.into_iter().enumerate() {
            let input = chunk(window, &pass.grid, k, half.normalised.then_some(&pass.norm))?;
            let separated = half.model.separate_chunk(&input)?;
            let own = separated.index((.., half.source..half.source + 1));
            pass.halves[i].add(k, &own)?;
        }
        Ok(())
    }

    fn stems(&self, pass: &Pass, mix: &Audio, from: usize, to: usize) -> Result<Stems> {
        check_geometry(mix, self.channels, self.sample_rate)?;
        check_range(mix, from, to)?;
        pass.check_final(from, to)?;
        let halves = self.halves_for(pass.parts);
        let mut planes = halves.iter().enumerate().map(|(i, half)| {
            source_planes(
                &pass.halves[i],
                0,
                from,
                to,
                self.channels,
                half.normalised.then_some(&pass.norm),
            )
        });
        let vocals = planes.next().context("the pass has no vocals half")??;
        let drums = planes.next().context("the pass has no drums half")??;
        let bass = planes.next().transpose()?;
        Ok(remainder_stems(mix, drums, vocals, bass))
    }

    fn separate(&mut self, mix: &Audio, parts: Parts, sink: &dyn ProgressSink) -> Result<Stems> {
        #[cfg(feature = "research")]
        if !self.segmentable() {
            return self.separate_in_halves(mix, parts, sink);
        }
        whole_track(self, mix, parts, sink, |_| Ok(()))
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            // Report the runtime: it is the same MLX as the single-model path, so a client that
            // checks the value needs no second name for it. `model` says which models ran.
            backend: "mlx".into(),
            model: self.model.clone(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            stems: crate::stems::SHIPPED.iter().map(|s| (*s).into()).collect(),
            // The device is detected, as in the single-model path.
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
    /// [`Half::demucs`] uses one number both for the checkpoint to load and for which of its four
    /// outputs to keep, since a fine-tuned set pairs them. Swapping the indices would load the
    /// wrong specialist and keep the wrong output, and nothing downstream would notice: the shapes
    /// agree and the parts still sum to the mix.
    #[test]
    fn the_specialists_are_the_sources_they_are_named_for() {
        assert_eq!(FOUR_STEM_SOURCES[DRUMS.unsigned_abs() as usize], "drums");
        assert_eq!(FOUR_STEM_SOURCES[BASS.unsigned_abs() as usize], "bass");
        assert_eq!(FOUR_STEM_SOURCES[VOCALS.unsigned_abs() as usize], "vocals");
    }
}
