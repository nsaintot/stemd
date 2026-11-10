//! One BS-RoFormer that separates every stem in one pass.
//!
//! stemd's own model emits bass, drums, other and vocals. Drums and vocals make the
//! shipped stems, with harmonics the remainder as in [`crate::hybrid`], so the
//! player's derived drums are the model's own drums estimate. Bass rides along for
//! a client that asks for it split out of harmonics.

use std::path::Path;

use anyhow::{Context, Result, bail};
use stemd_mlx::Precision;
use stemd_mlx::apply::{Accumulator, Chunked, Grid};
use stemd_mlx::roformer::{BsRoformer, Config};

use crate::backend::{BackendInfo, Separate};
use crate::hybrid::{check_geometry, open, remainder_stems};
use crate::mixture::Normalisation;
use crate::pass::{Pass, Segmenting, check_range, chunk, source_planes};
use crate::pcm::Audio;
use crate::stems::{Parts, Stems, all_parts};

/// The model's output order.
pub const STEMS: [&str; 4] = ["bass", "drums", "other", "vocals"];

/// Index of `name` in [`STEMS`].
fn index(name: &str) -> i32 {
    STEMS.iter().position(|s| *s == name).expect("one of STEMS") as i32
}

pub struct RoformerSeparator {
    model: BsRoformer,
    name: String,
    overlap: f32,
}

impl RoformerSeparator {
    pub fn load(dir: &Path, artefact: &str, precision: Precision, overlap: f32) -> Result<Self> {
        let weights = open(dir, artefact, precision)?;
        let config = Config {
            precision,
            ..Config::of(&weights)
        };
        let model = BsRoformer::load(&weights, config)
            .with_context(|| format!("loading {artefact} as a BS-RoFormer"))?;
        if model.stems() != STEMS.len() as i32 {
            bail!(
                "{artefact} separates {} stems, not the {} this path expects",
                model.stems(),
                STEMS.len()
            );
        }
        Ok(Self {
            model,
            name: artefact.to_owned(),
            overlap,
        })
    }

    fn channels(&self) -> usize {
        self.model.config().audio_channels.unsigned_abs() as usize
    }

    fn sample_rate(&self) -> u32 {
        self.model.config().sample_rate.unsigned_abs()
    }
}

impl Separate for RoformerSeparator {
    /// Trained on raw audio, so no track normalisation.
    fn normalises(&self) -> bool {
        false
    }

    fn segmenting(&self) -> Option<Segmenting> {
        Some(Segmenting {
            segment: self.model.chunk(),
            overlap: self.overlap,
        })
    }

    fn begin(&self, frames: usize, parts: Parts, norm: Normalisation) -> Result<Pass> {
        let grid = Grid::new(i32::try_from(frames)?, self.model.chunk(), self.overlap)?;
        let mut pass = Pass::new(grid, parts, norm);
        pass.halves.push(Accumulator::new(
            grid,
            1,
            self.model.sources(),
            i32::try_from(self.channels())?,
        )?);
        Ok(pass)
    }

    fn run(&mut self, pass: &mut Pass, window: &Audio, k: usize) -> Result<()> {
        check_geometry(window, self.channels(), self.sample_rate())?;
        let input = chunk(window, &pass.grid, k, None)?;
        let separated = self.model.separate_chunk(&input)?;
        pass.halves[0].add(k, &separated)
    }

    fn stems(&self, pass: &Pass, mix: &Audio, from: usize, to: usize) -> Result<Stems> {
        check_geometry(mix, self.channels(), self.sample_rate())?;
        check_range(mix, from, to)?;
        pass.check_final(from, to)?;
        let plane = |name: &str| {
            source_planes(
                &pass.halves[0],
                index(name),
                from,
                to,
                self.channels(),
                None,
            )
        };
        let (drums, vocals, bass) = (plane("drums")?, plane("vocals")?, plane("bass")?);
        Ok(remainder_stems(mix, drums, vocals, Some(bass)))
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            backend: "mlx".into(),
            model: self.name.clone(),
            sample_rate: self.sample_rate(),
            channels: self.channels(),
            stems: crate::stems::SHIPPED.iter().map(|s| (*s).into()).collect(),
            device: stemd_mlx::Accelerator::detect().as_str().into(),
            parts: all_parts(),
        }
    }
}
