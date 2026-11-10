//! One BS-RoFormer that separates every stem in one pass.
//!
//! stemd's own model emits bass, drums, other and vocals. Drums and vocals make the
//! shipped stems, with harmonics the remainder as in [`crate::hybrid`], so the
//! player's derived drums are the model's own drums estimate. Bass rides along for
//! a client that asks for it split out of harmonics.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_rs::ops::indexing::IndexOp;
use stemd_mlx::Precision;
use stemd_mlx::apply::over_track;
use stemd_mlx::roformer::{BsRoformer, Config};

use crate::backend::{BackendInfo, Separate};
use crate::hybrid::{check_geometry, open, remainder_stems, to_array, to_planes};
use crate::pcm::Audio;
use crate::progress::{Cancelled, Progress, ProgressSink, Stage};
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
    fn separate(&mut self, mix: &Audio, _parts: Parts, sink: &dyn ProgressSink) -> Result<Stems> {
        check_geometry(mix, self.channels(), self.sample_rate())?;
        sink.update(Progress::new(Stage::Analysing));

        let frames = mix.frames();
        if frames == 0 {
            bail!("nothing to separate: the mix has no frames");
        }

        // Trained on raw audio, so no track normalisation.
        let input = to_array(mix, None)?;
        let out = over_track(&self.model, &input, self.overlap, &mut |done, total| {
            if sink.cancelled() {
                return Err(Cancelled.into());
            }
            sink.update(Progress::counted(
                Stage::Separating,
                u32::try_from(done).unwrap_or(u32::MAX),
                u32::try_from(total).unwrap_or(u32::MAX),
            ));
            Ok(())
        })?;
        let drums = to_planes(
            &out.index((0, index("drums"))),
            self.channels(),
            None,
            frames,
        )?;
        let vocals = to_planes(
            &out.index((0, index("vocals"))),
            self.channels(),
            None,
            frames,
        )?;
        let bass = to_planes(
            &out.index((0, index("bass"))),
            self.channels(),
            None,
            frames,
        )?;

        sink.update(Progress::new(Stage::Reconstructing));
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
