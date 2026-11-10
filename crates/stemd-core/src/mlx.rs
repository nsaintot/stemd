//! demucs v4 on MLX, through [`stemd_mlx`].
//!
//! The model is built layer by layer from a safetensors artefact rather than
//! loaded as a traced graph, so this file is only the adapter between
//! [`Audio`](crate::pcm::Audio) and MLX arrays, plus the geometry checks and
//! progress reporting the server expects.
//!
//! Two things a caller has to know:
//!
//! * MLX belongs to one thread, and that thread is the one that built it, not
//!   merely the one using it. The CUDA backend keeps its stream registry in
//!   thread-local storage, so weights allocated on another thread fail their
//!   first `eval` with `There is no Stream(gpu, N) in current thread`. The
//!   server's queue worker constructs its own separator and never accepts one:
//!   see `stemd_server::queue::BuildSeparator`.
//! * The separated track is held as one array: the four sources of a six-minute
//!   stereo track are about 560 MB, the price of doing the overlap-add on the GPU.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_rs::Array;
use stemd_mlx::Precision;
use stemd_mlx::apply::{Accumulator, Chunked, DEFAULT_OVERLAP, Grid};
use stemd_mlx::htdemucs::{Config, HtDemucs};
use stemd_mlx::weights::Weights;

use crate::backend::{BackendInfo, Separate};
use crate::mixture::{Normalisation, fold_into_shipped};
use crate::pass::{Pass, Segmenting, check_range, chunk};
use crate::pcm::Audio;
use crate::stems::{FOUR_STEM_SOURCES, Parts, Stems, unexplained_db};

pub struct MlxConfig {
    /// Fraction of a segment shared with its neighbour. demucs uses 0.25.
    pub overlap: f32,
    /// What the encoder, transformer and decoder run at.
    ///
    /// [`Precision::F16`] is worth about 1.3x and costs -54 dB against the same model
    /// at [`Precision::F32`], 20 dB below htdemucs's own residual, which does not move.
    ///
    /// The normalisations accumulate in full precision whatever this says. Without
    /// that, half precision here produces NaN rather than separating slightly worse.
    pub precision: Precision,
}

impl Default for MlxConfig {
    fn default() -> Self {
        Self {
            overlap: DEFAULT_OVERLAP,
            precision: Precision::F32,
        }
    }
}

pub struct MlxSeparator {
    demucs: HtDemucs,
    /// The artefact's stem, e.g. `htdemucs`, for [`BackendInfo`].
    model: String,
    sources: Vec<String>,
    sample_rate: u32,
    channels: usize,
    config: MlxConfig,
}

impl MlxSeparator {
    /// Load `<dir>/<name>.safetensors`, which must hold exactly one model.
    ///
    /// There is no manifest beside it. The architecture is compiled in as [`Config`],
    /// every layer checks the shape of each tensor it pulls, and a file that is not
    /// this architecture fails at load naming the tensor that disagreed.
    pub fn load(dir: &Path, name: &str, config: MlxConfig) -> Result<Self> {
        let path = dir.join(format!("{name}.safetensors"));
        let weights = Weights::load(&path)?;
        //  Cast only on the way down. Widening a stored f16 artefact would compute the
        //  same, since mlx promotes and f16 to f32 is exact, but would double the memory
        //  the weights occupy, and every demucs artefact this loads is f32 already.
        //  `hybrid::open` casts unconditionally because its RoFormer half is published at
        //  f16.
        let weights = if config.precision == Precision::F16 {
            weights.cast(config.precision)?
        } else {
            weights
        };
        let architecture = || Config {
            precision: config.precision,
            ..Config::default()
        };

        //  How many models the file holds is asked of the tensor names rather than guessed
        //  from the filename. More than one means a fine-tuned set like `htdemucs_ft`,
        //  whose four checkpoints each produce a usable version of exactly one source:
        //  running it here would take all four sources from whichever was stored first.
        //  See [`crate::hybrid::HybridSeparator::demucs_specialists`].
        let demucs = match weights.models() {
            0 => bail!(
                "{} holds no model_N tensors; it is not a converted demucs artefact",
                path.display()
            ),
            1 => HtDemucs::load(&weights, "model_0", architecture())?,
            n => bail!(
                "{} holds {n} fine-tuned models; loading it as a single model would \
                 take every source from one of them. It needs a preset that says \
                 which model each source comes from.",
                path.display()
            ),
        };

        let architecture = Config::default();
        Ok(Self {
            demucs,
            model: name.to_owned(),
            sources: FOUR_STEM_SOURCES.iter().map(|s| (*s).into()).collect(),
            sample_rate: u32::try_from(architecture.sample_rate)?,
            channels: usize::try_from(architecture.audio_channels)?,
            config,
        })
    }

    pub const fn model(&self) -> &String {
        &self.model
    }

    /// Reject a mix the model cannot accept, before any work is done.
    fn check_geometry(&self, mix: &Audio) -> Result<()> {
        if mix.channels() != self.channels {
            bail!(
                "model expects {} channels, got {}",
                self.channels,
                mix.channels()
            );
        }
        if mix.sample_rate != self.sample_rate {
            bail!(
                "model expects {} Hz, got {} Hz",
                self.sample_rate,
                mix.sample_rate
            );
        }
        Ok(())
    }

    /// A `[1, S, C, n]` result back into one [`Audio`] per model source, with
    /// the normalisation undone.
    fn to_sources(&self, out: &Array, norm: &Normalisation, frames: usize) -> Result<Vec<Audio>> {
        // Reading the buffer of a lazy array is undefined rather than merely
        // slow, so this is not optional.
        mlx_rs::transforms::eval([out])?;
        let flat = out
            .try_as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("the separated track is not contiguous f32: {e:?}"))?;

        let stride = self.channels * frames;
        Ok((0..self.sources.len())
            .map(|source| {
                let plane = &flat[source * stride..(source + 1) * stride];
                let data = (0..self.channels)
                    .map(|c| {
                        plane[c * frames..(c + 1) * frames]
                            .iter()
                            .map(|&v| norm.restore(v))
                            .collect()
                    })
                    .collect();
                Audio::new(data, self.sample_rate)
            })
            .collect())
    }
}

impl Separate for MlxSeparator {
    fn normalises(&self) -> bool {
        true
    }

    fn segmenting(&self) -> Option<Segmenting> {
        Some(Segmenting {
            segment: self.demucs.chunk(),
            overlap: self.config.overlap,
        })
    }

    fn begin(&self, frames: usize, parts: Parts, norm: Normalisation) -> Result<Pass> {
        let grid = Grid::new(
            i32::try_from(frames)?,
            self.demucs.chunk(),
            self.config.overlap,
        )?;
        let mut pass = Pass::new(grid, parts, norm);
        pass.halves.push(Accumulator::new(
            grid,
            1,
            self.demucs.sources(),
            i32::try_from(self.channels)?,
        )?);
        Ok(pass)
    }

    fn run(&mut self, pass: &mut Pass, window: &Audio, k: usize) -> Result<()> {
        self.check_geometry(window)?;
        let input = chunk(window, &pass.grid, k, Some(&pass.norm))?;
        let separated = self.demucs.separate_chunk(&input)?;
        pass.halves[0].add(k, &separated)
    }

    fn stems(&self, pass: &Pass, mix: &Audio, from: usize, to: usize) -> Result<Stems> {
        self.check_geometry(mix)?;
        check_range(mix, from, to)?;
        pass.check_final(from, to)?;
        let out = pass.halves[0].range(i32::try_from(from)?, i32::try_from(to)?)?;
        let sources = self.to_sources(&out, &pass.norm, to - from)?;
        // The model's own sources, for a job asking for them one by one. They do not
        // sum to the mix; the leftover lands on whatever part the job leaves out.
        let more = ["drums", "bass", "other"]
            .into_iter()
            .filter_map(|name| {
                let index = self.sources.iter().position(|s| s == name)?;
                Some((name, sources[index].clone()))
            })
            .collect();

        Ok(Stems {
            model_residual_db: unexplained_db(mix, &sources),
            shipped: fold_into_shipped(&sources, &self.sources)
                .context("folding the model's sources into the shipped stems")?,
            more,
        })
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            backend: "mlx".into(),
            model: self.model.clone(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            stems: crate::stems::SHIPPED.iter().map(|s| (*s).into()).collect(),
            // Asked rather than asserted. This was the literal `"gpu"` for as
            // long as Metal was the only backend and the answer could not be
            // anything else; a CUDA build on a machine with no usable card
            // falls back to the CPU and used to say otherwise.
            device: stemd_mlx::Accelerator::detect().as_str().into(),
            parts: crate::stems::all_parts(),
        }
    }
}
