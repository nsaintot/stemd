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
//! * The whole track is held as one array. A six-minute stereo track is about
//!   140 MB in and the four separated sources about 560 MB out, which is the
//!   price of doing the overlap-add on the GPU.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlx_rs::Array;
use stemd_mlx::Precision;
use stemd_mlx::apply::{DEFAULT_OVERLAP, over_track};
use stemd_mlx::htdemucs::{Config, HtDemucs};
use stemd_mlx::weights::Weights;

use crate::backend::{BackendInfo, Separate};
use crate::mixture::{Normalisation, fold_into_shipped};
use crate::pcm::Audio;
use crate::progress::{Cancelled, Progress, ProgressSink, Stage};
use crate::stems::{FOUR_STEM_SOURCES, Stems, unexplained_db};

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

    /// The normalised mixture as a `[1, C, T]` array.
    fn to_array(&self, mix: &Audio, norm: &Normalisation) -> Result<Array> {
        let frames = mix.frames();
        let mut flat = Vec::with_capacity(self.channels * frames);
        for channel in &mix.data {
            flat.extend(channel.iter().map(|&v| norm.apply(v)));
        }
        let shape = [1, i32::try_from(self.channels)?, i32::try_from(frames)?];
        Ok(Array::from_slice(&flat, &shape))
    }

    /// A `[1, S, C, T]` result back into one [`Audio`] per model source, with
    /// the normalisation undone.
    fn to_sources(&self, out: &Array, norm: &Normalisation, frames: usize) -> Result<Vec<Audio>> {
        // Evaluated already by `separate`, but reading the buffer of a lazy
        // array is undefined rather than merely slow, so this is not optional.
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
    fn separate(&mut self, mix: &Audio, sink: &dyn ProgressSink) -> Result<Stems> {
        self.check_geometry(mix)?;
        sink.update(Progress::new(Stage::Analysing));

        let frames = mix.frames();
        if frames == 0 {
            bail!("nothing to separate: the mix has no frames");
        }
        let norm = Normalisation::of(mix);
        let input = self.to_array(mix, &norm)?;

        // Cancellation costs one segment rather than one track, because the
        // callback can refuse to continue and `over_track` stops where it is.
        // The partial overlap-add is worth nothing on its own and is dropped.
        let separated = over_track(
            &self.demucs,
            &input,
            self.config.overlap,
            &mut |done, total| {
                if sink.cancelled() {
                    return Err(Cancelled.into());
                }
                sink.update(Progress::counted(
                    Stage::Separating,
                    u32::try_from(done).unwrap_or(u32::MAX),
                    u32::try_from(total).unwrap_or(u32::MAX),
                ));
                Ok(())
            },
        )?;

        sink.update(Progress::new(Stage::Reconstructing));
        let sources = self.to_sources(&separated, &norm, frames)?;

        Ok(Stems {
            model_residual_db: unexplained_db(mix, &sources),
            shipped: fold_into_shipped(&sources, &self.sources)
                .context("folding the model's sources into the shipped stems")?,
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
        }
    }
}
