//! BS-RoFormer: band-split transformers over a spectrogram.
//!
//! No convolutions at all: linear layers, RMS norms and attention, so nothing has
//! to be rearranged between channels-first and channels-last, and every piece has
//! a fused kernel behind it.
//!
//! ```text
//! audio [B, 2, T]
//!   -> stft                      [B, 2, 1025, N] complex
//!   -> flattened per frame       [B, N, 4100]     (freq-major, then channel, then re/im)
//!   -> band split                [B, N, 62, 512]  one projection per frequency band
//!   -> 12 blocks, each a transformer over time then one over bands
//!   -> mask estimator per band   [B, N, 4100]
//!   -> multiply the spectrogram, inverse stft
//! ```
//!
//! It emits one stem, vocals; the instrumental is the mixture minus it. See
//! docs/evaluation.md for what that means for a server shipping `harmonics`.

use std::f32::consts::PI;

use anyhow::{Result, bail};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype, ops};

use crate::precision::Precision;
use crate::spectral::Stft;
use crate::weights::{Scope, Weights};

/// Everything the artefact was trained with. Not configurable, for the same
/// reason [`crate::htdemucs::Config`] is not: a different value here does not
/// make a different model, it makes one the weights no longer fit.
pub struct Config {
    pub dim: i32,
    pub depth: i32,
    pub heads: i32,
    pub dim_head: i32,
    pub ff_mult: i32,
    pub audio_channels: i32,
    pub n_fft: i32,
    pub hop: i32,
    /// Frequency bins per band. Sums to `n_fft / 2 + 1`.
    pub bands: Vec<i32>,
    /// Hidden width of each band's mask MLP, as a multiple of `dim`.
    pub mask_expansion: i32,
    /// Samples in one chunk. Unlike htdemucs this is not a hard constraint:
    /// position is rotary, so any length runs, but it is what the model was
    /// trained at and what the overlap-add is sized around.
    pub chunk: i32,
    pub sample_rate: i32,
    /// Precision the transformer stack runs at.
    ///
    /// [`Precision`] rather than a [`Dtype`] so a caller can ask for half
    /// without linking mlx into its own signatures; see [`crate::precision`].
    pub precision: Precision,
    /// How position enters attention.
    pub positional: Positional,
}

/// Which positional scheme the artefact was trained with.
///
/// Not a tuning knob: the weights were trained against one of these and the
/// other produces plausible noise. The artefact says which: a PoPE checkpoint
/// carries `pope_embed.*` tensors and no `rotary_embed.freqs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Positional {
    /// Rotary, interleaved, applied to q and k. The viperx checkpoints.
    Rope,
    /// Polar: q and k become magnitudes through softplus and are rotated into
    /// (cos, sin) pairs, with a learned per-head phase on the keys. See
    /// [`Pope`].
    Pope,
}

/// The band table every published BS-RoFormer uses: fine at the bottom, coarse
/// at the top, 62 bands over 1025 bins.
fn default_bands() -> Vec<i32> {
    let mut bands = Vec::with_capacity(62);
    for (width, count) in [(2, 24), (4, 12), (12, 8), (24, 8), (48, 8)] {
        bands.extend(std::iter::repeat_n(width, count));
    }
    bands.push(128);
    bands.push(129);
    bands
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dim: 512,
            depth: 12,
            heads: 8,
            dim_head: 64,
            ff_mult: 4,
            audio_channels: 2,
            n_fft: 2048,
            hop: 441,
            bands: default_bands(),
            mask_expansion: 4,
            chunk: 352_800,
            sample_rate: 44100,
            precision: Precision::default(),
            positional: Positional::Rope,
        }
    }
}

impl Config {
    /// BS PolarFormer, which is this architecture at half the width.
    ///
    /// Same 62-band table summing to 1025, same depth, same eight heads of 64, same
    /// `ff_mult`, same mask estimator shape. Four things move: the residual width,
    /// the hop, the chunk, and rotary becoming polar. That is why this is a
    /// constructor rather than a second model.
    pub fn polarformer() -> Self {
        Self {
            dim: 256,
            hop: 512,
            chunk: 588_800,
            positional: Positional::Pope,
            ..Self::default()
        }
    }

    /// Which variant an artefact is, asked of its tensors.
    ///
    /// The two differ in four values and no code, and nothing in either file names
    /// itself, but a polar artefact carries `pope_embed` tensors and a rotary one
    /// carries `rotary_embed.freqs`, and neither can be renamed without the load
    /// failing.
    pub fn of(weights: &Weights) -> Self {
        if weights
            .at("layers.0.0.layers.0.0")
            .has("pope_embed.inv_freqs")
        {
            Self::polarformer()
        } else {
            Self::default()
        }
    }

    /// Frequency bins the transform produces.
    pub fn bins(&self) -> i32 {
        self.n_fft / 2 + 1
    }

    /// Width of each band once complex parts and audio channels are unrolled
    /// into it, which is what the band split actually projects from.
    pub fn band_features(&self) -> Vec<i32> {
        self.bands
            .iter()
            .map(|f| 2 * f * self.audio_channels)
            .collect()
    }

    fn check(&self) -> Result<()> {
        let total: i32 = self.bands.iter().sum();
        if total != self.bins() {
            bail!(
                "the bands cover {total} bins but a {}-point transform has {}",
                self.n_fft,
                self.bins()
            );
        }
        Ok(())
    }
}

/// `x / rms(x) * gamma`, over the last axis.
///
/// The reference spells this `F.normalize(x, dim=-1) * sqrt(dim) * gamma`, which
/// is the same thing: dividing by the L2 norm and multiplying by the root of the
/// length is dividing by the root mean square. There is no bias, so reaching for a
/// layer norm here would load without complaint and be wrong.
struct RmsNorm {
    gamma: Array,
}

impl RmsNorm {
    fn load(w: &Scope<'_>, dim: i32) -> Result<Self> {
        Ok(Self {
            gamma: w.get_shaped("gamma", &[dim])?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::fast::rms_norm(x, &self.gamma, 1e-12)?)
    }
}

/// A linear layer whose bias is optional, because half of them here have none.
struct Linear {
    /// Stored `[in, out]`, transposed once at load. See the note in
    /// [`crate::transformer`].
    weight: Array,
    bias: Option<Array>,
}

impl Linear {
    fn load(w: &Scope<'_>, chin: i32, chout: i32) -> Result<Self> {
        let weight = w.get_shaped("weight", &[chout, chin])?;
        Ok(Self {
            weight: ops::transpose_axes(&weight, &[1, 0][..])?,
            bias: w
                .has("bias")
                .then(|| w.get_shaped("bias", &[chout]))
                .transpose()?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = ops::matmul(x, &self.weight)?;
        Ok(match &self.bias {
            Some(b) => ops::add(&y, b)?,
            None => y,
        })
    }
}

/// One projection per frequency band, from that band's bins to the model width.
///
/// The bands are not equal: 2 bins wide at the bottom and 129 at the top, so each
/// carries its own norm and its own matrix, and there are sixty-two. This is where
/// the model's frequency resolution is decided.
struct BandSplit {
    bands: Vec<(RmsNorm, Linear)>,
    widths: Vec<i32>,
}

impl BandSplit {
    fn load(w: &Scope<'_>, config: &Config) -> Result<Self> {
        let widths = config.band_features();
        let mut bands = Vec::with_capacity(widths.len());
        for (index, &width) in widths.iter().enumerate() {
            let scope = w.child(&index.to_string());
            bands.push((
                RmsNorm::load(&scope.child("0"), width)?,
                Linear::load(&scope.child("1"), width, config.dim)?,
            ));
        }
        Ok(Self { bands, widths })
    }

    /// `[B, N, sum(widths)] -> [B, N, bands, dim]`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = Vec::with_capacity(self.bands.len());
        let mut at = 0;
        for ((norm, project), &width) in self.bands.iter().zip(&self.widths) {
            let slice = x.index((.., .., at..at + width));
            out.push(project.forward(&norm.forward(&slice)?)?);
            at += width;
        }
        Ok(ops::stack_axis(&out, -2)?)
    }
}

/// Attention with a rotary position, and a gate per head.
///
/// After attending, each head's output is scaled by `sigmoid(W x)` for that head:
/// eight numbers per position, learned. It is not in the paper's diagram, and
/// leaving it out would load fine and separate slightly worse.
pub struct Attention {
    norm: RmsNorm,
    qkv: Linear,
    gates: Linear,
    out: Linear,
    heads: i32,
    dim_head: i32,
    /// `None` for a rotary artefact. See [`Positional`].
    pope: Option<Pope>,
}

/// Base of the rotary embedding. Checked against the artefact rather than
/// assumed: the stored `rotary_embed.freqs` are exactly
/// `1 / 10000^(arange(0, 64, 2) / 64)`.
const ROPE_BASE: f32 = 10_000.0;

/// Polar positional embedding: the one thing BS PolarFormer does differently.
///
/// Rotary turns q and k as vectors by an angle that depends on position. Polar
/// keeps only a magnitude, `softplus(q)`, and places it on the unit circle at the
/// angle position dictates:
///
/// ```text
/// phi[i, j] = i * inv_freqs[j]
/// q'[i, 2j], q'[i, 2j+1] = softplus(q)[i, j] * (cos phi, sin phi)
/// k'[i, 2j], k'[i, 2j+1] = softplus(k)[i, j] * (cos psi, sin psi)
///     where psi = phi + bias[head, j]
/// ```
///
/// The head depth doubles, 64 to 128, because each magnitude becomes a (cos, sin)
/// pair. `v` is untouched at 64, so this cannot go through a fused attention
/// kernel that wants one head size for all three.
///
/// The bias is on the keys only, learned per head and per frequency, and clamped
/// to `[-2pi, 0]` on the way out: the clamp is in the reference, and a checkpoint
/// is not a promise, so it is applied rather than assumed.
struct Pope {
    /// `[dim_head]`, stored rather than derived: the reference registers them
    /// as a buffer, so a checkpoint could have been trained with something
    /// other than the `theta ** -(arange(d) / d)` default.
    inv_freqs: Array,
    /// `[heads, dim_head]`, the learned phase offset on the keys.
    bias: Array,
}

impl Pope {
    fn load(w: &Scope<'_>, config: &Config) -> Result<Self> {
        let scope = w.child("pope_embed");
        Ok(Self {
            inv_freqs: scope.get_shaped("inv_freqs", &[config.dim_head])?,
            bias: scope.get_shaped("bias", &[config.heads, config.dim_head])?,
        })
    }

    /// `q, k: [B, heads, N, dim_head] -> [B, heads, N, 2 * dim_head]` each.
    fn apply(&self, q: &Array, k: &Array, positions: i32) -> Result<(Array, Array)> {
        // Angles in float32 whatever the model runs at. This is the same
        // lesson the normalisations taught: a cosine of a large position index
        // in half precision is not the cosine of that index.
        let steps =
            ops::arange::<_, f32>(0.0, f64::from(positions), 1.0)?.reshape(&[positions, 1])?;
        let inv = self.inv_freqs.as_dtype(Dtype::Float32)?.reshape(&[1, -1])?;
        let phi = ops::multiply(&steps, &inv)?;

        // The keys carry a per-head offset, so their angles are [heads, N, D]
        // where the queries' are [N, D] and broadcast.
        let bias = self.bias.as_dtype(Dtype::Float32)?;
        let bias = ops::clip(&bias, (Array::from_f32(-2.0 * PI), Array::from_f32(0.0)))?;
        let psi = ops::add(
            &phi.reshape(&[1, positions, -1])?,
            &bias.reshape(&[-1, 1, self.bias.shape()[1]])?,
        )?;

        let dtype = q.dtype();
        let polar = |magnitude: &Array, angle: &Array| -> Result<Array> {
            let (cos, sin) = (ops::cos(angle)?, ops::sin(angle)?);
            let (cos, sin) = (cos.as_dtype(dtype)?, sin.as_dtype(dtype)?);
            let real = ops::multiply(magnitude, &cos)?;
            let imag = ops::multiply(magnitude, &sin)?;
            // Interleaved: [..., D, 2] flattened is (cos_0, sin_0, cos_1, ...),
            // which is the order `rearrange(..., '... d two')` produces and the
            // order the trained weights expect.
            let stacked = ops::stack_axis(&[real, imag], -1)?;
            let mut shape = stacked.shape().to_vec();
            let two = shape.pop().expect("a last axis");
            let last = shape.last_mut().expect("a depth axis");
            *last *= two;
            Ok(stacked.reshape(&shape)?)
        };

        Ok((polar(&softplus(q)?, &phi)?, polar(&softplus(k)?, &psi)?))
    }
}

/// `log(1 + exp(x))`, in the form that does not overflow.
///
/// The direct spelling loses the whole tensor to `inf` for `x` past about 11 in
/// half precision, and these are pre-activation attention inputs, which are not
/// bounded.
fn softplus(x: &Array) -> Result<Array> {
    let zero = crate::layers::scalar_like(0.0, x)?;
    let positive = ops::maximum(x, &zero)?;
    let decayed = ops::exp(&ops::negative(&ops::abs(x)?)?)?;
    Ok(ops::add(&positive, &ops::log1p(&decayed)?)?)
}

impl Attention {
    fn load(w: &Scope<'_>, config: &Config) -> Result<Self> {
        let inner = config.heads * config.dim_head;
        Ok(Self {
            norm: RmsNorm::load(&w.child("norm"), config.dim)?,
            qkv: Linear::load(&w.child("to_qkv"), config.dim, inner * 3)?,
            gates: Linear::load(&w.child("to_gates"), config.dim, config.heads)?,
            out: Linear::load(&w.child("to_out.0"), inner, config.dim)?,
            heads: config.heads,
            dim_head: config.dim_head,
            pope: match config.positional {
                Positional::Rope => None,
                Positional::Pope => Some(Pope::load(w, config)?),
            },
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let normed = self.norm.forward(x)?;
        let s = normed.shape().to_vec();
        let (batch, positions) = (s[0], s[1]);

        // One matrix for all three, laid out query-major then head then depth.
        let qkv = self.qkv.forward(&normed)?.reshape(&[
            batch,
            positions,
            3,
            self.heads,
            self.dim_head,
        ])?;
        let qkv = ops::transpose_axes(&qkv, &[2, 0, 3, 1, 4][..])?;
        let part = |i: i32| qkv.index(i);

        // Rotary over the whole head depth, interleaved rather than split-half
        // -- `traditional` is mlx's name for the convention
        // `rotary_embedding_torch` uses, and the other one would rotate pairs
        // of dimensions that were never paired during training.
        let rope = |t: &Array| -> Result<Array> {
            Ok(mlx_rs::fast::rope(
                t,
                self.dim_head,
                true,
                Some(ROPE_BASE),
                1.0,
                0,
                None,
            )?)
        };
        let v = part(2);
        let scale = 1.0 / (self.dim_head as f32).sqrt();

        let attended = match &self.pope {
            None => {
                let q = rope(&part(0))?;
                let k = rope(&part(1))?;
                mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, scale, None)?
            }
            //  Polar doubles the depth of q and k and leaves v alone, so the fused kernel is
            //  not available: it wants one head size for all three. Written out, which costs
            //  an explicit N-by-N score matrix.
            //  The scale is `dim_head ** -0.5` and not `(2 * dim_head) ** -0.5`, which is what
            //  the weights were trained under.
            Some(pope) => {
                let (q, k) = pope.apply(&part(0), &part(1), positions)?;
                let scores = ops::matmul(&q, &ops::swap_axes(&k, -1, -2)?)?;
                let scores = ops::multiply(&scores, &crate::layers::scalar_like(scale, &scores)?)?;
                ops::matmul(&ops::softmax_axes(&scores, &[-1][..], true)?, &v)?
            }
        };

        // [B, N, heads] -> [B, heads, N, 1], so each head scales its own rows.
        let gates = ops::sigmoid(&self.gates.forward(&normed)?)?;
        let gates = ops::transpose_axes(&gates, &[0, 2, 1][..])?
            .reshape(&[batch, self.heads, positions, 1])?;
        let attended = ops::multiply(&attended, &gates)?;

        let merged = ops::transpose_axes(&attended, &[0, 2, 1, 3][..])?.reshape(&[
            batch,
            positions,
            self.heads * self.dim_head,
        ])?;
        self.out.forward(&merged)
    }
}

/// Norm, widen, gelu, narrow.
pub struct FeedForward {
    norm: RmsNorm,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn load(w: &Scope<'_>, config: &Config) -> Result<Self> {
        let inner = config.dim * config.ff_mult;
        // The indices are positions in a torch Sequential: 0 is the norm, 1 and
        // 4 the two linear layers, with an activation and two dropouts between
        // them that carry no weights.
        Ok(Self {
            norm: RmsNorm::load(&w.child("net.0"), config.dim)?,
            up: Linear::load(&w.child("net.1"), config.dim, inner)?,
            down: Linear::load(&w.child("net.4"), inner, config.dim)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = self.up.forward(&self.norm.forward(x)?)?;
        self.down.forward(&crate::layers::gelu(&y)?)
    }
}

/// A stack of attention-then-feed-forward, each residual.
///
/// No output norm, unlike the class it is ported from: these are built with
/// `norm_output` off and only [`BsRoformer`] carries a final one. A norm here
/// would need weights that are not in the artefact.
pub struct Transformer {
    layers: Vec<(Attention, FeedForward)>,
}

impl Transformer {
    fn load(w: &Scope<'_>, config: &Config, depth: i32) -> Result<Self> {
        let mut layers = Vec::with_capacity(depth as usize);
        for index in 0..depth {
            let scope = w.child("layers").child(&index.to_string());
            layers.push((
                Attention::load(&scope.child("0"), config)?,
                FeedForward::load(&scope.child("1"), config)?,
            ));
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for (attn, ff) in &self.layers {
            x = ops::add(&attn.forward(&x)?, &x)?;
            x = ops::add(&ff.forward(&x)?, &x)?;
        }
        Ok(x)
    }

    pub fn layer(&self, index: usize) -> (&Attention, &FeedForward) {
        let (a, f) = &self.layers[index];
        (a, f)
    }
}

/// One block: attention across time, then attention across bands.
///
/// Both are ordinary transformers; what differs is which axis is folded into the
/// batch before they run. Every band sees the whole timeline, then every instant
/// sees the whole spectrum.
pub struct Block {
    pub time: Transformer,
    pub freq: Transformer,
}

/// One small network per band, turning the model's features back into a complex
/// mask over that band's bins.
///
/// Each is `dim -> dim * 4 -> width * 2` with a tanh between, and a gated linear
/// unit then halves the output: the first half is the mask, the second gates it.
/// The tanh bounds the hidden layer, and the activation the rest of the model uses
/// would load fine and mask differently.
struct MaskEstimator {
    bands: Vec<(Linear, Linear)>,
}

impl MaskEstimator {
    fn load(w: &Scope<'_>, config: &Config) -> Result<Self> {
        let hidden = config.dim * config.mask_expansion;
        let mut bands = Vec::new();
        for (index, &width) in config.band_features().iter().enumerate() {
            // `to_freqs.<band>.0` is the MLP inside a Sequential(MLP, GLU);
            // inside it, 0 and 2 are the matrices and 1 is the tanh.
            let scope = w.child(&index.to_string()).child("0");
            bands.push((
                Linear::load(&scope.child("0"), config.dim, hidden)?,
                Linear::load(&scope.child("2"), hidden, width * 2)?,
            ));
        }
        Ok(Self { bands })
    }

    /// `[B, N, bands, dim] -> [B, N, sum(widths)]`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = Vec::with_capacity(self.bands.len());
        for (index, (up, down)) in self.bands.iter().enumerate() {
            let features = x.index((.., .., index as i32, ..));
            let hidden = ops::tanh(&up.forward(&features)?)?;
            let both = down.forward(&hidden)?;
            let half = *both.shape().last().expect("a last axis") / 2;
            let value = both.index((.., .., 0..half));
            let gate = both.index((.., .., half..2 * half));
            out.push(ops::multiply(&value, &ops::sigmoid(&gate)?)?);
        }
        Ok(ops::concatenate_axis(&out, -1)?)
    }
}

impl crate::apply::Chunked for BsRoformer {
    fn chunk(&self) -> i32 {
        self.config.chunk
    }

    /// One: vocals. The instrumental is the mixture minus it, which is not
    /// this model's business.
    fn sources(&self) -> i32 {
        1
    }

    fn separate_chunk(&self, chunk: &Array) -> Result<Array> {
        let out = self.forward(chunk)?;
        let s = out.shape().to_vec();
        Ok(out.reshape(&[s[0], 1, s[1], s[2]])?)
    }
}

/// The model.
pub struct BsRoformer {
    stft: Stft,
    band_split: BandSplit,
    blocks: Vec<Block>,
    final_norm: RmsNorm,
    mask: MaskEstimator,
    config: Config,
}

impl BsRoformer {
    pub fn load(weights: &Weights, config: Config) -> Result<Self> {
        config.check()?;
        let mut blocks = Vec::with_capacity(config.depth as usize);
        for index in 0..config.depth {
            let scope = weights.at("layers").child(&index.to_string());
            blocks.push(Block {
                time: Transformer::load(&scope.child("0"), &config, 1)?,
                freq: Transformer::load(&scope.child("1"), &config, 1)?,
            });
        }
        Ok(Self {
            stft: Stft::new(config.n_fft, config.hop),
            band_split: BandSplit::load(&weights.at("band_split.to_features"), &config)?,
            blocks,
            final_norm: RmsNorm::load(&weights.at("final_norm"), config.dim)?,
            mask: MaskEstimator::load(&weights.at("mask_estimators.0.to_freqs"), &config)?,
            config,
        })
    }

    pub fn block(&self, index: usize) -> &Block {
        &self.blocks[index]
    }

    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// `[B, C, T]` audio to `[B, C, F, N]` complex, as two real arrays.
    pub fn spectrogram(&self, audio: &Array) -> Result<(Array, Array)> {
        let z = self.stft.forward(audio, true)?;
        Ok((ops::real(&z)?, ops::imag(&z)?))
    }

    /// The spectrogram flattened the way the band split wants it.
    ///
    /// `[B, C, F, N] -> [B, N, F * C * 2]`, ordered by frequency, then audio channel,
    /// then real and imaginary. Getting that order wrong produces exactly the right
    /// shape and feeds every band a mixture of its neighbours' bins.
    pub fn interleave(&self, real: &Array, imag: &Array) -> Result<Array> {
        let shape = real.shape().to_vec();
        let (batch, channels, bins, frames) = (shape[0], shape[1], shape[2], shape[3]);
        // [B, C, F, N] each -> [B, C, F, N, 2]
        let paired = ops::stack_axis(&[real.clone(), imag.clone()], -1)?;
        // -> [B, N, F, C, 2] -> [B, N, F * C * 2]
        let ordered = ops::transpose_axes(&paired, &[0, 3, 2, 1, 4][..])?;
        Ok(ordered.reshape(&[batch, frames, bins * channels * 2])?)
    }

    /// Spectrogram, interleave, band split: `[B, C, T] -> [B, N, bands, dim]`.
    pub fn embed(&self, audio: &Array) -> Result<Array> {
        let (real, imag) = self.spectrogram(audio)?;
        let flat = self.interleave(&real, &imag)?;
        self.band_split.forward(&flat)
    }

    /// The twelve blocks, and the norm after them.
    ///
    /// `x` is `[B, N, bands, dim]` throughout. Each block folds one axis into
    /// the batch, runs a transformer over the other, and puts it back.
    pub fn transform(&self, x: &Array) -> Result<Array> {
        let s = x.shape().to_vec();
        let (batch, frames, bands, dim) = (s[0], s[1], s[2], s[3]);
        let mut x = x.clone();
        for block in &self.blocks {
            // Across time: every band attends over the whole timeline.
            let packed = ops::transpose_axes(&x, &[0, 2, 1, 3][..])?.reshape(&[
                batch * bands,
                frames,
                dim,
            ])?;
            let out = block.time.forward(&packed)?;
            x = ops::transpose_axes(
                &out.reshape(&[batch, bands, frames, dim])?,
                &[0, 2, 1, 3][..],
            )?;

            // Across bands: every instant attends over the whole spectrum.
            let packed = x.reshape(&[batch * frames, bands, dim])?;
            let out = block.freq.forward(&packed)?;
            x = out.reshape(&[batch, frames, bands, dim])?;
        }
        self.final_norm.forward(&x)
    }

    /// The mask, in the same interleaved layout the spectrogram was flattened
    /// into: `[B, N, bands, dim] -> [B, N, F * C * 2]`.
    pub fn mask(&self, x: &Array) -> Result<Array> {
        self.mask.forward(x)
    }

    /// Separate one chunk. `[B, C, T]` in, `[B, C, T]` of vocals out.
    pub fn forward(&self, audio: &Array) -> Result<Array> {
        let shape = audio.shape().to_vec();
        let (channels, samples) = (shape[1], shape[2]);
        if channels != self.config.audio_channels {
            bail!(
                "this model separates {}-channel audio, not {channels}",
                self.config.audio_channels
            );
        }

        let (real, imag) = self.spectrogram(audio)?;

        //  The transform and the mask multiply stay in full precision; the band split, the
        //  transformer stack and the mask estimators run at whatever the config asks for.
        //  This cast is the only place this model turns a precision into an mlx dtype.
        let flat = self
            .interleave(&real, &imag)?
            .as_dtype(self.config.precision.dtype())?;
        let mask = self.mask(&self.transform(&self.band_split.forward(&flat)?)?)?;

        self.apply(&mask, &real, &imag, samples)
    }

    /// Apply a mask to the spectrogram it was estimated from, and invert.
    ///
    /// Everything in [`Self::forward`] after the mask estimator: the complex multiply,
    /// dropping DC, and the inverse transform. `[B, N, F * C * 2]` of mask and
    /// `[B, C, F, N]` of spectrogram in, `[B, C, samples]` out.
    ///
    /// Split out so it can be measured directly; the stages do not sum to the forward.
    pub fn apply(&self, mask: &Array, real: &Array, imag: &Array, samples: i32) -> Result<Array> {
        let shape = real.shape().to_vec();
        let batch = shape[0];
        let channels = self.config.audio_channels;
        let bins = self.config.bins();
        let planes = bins * channels;
        let frames = *shape.last().expect("a last axis");

        let mask = mask.as_dtype(Dtype::Float32)?;

        // Both sides to [B, F * C, N] complex, in the same frequency-then-
        // channel order the flattening used.
        let mask = mask
            .reshape(&[batch, frames, planes, 2])?
            .transpose_axes(&[0, 2, 1, 3][..])?;
        let (mask_real, mask_imag) = (mask.index((.., .., .., 0)), mask.index((.., .., .., 1)));
        let ordered = |a: &Array| -> Result<Array> {
            Ok(ops::transpose_axes(a, &[0, 2, 1, 3][..])?.reshape(&[batch, planes, frames])?)
        };
        let (spec_real, spec_imag) = (ordered(real)?, ordered(imag)?);

        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i.
        let out_real = ops::subtract(
            &ops::multiply(&spec_real, &mask_real)?,
            &ops::multiply(&spec_imag, &mask_imag)?,
        )?;
        let out_imag = ops::add(
            &ops::multiply(&spec_real, &mask_imag)?,
            &ops::multiply(&spec_imag, &mask_real)?,
        )?;

        // Unpick the channels, drop DC, invert.
        let split = |a: &Array| -> Result<Array> {
            Ok(ops::transpose_axes(
                &a.reshape(&[batch, bins, channels, frames])?,
                &[0, 2, 1, 3][..],
            )?)
        };
        let (out_real, out_imag) = (split(&out_real)?, split(&out_imag)?);

        // The reference zeroes the DC bin before inverting. Left in, it is an
        // offset on the output rather than anything audible, but it is one the
        // reference does not have.
        let keep = Array::from_iter(1..bins, &[bins - 1]);
        let out_real = ops::pad(
            &out_real.take_axis(&keep, 2)?,
            &[(0, 0), (0, 0), (1, 0), (0, 0)][..],
            Array::from_f32(0.0),
            None,
        )?;
        let out_imag = ops::pad(
            &out_imag.take_axis(&keep, 2)?,
            &[(0, 0), (0, 0), (1, 0), (0, 0)][..],
            Array::from_f32(0.0),
            None,
        )?;

        let spectrum = ops::add(
            &out_real.as_dtype(Dtype::Complex64)?,
            &ops::multiply(
                &out_imag.as_dtype(Dtype::Complex64)?,
                Array::from_complex(mlx_rs::complex64::new(0.0, 1.0)),
            )?,
        )?;
        self.stft.inverse(&spectrum, samples, true)
    }
}
