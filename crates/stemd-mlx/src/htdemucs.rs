//! htdemucs itself: the two branches, the transformer between them, and the
//! bookkeeping that joins them back into audio.
//!
//! A spectrogram goes down four encoder layers while the waveform goes down four
//! of its own; the two meet in the transformer; they come back up through four
//! decoders each, with skip connections; and the two answers are added. The
//! branches are not independent: each encoder injects into the other, and the last
//! decoder of the frequency branch feeds the time branch, so the loops are
//! interleaved rather than sequential.
//!
//! Every constant is read from the artefact's configuration in the reference and
//! repeated here because the model is built rather than traced. They are stated
//! once, in [`Config`].

use anyhow::{Result, bail};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype, ops};

use crate::layers::{Conv1d, DecLayer, Domain, EncLayer, scalar_like};
use crate::precision::Precision;
use crate::spectral::Spectral;
use crate::transformer::CrossTransformer;
use crate::weights::{Scope, Weights};

/// What the traced artefact says about itself.
///
/// Not configurable: these are the numbers htdemucs was trained with, and a
/// different value here does not produce a different model, it produces a
/// model whose weights no longer fit.
pub struct Config {
    pub depth: i32,
    pub channels: i32,
    pub audio_channels: i32,
    pub sources: i32,
    pub kernel: i32,
    pub stride: i32,
    pub context: i32,
    pub context_enc: i32,
    pub bottom_channels: i32,
    pub transformer_layers: i32,
    pub transformer_heads: i32,
    pub transformer_ff: i32,
    pub freq_emb_scale: f32,
    pub freq_emb_weight_scale: f32,
    pub dconv_depth: i32,
    pub dconv_compress: i32,
    /// Length in seconds the model was trained on. Shorter input is padded up
    /// to it, because the transformer's position embedding was learned at that
    /// extent.
    pub segment: f32,
    pub sample_rate: i32,
    /// Precision the encoder, transformer and decoder run at.
    ///
    /// Everything outside them stays `f32`: the spectrogram and its inverse, the
    /// per-branch standardisation, and the overlap-add. Those accumulate over the
    /// whole track or take variances over hundreds of thousands of samples, which is
    /// where half precision would hurt.
    ///
    /// [`Precision`] rather than a [`Dtype`] so a caller need not link mlx.
    pub precision: Precision,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            depth: 4,
            channels: 48,
            audio_channels: 2,
            sources: 4,
            kernel: 8,
            stride: 4,
            context: 1,
            context_enc: 0,
            bottom_channels: 512,
            transformer_layers: 5,
            transformer_heads: 8,
            transformer_ff: 2048,
            freq_emb_scale: 0.2,
            freq_emb_weight_scale: 10.0,
            dconv_depth: 2,
            dconv_compress: 8,
            segment: 7.8,
            sample_rate: 44100,
            precision: Precision::default(),
        }
    }
}

impl Config {
    /// Samples in one training segment, which is the length the model runs on.
    pub fn training_length(&self) -> i32 {
        (self.segment * self.sample_rate as f32) as i32
    }
}

/// The frequency embedding added after the first encoder layer.
///
/// A lookup over frequency bins, scaled twice: once by the constant folded into
/// the stored weights and once by `freq_emb_scale` where it is added.
struct FreqEmbedding {
    weight: Array,
    scale: f32,
}

impl FreqEmbedding {
    fn load(w: &Scope<'_>, bins: i32, channels: i32, scale: f32) -> Result<Self> {
        Ok(Self {
            weight: w.get_shaped("embedding.weight", &[bins, channels])?,
            scale,
        })
    }

    /// `[bins, channels] -> [1, channels, bins, 1]`, ready to broadcast over a
    /// `[B, C, F, T]` tensor.
    fn forward(&self, bins: i32) -> Result<Array> {
        let rows = self.weight.index((0..bins, ..));
        let scaled = ops::multiply(&rows, scalar_like(self.scale, &rows)?)?;
        let t = ops::transpose_axes(&scaled, &[1, 0][..])?;
        let s = t.shape().to_vec();
        Ok(t.reshape(&[1, s[0], s[1], 1])?)
    }
}

pub struct HtDemucs {
    config: Config,
    spectral: Spectral,
    encoder: Vec<EncLayer>,
    tencoder: Vec<EncLayer>,
    decoder: Vec<DecLayer>,
    tdecoder: Vec<DecLayer>,
    freq_emb: FreqEmbedding,
    upsampler: Conv1d,
    upsampler_t: Conv1d,
    downsampler: Conv1d,
    downsampler_t: Conv1d,
    transformer: CrossTransformer,
}

impl HtDemucs {
    /// Build the model from one artefact's tensors.
    ///
    /// `prefix` names the sub-model. A single-model artefact holds only
    /// `model_0`; a fine-tuned set holds one per source, and a caller that
    /// wants a specific source asks for that source's model by index.
    pub const fn config(&self) -> &Config {
        &self.config
    }

    pub fn load(weights: &Weights, prefix: &str, config: Config) -> Result<Self> {
        let w = weights.at(prefix);
        let dconv = Some((config.dconv_depth, config.dconv_compress));
        let cac = 2; // complex-as-channels doubles the audio channels

        let mut encoder = Vec::new();
        let mut tencoder = Vec::new();
        let mut decoder = Vec::new();
        let mut tdecoder = Vec::new();

        // Channels double at every step: 48, 96, 192, 384 out of 4 and 2 in.
        for index in 0..config.depth {
            let chout = config.channels * (1 << index);
            let chin_z = if index == 0 {
                config.audio_channels * cac
            } else {
                chout / 2
            };
            let chin_t = if index == 0 {
                config.audio_channels
            } else {
                chout / 2
            };
            let name = index.to_string();

            encoder.push(EncLayer::load(
                &w.child("encoder").child(&name),
                chin_z,
                chout,
                config.kernel,
                config.stride,
                Domain::Frequency,
                true,
                false,
                config.context_enc,
                dconv,
            )?);
            tencoder.push(EncLayer::load(
                &w.child("tencoder").child(&name),
                chin_t,
                chout,
                config.kernel,
                config.stride,
                Domain::Time,
                true,
                false,
                config.context_enc,
                dconv,
            )?);

            // Decoders mirror the encoders and are stored outermost-first, so
            // index 0 here is the *last* decoder to run.
            let last = index == 0;
            let dec_out_z = if last {
                config.audio_channels * config.sources * cac
            } else {
                chin_z
            };
            let dec_out_t = if last {
                config.audio_channels * config.sources
            } else {
                chin_t
            };
            let position = (config.depth - 1 - index).to_string();
            decoder.insert(
                0,
                DecLayer::load(
                    &w.child("decoder").child(&position),
                    chout,
                    dec_out_z,
                    config.kernel,
                    config.stride,
                    Domain::Frequency,
                    true,
                    false,
                    last,
                    config.context,
                    dconv,
                )?,
            );
            tdecoder.insert(
                0,
                DecLayer::load(
                    &w.child("tdecoder").child(&position),
                    chout,
                    dec_out_t,
                    config.kernel,
                    config.stride,
                    Domain::Time,
                    true,
                    false,
                    last,
                    config.context,
                    dconv,
                )?,
            );
        }

        let bottom = config.channels * (1 << (config.depth - 1));
        Ok(Self {
            spectral: Spectral::new(),
            encoder,
            tencoder,
            decoder,
            tdecoder,
            freq_emb: FreqEmbedding::load(
                &w.child("freq_emb"),
                512,
                config.channels,
                config.freq_emb_weight_scale,
            )?,
            upsampler: Conv1d::load(
                &w.child("channel_upsampler.conv"),
                bottom,
                config.bottom_channels,
                1,
                1,
                0,
                1,
            )?,
            upsampler_t: Conv1d::load(
                &w.child("channel_upsampler_t.conv"),
                bottom,
                config.bottom_channels,
                1,
                1,
                0,
                1,
            )?,
            downsampler: Conv1d::load(
                &w.child("channel_downsampler.conv"),
                config.bottom_channels,
                bottom,
                1,
                1,
                0,
                1,
            )?,
            downsampler_t: Conv1d::load(
                &w.child("channel_downsampler_t.conv"),
                config.bottom_channels,
                bottom,
                1,
                1,
                0,
                1,
            )?,
            transformer: CrossTransformer::load(
                &w.child("crosstransformer"),
                config.bottom_channels,
                config.transformer_heads,
                config.transformer_ff,
                config.transformer_layers,
            )?,
            config,
        })
    }

    /// Separate one segment. `mix` is `[B, C, T]`; the result is
    /// `[B, sources, C, T]`.
    pub fn forward(&self, mix: &Array) -> Result<Array> {
        let length = *mix.shape().last().expect("a last axis");
        let training = self.config.training_length();
        if length > training {
            bail!(
                "{length} samples is longer than the {training}-sample segment this model runs on"
            );
        }
        // Padded up rather than run short: the transformer's position embedding
        // was learned at the training extent, so a shorter input is a different
        // model, not a faster one.
        let padded = ops::pad(
            mix,
            &[(0, 0), (0, 0), (0, training - length)][..],
            Array::from_f32(0.0),
            None,
        )?;

        let z = self.spectral.forward(&padded)?;
        let mag = complex_as_channels(&z)?;

        // Each branch is normalised by its own statistics and put back
        // afterwards, so the model sees a consistent scale whatever the track.
        let (x, mean, std) = standardise(&mag, &[1, 2, 3])?;
        let (xt, mean_t, std_t) = standardise(&padded, &[1, 2])?;
        // The one place the configured precision becomes an mlx dtype on this
        // path: everything above is f32 by construction, everything below runs
        // at whatever these two casts produce.
        let dtype = self.config.precision.dtype();
        let (x, xt) = (x.as_dtype(dtype)?, xt.as_dtype(dtype)?);

        let (x, xt, saved, saved_t, lengths, lengths_t) = self.encode(x, xt)?;
        let (x, xt) = self.bottleneck(x, xt)?;
        let (x, xt) = self.decode(x, xt, saved, saved_t, lengths, lengths_t)?;

        // Frequency branch: back to a spectrogram, masked onto the original
        // complex bins, and inverted.
        let (x, xt) = (x.as_dtype(Dtype::Float32)?, xt.as_dtype(Dtype::Float32)?);
        let s = mag.shape().to_vec();
        let (b, fq, t) = (s[0], s[2], s[3]);
        let x = x.reshape(&[b, self.config.sources, -1, fq, t])?;
        let x = destandardise(&x, &mean, &std)?;
        let x = self.spectral.inverse(&channels_as_complex(&x)?, training)?;

        // Time branch: the same, then the two are added.
        let actual = *xt.shape().last().expect("a last axis");
        let xt = xt.reshape(&[b, self.config.sources, -1, actual])?;
        let xt = destandardise(&xt, &mean_t, &std_t)?;

        let out = ops::add(&xt, &centre_trim(&x, actual)?)?;
        Ok(out.index((.., .., .., 0..length)))
    }

    /// Both encoder stacks, interleaved: the time branch injects into the
    /// frequency branch at the depth where their shapes meet.
    #[allow(clippy::type_complexity)]
    fn encode(
        &self,
        mut x: Array,
        mut xt: Array,
    ) -> Result<(Array, Array, Vec<Array>, Vec<Array>, Vec<i32>, Vec<i32>)> {
        let mut saved = Vec::new();
        let mut saved_t = Vec::new();
        let mut lengths = Vec::new();
        let mut lengths_t = Vec::new();

        for (index, encode) in self.encoder.iter().enumerate() {
            lengths.push(*x.shape().last().expect("a last axis"));
            let mut inject = None;
            if let Some(tenc) = self.tencoder.get(index) {
                lengths_t.push(*xt.shape().last().expect("a last axis"));
                xt = tenc.forward(&xt, None)?;
                if tenc.empty {
                    inject = Some(xt.clone());
                } else {
                    saved_t.push(xt.clone());
                }
            }
            x = encode.forward(&x, inject.as_ref())?;
            if index == 0 {
                let bins = x.shape()[2];
                let emb = self.freq_emb.forward(bins)?;
                let scaled = ops::multiply(&emb, scalar_like(self.config.freq_emb_scale, &x)?)?;
                x = ops::add(&x, &scaled)?;
            }
            saved.push(x.clone());
        }
        Ok((x, xt, saved, saved_t, lengths, lengths_t))
    }

    /// Up to the transformer's width, across it, and back down.
    fn bottleneck(&self, x: Array, xt: Array) -> Result<(Array, Array)> {
        let s = x.shape().to_vec();
        let (b, c, f, t) = (s[0], s[1], s[2], s[3]);

        let flat = x.reshape(&[b, c, f * t])?;
        let up = self
            .upsampler
            .forward(&flat)?
            .reshape(&[b, self.config.bottom_channels, f, t])?;
        let up_t = self.upsampler_t.forward(&xt)?;

        let (x, xt) = self.transformer.forward(&up, &up_t)?;

        let flat = x.reshape(&[b, self.config.bottom_channels, f * t])?;
        let down = self.downsampler.forward(&flat)?.reshape(&[b, c, f, t])?;
        let down_t = self.downsampler_t.forward(&xt)?;
        Ok((down, down_t))
    }

    /// Both decoder stacks, consuming the skips in reverse.
    fn decode(
        &self,
        mut x: Array,
        mut xt: Array,
        mut saved: Vec<Array>,
        mut saved_t: Vec<Array>,
        mut lengths: Vec<i32>,
        mut lengths_t: Vec<i32>,
    ) -> Result<(Array, Array)> {
        let offset = self.config.depth as usize - self.tdecoder.len();
        for (index, decode) in self.decoder.iter().enumerate() {
            let skip = saved.pop().expect("one skip per encoder layer");
            let length = lengths.pop().expect("one length per encoder layer");
            let (out, pre) = decode.forward(&x, Some(&skip), length)?;
            x = out;

            if index >= offset {
                let tdec = &self.tdecoder[index - offset];
                let length_t = lengths_t.pop().expect("one length per time encoder");
                if tdec.empty {
                    // The frequency branch's own output seeds the time branch.
                    let pre = pre.index((.., .., 0));
                    xt = tdec.forward(&pre, None, length_t)?.0;
                } else {
                    let skip_t = saved_t.pop().expect("one skip per time encoder");
                    xt = tdec.forward(&xt, Some(&skip_t), length_t)?.0;
                }
            }
        }
        if !saved.is_empty() || !saved_t.is_empty() {
            bail!("skip connections were not fully consumed");
        }
        Ok((x, xt))
    }
}

/// Real and imaginary parts interleaved as channels: `[B, C, F, T]` complex
/// becomes `[B, 2C, F, T]` real.
fn complex_as_channels(z: &Array) -> Result<Array> {
    let s = z.shape().to_vec();
    let (b, c, f, t) = (s[0], s[1], s[2], s[3]);
    let stacked = ops::stack_axis(&[ops::real(z)?, ops::imag(z)?], 2)?;
    Ok(stacked.reshape(&[b, c * 2, f, t])?)
}

/// The inverse: `[B, S, 2C, F, T]` real becomes `[B, S, C, F, T]` complex.
fn channels_as_complex(x: &Array) -> Result<Array> {
    let s = x.shape().to_vec();
    let (b, sources, c2, f, t) = (s[0], s[1], s[2], s[3], s[4]);
    let split = x.reshape(&[b, sources, c2 / 2, 2, f, t])?;
    let moved = ops::transpose_axes(&split, &[0, 1, 2, 4, 5, 3][..])?;
    let real = moved.index((.., .., .., .., .., 0));
    let imag = moved.index((.., .., .., .., .., 1));
    // No `complex(re, im)` in mlx-rs; multiplying by i and adding is the same.
    let i = Array::from_complex(mlx_rs::complex64::new(0.0, 1.0));
    Ok(ops::add(&real, &ops::multiply(&imag, &i)?)?)
}

/// Subtract the mean and divide by the deviation over `axes`, returning both so
/// the operation can be undone after the model has run.
fn standardise(x: &Array, axes: &[i32]) -> Result<(Array, Array, Array)> {
    let mean = ops::mean_axes(x, axes, true)?;
    let centred = ops::subtract(x, &mean)?;
    let var = ops::mean_axes(&ops::multiply(&centred, &centred)?, axes, true)?;
    let std = ops::sqrt(&var)?;
    let scaled = ops::divide(&centred, &ops::add(&std, Array::from_f32(1e-5))?)?;
    Ok((scaled, mean, std))
}

/// Undo [`standardise`] for a tensor that has gained a sources axis, so the
/// statistics broadcast one axis further out than they were taken over.
fn destandardise(x: &Array, mean: &Array, std: &Array) -> Result<Array> {
    let widen = |a: &Array| -> Result<Array> {
        let mut shape = a.shape().to_vec();
        shape.insert(1, 1);
        Ok(a.reshape(&shape)?)
    };
    Ok(ops::add(&ops::multiply(x, &widen(std)?)?, &widen(mean)?)?)
}

/// Trim a tensor to `length` about its centre.
fn centre_trim(x: &Array, length: i32) -> Result<Array> {
    let have = *x.shape().last().expect("a last axis");
    if have == length {
        return Ok(x.clone());
    }
    if have < length {
        bail!("cannot centre-trim {have} samples to {length}");
    }
    let start = (have - length) / 2;
    Ok(x.index((.., .., .., start..start + length)))
}
