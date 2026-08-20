//! The convolutional blocks demucs is built from.
//!
//! Two things here are easy to get wrong in ways nothing reports.
//!
//! **Layout.** demucs thinks in channels-first, `[B, C, T]` and `[B, C, F, T]`,
//! and MLX convolutions are channels-last, so every convolution is wrapped in a
//! transpose on the way in and out. A missed transpose does not fail: it convolves
//! over the wrong axis and produces the right shape containing nothing useful.
//!
//! **Which norms exist.** htdemucs sets `norm_starts = 4` with `depth = 4`, so no
//! encoder or decoder layer carries a `GroupNorm` at all. The only normalisation
//! here is inside [`DConv`], at one group. Adding the norms the class could have
//! would load fine and change every number.

use anyhow::Result;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, ops};

use crate::weights::Scope;

/// A convolution over `[B, C, L]`, transposing into MLX's `[B, L, C]`.
pub struct Conv1d {
    weight: Array,
    bias: Option<Array>,
    stride: i32,
    padding: i32,
    dilation: i32,
}

impl Conv1d {
    /// `weight` is `[out, kernel, in]`, which is what the converted artefacts
    /// already hold.
    pub fn load(
        w: &Scope<'_>,
        chin: i32,
        chout: i32,
        kernel: i32,
        stride: i32,
        padding: i32,
        dilation: i32,
    ) -> Result<Self> {
        Ok(Self {
            weight: w.get_shaped("weight", &[chout, kernel, chin])?,
            bias: w.has("bias").then(|| w.get("bias")).transpose()?,
            stride,
            padding,
            dilation,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = ops::swap_axes(x, 1, 2)?;
        let y = ops::conv1d(
            &x,
            &self.weight,
            self.stride,
            self.padding,
            self.dilation,
            1,
        )?;
        let y = match &self.bias {
            Some(b) => ops::add(&y, b)?,
            None => y,
        };
        Ok(ops::swap_axes(&y, 1, 2)?)
    }
}

/// A transposed convolution over `[B, C, L]`.
pub struct ConvTranspose1d {
    weight: Array,
    bias: Option<Array>,
    stride: i32,
}

impl ConvTranspose1d {
    /// `weight` is `[out, kernel, in]`: the same layout as the forward
    /// convolution, not the transpose of it. Worth stating because the obvious
    /// guess is wrong and the shape check is what said so.
    pub fn load(w: &Scope<'_>, chin: i32, chout: i32, kernel: i32, stride: i32) -> Result<Self> {
        Ok(Self {
            weight: w.get_shaped("weight", &[chout, kernel, chin])?,
            bias: w.has("bias").then(|| w.get("bias")).transpose()?,
            stride,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = ops::swap_axes(x, 1, 2)?;
        let y = ops::conv_transpose1d(&x, &self.weight, self.stride, 0, 1, 0, 1)?;
        let y = match &self.bias {
            Some(b) => ops::add(&y, b)?,
            None => y,
        };
        Ok(ops::swap_axes(&y, 1, 2)?)
    }
}

/// A convolution over `[B, C, H, W]`, transposing into MLX's `[B, H, W, C]`.
pub struct Conv2d {
    weight: Array,
    bias: Option<Array>,
    stride: (i32, i32),
    padding: (i32, i32),
}

impl Conv2d {
    pub fn load(
        w: &Scope<'_>,
        chin: i32,
        chout: i32,
        kernel: (i32, i32),
        stride: (i32, i32),
        padding: (i32, i32),
    ) -> Result<Self> {
        Ok(Self {
            weight: w.get_shaped("weight", &[chout, kernel.0, kernel.1, chin])?,
            bias: w.has("bias").then(|| w.get("bias")).transpose()?,
            stride,
            padding,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = to_nhwc(x)?;
        let y = ops::conv2d(&x, &self.weight, self.stride, self.padding, (1, 1), 1)?;
        let y = match &self.bias {
            Some(b) => ops::add(&y, b)?,
            None => y,
        };
        to_nchw(&y)
    }
}

/// A transposed convolution over `[B, C, H, W]`.
pub struct ConvTranspose2d {
    weight: Array,
    bias: Option<Array>,
    stride: (i32, i32),
}

impl ConvTranspose2d {
    pub fn load(
        w: &Scope<'_>,
        chin: i32,
        chout: i32,
        kernel: (i32, i32),
        stride: (i32, i32),
    ) -> Result<Self> {
        Ok(Self {
            weight: w.get_shaped("weight", &[chout, kernel.0, kernel.1, chin])?,
            bias: w.has("bias").then(|| w.get("bias")).transpose()?,
            stride,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x = to_nhwc(x)?;
        let y = ops::conv_transpose2d(&x, &self.weight, self.stride, (0, 0), (1, 1), (0, 0), 1)?;
        let y = match &self.bias {
            Some(b) => ops::add(&y, b)?,
            None => y,
        };
        to_nchw(&y)
    }
}

fn to_nhwc(x: &Array) -> Result<Array> {
    Ok(ops::transpose_axes(x, &[0, 2, 3, 1][..])?)
}

fn to_nchw(x: &Array) -> Result<Array> {
    Ok(ops::transpose_axes(x, &[0, 3, 1, 2][..])?)
}

/// Group normalisation over the channel axis of `[B, C, ...]`.
///
/// Written out rather than taken from `nn::GroupNorm`, which normalises the
/// *last* axis. Here the channels are axis 1 and everything after them is the
/// spatial extent to average over.
pub struct GroupNorm {
    groups: i32,
    weight: Array,
    bias: Array,
    eps: f32,
}

impl GroupNorm {
    pub fn load(w: &Scope<'_>, groups: i32, channels: i32) -> Result<Self> {
        Ok(Self {
            groups,
            weight: w.get_shaped("weight", &[channels])?,
            bias: w.get_shaped("bias", &[channels])?,
            eps: 1e-5,
        })
    }

    /// Normalising a group is normalising a last axis, so this is `layer_norm`.
    ///
    /// Once the channels and the extent after them are folded into one axis a group
    /// norm is exactly a layer norm over it, and mlx has that as a fused kernel
    /// accumulating in full precision. Written out it is eleven passes plus two dtype
    /// conversions, sixty-four times a forward pass. The affine is applied afterwards,
    /// being per channel rather than per element of the normalised axis.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape().to_vec();
        let (b, c) = (shape[0], shape[1]);
        let rest: i32 = shape[2..].iter().product();
        let grouped = x.reshape(&[b, self.groups, c / self.groups * rest])?;
        let normed = mlx_rs::fast::layer_norm(&grouped, None, None, self.eps)?.reshape(&shape)?;

        // Broadcast the per-channel affine over whatever trails the channels.
        let mut affine = vec![1; shape.len()];
        affine[1] = c;
        let weight = self.weight.reshape(&affine)?;
        let bias = self.bias.reshape(&affine)?;
        Ok(ops::add(&ops::multiply(&normed, &weight)?, &bias)?)
    }
}

/// A scalar constant in the same precision as the tensor it will meet.
///
/// `Array::from_f32` builds a float32 array, not a weak scalar, and mlx promotes,
/// so a single epsilon inside a normalisation puts the whole network back into
/// full precision one layer in.
pub fn scalar_like(value: f32, like: &Array) -> Result<Array> {
    Ok(Array::from_f32(value).as_dtype(like.dtype())?)
}

/// `x * (1 + erf(x / sqrt(2))) / 2`, the exact gelu mlx computes.
///
/// Written out rather than calling `nn::gelu`, which builds its constants as
/// float32 arrays, so a half-precision activation meeting it comes back float32.
/// It sits inside every `DConv`.
pub fn gelu(x: &Array) -> Result<Array> {
    let scaled = ops::divide(x, &scalar_like(std::f32::consts::SQRT_2, x)?)?;
    let gated = ops::add(&scalar_like(1.0, x)?, &ops::erf(&scaled)?)?;
    Ok(ops::divide(
        &ops::multiply(x, &gated)?,
        &scalar_like(2.0, x)?,
    )?)
}

/// Gated linear unit over the channel axis: halve the channels, gate with the
/// sigmoid of the other half.
pub fn glu(x: &Array) -> Result<Array> {
    let channels = x.shape()[1];
    let half = channels / 2;
    let a = x.index((.., 0..half));
    let b = x.index((.., half..channels));
    Ok(ops::multiply(&a, &ops::sigmoid(&b)?)?)
}

/// A learned per-channel scale, initialised near zero so a residual branch
/// starts as a no-op.
pub struct LayerScale {
    scale: Array,
}

impl LayerScale {
    pub fn load(w: &Scope<'_>, channels: i32) -> Result<Self> {
        Ok(Self {
            scale: w.get_shaped("scale", &[channels])?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut shape = vec![1; x.ndim()];
        shape[1] = x.shape()[1];
        Ok(ops::multiply(x, &self.scale.reshape(&shape)?)?)
    }
}

/// The dilated residual branch, `depth` blocks of
/// conv → norm → gelu → conv → norm → glu → scale, each added to its input.
pub struct DConv {
    blocks: Vec<DConvBlock>,
}

struct DConvBlock {
    conv1: Conv1d,
    norm1: GroupNorm,
    conv2: Conv1d,
    norm2: GroupNorm,
    scale: LayerScale,
}

impl DConv {
    pub fn load(w: &Scope<'_>, channels: i32, depth: i32, compress: i32) -> Result<Self> {
        let hidden = channels / compress;
        let mut blocks = Vec::new();
        for d in 0..depth {
            let dilation = 1 << d;
            let padding = dilation; // kernel 3, so dilation * (3 / 2)
            // `dconv.layers.<d>.layers.<i>`: the block is a Sequential, and the
            // indices are the positions in it: 2 and 5 are the activations,
            // which carry no weights but still occupy a slot.
            let b = w.child("layers").child(&d.to_string()).child("layers");
            blocks.push(DConvBlock {
                conv1: Conv1d::load(
                    &b.child("0").child("conv"),
                    channels,
                    hidden,
                    3,
                    1,
                    padding,
                    dilation,
                )?,
                norm1: GroupNorm::load(&b.child("1"), 1, hidden)?,
                conv2: Conv1d::load(
                    &b.child("3").child("conv"),
                    hidden,
                    2 * channels,
                    1,
                    1,
                    0,
                    1,
                )?,
                norm2: GroupNorm::load(&b.child("4"), 1, 2 * channels)?,
                scale: LayerScale::load(&b.child("6"), channels)?,
            });
        }
        Ok(Self { blocks })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for block in &self.blocks {
            let y = block.conv1.forward(&x)?;
            let y = gelu(&block.norm1.forward(&y)?)?;
            let y = block.conv2.forward(&y)?;
            let y = glu(&block.norm2.forward(&y)?)?;
            let y = block.scale.forward(&y)?;
            x = ops::add(&x, &y)?;
        }
        Ok(x)
    }
}

/// Which domain a layer works in. The frequency branch is two-dimensional over
/// `[B, C, F, T]`; the time branch is one-dimensional over `[B, C, T]`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Frequency,
    Time,
}

enum Conv {
    Freq(Conv2d),
    Time(Conv1d),
}

enum ConvTr {
    Freq(ConvTranspose2d),
    Time(ConvTranspose1d),
}

/// One encoder step: strided convolution, activation, dilated branch, and a
/// gated rewrite that keeps the channel count.
pub struct EncLayer {
    conv: Conv,
    rewrite: Option<Conv>,
    dconv: Option<DConv>,
    domain: Domain,
    stride: i32,
    pub empty: bool,
}

impl EncLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        w: &Scope<'_>,
        chin: i32,
        chout: i32,
        kernel: i32,
        stride: i32,
        domain: Domain,
        pad: bool,
        empty: bool,
        context: i32,
        dconv: Option<(i32, i32)>,
    ) -> Result<Self> {
        let padding = if pad { kernel / 4 } else { 0 };
        let conv = match domain {
            Domain::Frequency => Conv::Freq(Conv2d::load(
                &w.child("conv.conv"),
                chin,
                chout,
                (kernel, 1),
                (stride, 1),
                (padding, 0),
            )?),
            Domain::Time => Conv::Time(Conv1d::load(
                &w.child("conv.conv"),
                chin,
                chout,
                kernel,
                stride,
                padding,
                1,
            )?),
        };
        if empty {
            return Ok(Self {
                conv,
                rewrite: None,
                dconv: None,
                domain,
                stride,
                empty,
            });
        }
        let rewrite = Some(match domain {
            // Square: the reference passes `1 + 2 * context` as a scalar kernel,
            // so it applies over frequency *and* time. The encoder's context is
            // zero, which makes 1x1 square either way and hides the difference;
            // the decoder's is one, and there it is a 3x3.
            Domain::Frequency => Conv::Freq(Conv2d::load(
                &w.child("rewrite.conv"),
                chout,
                2 * chout,
                (1 + 2 * context, 1 + 2 * context),
                (1, 1),
                (context, context),
            )?),
            Domain::Time => Conv::Time(Conv1d::load(
                &w.child("rewrite.conv"),
                chout,
                2 * chout,
                1 + 2 * context,
                1,
                context,
                1,
            )?),
        });
        let dconv = dconv
            .map(|(depth, compress)| DConv::load(&w.child("dconv"), chout, depth, compress))
            .transpose()?;
        Ok(Self {
            conv,
            rewrite,
            dconv,
            domain,
            stride,
            empty,
        })
    }

    pub fn forward(&self, x: &Array, inject: Option<&Array>) -> Result<Array> {
        let mut x = x.clone();
        if self.domain == Domain::Time {
            if x.ndim() == 4 {
                let s = x.shape().to_vec();
                x = x.reshape(&[s[0], -1, s[3]])?;
            }
            let le = *x.shape().last().expect("a last axis");
            if le % self.stride != 0 {
                let extra = self.stride - le % self.stride;
                let zero = scalar_like(0.0, &x)?;
                x = ops::pad(&x, &[(0, 0), (0, 0), (0, extra)][..], zero, None)?;
            }
        }
        let mut y = match &self.conv {
            Conv::Freq(c) => c.forward(&x)?,
            Conv::Time(c) => c.forward(&x)?,
        };
        if self.empty {
            return Ok(y);
        }
        if let Some(inject) = inject {
            let inject = if inject.ndim() == 3 && y.ndim() == 4 {
                let s = inject.shape().to_vec();
                inject.reshape(&[s[0], s[1], 1, s[2]])?
            } else {
                inject.clone()
            };
            y = ops::add(&y, &inject)?;
        }
        // No GroupNorm here: see the module docs.
        y = gelu(&y)?;
        if let Some(dconv) = &self.dconv {
            y = self.through_dconv(dconv, &y)?;
        }
        match &self.rewrite {
            Some(Conv::Freq(c)) => glu(&c.forward(&y)?),
            Some(Conv::Time(c)) => glu(&c.forward(&y)?),
            None => Ok(y),
        }
    }

    /// The dilated branch is one-dimensional, so a frequency-domain tensor is
    /// folded so each frequency bin becomes its own row before it runs.
    fn through_dconv(&self, dconv: &DConv, y: &Array) -> Result<Array> {
        if self.domain == Domain::Time {
            return dconv.forward(y);
        }
        let s = y.shape().to_vec();
        let (b, c, fr, t) = (s[0], s[1], s[2], s[3]);
        let folded = ops::transpose_axes(y, &[0, 2, 1, 3][..])?.reshape(&[b * fr, c, t])?;
        let out = dconv.forward(&folded)?;
        Ok(ops::transpose_axes(
            &out.reshape(&[b, fr, c, t])?,
            &[0, 2, 1, 3][..],
        )?)
    }
}

/// One decoder step: gated rewrite, dilated branch, transposed convolution.
pub struct DecLayer {
    conv_tr: ConvTr,
    rewrite: Option<Conv>,
    dconv: Option<DConv>,
    domain: Domain,
    chin: i32,
    pad: i32,
    last: bool,
    pub empty: bool,
}

impl DecLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        w: &Scope<'_>,
        chin: i32,
        chout: i32,
        kernel: i32,
        stride: i32,
        domain: Domain,
        pad: bool,
        empty: bool,
        last: bool,
        context: i32,
        dconv: Option<(i32, i32)>,
    ) -> Result<Self> {
        let padding = if pad { kernel / 4 } else { 0 };
        let conv_tr = match domain {
            Domain::Frequency => ConvTr::Freq(ConvTranspose2d::load(
                &w.child("conv_tr.conv"),
                chin,
                chout,
                (kernel, 1),
                (stride, 1),
            )?),
            Domain::Time => ConvTr::Time(ConvTranspose1d::load(
                &w.child("conv_tr.conv"),
                chin,
                chout,
                kernel,
                stride,
            )?),
        };
        if empty {
            return Ok(Self {
                conv_tr,
                rewrite: None,
                dconv: None,
                domain,
                chin,
                pad: padding,
                last,
                empty,
            });
        }
        let rewrite = Some(match domain {
            Domain::Frequency => Conv::Freq(Conv2d::load(
                &w.child("rewrite.conv"),
                chin,
                2 * chin,
                (1 + 2 * context, 1 + 2 * context),
                (1, 1),
                (context, context),
            )?),
            Domain::Time => Conv::Time(Conv1d::load(
                &w.child("rewrite.conv"),
                chin,
                2 * chin,
                1 + 2 * context,
                1,
                context,
                1,
            )?),
        });
        let dconv = dconv
            .map(|(depth, compress)| DConv::load(&w.child("dconv"), chin, depth, compress))
            .transpose()?;
        Ok(Self {
            conv_tr,
            rewrite,
            dconv,
            domain,
            chin,
            pad: padding,
            last,
            empty,
        })
    }

    /// Returns the layer's output and the value before the transposed
    /// convolution, which the time branch injects into.
    pub fn forward(&self, x: &Array, skip: Option<&Array>, length: i32) -> Result<(Array, Array)> {
        let mut x = x.clone();
        if self.domain == Domain::Frequency && x.ndim() == 3 {
            let s = x.shape().to_vec();
            x = x.reshape(&[s[0], self.chin, -1, s[2]])?;
        }
        let y = if self.empty {
            x
        } else {
            let mut x = match skip {
                Some(skip) => ops::add(&x, skip)?,
                None => x,
            };
            x = match &self.rewrite {
                Some(Conv::Freq(c)) => glu(&c.forward(&x)?)?,
                Some(Conv::Time(c)) => glu(&c.forward(&x)?)?,
                None => x,
            };
            if let Some(dconv) = &self.dconv {
                x = self.through_dconv(dconv, &x)?;
            }
            x
        };

        let mut z = match &self.conv_tr {
            ConvTr::Freq(c) => c.forward(&y)?,
            ConvTr::Time(c) => c.forward(&y)?,
        };
        z = match self.domain {
            Domain::Frequency if self.pad > 0 => {
                let f = z.shape()[2];
                z.index((.., .., self.pad..f - self.pad, ..))
            }
            Domain::Frequency => z,
            Domain::Time => z.index((.., .., self.pad..self.pad + length)),
        };
        if !self.last {
            z = gelu(&z)?;
        }
        Ok((z, y))
    }

    fn through_dconv(&self, dconv: &DConv, y: &Array) -> Result<Array> {
        if self.domain == Domain::Time {
            return dconv.forward(y);
        }
        let s = y.shape().to_vec();
        let (b, c, fr, t) = (s[0], s[1], s[2], s[3]);
        let folded = ops::transpose_axes(y, &[0, 2, 1, 3][..])?.reshape(&[b * fr, c, t])?;
        let out = dconv.forward(&folded)?;
        Ok(ops::transpose_axes(
            &out.reshape(&[b, fr, c, t])?,
            &[0, 2, 1, 3][..],
        )?)
    }
}
