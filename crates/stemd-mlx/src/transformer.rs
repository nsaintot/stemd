//! The cross-domain transformer that joins the two branches.
//!
//! Five layers, alternating: even indices attend within a branch, odd indices
//! attend across them, with the frequency branch reading the time branch and the
//! time branch reading the frequency branch's previous value. Both carry
//! sinusoidal position, two-dimensional over `[frequency, time]` for one and
//! one-dimensional for the other.
//!
//! Everything here works channel-last, `[B, T, C]`, unlike [`crate::layers`],
//! which is channel-first. That is the reference's arrangement, and it is why
//! `LayerScale` appears again here rather than being reused: the one in `layers`
//! broadcasts over axis 1.
//!
//! The `cape` position embedding and the sparse-attention paths in the reference
//! are dead for htdemucs and are not ported.

use anyhow::Result;
use mlx_rs::{Array, ops};

use crate::layers::gelu;
use crate::weights::Scope;

const MAX_PERIOD: f32 = 10_000.0;

/// The epsilon every normalisation in the reference carries.
const EPS: f32 = 1e-5;

/// One-dimensional sinusoidal position: `[T, 1, dim]`, cosines then sines.
fn sin_embedding(length: i32, dim: i32) -> Result<Array> {
    let half = dim / 2;
    let pos = Array::from_iter((0..length).map(|i| i as f32), &[length, 1, 1]);
    let adim = Array::from_iter((0..half).map(|i| i as f32), &[1, 1, half]);
    let denom = ops::power(
        Array::from_f32(MAX_PERIOD),
        &ops::divide(&adim, Array::from_f32((half - 1) as f32))?,
    )?;
    let phase = ops::divide(&pos, &denom)?;
    Ok(ops::concatenate_axis(
        &[ops::cos(&phase)?, ops::sin(&phase)?],
        -1,
    )?)
}

/// Two-dimensional sinusoidal position: `[1, d_model, height, width]`.
///
/// Half the channels encode width and half encode height, and within each half
/// sine and cosine alternate channel by channel. A version that concatenated
/// instead of interleaving would have the right shape and the wrong meaning.
fn sin_embedding_2d(d_model: i32, height: i32, width: i32) -> Result<Array> {
    let half = d_model / 2;
    let terms = half / 2;
    let div = Array::from_iter(
        (0..terms).map(|i| (-(MAX_PERIOD.ln()) / half as f32 * (2 * i) as f32).exp()),
        &[1, terms],
    );

    // One axis at a time: [len, 1] * [1, terms] -> [len, terms] -> [terms, len].
    let axis = |len: i32| -> Result<(Array, Array)> {
        let pos = Array::from_iter((0..len).map(|i| i as f32), &[len, 1]);
        let angle = ops::multiply(&pos, &div)?;
        Ok((
            ops::transpose_axes(&ops::sin(&angle)?, &[1, 0][..])?,
            ops::transpose_axes(&ops::cos(&angle)?, &[1, 0][..])?,
        ))
    };

    let (sin_w, cos_w) = axis(width)?;
    let sin_w = ops::broadcast_to(&sin_w.reshape(&[terms, 1, width])?, &[terms, height, width])?;
    let cos_w = ops::broadcast_to(&cos_w.reshape(&[terms, 1, width])?, &[terms, height, width])?;
    let pe_w = ops::stack_axis(&[sin_w, cos_w], 1)?.reshape(&[half, height, width])?;

    let (sin_h, cos_h) = axis(height)?;
    let sin_h = ops::broadcast_to(
        &sin_h.reshape(&[terms, height, 1])?,
        &[terms, height, width],
    )?;
    let cos_h = ops::broadcast_to(
        &cos_h.reshape(&[terms, height, 1])?,
        &[terms, height, width],
    )?;
    let pe_h = ops::stack_axis(&[sin_h, cos_h], 1)?.reshape(&[half, height, width])?;

    let pe = ops::concatenate_axis(&[pe_w, pe_h], 0)?;
    Ok(pe.reshape(&[1, d_model, height, width])?)
}

/// Normalisation over the last axis of `[B, T, C]`.
struct LayerNorm {
    weight: Array,
    bias: Array,
}

impl LayerNorm {
    fn load(w: &Scope<'_>, dim: i32) -> Result<Self> {
        Ok(Self {
            weight: w.get_shaped("weight", &[dim])?,
            bias: w.get_shaped("bias", &[dim])?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(mlx_rs::fast::layer_norm(
            x,
            Some(&self.weight),
            Some(&self.bias),
            EPS,
        )?)
    }
}

/// The output norm: a group norm at *one* group over `[B, T, C]`, so the
/// statistics are taken across time and channels together and only the affine
/// is per channel.
struct OutNorm {
    weight: Array,
    bias: Array,
}

impl OutNorm {
    fn load(w: &Scope<'_>, dim: i32) -> Result<Self> {
        Ok(Self {
            weight: w.get_shaped("weight", &[dim])?,
            bias: w.get_shaped("bias", &[dim])?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        // Statistics over time and channels together, so both axes fold into
        // the one `layer_norm` normalises. The affine is per channel and is
        // applied after, not by the kernel.
        let shape = x.shape().to_vec();
        let (b, t, c) = (shape[0], shape[1], shape[2]);
        let flat = x.reshape(&[b, 1, t * c])?;
        let normed = mlx_rs::fast::layer_norm(&flat, None, None, EPS)?.reshape(&shape)?;
        Ok(ops::add(
            &ops::multiply(&normed, &self.weight)?,
            &self.bias,
        )?)
    }
}

/// A per-channel scale over the last axis.
struct LayerScale {
    scale: Array,
}

impl LayerScale {
    fn load(w: &Scope<'_>, dim: i32) -> Result<Self> {
        Ok(Self {
            scale: w.get_shaped("scale", &[dim])?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(ops::multiply(x, &self.scale)?)
    }
}

struct Linear {
    /// Stored `[chin, chout]`, which is the transpose of how it arrives.
    ///
    /// Torch keeps a linear weight as `[out, in]` and matmul wants `[in, out]`. Doing
    /// it once at load rather than inside `forward` measures the same, since mlx is
    /// lazy, and puts the rearrangement on the weight rather than the pass.
    weight: Array,
    bias: Array,
}

impl Linear {
    fn load(w: &Scope<'_>, chin: i32, chout: i32) -> Result<Self> {
        let weight = w.get_shaped("weight", &[chout, chin])?;
        Ok(Self {
            weight: ops::transpose_axes(&weight, &[1, 0][..])?,
            bias: w.get_shaped("bias", &[chout])?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = ops::matmul(x, &self.weight)?;
        Ok(ops::add(&y, &self.bias)?)
    }
}

/// Multi-head attention: the projections and the head split written out, the
/// attention itself handed to mlx. The projections stay explicit because their
/// scaling and layout would have to be verified against a library's conventions
/// anyway.
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    heads: i32,
}

impl Attention {
    fn load(w: &Scope<'_>, dim: i32, heads: i32) -> Result<Self> {
        Ok(Self {
            query: Linear::load(&w.child("query_proj"), dim, dim)?,
            key: Linear::load(&w.child("key_proj"), dim, dim)?,
            value: Linear::load(&w.child("value_proj"), dim, dim)?,
            out: Linear::load(&w.child("out_proj"), dim, dim)?,
            heads,
        })
    }

    fn forward(&self, q: &Array, kv: &Array) -> Result<Array> {
        let split = |x: &Array| -> Result<Array> {
            let s = x.shape().to_vec();
            let (b, t, c) = (s[0], s[1], s[2]);
            let heads = x.reshape(&[b, t, self.heads, c / self.heads])?;
            Ok(ops::transpose_axes(&heads, &[0, 2, 1, 3][..])?)
        };

        let qh = split(&self.query.forward(q)?)?;
        let kh = split(&self.key.forward(kv)?)?;
        let vh = split(&self.value.forward(kv)?)?;

        //  mlx's own, rather than matmul/softmax/matmul written out. Tried as a speed fix
        //  on the theory that materialising the scores, a 231 MB tensor twenty times a
        //  forward pass, was worth avoiding. It measured identical, because the fused
        //  Metal kernel engages only when the query length is one.
        let head_dim = *qh.shape().last().expect("a last axis");
        let scale = 1.0 / (head_dim as f32).sqrt();
        let attended = mlx_rs::fast::scaled_dot_product_attention(&qh, &kh, &vh, scale, None)?;

        let s = attended.shape().to_vec();
        let merged = ops::transpose_axes(&attended, &[0, 2, 1, 3][..])?.reshape(&[
            s[0],
            s[2],
            s[1] * s[3],
        ])?;
        self.out.forward(&merged)
    }
}

/// One layer, self-attending or cross-attending.
///
/// The two differ only in where the keys come from and in carrying a third
/// norm, so they are one struct: a `norm3` that is `None` means self-attention.
struct Layer {
    attn: Attention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: Option<LayerNorm>,
    linear1: Linear,
    linear2: Linear,
    gamma1: LayerScale,
    gamma2: LayerScale,
    norm_out: OutNorm,
}

impl Layer {
    fn load(w: &Scope<'_>, dim: i32, heads: i32, ff: i32, cross: bool) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(
                &w.child(if cross { "cross_attn" } else { "attn" }),
                dim,
                heads,
            )?,
            norm1: LayerNorm::load(&w.child("norm1"), dim)?,
            norm2: LayerNorm::load(&w.child("norm2"), dim)?,
            norm3: cross
                .then(|| LayerNorm::load(&w.child("norm3"), dim))
                .transpose()?,
            linear1: Linear::load(&w.child("linear1"), dim, ff)?,
            linear2: Linear::load(&w.child("linear2"), ff, dim)?,
            gamma1: LayerScale::load(&w.child("gamma_1"), dim)?,
            gamma2: LayerScale::load(&w.child("gamma_2"), dim)?,
            norm_out: OutNorm::load(&w.child("norm_out.gn"), dim)?,
        })
    }

    /// `other` is `None` for self-attention. Norm-first throughout.
    fn forward(&self, x: &Array, other: Option<&Array>) -> Result<Array> {
        let (attended, feed_norm) = match (other, &self.norm3) {
            // Cross: queries from this branch, keys and values from the other,
            // and the feed-forward reads norm3 rather than norm2.
            (Some(other), Some(norm3)) => {
                let kv = self.norm2.forward(other)?;
                (self.attn.forward(&self.norm1.forward(x)?, &kv)?, norm3)
            }
            _ => {
                let normed = self.norm1.forward(x)?;
                (self.attn.forward(&normed, &normed)?, &self.norm2)
            }
        };
        let x = ops::add(x, &self.gamma1.forward(&attended)?)?;
        let fed = self
            .linear2
            .forward(&gelu(&self.linear1.forward(&feed_norm.forward(&x)?)?)?)?;
        let x = ops::add(&x, &self.gamma2.forward(&fed)?)?;
        self.norm_out.forward(&x)
    }
}

/// The whole cross-domain block.
pub struct CrossTransformer {
    norm_in: LayerNorm,
    norm_in_t: LayerNorm,
    layers: Vec<Layer>,
    layers_t: Vec<Layer>,
    dim: i32,
}

impl CrossTransformer {
    pub fn load(w: &Scope<'_>, dim: i32, heads: i32, ff: i32, depth: i32) -> Result<Self> {
        let mut layers = Vec::new();
        let mut layers_t = Vec::new();
        for idx in 0..depth {
            // Even indices attend within a branch, odd across.
            let cross = idx % 2 == 1;
            let name = idx.to_string();
            layers.push(Layer::load(
                &w.child("layers").child(&name),
                dim,
                heads,
                ff,
                cross,
            )?);
            layers_t.push(Layer::load(
                &w.child("layers_t").child(&name),
                dim,
                heads,
                ff,
                cross,
            )?);
        }
        Ok(Self {
            norm_in: LayerNorm::load(&w.child("norm_in"), dim)?,
            norm_in_t: LayerNorm::load(&w.child("norm_in_t"), dim)?,
            layers,
            layers_t,
            dim,
        })
    }

    /// `x` is `[B, C, F, T1]` and `xt` is `[B, C, T2]`; both come back in the
    /// shape they went in.
    pub fn forward(&self, x: &Array, xt: &Array) -> Result<(Array, Array)> {
        let s = x.shape().to_vec();
        let (b, c, fr, t1) = (s[0], s[1], s[2], s[3]);

        // [B, C, F, T] -> [B, T*F, C], which is the order the embedding is
        // flattened in too.
        let flatten = |a: &Array| -> Result<Array> {
            Ok(ops::transpose_axes(a, &[0, 3, 2, 1][..])?.reshape(&[b, t1 * fr, c])?)
        };
        let pos2d = ops::broadcast_to(&sin_embedding_2d(c, fr, t1)?, &[b, c, fr, t1])?;
        // Built in f32 from shapes alone, so it has to adopt the branch's
        // precision rather than dictate it.
        let pos2d = flatten(&pos2d)?.as_dtype(x.dtype())?;
        let mut x = ops::add(&self.norm_in.forward(&flatten(x)?)?, &pos2d)?;

        let t2 = xt.shape()[2];
        let pos = sin_embedding(t2, self.dim)?;
        let pos = ops::transpose_axes(&pos, &[1, 0, 2][..])?.as_dtype(xt.dtype())?;
        let mut xt = ops::add(&self.norm_in_t.forward(&ops::swap_axes(xt, 1, 2)?)?, &pos)?;

        for (layer, layer_t) in self.layers.iter().zip(&self.layers_t) {
            if layer.norm3.is_none() {
                x = layer.forward(&x, None)?;
                xt = layer_t.forward(&xt, None)?;
            } else {
                // The time branch reads the frequency branch's *previous*
                // value, not the one this layer just produced.
                let previous = x.clone();
                x = layer.forward(&x, Some(&xt))?;
                xt = layer_t.forward(&xt, Some(&previous))?;
            }
        }

        let x = ops::transpose_axes(&x.reshape(&[b, t1, fr, c])?, &[0, 3, 2, 1][..])?;
        Ok((x, ops::swap_axes(&xt, 1, 2)?))
    }
}

/// Exposed for the tests, which check the embeddings on their own: they are
/// pure functions of the shape, so a mistake in one is easy to find here and
/// very hard to find once it has been added to a tensor.
pub mod embeddings {
    use super::{Result, sin_embedding, sin_embedding_2d};
    use mlx_rs::Array;

    pub fn one_dimensional(length: i32, dim: i32) -> Result<Array> {
        sin_embedding(length, dim)
    }

    pub fn two_dimensional(d_model: i32, height: i32, width: i32) -> Result<Array> {
        sin_embedding_2d(d_model, height, width)
    }
}
