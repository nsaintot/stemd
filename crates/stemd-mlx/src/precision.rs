//! What precision a model runs at, said without naming a backend.
//!
//! [`mlx_rs::Dtype`] used to sit in both models' public config, which put `mlx_rs`
//! in the signature of anything wanting to ask for half precision, and on a
//! machine without Metal none of those can name one.
//!
//! Two values, because two is what the models were measured at.

use std::fmt;

use crate::device::Accelerator;

/// Which architecture, for the purpose of choosing a precision.
///
/// The two want different things on the same card, so a precision cannot be
/// chosen from the hardware alone. See [`Precision::preferred`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// demucs v4, in any of its variants. Convolutions, through cuDNN on CUDA.
    HtDemucs,
    /// BS-RoFormer and BS-PolarFormer. Matmul, attention and norms.
    Roformer,
}

/// Precision the encoder, transformer and decoder run at.
///
/// The rest of each model stays `f32` regardless: see
/// [`Config::precision`](crate::htdemucs::Config::precision) for which parts and
/// why.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Precision {
    /// Half. Worth about 1.3x on htdemucs and 1.2x on BS-RoFormer, at -54 dB
    /// against [`Self::F32`] on the stems that ship.
    F16,
    /// Full. The default here because it is the neutral answer for a `Config`
    /// built by hand; the server asks [`Self::preferred`] unless told otherwise.
    #[default]
    F32,
}

impl Precision {
    /// What a given model is fastest at on a given backend.
    ///
    /// Measured, one chunk each, and the table does not factorise:
    ///
    /// ```text
    ///                    Metal              CUDA
    ///   htdemucs         f16, ~1.3x         f16, 22.8x
    ///   BS-PolarFormer   f16, 1.57x         f32, 3.76x
    /// ```
    ///
    /// Three of the four want half. PolarFormer on CUDA does not: full precision is
    /// 3.76x faster there, 0.99 s against 3.72 s on a 3090 Ti. It is not the matmuls,
    /// which reach 66.7 TFLOP/s at f16 against 27.7 at f32. **This is a workaround
    /// for a defect in something else**, not a property of the model; if MLX's CUDA
    /// f16 path is fixed, this arm should go back to `F16`.
    ///
    /// Full on the CPU everywhere, decided on accuracy rather than speed: MLX's CPU
    /// `layer_norm` loses ground as the normalised axis widens, -48 dB at 8.4M
    /// elements against -136 for either GPU, and htdemucs normalises over 8.2M in its
    /// time branch.
    ///
    /// This changes what a server separates, so it changes the model id a client
    /// caches on.
    pub const fn preferred(family: Family, on: Accelerator) -> Self {
        match (family, on) {
            (_, Accelerator::Cpu) => Self::F32,
            (Family::HtDemucs, _) => Self::F16,
            (Family::Roformer, Accelerator::Metal) => Self::F16,
            (Family::Roformer, Accelerator::Cuda) => Self::F32,
        }
    }

    /// The mlx dtype this means.
    ///
    /// `pub(crate)` on purpose: it is the one place the two vocabularies meet,
    /// and a caller outside this crate reaching for it would be re-introducing
    /// exactly the coupling this type exists to remove.
    pub(crate) const fn dtype(self) -> mlx_rs::Dtype {
        match self {
            Self::F16 => mlx_rs::Dtype::Float16,
            Self::F32 => mlx_rs::Dtype::Float32,
        }
    }
}

impl fmt::Display for Precision {
    /// Short and stable. This lands in the server's cache key, so changing
    /// either string silently invalidates every entry separated before it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_precision_names_itself_distinctly() {
        assert_eq!(Precision::F16.to_string(), "f16");
        assert_eq!(Precision::F32.to_string(), "f32");
    }

    #[test]
    fn the_dtypes_are_the_ones_the_models_were_measured_at() {
        assert_eq!(Precision::F16.dtype(), mlx_rs::Dtype::Float16);
        assert_eq!(Precision::F32.dtype(), mlx_rs::Dtype::Float32);
    }
}
