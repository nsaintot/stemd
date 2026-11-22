//! The precision a model runs at, named without reference to a backend.
//!
//! It keeps `mlx_rs::Dtype` out of the models' public config. There are two values, the two the
//! models are measured at.

use std::fmt;

use crate::device::Accelerator;

/// The model architecture, for choosing a precision.
///
/// The two architectures want different precisions on the same card, so the hardware alone cannot
/// decide. See [`Precision::preferred`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// demucs v4, in any variant: convolutions, through cuDNN on CUDA.
    HtDemucs,
    /// BS-RoFormer and BS-PolarFormer: matmul, attention and norms.
    Roformer,
}

/// The precision the encoder, transformer and decoder run at.
///
/// The rest of each model always stays `f32`: see
/// [`Config::precision`](crate::htdemucs::Config::precision) for which parts and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Precision {
    /// Half precision: about 1.3x faster on htdemucs and 1.2x on BS-RoFormer, at -54 dB against
    /// [`Self::F32`] on the shipped stems.
    F16,
    /// Full precision. It is the default because it is the neutral choice for a `Config` built by
    /// hand; the server asks [`Self::preferred`] unless told otherwise.
    #[default]
    F32,
}

impl Precision {
    /// The fastest precision for a model on a backend. This changes what a server separates, so it
    /// changes the model id a client caches on.
    // Speed of f16 against f32, on one chunk each; the table does not factorise:
    //
    //                        Metal              CUDA
    //       htdemucs         f16, ~1.3x         f16, 22.8x
    //       BS-PolarFormer   f16, 1.57x         f32, 3.76x
    //
    // Three of the four want half. PolarFormer on CUDA wants full: it is 3.76x faster there, 0.99 s
    // against 3.72 s on a 3090 Ti. The matmuls are not the cause; they reach 66.7 TFLOP/s at f16
    // against 27.7 at f32. This arm works around a defect in MLX's CUDA f16 path; once that is
    // fixed, it should go back to `F16`.
    //
    // The CPU always gets full precision, for accuracy: MLX's CPU `layer_norm` loses accuracy as
    // the normalised axis widens, -48 dB at 8.4M elements against -136 dB on either GPU, and
    // htdemucs normalises over 8.2M in its time branch.
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
    /// Crate-private so mlx types stay out of the public API.
    pub(crate) const fn dtype(self) -> mlx_rs::Dtype {
        match self {
            Self::F16 => mlx_rs::Dtype::Float16,
            Self::F32 => mlx_rs::Dtype::Float32,
        }
    }
}

impl fmt::Display for Precision {
    /// Short and stable: this lands in the server's cache key, so changing either string silently
    /// invalidates every entry separated before.
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
