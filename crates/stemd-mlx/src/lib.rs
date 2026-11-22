//! demucs v4 and BS-RoFormer on MLX, in Rust, on Metal or CUDA.
//!
//! The model is built here layer by layer, because MLX has no equivalent of TorchScript for loading
//! a traced graph. Every layer can therefore differ from the original, so each stage is nulled
//! against the reference.
//!
//! MLX cannot be driven from two threads at once; callers must serialise.

pub mod apply;
pub mod device;
pub mod htdemucs;
pub mod layers;
pub mod memory;
pub mod precision;
pub mod roformer;
pub mod spectral;
pub mod transformer;
pub mod weights;

pub use device::Accelerator;
pub use precision::{Family, Precision};
