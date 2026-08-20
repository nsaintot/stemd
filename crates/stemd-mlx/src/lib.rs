//! demucs v4 on Apple's MLX, in Rust.
//!
//! The model is built here rather than loaded as a traced graph, because MLX has
//! no equivalent of TorchScript. Every layer is therefore a place this can differ
//! from the original, which is why each stage is nulled against the reference.
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
