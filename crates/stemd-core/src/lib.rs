//! Core separation pipeline: PCM in, stems out.
//!
//! One backend: [`mlx`], demucs v4 built layer by layer on Apple's MLX. The
//! [`Separate`] trait exists so a second model can be slotted in without touching
//! the server.
//!
//! [`mixture`] holds what the model does not own: the track-level normalisation,
//! and folding four sources into the two stems that ship.

pub mod backend;
pub mod hybrid;
pub mod mixture;
pub mod mlx;
pub mod pcm;
pub mod progress;
pub mod resample;
pub mod stemfmt;
pub mod stems;

pub use backend::{BackendInfo, Separate};
pub use hybrid::{HybridConfig, HybridSeparator};
pub use mlx::{MlxConfig, MlxSeparator};
pub use pcm::{Audio, PcmFormat};
pub use progress::{Cancelled, Progress, ProgressSink, Silent, Stage};
pub use resample::{DspMode, OutputRate};
/// Re-exported so a caller can name a precision without depending on
/// `stemd-mlx`. stemd-server does exactly that: it has to put the precision in
/// its cache key and never links the model crate.
pub use stemd_mlx::{Accelerator, Family, Precision, memory};
pub use stemfmt::{StemFormat, encode as encode_stem};
pub use stems::{DERIVED, PARTS, PartGains, SHIPPED, Stems};
