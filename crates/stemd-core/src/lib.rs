//! The separation pipeline: it takes PCM audio and returns stems.
//!
//! The models run on MLX: [`mlx`] holds demucs v4, [`roformer`] the BS-RoFormer, and [`hybrid`]
//! chains two models. Each implements [`Separate`], which the server calls.
//!
//! [`mixture`] holds what the model does not own: the track-level normalisation,
//! and folding four sources into the two stems that ship.

pub mod backend;
pub mod hybrid;
pub mod mixture;
pub mod mlx;
pub mod pass;
pub mod pcm;
pub mod progress;
pub mod resample;
pub mod roformer;
pub mod stemfmt;
pub mod stems;
#[cfg(feature = "toy")]
pub mod toy;

pub use backend::{BackendInfo, Separate};
pub use hybrid::{HybridConfig, HybridSeparator};
pub use mixture::Normalisation;
pub use mlx::{MlxConfig, MlxSeparator};
pub use pass::{Pass, Segmenting, whole_track};
pub use pcm::{Audio, PcmFormat};
pub use progress::{Cancelled, Progress, ProgressSink, Silent, Stage};
pub use resample::{DspMode, OutputRate, polyphase::Polyphase};
pub use roformer::RoformerSeparator;
/// Re-exported so the server can plan segments without linking `stemd-mlx`.
pub use stemd_mlx::apply::{Grid, Placement};
/// Re-exported so a caller can name a precision without depending on `stemd-mlx`. stemd-server puts
/// the precision in its cache key and does not link the model crate.
pub use stemd_mlx::{Accelerator, Family, Precision, memory};
pub use stemfmt::{StemFormat, encode as encode_stem};
pub use stems::{DERIVED, PART_NAMES, PARTS, PartGains, PartSet, Parts, SHIPPED, Stems};
