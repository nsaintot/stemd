//! Backend interface.
//!
//! [`MlxSeparator`](crate::mlx::MlxSeparator) is the only implementation; the
//! trait exists so a second model can be added without touching the server.

use anyhow::Result;
use serde::Serialize;

use crate::pcm::Audio;
use crate::progress::ProgressSink;
use crate::stems::Stems;

/// What a backend advertises about itself.
#[derive(Debug, Clone, Serialize)]
pub struct BackendInfo {
    pub backend: String,
    pub model: String,
    pub sample_rate: u32,
    pub channels: usize,
    /// Stems the backend ships, i.e. what crosses the wire. One fewer than
    /// [`crate::stems::PARTS`]: the missing one is [`crate::stems::DERIVED`],
    /// which the client reconstructs from the mix.
    pub stems: Vec<String>,
    /// Where the work runs: "gpu", "cpu".
    pub device: String,
}

pub trait Separate: Send {
    fn separate(&mut self, mix: &Audio, sink: &dyn ProgressSink) -> Result<Stems>;
    fn info(&self) -> BackendInfo;
}
