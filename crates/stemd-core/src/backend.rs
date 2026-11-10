//! Backend interface.
//!
//! [`MlxSeparator`](crate::mlx::MlxSeparator) is the only implementation; the
//! trait exists so a second model can be added without touching the server.

use anyhow::Result;
use serde::Serialize;

use crate::mixture::Normalisation;
use crate::pass::{Pass, Segmenting, whole_track};
use crate::pcm::Audio;
use crate::progress::ProgressSink;
use crate::stems::{Parts, Stems};

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
    /// Parts a job may ask for, from [`crate::stems::PART_NAMES`].
    pub parts: Vec<String>,
}

pub trait Separate: Send {
    fn info(&self) -> BackendInfo;

    /// True when the models expect the whole mix's [`Normalisation`].
    fn normalises(&self) -> bool;

    /// How the models cut a track into segments, or `None` when they cannot run
    /// segment by segment.
    fn segmenting(&self) -> Option<Segmenting>;

    /// Start separating a track of `frames` samples into `parts`.
    ///
    /// `norm` is the whole mix's normalisation, ignored by models trained on raw
    /// audio.
    fn begin(&self, frames: usize, parts: Parts, norm: Normalisation) -> Result<Pass>;

    /// Run segment `k` on `window`, the mix over the samples [`Pass::window`] names.
    fn run(&mut self, pass: &mut Pass, window: &Audio, k: usize) -> Result<()>;

    /// The stems over samples `from..to`, given the mix over the same samples.
    /// [`Pass::is_final`] must hold for them.
    fn stems(&self, pass: &Pass, mix: &Audio, from: usize, to: usize) -> Result<Stems>;

    /// Separate `mix` into at least [`crate::SHIPPED`] and whatever `parts` adds.
    fn separate(&mut self, mix: &Audio, parts: Parts, sink: &dyn ProgressSink) -> Result<Stems> {
        whole_track(self, mix, parts, sink, |_| Ok(()))
    }
}
