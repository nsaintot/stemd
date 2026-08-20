//! Progress reporting.
//!
//! Separation is chunked, so progress is a real count of completed work rather
//! than a spinner. The client polls `GET /v1/jobs/{id}` and sees the chunk
//! counter advance.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Accepted, not yet picked up by a worker.
    Queued,
    /// Decoding the uploaded PCM and computing the mix spectrogram.
    Analysing,
    /// Running the model, chunk by chunk. This is where the time goes.
    Separating,
    /// Inverse STFT and residual.
    Reconstructing,
    /// Writing stems to their temporary paths.
    Writing,
    Done,
    Failed,
    /// Stopped between segments because nobody is waiting for it any more.
    /// Terminal and distinct from `Failed`: nothing went wrong.
    Cancelled,
}

impl Stage {
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    pub stage: Stage,
    /// Units of work finished within the current stage.
    pub completed: u32,
    /// Total units for the current stage; 0 when the stage is not countable.
    pub total: u32,
    /// Overall completion in 0.0..=1.0. Stage weights are rough by design:
    /// separation dominates, so the others are given a token share.
    pub fraction: f32,
    pub detail: Option<String>,
}

impl Progress {
    pub fn new(stage: Stage) -> Self {
        Self {
            stage,
            completed: 0,
            total: 0,
            fraction: stage_floor(stage),
            detail: None,
        }
    }

    pub fn counted(stage: Stage, completed: u32, total: u32) -> Self {
        let floor = stage_floor(stage);
        let ceil = stage_ceiling(stage);
        let within = if total == 0 {
            0.0
        } else {
            f64::from(completed) / f64::from(total)
        };
        Self {
            stage,
            completed,
            total,
            fraction: (floor as f64 + within * f64::from(ceil - floor)) as f32,
            detail: None,
        }
    }

    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}

const fn stage_floor(stage: Stage) -> f32 {
    match stage {
        Stage::Queued => 0.0,
        Stage::Analysing => 0.02,
        Stage::Separating => 0.10,
        Stage::Reconstructing => 0.90,
        Stage::Writing => 0.96,
        Stage::Done | Stage::Failed | Stage::Cancelled => 1.0,
    }
}

const fn stage_ceiling(stage: Stage) -> f32 {
    match stage {
        Stage::Queued => 0.02,
        Stage::Analysing => 0.10,
        Stage::Separating => 0.90,
        Stage::Reconstructing => 0.96,
        Stage::Writing => 1.0,
        Stage::Done | Stage::Failed | Stage::Cancelled => 1.0,
    }
}

/// Sink for progress updates, and the channel a caller uses to call a
/// separation off. Implemented for closures.
pub trait ProgressSink: Send + Sync {
    fn update(&self, progress: Progress);

    /// True once nobody is waiting for this separation any more.
    ///
    /// Polled by the backend between segments, so a caller that stops caring
    /// frees the worker within one segment rather than one track. Defaults to
    /// never, which is what a closure sink and [`Silent`] want.
    fn cancelled(&self) -> bool {
        false
    }
}

impl<F: Fn(Progress) + Send + Sync> ProgressSink for F {
    fn update(&self, progress: Progress) {
        self(progress);
    }
}

/// A separation that stopped because its sink asked it to.
///
/// A distinct type rather than a message, so a caller can tell a routine
/// cancellation from a model failure without matching on strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl Cancelled {
    /// True when `err` is a cancellation rather than a failure. Walks the chain,
    /// so adding context later cannot turn a cancelled job into a reported
    /// failure.
    pub fn caused(err: &anyhow::Error) -> bool {
        err.chain().any(|e| e.is::<Self>())
    }
}

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// A sink that discards updates, for tests and one-shot CLI use.
pub struct Silent;

impl ProgressSink for Silent {
    fn update(&self, _progress: Progress) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cancellation_is_recognised_through_a_context_layer() {
        // The worker tells cancelled from failed by this alone, so a `.context`
        // added anywhere on the way up must not turn a skip into a reported
        // failure.
        let wrapped = anyhow::Error::from(Cancelled).context("separating");
        assert!(Cancelled::caused(&wrapped));
        assert!(!Cancelled::caused(&anyhow::anyhow!("model exploded")));
    }

    #[test]
    fn cancelled_is_terminal_but_is_not_a_failure() {
        assert!(Stage::Cancelled.is_terminal());
        assert_ne!(Stage::Cancelled, Stage::Failed);
    }

    #[test]
    fn fraction_advances_monotonically_across_stages() {
        let a = Progress::counted(Stage::Separating, 0, 50).fraction;
        let b = Progress::counted(Stage::Separating, 25, 50).fraction;
        let c = Progress::counted(Stage::Separating, 50, 50).fraction;
        assert!(a < b && b < c, "{a} {b} {c}");
        assert!(c <= Progress::new(Stage::Reconstructing).fraction);
    }
}
