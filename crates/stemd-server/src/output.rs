//! What a request asks for on the way out.

use stemd_core::{OutputRate, Parts, StemFormat};

use crate::filters::Dsp;

/// What a job asks for on the way out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    pub format: StemFormat,
    pub rate: OutputRate,
    /// Ship the derived part as well, so the client need not rebuild it from the mix.
    pub derived: bool,
    /// Filter the conversion to `rate` runs through.
    pub dsp: Dsp,
    /// Parts to ship. The derived part is the one they leave out.
    pub parts: Parts,
}

impl Output {
    /// How many parts this asks for, for a log line.
    pub fn parts(&self) -> usize {
        self.parts.len() + usize::from(self.ships_derived())
    }

    /// Whether the derived part ships: asked for, and something is left out.
    pub fn ships_derived(&self) -> bool {
        self.derived && self.parts.derived().is_some()
    }
}
