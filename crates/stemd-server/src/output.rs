//! What a request asks for on the way out.

use serde::Serialize;
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

    /// The transfer gain for every stem: 3 dB of headroom over the mix's peak, or over full scale when it is unknown.
    pub fn gain(&self, peak: Option<f32>) -> f32 {
        if self.format.has_headroom() {
            return 1.0;
        }
        let ceiling = peak.unwrap_or(1.0) * std::f32::consts::SQRT_2;
        if ceiling <= 1.0 { 1.0 } else { 1.0 / ceiling }
    }

    /// The parts this serves: those asked for, then the derived one if it ships.
    pub fn served(&self) -> Vec<&'static str> {
        let derived = self.ships_derived().then(|| self.parts.derived()).flatten();
        self.parts.names().chain(derived).collect()
    }

    /// Every part a piece directory of this output holds: the parts, and the derived one.
    pub fn stored_parts(&self) -> Vec<&'static str> {
        self.parts.names().chain(self.parts.derived()).collect()
    }
}

/// The output stems are in, as sessions, jobs and lookups report it.
#[derive(Debug, Clone, Serialize)]
pub struct OutputView {
    pub format: String,
    pub sample_rate: u32,
    /// Multiply by `1.0 / gain`; absent for an open track and for a job, whose stems carry it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gain: Option<f32>,
    pub parts: Vec<String>,
    /// The part the client rebuilds as `mix` minus the parts, when one is left out.
    pub derived: Option<&'static str>,
}

impl OutputView {
    pub fn of(output: &Output, gain: Option<f32>) -> Self {
        Self {
            format: output.format.to_string(),
            sample_rate: output.rate.hz(),
            gain,
            parts: output.parts.names().map(str::to_owned).collect(),
            derived: output.parts.derived(),
        }
    }
}

/// How a stem is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Storage {
    Whole,
    Pieces,
    None,
}

/// Bits of [`sources`]: `drums`, `bass`, `other`, `vocals`.
const SOURCE_BITS: [(&str, u16); 4] = [("drums", 1), ("bass", 2), ("other", 4), ("vocals", 8)];

/// The sources a part holds, one bit each: `1` drums, `2` bass, `4` other,
/// `8` vocals. `harmonics` is `bass` and `other`, so `6`.
pub fn sources(part: &str) -> u16 {
    match part {
        "harmonics" => 2 | 4,
        name => SOURCE_BITS
            .iter()
            .find(|(n, _)| *n == name)
            .map_or(0, |(_, bit)| *bit),
    }
}

/// Output-rate sample `j` of the track sits at `j * model / out`, so model-rate
/// sample `i` maps to the first output sample at or after it.
pub fn output_index(i: usize, model: u64, out: u64) -> usize {
    (i as u64 * out).div_ceil(model) as usize
}

/// Model-rate samples `span` as the matching output-rate samples.
pub fn output_span(span: &std::ops::Range<usize>, model: u64, out: u64) -> std::ops::Range<usize> {
    output_index(span.start, model, out)..output_index(span.end, model, out)
}

/// Piece `k` of a `frames`-sample track cut every `stride` samples.
pub fn piece_span(k: usize, stride: usize, frames: usize) -> std::ops::Range<usize> {
    (k * stride).min(frames)..((k + 1) * stride).min(frames)
}
