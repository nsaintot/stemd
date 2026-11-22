//! What a client and the window see of a session.

use std::time::Instant;

use serde::Serialize;

/// A session as its client sees it.
#[derive(Debug, Serialize)]
pub struct View {
    pub id: String,
    /// `open`, or `closed` with `error` saying why.
    pub state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Changes whenever another field does.
    pub rev: u64,
    pub track: TrackView,
    pub output: crate::output::OutputView,
    pub grid: GridView,
    pub arrived: Vec<[usize; 2]>,
    /// Samples separated in memory.
    pub ready: Vec<[usize; 2]>,
    pub stems: Vec<StemView>,
    /// True when every sample is ready or stored, so any range reads.
    pub complete: bool,
    /// Segments asked for and not run yet.
    pub planned: usize,
    /// Set while segments wait on something other than the worker.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waiting_for: Option<&'static str>,
    pub idle_timeout_secs: u64,
}

#[derive(Debug, Serialize)]
pub struct TrackView {
    /// The track id, `null` until the whole upload has been hashed.
    pub id: Option<String>,
    pub frames: usize,
    /// The model's rate.
    pub sample_rate: u32,
    pub channels: usize,
}

#[derive(Debug, Serialize)]
pub struct GridView {
    pub segment_frames: i32,
    pub stride_frames: i32,
    pub segments: usize,
    /// Median seconds of the last [`crate::queue::SPEED_KEPT`] segments, `null` until one ran.
    pub segment_run_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StemView {
    pub name: &'static str,
    /// See [`crate::output::sources`].
    pub sources: u16,
    pub stored: crate::output::Storage,
    /// The whole file's size, when stored whole.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// Stored pieces as half-open spans of piece indices, when stored as pieces.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pieces: Vec<[usize; 2]>,
}

/// Sorted piece indices as half-open spans.
pub(super) fn spans_of(pieces: &[usize]) -> Vec<[usize; 2]> {
    let mut out: Vec<[usize; 2]> = Vec::new();
    for &k in pieces {
        match out.last_mut() {
            Some(last) if last[1] == k => last[1] = k + 1,
            _ => out.push([k, k + 1]),
        }
    }
    out
}

/// What the window shows of a session. It carries no track id or name.
#[derive(Debug, Clone, PartialEq)]
pub struct Activity {
    /// Counts sessions from 1 since the server started.
    pub number: u64,
    pub frames: usize,
    pub sample_rate: u32,
    /// Pieces in the track.
    pub pieces: usize,
    /// Pieces ready in memory or in the cache.
    pub ready: usize,
    /// Samples uploaded.
    pub arrived: usize,
    /// Samples ready in memory or in a stored piece.
    pub covered: usize,
    /// Planned segments not run yet that run before jobs: priority below [`BEFORE_JOBS`].
    pub high: usize,
    /// Planned segments not run yet that wait for jobs.
    pub normal: usize,
    /// True when a planned segment can run now.
    pub runnable: bool,
}

/// A session that ended, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ended {
    pub number: u64,
    pub reason: String,
    pub at: Instant,
}
