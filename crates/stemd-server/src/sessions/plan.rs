//! Which segments a session's window asks for, and in what order they run.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::Deserialize;
use stemd_core::Grid;

/// Priorities below this run before whole-track jobs, the rest when no job waits.
pub const BEFORE_JOBS: u8 = 50;

/// The highest priority a region may ask for.
pub const MAX_PRIORITY: u8 = 99;

/// The most regions one window may hold.
pub const MAX_REGIONS: usize = 32;

/// Which planned segments the worker takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Priorities below [`BEFORE_JOBS`]: ahead of whole-track jobs.
    BeforeJobs,
    /// The rest: when no job is waiting.
    WhenIdle,
}

/// Samples around `at` a client wants separated, and how urgently: `0` first.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Region {
    /// Track sample the region is centred on, such as the needle or a cue.
    pub at: usize,
    #[serde(default)]
    pub ahead: usize,
    #[serde(default)]
    pub behind: usize,
    pub priority: u8,
}

/// What a client wants separated next. Replaces the session's last window.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WindowRequest {
    pub regions: Vec<Region>,
}

impl WindowRequest {
    /// Fail on too many regions or a priority past [`MAX_PRIORITY`].
    pub fn check(&self) -> Result<()> {
        if self.regions.len() > MAX_REGIONS {
            bail!(
                "{} regions; a window holds up to {MAX_REGIONS}",
                self.regions.len()
            );
        }
        if let Some(region) = self.regions.iter().find(|r| r.priority > MAX_PRIORITY) {
            bail!("priority {} is past {MAX_PRIORITY}", region.priority);
        }
        Ok(())
    }
}

/// Where a planned segment stands: the lower runs first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rank {
    pub priority: u8,
    /// Samples from `at`: `0` under it, `2d + 1` for `d` samples behind, so ahead wins a tie.
    pub distance: usize,
}

impl Rank {
    pub const fn class(self) -> Class {
        if self.priority < BEFORE_JOBS {
            Class::BeforeJobs
        } else {
            Class::WhenIdle
        }
    }
}

/// The segments a window asks for, each at its best rank, in the order to run them.
pub fn plan(grid: &Grid, request: &WindowRequest) -> Vec<(usize, Rank)> {
    let frames = grid.length().unsigned_abs() as usize;
    if frames == 0 {
        return Vec::new();
    }
    let as_i32 = |v: usize| i32::try_from(v.min(frames)).unwrap_or(i32::MAX);
    let mut best: BTreeMap<usize, Rank> = BTreeMap::new();
    for region in &request.regions {
        let at = region.at.min(frames - 1);
        let from = at.saturating_sub(region.behind);
        let to = at.saturating_add(region.ahead).max(at + 1);
        for k in grid.covering(as_i32(from), as_i32(to)) {
            let placement = grid.placement(k);
            let start = placement.offset.unsigned_abs() as usize;
            let end = placement.end().unsigned_abs() as usize;
            let distance = if start > at {
                start - at
            } else if end <= at {
                2 * (at - end) + 1
            } else {
                0
            };
            let rank = Rank {
                priority: region.priority,
                distance,
            };
            best.entry(k)
                .and_modify(|r| *r = (*r).min(rank))
                .or_insert(rank);
        }
    }
    let mut order: Vec<(usize, Rank)> = best.into_iter().collect();
    order.sort_by_key(|&(k, rank)| (rank, k));
    order
}
