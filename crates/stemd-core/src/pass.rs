//! Separating a track one segment at a time, in any order.
//!
//! A [`Pass`] holds the crossfaded output of every model a separator runs. Each
//! segment reads only its own input window, so segments can run in any order and
//! before the rest of the track has arrived. A range is final once every segment
//! covering it has run, and its samples are then the same as a whole-track run's.

use anyhow::{Result, bail};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use stemd_mlx::apply::{Accumulator, Grid, pad_window};

use crate::backend::Separate;
use crate::hybrid::to_planes;
use crate::mixture::Normalisation;
use crate::pcm::Audio;
use crate::progress::{Cancelled, Progress, ProgressSink, Stage};
use crate::stems::{Parts, Stems};

/// How a separator cuts a track: its segment length and the share of it
/// overlapping a neighbour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segmenting {
    pub segment: i32,
    pub overlap: f32,
}

impl Segmenting {
    /// The grid over a track of `frames` samples.
    pub fn grid(self, frames: usize) -> Result<Grid> {
        Grid::new(i32::try_from(frames)?, self.segment, self.overlap)
    }
}

/// One track's separation in progress.
pub struct Pass {
    pub(crate) grid: Grid,
    pub(crate) parts: Parts,
    pub(crate) norm: Normalisation,
    /// One per model the separator runs, in the separator's own order.
    pub(crate) halves: Vec<Accumulator>,
}

impl Pass {
    pub(crate) fn new(grid: Grid, parts: Parts, norm: Normalisation) -> Self {
        Self {
            grid,
            parts,
            norm,
            halves: Vec::new(),
        }
    }

    pub const fn grid(&self) -> &Grid {
        &self.grid
    }

    pub const fn parts(&self) -> Parts {
        self.parts
    }

    /// Number of segments.
    pub const fn segments(&self) -> usize {
        self.grid.len()
    }

    /// True once every model has run segment `k`.
    pub fn is_done(&self, k: usize) -> bool {
        self.halves.iter().all(|half| half.is_done(k))
    }

    /// True when every segment covering samples `from..to` has run.
    pub fn is_final(&self, from: usize, to: usize) -> bool {
        let (Ok(from), Ok(to)) = (i32::try_from(from), i32::try_from(to)) else {
            return false;
        };
        self.halves.iter().all(|half| half.is_final(from, to))
    }

    /// Track samples segment `k` reads.
    pub fn window(&self, k: usize) -> std::ops::Range<usize> {
        let window = self.grid.window(k);
        window.from.unsigned_abs() as usize..window.to.unsigned_abs() as usize
    }

    /// Fail unless samples `from..to` can be read out.
    pub(crate) fn check_final(&self, from: usize, to: usize) -> Result<()> {
        if from >= to || to > self.grid.length().unsigned_abs() as usize {
            bail!(
                "{from}..{to} is not a range of this {}-sample track",
                self.grid.length()
            );
        }
        if !self.is_final(from, to) {
            bail!("samples {from}..{to} still wait on a segment");
        }
        Ok(())
    }
}

/// Segment `k`'s model input, `[1, C, segment]`, normalised when `norm` is given.
///
/// `window` holds exactly the samples [`Grid::window`] names.
pub(crate) fn chunk(
    window: &Audio,
    grid: &Grid,
    k: usize,
    norm: Option<&Normalisation>,
) -> Result<Array> {
    let place = grid.window(k);
    let frames = (place.to - place.from).unsigned_abs() as usize;
    if window.frames() != frames {
        bail!(
            "segment {k} reads {frames} samples, given {}",
            window.frames()
        );
    }
    let mut flat = Vec::with_capacity(window.channels() * frames);
    for channel in &window.data {
        match norm {
            Some(n) => flat.extend(channel.iter().map(|&v| n.apply(v))),
            None => flat.extend_from_slice(channel),
        }
    }
    let shape = [1, i32::try_from(window.channels())?, place.to - place.from];
    pad_window(&Array::from_slice(&flat, &shape), &place)
}

/// Fail unless `mix` holds exactly the `to - from` samples of a range.
pub(crate) fn check_range(mix: &Audio, from: usize, to: usize) -> Result<()> {
    if mix.frames() != to - from {
        bail!(
            "{from}..{to} needs {} samples of mix, given {}",
            to - from,
            mix.frames()
        );
    }
    Ok(())
}

/// One source of a half over `from..to`, as planes, denormalised when `norm` is given.
pub(crate) fn source_planes(
    half: &Accumulator,
    source: i32,
    from: usize,
    to: usize,
    channels: usize,
    norm: Option<&Normalisation>,
) -> Result<Vec<Vec<f32>>> {
    let out = half.range(i32::try_from(from)?, i32::try_from(to)?)?;
    to_planes(&out.index((0, source)), channels, norm, to - from)
}

/// `mix` cut to samples `from..to`.
pub(crate) fn excerpt(mix: &Audio, from: usize, to: usize) -> Audio {
    Audio::new(
        mix.data.iter().map(|c| c[from..to].to_vec()).collect(),
        mix.sample_rate,
    )
}

/// Separate a whole track through `separator`'s segments, in order.
///
/// `between` runs before each segment, with the separator, so a caller can
/// slip other work in.
pub fn whole_track<S: Separate + ?Sized>(
    separator: &mut S,
    mix: &Audio,
    parts: Parts,
    sink: &dyn ProgressSink,
    mut between: impl FnMut(&mut S) -> Result<()>,
) -> Result<Stems> {
    let frames = mix.frames();
    if frames == 0 {
        bail!("nothing to separate: the mix has no frames");
    }
    sink.update(Progress::new(Stage::Analysing));
    let mut pass = separator.begin(frames, parts, Normalisation::of(mix))?;

    let total = pass.segments();
    let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
    for k in 0..total {
        // Checked before each segment, so cancelling costs at most one.
        if sink.cancelled() {
            return Err(Cancelled.into());
        }
        sink.update(Progress::counted(Stage::Separating, count(k), count(total)));
        between(separator)?;
        let window = pass.window(k);
        separator.run(&mut pass, &excerpt(mix, window.start, window.end), k)?;
    }
    sink.update(Progress::counted(
        Stage::Separating,
        count(total),
        count(total),
    ));

    sink.update(Progress::new(Stage::Reconstructing));
    separator.stems(&pass, mix, 0, frames)
}
