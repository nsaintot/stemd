//! Running a model over a whole track.
//!
//! htdemucs sees 7.8 seconds at a time, so a track is cut into overlapping
//! segments, each separated on its own, and the results crossfaded back together.
//! The crossfade is a triangular weight over each segment and a running sum of
//! those weights to divide by at the end.
//!
//! This used to carry a bag as well: several models over the same audio, combined
//! per source by a weight matrix, which is how `htdemucs_ft` is stored. Nothing
//! runs one any more: every arrangement that wants more than one model wants
//! specific sources from specific models. See `stemd_core::hybrid`.

use anyhow::{Result, bail};
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp};
use mlx_rs::{Array, ops};

use crate::htdemucs::HtDemucs;

/// Fraction of a segment shared with its neighbour.
pub const DEFAULT_OVERLAP: f32 = 0.25;

/// A model that separates one fixed-length chunk of audio.
///
/// The segmenting and crossfade below are the same for any of them, and they
/// are the part that took three attempts to get right, so a second model gets
/// this trait rather than a second copy.
pub trait Chunked {
    /// Samples the model runs on at once.
    fn chunk(&self) -> i32;
    /// Stems it produces.
    fn sources(&self) -> i32;
    /// `[B, C, chunk] -> [B, sources, C, chunk]`.
    fn separate_chunk(&self, chunk: &Array) -> Result<Array>;
}

impl Chunked for HtDemucs {
    fn chunk(&self) -> i32 {
        self.config().training_length()
    }
    fn sources(&self) -> i32 {
        self.config().sources
    }
    fn separate_chunk(&self, chunk: &Array) -> Result<Array> {
        self.forward(chunk)
    }
}

/// Where each segment of a track sits.
///
/// Segments start every `stride` samples from sample 0. A short last segment is
/// centred on what is left of the track, so its window reaches back before its
/// offset. Each segment depends only on its own input window, so segments can run
/// in any order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    length: i32,
    segment: i32,
    stride: i32,
}

/// One segment's place on the track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    /// First output sample the segment contributes to.
    pub offset: i32,
    /// Output samples it contributes, from `offset`.
    pub take: i32,
    /// First input sample the model sees. Negative where the window starts
    /// before the track.
    pub start: i32,
}

impl Placement {
    /// One past the last output sample the segment contributes to.
    pub const fn end(&self) -> i32 {
        self.offset + self.take
    }
}

/// The input samples one segment reads, and the zeros around them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// First track sample read.
    pub from: i32,
    /// One past the last track sample read.
    pub to: i32,
    pub pad_left: i32,
    pub pad_right: i32,
}

impl Grid {
    pub fn new(length: i32, segment: i32, overlap: f32) -> Result<Self> {
        let stride = ((1.0 - overlap) * segment as f32) as i32;
        if stride < 1 {
            bail!("an overlap of {overlap} leaves no stride");
        }
        if segment < 1 || length < 0 {
            bail!("a grid needs a positive segment and a length, got {segment} and {length}");
        }
        Ok(Self {
            length,
            segment,
            stride,
        })
    }

    pub const fn length(&self) -> i32 {
        self.length
    }

    pub const fn segment(&self) -> i32 {
        self.segment
    }

    pub const fn stride(&self) -> i32 {
        self.stride
    }

    /// Number of segments.
    pub const fn len(&self) -> usize {
        ((self.length + self.stride - 1) / self.stride) as usize
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// True when no sample is covered by more than two segments, as with any
    /// overlap up to one half.
    pub const fn order_free(&self) -> bool {
        2 * self.stride >= self.segment
    }

    /// Where segment `k` sits.
    pub fn placement(&self, k: usize) -> Placement {
        let offset = k as i32 * self.stride;
        // A short final chunk is not zero-padded on the right. The window
        // is centred on it and real audio pulled in from *before* the
        // offset, with zeros only where that runs off the track. Padding at
        // the end instead shifts the last segment by half the shortfall,
        // which is inaudible on its own and nulls at -3 dB.
        let take = self.segment.min(self.length - offset);
        let delta = self.segment - take;
        Placement {
            offset,
            take,
            start: offset - delta / 2,
        }
    }

    /// The input segment `k` reads.
    pub fn window(&self, k: usize) -> Window {
        let start = self.placement(k).start;
        let end = start + self.segment;
        let (from, to) = (start.max(0), end.min(self.length));
        Window {
            from,
            to,
            pad_left: from - start,
            pad_right: end - to,
        }
    }

    /// Segment `k`'s model input, cut from the whole track `[B, C, T]`.
    pub fn chunk(&self, mix: &Array, k: usize) -> Result<Array> {
        let window = self.window(k);
        pad_window(&mix.index((.., .., window.from..window.to)), &window)
    }

    /// The segments that contribute to output samples `from..to`.
    pub fn covering(&self, from: i32, to: i32) -> std::ops::Range<usize> {
        let (from, to) = (from.max(0), to.min(self.length));
        if from >= to {
            return 0..0;
        }
        // Segment k reaches up to `k * stride + segment`, clipped to the track.
        let first = if from < self.segment {
            0
        } else {
            ((from - self.segment) / self.stride + 1) as usize
        };
        let last = ((to + self.stride - 1) / self.stride) as usize;
        first..last.min(self.len())
    }
}

/// Pad one segment's input, read from the track at `window`, to a full segment.
pub fn pad_window(input: &Array, window: &Window) -> Result<Array> {
    if window.pad_left > 0 || window.pad_right > 0 {
        Ok(ops::pad(
            input,
            &[(0, 0), (0, 0), (window.pad_left, window.pad_right)][..],
            Array::from_f32(0.0),
            None,
        )?)
    } else {
        Ok(input.clone())
    }
}

/// The crossfaded output of a track, filled in one segment at a time.
pub struct Accumulator {
    grid: Grid,
    weight: Array,
    out: Array,
    sum_weight: Array,
    done: Vec<bool>,
}

impl Accumulator {
    /// An empty output of `[batch, sources, channels, length]`.
    pub fn new(grid: Grid, batch: i32, sources: i32, channels: i32) -> Result<Self> {
        Ok(Self {
            grid,
            weight: transition_weight(grid.segment)?,
            out: ops::zeros::<f32>(&[batch, sources, channels, grid.length])?,
            sum_weight: ops::zeros::<f32>(&[grid.length])?,
            done: vec![false; grid.len()],
        })
    }

    pub const fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn is_done(&self, k: usize) -> bool {
        self.done.get(k).copied().unwrap_or(false)
    }

    /// Add segment `k`'s model output, `[B, sources, C, segment]`.
    pub fn add(&mut self, k: usize, separated: &Array) -> Result<()> {
        if k >= self.done.len() {
            bail!("segment {k} is past the last of {}", self.done.len());
        }
        if self.done[k] {
            bail!("segment {k} was already added");
        }
        let placement = self.grid.placement(k);
        let (offset, take) = (placement.offset, placement.take);

        // The window was centred on the chunk, so the output is centred
        // too: sample `delta / 2` is the one that belongs at `offset`.
        // Taking from zero instead shifts the tail segment half the
        // shortfall early: inaudible on its own, and worth -4 dB.
        let trim = offset - placement.start;
        let piece = separated.index((.., .., .., trim..trim + take));
        let w = self.weight.index(0..take).reshape(&[1, 1, 1, take])?;

        //  Read the slice, add into it, write it back. Concatenating zero tensors either
        //  side would allocate two whole-track tensors per segment: on a six-minute track,
        //  sixty-eight rounds of half a gigabyte to place seven seconds of audio. The
        //  scatter-add spelling corrupts strided slices on the MLX this was built against.
        let end = offset + take;
        let slice = self.out.index((.., .., .., offset..end));
        self.out.try_index_mut(
            (.., .., .., offset..end),
            ops::add(&slice, &ops::multiply(&piece, &w)?)?,
        )?;
        let covered = self.sum_weight.index(offset..end);
        self.sum_weight
            .try_index_mut(offset..end, ops::add(&covered, self.weight.index(0..take))?)?;

        // MLX is lazy. Without this the segments pile into one unevaluated
        // expression and the memory it needs is every intermediate of the
        // whole track at once rather than one segment's. Three segments
        // would never show it; sixty-eight would.
        mlx_rs::transforms::eval([&self.out, &self.sum_weight])?;
        self.done[k] = true;
        Ok(())
    }

    /// True when every segment contributing to `from..to` has been added.
    pub fn is_final(&self, from: i32, to: i32) -> bool {
        self.grid.covering(from, to).all(|k| self.done[k])
    }

    /// The finished output over `from..to`, `[B, sources, C, to - from]`.
    pub fn range(&self, from: i32, to: i32) -> Result<Array> {
        if !self.is_final(from, to) {
            bail!("samples {from}..{to} still wait on a segment");
        }
        let out = self.out.index((.., .., .., from..to));
        Ok(ops::divide(&out, self.sum_weight.index(from..to))?)
    }

    /// The finished output of the whole track.
    pub fn finish(self) -> Result<Array> {
        if let Some(k) = self.done.iter().position(|d| !d) {
            bail!("segment {k} was never added");
        }
        let smallest = ops::min(&self.sum_weight, None)?.item::<f32>();
        if smallest <= 0.0 {
            bail!("some samples were covered by no segment; the overlap is too small");
        }
        Ok(ops::divide(&self.out, &self.sum_weight)?)
    }
}

/// One model over a whole track, chunk by chunk, crossfaded back together.
///
/// `mix` is `[B, C, T]`, the result `[B, sources, C, T]`.
pub fn over_track(
    model: &dyn Chunked,
    mix: &Array,
    overlap: f32,
    progress: &mut impl FnMut(usize, usize) -> Result<()>,
) -> Result<Array> {
    let shape = mix.shape().to_vec();
    let (batch, channels, length) = (shape[0], shape[1], shape[2]);
    let grid = Grid::new(length, model.chunk(), overlap)?;
    let mut acc = Accumulator::new(grid, batch, model.sources(), channels)?;

    let total = grid.len();
    for k in 0..total {
        progress(k, total)?;
        acc.add(k, &model.separate_chunk(&grid.chunk(mix, k)?)?)?;
    }
    progress(total, total)?;
    acc.finish()
}

/// A triangular crossfade over one segment, peaking in the middle.
///
/// Rising `1..=n/2` then falling, normalised to a maximum of one. Overlapping
/// segments sum to a constant once divided by the accumulated weight, so the
/// joins do not modulate the level.
fn transition_weight(segment: i32) -> Result<Array> {
    let half = segment / 2;
    let rising = (1..=half).map(|i| i as f32);
    let falling = (1..=segment - half).rev().map(|i| i as f32);
    let values: Vec<f32> = rising.chain(falling).collect();
    let weight = Array::from_slice(&values, &[segment]);
    let peak = ops::max(&weight, None)?;
    Ok(ops::divide(&weight, &peak)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::{Mutex, MutexGuard};

    /// Metal trips on a second command encoder, so anything touching MLX takes
    /// this first. The integration tests have their own copy of this; two
    /// unit tests in one binary run in parallel by default and abort the
    /// process rather than failing, which is a confusing way to find out.
    static GPU: Mutex<()> = Mutex::new(());

    fn one_at_a_time() -> MutexGuard<'static, ()> {
        GPU.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A model that separates nothing and counts how often it was asked to.
    struct Counting {
        chunk: i32,
        runs: Cell<usize>,
    }

    impl Chunked for Counting {
        fn chunk(&self) -> i32 {
            self.chunk
        }
        fn sources(&self) -> i32 {
            1
        }
        fn separate_chunk(&self, chunk: &Array) -> Result<Array> {
            self.runs.set(self.runs.get() + 1);
            let s = chunk.shape().to_vec();
            Ok(ops::zeros::<f32>(&[s[0], 1, s[1], s[2]])?)
        }
    }

    /// A caller that refuses to continue stops the track where it is.
    ///
    /// This is what cancellation is built on. A progress callback that returned
    /// nothing would let a caller set a flag only readable after `over_track`
    /// returned, by which point the whole track has been separated.
    #[test]
    fn refusing_to_continue_stops_at_that_chunk() {
        let _gpu = one_at_a_time();
        let model = Counting {
            chunk: 1000,
            runs: Cell::new(0),
        };
        // Twenty chunks' worth of track, so stopping early is unambiguous.
        let mix = ops::zeros::<f32>(&[1, 2, 20_000]).expect("a silent track");

        let mut seen = 0;
        let result = over_track(&model, &mix, DEFAULT_OVERLAP, &mut |done, _| {
            seen = done;
            if done == 3 {
                bail!("stop here");
            }
            Ok(())
        });

        assert!(result.is_err(), "over_track ran to completion anyway");
        assert_eq!(seen, 3, "it kept reporting past the refusal");
        // Three chunks ran before the fourth was refused. The bug this guards
        // against is not "a few too many" but "all of them", so the assertion
        // is against the total rather than the exact count.
        assert_eq!(
            model.runs.get(),
            3,
            "the model ran {} times after being told to stop at chunk 3",
            model.runs.get()
        );
    }

    /// The ordinary path still reaches the end and reports it.
    #[test]
    fn a_track_that_is_never_refused_runs_every_chunk() {
        let _gpu = one_at_a_time();
        let model = Counting {
            chunk: 1000,
            runs: Cell::new(0),
        };
        let mix = ops::zeros::<f32>(&[1, 2, 20_000]).expect("a silent track");

        let mut last = (0, 0);
        let out = over_track(&model, &mix, DEFAULT_OVERLAP, &mut |done, total| {
            last = (done, total);
            Ok(())
        })
        .expect("separating");

        assert_eq!(last.0, last.1, "the final report is not the total");
        assert_eq!(model.runs.get(), last.1, "a chunk went unreported");
        assert_eq!(out.shape(), [1, 1, 2, 20_000]);
    }

    /// A model whose output depends on its whole input window, as a real one does.
    struct Contextual;

    impl Chunked for Contextual {
        fn chunk(&self) -> i32 {
            1000
        }
        fn sources(&self) -> i32 {
            2
        }
        fn separate_chunk(&self, chunk: &Array) -> Result<Array> {
            let mean = ops::mean_axes(chunk, &[2], true)?;
            let a = ops::multiply(chunk, Array::from_f32(1.5))?;
            let b = ops::add(&ops::sin(chunk)?, &mean)?;
            Ok(ops::stack_axis(&[a, b], 1)?)
        }
    }

    /// A track with a short tail, so the centred last segment is exercised.
    fn track(length: i32) -> Array {
        let mut state = 0x2545_f491_u32;
        let values: Vec<f32> = (0..2 * length)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        Array::from_slice(&values, &[1, 2, length])
    }

    fn bits(a: &Array) -> Vec<u32> {
        // A multiply by one copies a strided slice into a contiguous buffer
        // without changing any value's bits.
        let a = &ops::multiply(a, Array::from_f32(1.0)).expect("copying");
        mlx_rs::transforms::eval([a]).expect("evaluating");
        a.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
    }

    fn in_order(mix: &Array, overlap: f32) -> Array {
        over_track(&Contextual, mix, overlap, &mut |_, _| Ok(())).expect("separating")
    }

    fn in_this_order(mix: &Array, overlap: f32, order: &[usize]) -> Accumulator {
        let grid = Grid::new(mix.shape()[2], Contextual.chunk(), overlap).expect("a grid");
        let mut acc = Accumulator::new(grid, 1, 2, 2).expect("an accumulator");
        for &k in order {
            let out = Contextual
                .separate_chunk(&grid.chunk(mix, k).expect("a chunk"))
                .expect("separating");
            acc.add(k, &out).expect("adding");
        }
        acc
    }

    #[test]
    fn segments_run_out_of_order_match_an_in_order_run_bit_for_bit() {
        let _gpu = one_at_a_time();
        let mix = track(20_350);
        let expected = bits(&in_order(&mix, DEFAULT_OVERLAP));
        let n = Grid::new(20_350, 1000, DEFAULT_OVERLAP)
            .expect("a grid")
            .len();

        let reversed: Vec<usize> = (0..n).rev().collect();
        let from_the_middle: Vec<usize> = (n / 2..n).chain((0..n / 2).rev()).collect();
        for order in [reversed, from_the_middle] {
            let out = in_this_order(&mix, DEFAULT_OVERLAP, &order)
                .finish()
                .expect("finishing");
            assert_eq!(bits(&out), expected, "order {order:?} changed the output");
        }
    }

    #[test]
    fn a_range_holds_the_same_samples_as_the_whole_track() {
        let _gpu = one_at_a_time();
        let mix = track(20_350);
        let whole = in_order(&mix, DEFAULT_OVERLAP);
        let grid = Grid::new(20_350, 1000, DEFAULT_OVERLAP).expect("a grid");

        for (from, to) in [(0, 400), (7_321, 9_000), (19_900, 20_350)] {
            let order: Vec<usize> = grid.covering(from, to).rev().collect();
            let acc = in_this_order(&mix, DEFAULT_OVERLAP, &order);
            assert!(acc.is_final(from, to), "{from}..{to} is not final");
            let range = acc.range(from, to).expect("a range");
            let slice = whole.index((.., .., .., from..to));
            assert_eq!(bits(&range), bits(&slice), "{from}..{to} differs");
        }
    }

    #[test]
    fn a_range_waiting_on_a_segment_is_refused() {
        let _gpu = one_at_a_time();
        let mix = track(20_350);
        let grid = Grid::new(20_350, 1000, DEFAULT_OVERLAP).expect("a grid");
        let covering: Vec<usize> = grid.covering(5_000, 6_000).collect();
        let acc = in_this_order(&mix, DEFAULT_OVERLAP, &covering[1..]);
        assert!(!acc.is_final(5_000, 6_000));
        assert!(acc.range(5_000, 6_000).is_err());
    }

    #[test]
    fn covering_is_exactly_the_segments_that_touch_the_range() {
        for (length, overlap) in [(20_350, 0.25), (20_000, 0.5), (999, 0.25), (5_001, 0.1)] {
            let grid = Grid::new(length, 1000, overlap).expect("a grid");
            for from in (0..length).step_by(37) {
                for to in [from + 1, from + 250, from + 1000, from + 3001] {
                    let touching: Vec<usize> = (0..grid.len())
                        .filter(|&k| {
                            let p = grid.placement(k);
                            p.offset < to.min(length) && p.end() > from
                        })
                        .collect();
                    let covering: Vec<usize> = grid.covering(from, to).collect();
                    assert_eq!(covering, touching, "{length} at {overlap}: {from}..{to}");
                }
            }
        }
    }

    #[test]
    fn any_order_matches_up_to_half_overlap() {
        assert!(Grid::new(10_000, 1000, 0.25).expect("a grid").order_free());
        assert!(Grid::new(10_000, 1000, 0.5).expect("a grid").order_free());
        assert!(!Grid::new(10_000, 1000, 0.6).expect("a grid").order_free());
    }
}
