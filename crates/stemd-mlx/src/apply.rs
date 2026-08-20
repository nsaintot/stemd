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
    let sources = model.sources();
    let segment = model.chunk();
    let stride = ((1.0 - overlap) * segment as f32) as i32;
    if stride < 1 {
        bail!("an overlap of {overlap} leaves no stride");
    }

    let weight = transition_weight(segment)?;
    let mut out = ops::zeros::<f32>(&[batch, sources, channels, length])?;
    let mut sum_weight = ops::zeros::<f32>(&[length])?;

    let offsets: Vec<i32> = (0..length).step_by(stride as usize).collect();
    for (done, &offset) in offsets.iter().enumerate() {
        progress(done, offsets.len())?;

        // A short final chunk is not zero-padded on the right. The window
        // is centred on it and real audio pulled in from *before* the
        // offset, with zeros only where that runs off the track. Padding at
        // the end instead shifts the last segment by half the shortfall,
        // which is inaudible on its own and nulls at -3 dB.
        let take = segment.min(length - offset);
        let delta = segment - take;
        let start = offset - delta / 2;
        let end = start + segment;
        let (from, to) = (start.max(0), end.min(length));
        let chunk = mix.index((.., .., from..to));
        let (pad_left, pad_right) = (from - start, end - to);
        let chunk = if pad_left > 0 || pad_right > 0 {
            ops::pad(
                &chunk,
                &[(0, 0), (0, 0), (pad_left, pad_right)][..],
                Array::from_f32(0.0),
                None,
            )?
        } else {
            chunk
        };

        let separated = model.separate_chunk(&chunk)?;

        // The window was centred on the chunk, so the output is centred
        // too: sample `delta / 2` is the one that belongs at `offset`.
        // Taking from zero instead shifts the tail segment half the
        // shortfall early: inaudible on its own, and worth -4 dB.
        let trim = delta / 2;
        let piece = separated.index((.., .., .., trim..trim + take));
        let w = weight.index(0..take).reshape(&[1, 1, 1, take])?;

        //  Read the slice, add into it, write it back. Concatenating zero tensors either
        //  side would allocate two whole-track tensors per segment: on a six-minute track,
        //  sixty-eight rounds of half a gigabyte to place seven seconds of audio. The
        //  scatter-add spelling corrupts strided slices on the MLX this was built against.
        let end = offset + take;
        let slice = out.index((.., .., .., offset..end));
        out.try_index_mut(
            (.., .., .., offset..end),
            ops::add(&slice, &ops::multiply(&piece, &w)?)?,
        )?;
        let covered = sum_weight.index(offset..end);
        sum_weight.try_index_mut(offset..end, ops::add(&covered, weight.index(0..take))?)?;

        // MLX is lazy. Without this the segments pile into one unevaluated
        // expression and the memory it needs is every intermediate of the
        // whole track at once rather than one segment's. Three segments
        // would never show it; sixty-eight would.
        mlx_rs::transforms::eval([&out, &sum_weight])?;
    }
    progress(offsets.len(), offsets.len())?;

    let smallest = ops::min(&sum_weight, None)?.item::<f32>();
    if smallest <= 0.0 {
        bail!("some samples were covered by no segment; overlap {overlap} is too small");
    }
    Ok(ops::divide(&out, &sum_weight)?)
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
}
