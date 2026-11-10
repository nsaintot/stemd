//! A separator with no weights, for testing what drives one.
//!
//! Its model is a few array operations per segment, each depending on the whole
//! input window as a real model's output does. It normalises like demucs, so
//! code waiting on the whole mix's statistics is exercised too.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use mlx_rs::{Array, ops};
use stemd_mlx::apply::{Accumulator, Chunked, Grid};

use crate::backend::{BackendInfo, Separate};
use crate::hybrid::{check_geometry, remainder_stems};
use crate::mixture::Normalisation;
use crate::pass::{Pass, Segmenting, check_range, chunk, source_planes};
use crate::pcm::Audio;
use crate::stems::{Parts, Stems, all_parts};

pub const RATE: u32 = 44_100;

struct Model {
    segment: i32,
}

impl Chunked for Model {
    fn chunk(&self) -> i32 {
        self.segment
    }
    fn sources(&self) -> i32 {
        3
    }
    /// Drums, vocals and bass.
    fn separate_chunk(&self, chunk: &Array) -> Result<Array> {
        let mean = ops::mean_axes(chunk, &[2], true)?;
        let drums = ops::multiply(chunk, Array::from_f32(0.5))?;
        let vocals = ops::add(
            &ops::multiply(&ops::sin(chunk)?, Array::from_f32(0.25))?,
            &mean,
        )?;
        let bass = ops::multiply(&ops::multiply(chunk, chunk)?, Array::from_f32(0.1))?;
        Ok(ops::stack_axis(&[drums, vocals, bass], 1)?)
    }
}

pub struct Toy {
    model: Model,
    overlap: f32,
    /// Slept before each segment, so a test can see work interleave.
    pub delay: Duration,
    /// Segments run so far.
    pub runs: Arc<AtomicUsize>,
}

impl Toy {
    pub fn new(segment: i32, overlap: f32) -> Self {
        Self {
            model: Model { segment },
            overlap,
            delay: Duration::ZERO,
            runs: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Separate for Toy {
    fn info(&self) -> BackendInfo {
        BackendInfo {
            backend: "toy".into(),
            model: "toy".into(),
            sample_rate: RATE,
            channels: 2,
            stems: crate::stems::SHIPPED.iter().map(|s| (*s).into()).collect(),
            device: "cpu".into(),
            parts: all_parts(),
        }
    }

    fn normalises(&self) -> bool {
        true
    }

    fn segmenting(&self) -> Option<Segmenting> {
        Some(Segmenting {
            segment: self.model.segment,
            overlap: self.overlap,
        })
    }

    fn begin(&self, frames: usize, parts: Parts, norm: Normalisation) -> Result<Pass> {
        let grid = Grid::new(i32::try_from(frames)?, self.model.segment, self.overlap)?;
        let mut pass = Pass::new(grid, parts, norm);
        pass.halves.push(Accumulator::new(grid, 1, 3, 2)?);
        Ok(pass)
    }

    fn run(&mut self, pass: &mut Pass, window: &Audio, k: usize) -> Result<()> {
        check_geometry(window, 2, RATE)?;
        std::thread::sleep(self.delay);
        let input = chunk(window, &pass.grid, k, Some(&pass.norm))?;
        let separated = self.model.separate_chunk(&input)?;
        pass.halves[0].add(k, &separated)?;
        self.runs.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn stems(&self, pass: &Pass, mix: &Audio, from: usize, to: usize) -> Result<Stems> {
        check_geometry(mix, 2, RATE)?;
        check_range(mix, from, to)?;
        pass.check_final(from, to)?;
        let norm = Some(&pass.norm);
        let plane = |s| source_planes(&pass.halves[0], s, from, to, 2, norm);
        Ok(remainder_stems(mix, plane(0)?, plane(1)?, Some(plane(2)?)))
    }
}
