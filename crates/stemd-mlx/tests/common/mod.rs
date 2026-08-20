//! Shared plumbing for the null tests.
//!
//! Each test binary compiles this separately, so a helper only one of them uses
//! reads as dead code in the others. That is what the allow is for.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use mlx_rs::Array;
use mlx_rs::ops;
use stemd_mlx::weights::Weights;

/// MLX is not safe to drive from two threads at once: doing so trips
/// `A command encoder is already encoding to this command buffer` inside Metal.
/// Taking this keeps `cargo test --workspace` working without anyone having to
/// remember `--test-threads=1`. The server serialises separation through its
/// queue already, so this constrains the tests rather than the design.
static GPU: Mutex<()> = Mutex::new(());

pub fn one_at_a_time() -> MutexGuard<'static, ()> {
    GPU.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// A dumped reference array.
pub fn array(name: &str) -> Array {
    let dir = fixtures();
    let shape: Vec<i32> = std::fs::read_to_string(dir.join(format!("{name}.shape")))
        .unwrap_or_else(|e| panic!("reading {name}.shape: {e}"))
        .split_whitespace()
        .map(|d| d.parse().expect("a shape is integers"))
        .collect();
    let bytes = std::fs::read(dir.join(format!("{name}.f32")))
        .unwrap_or_else(|e| panic!("reading {name}.f32: {e}"));
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let expected: i32 = shape.iter().product();
    assert_eq!(data.len() as i32, expected, "{name} is not its own shape");
    Array::from_slice(&data, &shape)
}

/// The real artefact, if it is on this machine.
///
/// 168 MB of weights are not something to check into a repository, so the tests
/// that need them skip when they are absent and say so. `STEMD_MLX_WEIGHTS` points
/// at an `htdemucs.safetensors`.
pub fn weights() -> Option<Weights> {
    from_env("STEMD_MLX_WEIGHTS", "an htdemucs.safetensors")
}

/// A converted BS-RoFormer, from `tools/export/convert_roformer.py`.
pub fn roformer_weights() -> Option<Weights> {
    from_env("STEMD_MLX_ROFORMER", "a converted bs_roformer .safetensors")
}

fn from_env(variable: &str, what: &str) -> Option<Weights> {
    let Ok(path) = std::env::var(variable) else {
        eprintln!(
            "SKIPPED: set {variable} to {what} to run this. \
             Without it nothing here is checked."
        );
        return None;
    };
    Some(Weights::load(std::path::Path::new(&path)).expect("loading the weights"))
}

/// A dumped array from a directory named by an environment variable, for the
/// fixtures too big to keep in the repository.
pub fn external(variable: &str, name: &str) -> Option<Array> {
    let Ok(dir) = std::env::var(variable) else {
        eprintln!("SKIPPED: set {variable} to a fixture directory to run this.");
        return None;
    };
    let dir = std::path::Path::new(&dir);
    let shape: Vec<i32> = std::fs::read_to_string(dir.join(format!("{name}.shape")))
        .unwrap_or_else(|e| panic!("reading {name}.shape: {e}"))
        .split_whitespace()
        .map(|d| d.parse().expect("a shape is integers"))
        .collect();
    let bytes = std::fs::read(dir.join(format!("{name}.f32")))
        .unwrap_or_else(|e| panic!("reading {name}.f32: {e}"));
    let data: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let expected: i32 = shape.iter().product();
    assert_eq!(data.len() as i32, expected, "{name} is not its own shape");
    Some(Array::from_slice(&data, &shape))
}

/// How far `got` sits below `want`, in dB. `-inf` is a perfect match.
pub fn null_db(got: &Array, want: &Array) -> f64 {
    let diff = ops::subtract(got, want).expect("same shapes");
    let rms = |a: &Array| -> f64 {
        let sq = ops::multiply(a, a).expect("square");
        f64::from(ops::mean(&sq, None).expect("mean").item::<f32>()).sqrt()
    };
    let (e, r) = (rms(&diff), rms(want));
    if e == 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * (e / r).log10()
}
