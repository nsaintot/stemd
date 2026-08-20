//! The fused normalisations, against the arithmetic they stand for.
//!
//! `fast::rms_norm` and `fast::layer_norm` are single kernels doing what four or
//! five ops would do more slowly, and every backend picks the kernel by row width.
//! That dispatch is where a backend gets a band of widths wrong, and it is silent
//! when it does: the shapes still fit, nothing errors, and the model separates
//! audio that is merely wrong.
//!
//! MLX v0.31.2's CUDA RMSNorm gave one block two rows to reduce while sizing its
//! shared storage for one, for widths 129 to 256 at f32 and 257 to 512 at f16: a
//! band the ordinary suite only crosses from inside a loaded model.
//!
//! So this sweeps every width these kernels could be handed rather than the ones
//! today's artefacts happen to hand them. It is deliberately not `#[ignore]`d: it
//! needs a GPU and nothing else, no artefact and no fixture.
//!
//! `layer_norm` is swept too. Only `rms_norm` was ever wrong, and `layer_norm` has
//! no dispatch-by-group in the version this pins, so today it can only pass.

mod common;

use common::{null_db, one_at_a_time};
use mlx_rs::{Array, Dtype, ops};

/// Widths to sweep. Past what either model asks for: roformer normalises at
/// `dim` 256 and 512, htdemucs at 384 and 512, and the point is the widths
/// nobody has tried yet.
const WIDEST: i32 = 1024;

/// The eps each caller passes, so a kernel is checked at the magnitude it is
/// actually used at: roformer.rs asks for 1e-12, which is small enough to be
/// doing nothing, and the layer norms ask for something that is not.
const RMS_EPS: f32 = 1e-12;
const LAYER_EPS: f32 = 1e-5;

/// Four rows that disagree in scale by 4x a step.
///
/// One row would have caught nothing: the CUDA bug reduced *two* rows into one
/// row's worth of storage, so it needs a neighbour to be wrong about, and rows
/// that hold the same distribution would let it be wrong and still land within
/// tolerance. Four rows spanning 64x in RMS means any merging of two of them
/// misses by a factor, not by rounding.
const ROWS: i32 = 4;

fn input(width: i32) -> Array {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut data = Vec::with_capacity((ROWS * width) as usize);
    for row in 0..ROWS {
        let scale = 4.0_f32.powi(row);
        for _ in 0..width {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            data.push(((state >> 40) as f32 / 8_388_608.0 - 1.0) * scale);
        }
    }
    Array::from_slice(&data, &[ROWS, width])
}

/// A weight that is not all ones, so a kernel that ignored it would fail here
/// rather than pass and leave the affine untested.
fn gamma(width: i32) -> Array {
    let data: Vec<f32> = (0..width).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
    Array::from_slice(&data, &[width])
}

/// The reference is computed at f32 whatever the kernel ran at, because the
/// kernels accumulate at f32 too. Writing the f16 reference in f16 would compare
/// two different roundings and call the difference a kernel bug.
fn as_f32(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32).expect("upcast to f32")
}

fn rms_norm_written_out(x: &Array, gamma: &Array, eps: f32) -> Array {
    let (x, gamma) = (as_f32(x), as_f32(gamma));
    let mean_square = ops::multiply(&x, &x)
        .expect("square")
        .mean_axis(-1, true)
        .expect("mean over the last axis");
    let scale = ops::add(&mean_square, Array::from_f32(eps))
        .expect("eps")
        .rsqrt()
        .expect("rsqrt");
    ops::multiply(ops::multiply(&x, &scale).expect("normalise"), &gamma).expect("weight")
}

/// The unweighted form, which is the one htdemucs uses at both of its call
/// sites. The affine is elementwise and lands after the reduction; the
/// reduction is the part a dispatch bug corrupts.
fn layer_norm_written_out(x: &Array, eps: f32) -> Array {
    let x = as_f32(x);
    let centred = ops::subtract(&x, x.mean_axis(-1, true).expect("mean")).expect("centre");
    let variance = ops::multiply(&centred, &centred)
        .expect("square")
        .mean_axis(-1, true)
        .expect("mean over the last axis");
    let scale = ops::add(&variance, Array::from_f32(eps))
        .expect("eps")
        .rsqrt()
        .expect("rsqrt");
    ops::multiply(&centred, &scale).expect("normalise")
}

/// Sweep one kernel at one dtype and assert the worst width still nulls.
///
/// Measured over all 1024 widths. Metal, M-series: rms_norm -137.2 dB at f32 and
/// -65.8 at f16, layer_norm -95.1 and -72.4. CUDA, RTX 3090 Ti against the patched
/// MLX: rms_norm -139.0 and -65.8, layer_norm -107.7 and -72.4. The floors sit
/// 15 dB under the worst of either and some 60 dB above what the CUDA bug
/// produced, so they are set for immunity to backend jitter rather than as tight
/// as one machine allows.
///
/// The f32 floor looks slack against -137 dB because the binding case is
/// conditioning, not the kernel: below about width 16 a random row can come out
/// nearly constant, centring it cancels almost everything, and layer_norm's
/// relative error grows accordingly. rms_norm subtracts no mean and shows none of
/// it.
fn sweep(
    what: &str,
    dtype: Dtype,
    floor: f64,
    kernel: impl Fn(&Array, &Array) -> Array,
    written_out: impl Fn(&Array, &Array) -> Array,
) {
    let mut worst = (f64::NEG_INFINITY, 0);
    for width in 1..=WIDEST {
        let x = input(width).as_dtype(dtype).expect("cast the input");
        let g = gamma(width).as_dtype(dtype).expect("cast the weight");
        let null = null_db(&as_f32(&kernel(&x, &g)), &written_out(&x, &g));
        if null > worst.0 {
            worst = (null, width);
        }
    }
    let (null, width) = worst;
    println!("{what} at {dtype:?}: worst {null:.1} dB at width {width}");
    assert!(
        null <= floor,
        "fast::{what} at {dtype:?} disagrees with the same arithmetic \
         written out: {null:.1} dB at width {width}, floor {floor:.1} dB. \
         A fused normalisation that is wrong for one band of widths is an \
         MLX backend bug, not a stemd one — see the note at the top of this file."
    );
}

#[test]
fn rms_norm_agrees_with_its_arithmetic_at_every_width() {
    let _gpu = one_at_a_time();
    for (dtype, floor) in [(Dtype::Float32, -80.0), (Dtype::Float16, -50.0)] {
        sweep(
            "rms_norm",
            dtype,
            floor,
            |x, g| mlx_rs::fast::rms_norm(x, g, RMS_EPS).expect("rms_norm"),
            |x, g| rms_norm_written_out(x, g, RMS_EPS),
        );
    }
}

#[test]
fn layer_norm_agrees_with_its_arithmetic_at_every_width() {
    let _gpu = one_at_a_time();
    for (dtype, floor) in [(Dtype::Float32, -80.0), (Dtype::Float16, -50.0)] {
        sweep(
            "layer_norm",
            dtype,
            floor,
            |x, _| mlx_rs::fast::layer_norm(x, None, None, LAYER_EPS).expect("layer_norm"),
            |x, _| layer_norm_written_out(x, LAYER_EPS),
        );
    }
}
