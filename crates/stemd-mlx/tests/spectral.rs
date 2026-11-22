//! Null the Rust spectrogram against the reference implementation's.
//!
//! The fixtures are the Python MLX model's own `_spec`, `_magnitude` and `_ispec` for a known
//! input, dumped by `make_spec_fixture.py`. The tests check that the transform is the one the
//! traced model was built around, the property every later layer depends on, and say nothing about
//! its quality as a spectrogram.

mod common;

use common::{array, null_db, one_at_a_time};
use mlx_rs::ops;
use stemd_mlx::spectral::Spectral;

/// The forward transform, real and imaginary parts separately.
///
/// The test uses a tolerance: MLX's FFT may associate its butterflies differently between calls,
/// and the surrounding framing is two independent implementations. −100 dB is far below anything
/// the model can notice and far above float32 noise.
#[test]
fn the_spectrogram_matches_the_reference() {
    let _gpu = one_at_a_time();
    let input = array("spec_input");
    let z = Spectral::new().forward(&input).expect("forward");

    let (want_r, want_i) = (array("spec_real"), array("spec_imag"));
    assert_eq!(
        z.shape(),
        want_r.shape(),
        "shape differs before the values could"
    );

    let got_r = ops::real(&z).expect("real part");
    let got_i = ops::imag(&z).expect("imaginary part");
    let (nr, ni) = (null_db(&got_r, &want_r), null_db(&got_i, &want_i));
    println!("spectrogram null: real {nr:.1} dB, imag {ni:.1} dB");
    assert!(nr < -100.0, "real part nulls at only {nr:.1} dB");
    assert!(ni < -100.0, "imaginary part nulls at only {ni:.1} dB");
}

/// The inverse, including the trimming that makes it deliberately lossy.
///
/// Checked against the reference's output: `_spec` drops the Nyquist bin and two frames, so a round
/// trip loses about 28 dB by design, and a comparison with the original input would measure that
/// designed loss.
#[test]
fn the_inverse_matches_the_reference() {
    let _gpu = one_at_a_time();
    let input = array("spec_input");
    let length = *input.shape().last().expect("the input has samples");

    let spectral = Spectral::new();
    let z = spectral.forward(&input).expect("forward");
    let back = spectral.inverse(&z, length).expect("inverse");

    let want = array("ispec_output");
    assert_eq!(back.shape(), want.shape(), "shape differs");

    let null = null_db(&back, &want);
    println!("inverse null: {null:.1} dB");
    assert!(null < -80.0, "the inverse nulls at only {null:.1} dB");
}

/// `_magnitude` under `cac`: real and imaginary parts interleaved as channels, with no modulus. It
/// is easy to get subtly wrong: the two parts can be stacked in either order, and everything
/// downstream reads the result as the input image.
#[test]
fn complex_as_channels_interleaves_the_way_the_model_expects() {
    let _gpu = one_at_a_time();
    let input = array("spec_input");
    let z = Spectral::new().forward(&input).expect("forward");

    let shape = z.shape().to_vec();
    let (b, c, f, t) = (shape[0], shape[1], shape[2], shape[3]);
    let stacked = ops::stack_axis(
        &[
            ops::real(&z).expect("real part"),
            ops::imag(&z).expect("imaginary part"),
        ],
        2,
    )
    .expect("stack");
    let got = stacked.reshape(&[b, c * 2, f, t]).expect("reshape");

    let want = array("spec_magnitude");
    assert_eq!(got.shape(), want.shape(), "shape differs");
    let null = null_db(&got, &want);
    println!("magnitude null: {null:.1} dB");
    assert!(null < -100.0, "magnitude nulls at only {null:.1} dB");
}
