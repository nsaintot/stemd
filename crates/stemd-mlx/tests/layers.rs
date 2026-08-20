//! Null the first encoder layer of each branch against the reference.
//!
//! This is the step where a mistake is silent. demucs is channels-first and MLX
//! is channels-last, so a missed transpose convolves over frequency instead of
//! channels and returns a tensor of exactly the right shape. Nothing errors; the
//! model just separates nothing. The whole point of this file is that the
//! weights are real, the input is real, and the output has to match.

mod common;

use common::{array, null_db, one_at_a_time, weights};
use stemd_mlx::layers::{DecLayer, Domain, EncLayer};

/// The frequency branch: a 2-D strided convolution, gelu, the dilated branch
/// folded per frequency bin, then the gated rewrite.
#[test]
fn the_first_frequency_encoder_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = weights() else { return };

    let layer = EncLayer::load(
        &w.at("model_0.encoder.0"),
        4,  // cac doubles the two audio channels
        48, // `channels`
        8,  // `kernel_size`
        4,  // `stride`
        Domain::Frequency,
        true,         // padded
        false,        // not empty
        0,            // `context_enc`
        Some((2, 8)), // dconv depth and compression
    )
    .expect("loading encoder 0");

    let got = layer
        .forward(&array("enc0_input"), None)
        .expect("encoder 0 forward");
    let want = array("enc0_output");
    assert_eq!(got.shape(), want.shape(), "shape differs");

    let null = null_db(&got, &want);
    println!("frequency encoder 0 null: {null:.1} dB");
    assert!(null < -80.0, "encoder 0 nulls at only {null:.1} dB");
}

/// The time branch: the same block in one dimension, over `[B, C, T]`.
#[test]
fn the_first_time_encoder_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = weights() else { return };

    let layer = EncLayer::load(
        &w.at("model_0.tencoder.0"),
        2,
        48,
        8,
        4,
        Domain::Time,
        true,
        false,
        0,
        Some((2, 8)),
    )
    .expect("loading tencoder 0");

    let got = layer
        .forward(&array("tenc0_input"), None)
        .expect("tencoder 0 forward");
    let want = array("tenc0_output");
    assert_eq!(got.shape(), want.shape(), "shape differs");

    let null = null_db(&got, &want);
    println!("time encoder 0 null: {null:.1} dB");
    assert!(null < -80.0, "tencoder 0 nulls at only {null:.1} dB");
}

/// The frequency decoder: gated rewrite, dilated branch, transposed
/// convolution, then the frequency axis trimmed back by the padding.
///
/// Both returns are checked. `pre` is what the time branch injects into, so a
/// decoder that produced the right output from the wrong intermediate would
/// pass a laxer test and break the branch that reads it.
#[test]
fn the_first_frequency_decoder_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = weights() else { return };

    let layer = DecLayer::load(
        &w.at("model_0.decoder.0"),
        384,
        192,
        8,
        4,
        Domain::Frequency,
        true,  // padded
        false, // not empty
        false, // not the last decoder
        1,     // `context`
        Some((2, 8)),
    )
    .expect("loading decoder 0");

    let (got, pre) = layer
        .forward(&array("dec0_input"), Some(&array("dec0_skip")), 8)
        .expect("decoder 0 forward");

    let want = array("dec0_output");
    assert_eq!(got.shape(), want.shape(), "output shape differs");
    let null = null_db(&got, &want);
    println!("frequency decoder 0 null: {null:.1} dB");
    assert!(null < -80.0, "decoder 0 nulls at only {null:.1} dB");

    let want_pre = array("dec0_pre");
    assert_eq!(pre.shape(), want_pre.shape(), "pre shape differs");
    let null = null_db(&pre, &want_pre);
    println!("frequency decoder 0 pre null: {null:.1} dB");
    assert!(null < -80.0, "decoder 0 pre nulls at only {null:.1} dB");
}

/// The time decoder, which additionally trims its output to a given length.
#[test]
fn the_first_time_decoder_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = weights() else { return };

    let layer = DecLayer::load(
        &w.at("model_0.tdecoder.0"),
        384,
        192,
        8,
        4,
        Domain::Time,
        true,
        false,
        false,
        1,
        Some((2, 8)),
    )
    .expect("loading tdecoder 0");

    let (got, pre) = layer
        .forward(&array("tdec0_input"), Some(&array("tdec0_skip")), 128)
        .expect("tdecoder 0 forward");

    let want = array("tdec0_output");
    assert_eq!(got.shape(), want.shape(), "output shape differs");
    let null = null_db(&got, &want);
    println!("time decoder 0 null: {null:.1} dB");
    assert!(null < -80.0, "tdecoder 0 nulls at only {null:.1} dB");

    let null = null_db(&pre, &array("tdec0_pre"));
    println!("time decoder 0 pre null: {null:.1} dB");
    assert!(null < -80.0, "tdecoder 0 pre nulls at only {null:.1} dB");
}
