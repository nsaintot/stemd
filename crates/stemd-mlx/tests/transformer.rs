//! Null the cross-domain transformer against the reference.
//!
//! The position embeddings are checked on their own first. They are pure
//! functions of the shape, so a mistake in one: the sine/cosine interleaving
//! is the likely candidate: shows up here as a clear failure rather than as a
//! transformer that is inexplicably twenty decibels off.

mod common;

use common::{array, null_db, one_at_a_time, weights};
use stemd_mlx::transformer::{CrossTransformer, embeddings};

/// The one-dimensional embedding the time branch adds.
#[test]
fn the_time_position_embedding_matches() {
    let _gpu = one_at_a_time();
    let got = embeddings::one_dimensional(13, 512).expect("1-d embedding");
    let want = array("emb_1d");
    assert_eq!(got.shape(), want.shape(), "shape differs");
    let null = null_db(&got, &want);
    println!("1-d embedding null: {null:.1} dB");
    assert!(null < -100.0, "1-d embedding nulls at only {null:.1} dB");
}

/// The two-dimensional one, where half the channels encode width and half
/// height, sine and cosine alternating within each half.
#[test]
fn the_frequency_position_embedding_matches() {
    let _gpu = one_at_a_time();
    let got = embeddings::two_dimensional(512, 6, 5).expect("2-d embedding");
    let want = array("emb_2d");
    assert_eq!(got.shape(), want.shape(), "shape differs");
    let null = null_db(&got, &want);
    println!("2-d embedding null: {null:.1} dB");
    assert!(null < -100.0, "2-d embedding nulls at only {null:.1} dB");
}

/// All five layers, both branches, with the real weights.
#[test]
fn the_cross_transformer_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = weights() else { return };

    let tf = CrossTransformer::load(&w.at("model_0.crosstransformer"), 512, 8, 2048, 5)
        .expect("loading the transformer");

    let (x, xt) = tf
        .forward(&array("tf_x_input"), &array("tf_xt_input"))
        .expect("transformer forward");

    let want_x = array("tf_x_output");
    let want_xt = array("tf_xt_output");
    assert_eq!(x.shape(), want_x.shape(), "frequency shape differs");
    assert_eq!(xt.shape(), want_xt.shape(), "time shape differs");

    let (nx, nxt) = (null_db(&x, &want_x), null_db(&xt, &want_xt));
    println!("transformer null: frequency {nx:.1} dB, time {nxt:.1} dB");
    assert!(nx < -70.0, "the frequency branch nulls at only {nx:.1} dB");
    assert!(nxt < -70.0, "the time branch nulls at only {nxt:.1} dB");
}
