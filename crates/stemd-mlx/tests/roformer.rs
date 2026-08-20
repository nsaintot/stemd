//! BS-RoFormer, as far as it has been ported.
//!
//! The model is 159.8M parameters against htdemucs's 42M and separates one
//! stem, vocals, with the instrumental being the remainder. See
//! docs/evaluation.md for why it is worth having and how it has to be paired.
//!
//! Ignored: needs `STEMD_MLX_ROFORMER` pointing at a converted artefact.

mod common;

use common::{external, null_db, one_at_a_time, roformer_weights};
use mlx_rs::ops::indexing::IndexOp;
use stemd_mlx::roformer::{BsRoformer, Config};

/// The fixtures are ~40 MB, so they live outside the repository.
const FIXTURES: &str = "STEMD_MLX_ROFORMER_FIXTURES";

/// Which of the two artefacts is under test, asked of its tensors.
///
/// Both are this architecture: BS-RoFormer at `dim` 512 with rotary position,
/// and BS PolarFormer at 256 with polar. They differ in four config values and
/// no code, so every null test below runs against whichever artefact
/// `STEMD_MLX_ROFORMER` points at, and the fixtures beside it decide which one
/// is being checked. Guessing from the filename would work until someone
/// renamed a file; the presence of `pope_embed` cannot be renamed.
fn config_of(w: &stemd_mlx::weights::Weights) -> Config {
    let config = Config::of(w);
    println!("(dim {}, {:?} position)", config.dim, config.positional);
    config
}

/// The artefact as the server would load it: cast to the precision the config
/// asks for, rather than left at whatever the file happened to store.
///
/// The PolarFormer artefact is published as float16 and the viperx one as
/// float32, so this matters for saying what was tested, but it changes none of
/// the numbers below, which is worth recording because it was expected to.
/// mlx promotes, and these fixtures are float32, so a half weight meeting a
/// float32 activation computes in float32 from the first matmul on. The cast is
/// here so the test says what it runs at rather than inheriting it from how a
/// publisher happened to save a file.
fn model_of(w: &stemd_mlx::weights::Weights) -> BsRoformer {
    let config = config_of(w);
    let cast = w.cast(config.precision).expect("casting the artefact");
    BsRoformer::load(&cast, config).expect("loading")
}

/// The converted artefact is readable, complete, and shaped as expected.
///
/// Worth its own test before a line of the model is written against it: the
/// conversion had to clone twenty-four aliased rotary buffers to be writable
/// at all, and a silent truncation there would show up much later as a layer
/// that would not load.
#[test]
#[ignore]
fn the_converted_artefact_is_what_the_checkpoint_said() {
    let _gpu = one_at_a_time();
    let Some(w) = roformer_weights() else { return };

    // 699 for a rotary artefact. A polar one carries one shared `freqs` buffer
    // per attention less and a pair of `pope_embed` tensors more, so
    // 699 - 24 + 48 = 723: arithmetic rather than a second magic number, so a
    // truncated conversion still fails here instead of matching by accident.
    let polar = w.at("layers.0.0.layers.0.0").has("pope_embed.inv_freqs");
    let expected = if polar { 699 - 24 + 48 } else { 699 };
    assert_eq!(w.len(), expected, "the checkpoint holds {expected} tensors");

    // Twelve blocks, each a time transformer then a frequency transformer.
    for block in 0..12 {
        for (branch, what) in [(0, "time"), (1, "frequency")] {
            let attn = w.at(&format!("layers.{block}.{branch}.layers.0.0"));
            assert!(
                attn.has("to_qkv.weight"),
                "block {block}'s {what} transformer has no fused qkv"
            );
            // Gamma and no beta: these are RMS norms, not layer norms, and
            // reaching for the wrong one would load fine and be wrong.
            assert!(
                attn.has("norm.gamma"),
                "block {block} {what}: no norm gamma"
            );
            assert!(
                !attn.has("norm.beta"),
                "block {block} {what}: RMSNorm has no beta"
            );
            // Per-head gating, which is easy to miss reading the paper rather
            // than the weights: eight gates for eight heads.
            assert!(
                attn.has("to_gates.weight"),
                "block {block} {what}: no head gates"
            );
        }
    }

    // Sixty-two bands, each with its own norm, projection and mask estimator.
    for band in 0..62 {
        let split = w.at(&format!("band_split.to_features.{band}"));
        assert!(split.has("0.gamma"), "band {band} has no norm");
        assert!(split.has("1.weight"), "band {band} has no projection");
    }
    assert!(w.at("final_norm").has("gamma"));
}

/// The spectrogram, before anything is built on it.
///
/// BS-RoFormer takes an ordinary centred STFT: 2048-point, 441 hop, periodic
/// Hann, which is the same transform `crate::spectral::Stft` already does for
/// demucs at different constants. So this is really a test that reusing it is
/// legitimate, and it is cheap enough to be worth being sure about: everything
/// downstream reads these numbers.
#[test]
#[ignore]
fn the_spectrogram_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = roformer_weights() else { return };
    let Some(input) = external(FIXTURES, "rof_input") else {
        return;
    };

    let model = model_of(&w);
    let s = input.shape().to_vec();
    let audio = input.reshape(&[1, s[0], s[1]]).expect("add a batch axis");
    let (real, imag) = model.spectrogram(&audio).expect("stft");

    for (got, name) in [(&real, "rof_stft_real"), (&imag, "rof_stft_imag")] {
        let want = external(FIXTURES, name).expect("fixture");
        // The reference keeps [C, F, N]; this carries the batch axis through.
        let want = want
            .reshape(&[1, s[0], want.shape()[1], want.shape()[2]])
            .expect("add a batch axis");
        assert_eq!(got.shape(), want.shape(), "{name}: shape differs");
        let null = null_db(got, &want);
        println!("{name}: {null:.1} dB");
        assert!(null < -100.0, "{name} nulls at only {null:.1} dB");
    }
}

/// The band split: sixty-two projections, and the interleaving that feeds them.
///
/// Two things are being checked at once and only one of them is arithmetic.
/// The interleaving has to order the last axis by frequency, then audio
/// channel, then real and imaginary: get that wrong and every band is handed
/// a mixture of its neighbours' bins, at exactly the right shape. The
/// projections then have to be applied to the right slices, which is the same
/// mistake one level up.
#[test]
#[ignore]
fn the_band_split_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = roformer_weights() else { return };
    let Some(input) = external(FIXTURES, "rof_input") else {
        return;
    };
    let Some(want) = external(FIXTURES, "rof_bandsplit") else {
        return;
    };

    let model = model_of(&w);
    let s = input.shape().to_vec();
    let audio = input.reshape(&[1, s[0], s[1]]).expect("add a batch axis");
    let got = model.embed(&audio).expect("band split");

    assert_eq!(got.shape(), want.shape(), "shape differs");
    let null = null_db(&got, &want);
    println!("band split: {null:.1} dB");
    assert!(null < -100.0, "the band split nulls at only {null:.1} dB");
}

/// The two halves of a transformer layer, separately.
///
/// A whole block agreeing tells you less than knowing which half did, and
/// these two fail in different ways: attention has a rotary convention and a
/// per-head gate to get wrong, the feed-forward has only the order of two
/// matrices. Both take the packed shape the reference hands them:
/// `[bands, frames, dim]` for the time transformer, because that is what the
/// hooks captured.
#[test]
#[ignore]
fn attention_and_feed_forward_match_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = roformer_weights() else { return };
    let Some(input) = external(FIXTURES, "rof_bandsplit") else {
        return;
    };

    let model = model_of(&w);
    let (attn, ff) = model.block(0).time.layer(0);

    // [1, N, bands, dim] -> [bands, N, dim], which is how the time transformer
    // sees it: every band a row of the batch, attending over the timeline.
    let s = input.shape().to_vec();
    let packed = mlx_rs::ops::transpose_axes(&input, &[0, 2, 1, 3][..])
        .expect("pack")
        .reshape(&[s[0] * s[2], s[1], s[3]])
        .expect("reshape");

    let got_attn = attn.forward(&packed).expect("attention");
    let want_attn = external(FIXTURES, "rof_attn0").expect("fixture");
    assert_eq!(
        got_attn.shape(),
        want_attn.shape(),
        "attention: shape differs"
    );
    let null = null_db(&got_attn, &want_attn);
    println!("attention: {null:.1} dB");
    assert!(null < -100.0, "attention nulls at only {null:.1} dB");

    // The feed-forward is fed the residual sum, which is what it sees in place.
    let after = mlx_rs::ops::add(&got_attn, &packed).expect("residual");
    let got_ff = ff.forward(&after).expect("feed-forward");
    let want_ff = external(FIXTURES, "rof_ff0").expect("fixture");
    assert_eq!(
        got_ff.shape(),
        want_ff.shape(),
        "feed-forward: shape differs"
    );
    let null = null_db(&got_ff, &want_ff);
    println!("feed-forward: {null:.1} dB");
    assert!(null < -100.0, "the feed-forward nulls at only {null:.1} dB");
}

/// A whole block: a transformer over time, then one over bands.
#[test]
#[ignore]
fn a_block_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = roformer_weights() else { return };
    let Some(input) = external(FIXTURES, "rof_bandsplit") else {
        return;
    };

    let model = model_of(&w);
    let block = model.block(0);
    let s = input.shape().to_vec();
    let (batch, frames, bands, dim) = (s[0], s[1], s[2], s[3]);

    let packed = mlx_rs::ops::transpose_axes(&input, &[0, 2, 1, 3][..])
        .expect("pack")
        .reshape(&[batch * bands, frames, dim])
        .expect("reshape");
    let over_time = block.time.forward(&packed).expect("time transformer");
    let want = external(FIXTURES, "rof_block0_time").expect("fixture");
    assert_eq!(over_time.shape(), want.shape(), "time: shape differs");
    let null = null_db(&over_time, &want);
    println!("across time: {null:.1} dB");
    assert!(
        null < -100.0,
        "the time transformer nulls at only {null:.1} dB"
    );

    let unpacked = mlx_rs::ops::transpose_axes(
        over_time
            .reshape(&[batch, bands, frames, dim])
            .expect("unpack"),
        &[0, 2, 1, 3][..],
    )
    .expect("transpose");
    let packed = unpacked
        .reshape(&[batch * frames, bands, dim])
        .expect("repack");
    let over_bands = block.freq.forward(&packed).expect("freq transformer");
    let want = external(FIXTURES, "rof_block0_freq").expect("fixture");
    assert_eq!(over_bands.shape(), want.shape(), "freq: shape differs");
    let null = null_db(&over_bands, &want);
    println!("across bands: {null:.1} dB");
    assert!(
        null < -100.0,
        "the band transformer nulls at only {null:.1} dB"
    );
}

/// All twelve blocks and the norm after them.
#[test]
#[ignore]
fn the_whole_stack_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = roformer_weights() else { return };
    let Some(input) = external(FIXTURES, "rof_input") else {
        return;
    };
    let Some(want) = external(FIXTURES, "rof_final_norm") else {
        return;
    };

    let model = model_of(&w);
    let s = input.shape().to_vec();
    let audio = input.reshape(&[1, s[0], s[1]]).expect("add a batch axis");
    let embedded = model.embed(&audio).expect("band split");
    let got = model.transform(&embedded).expect("the stack");

    assert_eq!(got.shape(), want.shape(), "shape differs");
    let null = null_db(&got, &want);
    println!("twelve blocks: {null:.1} dB");
    assert!(null < -80.0, "the stack nulls at only {null:.1} dB");
}

/// The mask estimators: sixty-two small networks, one per band.
#[test]
#[ignore]
fn the_mask_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = roformer_weights() else { return };
    let Some(features) = external(FIXTURES, "rof_final_norm") else {
        return;
    };
    let Some(want) = external(FIXTURES, "rof_mask") else {
        return;
    };

    let model = model_of(&w);
    let got = model.mask(&features).expect("mask");

    assert_eq!(got.shape(), want.shape(), "shape differs");
    let null = null_db(&got, &want);
    println!("mask: {null:.1} dB");

    // -90 rather than the -100 every other stage clears, and the reason is not
    // "PolarFormer needed it": this stage is a tanh and a sigmoid over 62 bands
    // of 1024 hidden units, and both models land shallower here than either
    // does on a plain matmul. PolarFormer measures -95.9 dB from an exact input
    // -- the fixture -- so what is being seen is this stage's own arithmetic
    // and not error arriving from upstream.
    //
    // It is not fully explained and it does not need to be to be bounded:
    // -95.9 dB is 1.6e-5 of the signal, the whole model nulls at -88.7 dB
    // against a -60 dB bar, and the model's own separation error is thirty
    // decibels above either. A threshold set where a different artefact
    // happened to land is not a threshold, so this one is set where the claim
    // is -- inaudible, and far under the model's own error.
    assert!(null < -90.0, "the mask nulls at only {null:.1} dB");
}

/// The whole model: audio in, vocals out.
///
/// Everything above this passes on its own, so what is left for this to catch
/// is the wiring between them: the complex multiply, the order the frequency
/// and channel axes are unpicked in, the dropped DC bin, and the inverse.
/// Those are exactly the steps with no learned weights to disagree about, and
/// therefore the ones that fail silently.
#[test]
#[ignore]
fn the_whole_model_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = roformer_weights() else { return };
    let Some(input) = external(FIXTURES, "rof_input") else {
        return;
    };
    let Some(want) = external(FIXTURES, "rof_output") else {
        return;
    };

    let model = model_of(&w);
    let s = input.shape().to_vec();
    let audio = input.reshape(&[1, s[0], s[1]]).expect("add a batch axis");
    let got = model.forward(&audio).expect("separating");

    // The reference keeps a stem axis it only ever has one of.
    let want = want.reshape(&[1, s[0], s[1]]).expect("drop the stem axis");
    assert_eq!(got.shape(), want.shape(), "shape differs");
    let null = null_db(&got, &want);
    println!("whole model: {null:.1} dB");
    assert!(null < -60.0, "the model nulls at only {null:.1} dB");
}

/// A track shorter than one chunk still goes through the window protocol.
///
/// The comparison that suggests itself: segmenting a one-second track should
/// equal running the model on that second: is false here for the same reason
/// it is false for htdemucs: the chunk is centred in a full window and the
/// model sees eight seconds of mostly silence, which is different audio. What
/// has to hold is that the crossfade weighting and the division by the
/// accumulated weight cancel on top of whatever the model then says.
#[test]
#[ignore]
fn a_short_track_survives_the_crossfade() {
    let _gpu = one_at_a_time();
    let Some(w) = roformer_weights() else { return };
    let Some(input) = external(FIXTURES, "rof_input") else {
        return;
    };

    let model = model_of(&w);
    let s = input.shape().to_vec();
    let audio = input.reshape(&[1, s[0], s[1]]).expect("batch");

    let chunk = <BsRoformer as stemd_mlx::apply::Chunked>::chunk(&model);
    let length = s[1];
    let delta = chunk - length;
    let (left, right) = (delta / 2, delta - delta / 2);
    let centred = mlx_rs::ops::pad(
        &audio,
        &[(0, 0), (0, 0), (left, right)][..],
        mlx_rs::Array::from_f32(0.0),
        None,
    )
    .expect("centre the window");
    let want = model
        .forward(&centred)
        .expect("forward")
        .index((.., .., left..left + length));

    let got = stemd_mlx::apply::over_track(&model, &audio, 0.25, &mut |_, _| Ok(()))
        .expect("over_track")
        .index((.., 0, .., ..));

    assert_eq!(got.shape(), want.shape(), "shape differs");
    let null = null_db(&got, &want);
    println!("short track: {null:.1} dB");
    assert!(null < -100.0, "the crossfade changed it by {null:.1} dB");
}
