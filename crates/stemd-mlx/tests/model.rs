//! One whole segment through the assembled model.
//!
//! Every component has its own passing null below this, so a failure here is in
//! the wiring rather than the arithmetic: the interleaved encoder loops, the
//! order the skips are consumed in, the per-branch normalisation, or the mask
//! that puts the frequency branch back onto complex bins.

mod common;

use common::{array, null_db, one_at_a_time, weights};
use stemd_mlx::htdemucs::{Config, HtDemucs};

#[test]
fn a_whole_segment_matches_the_reference() {
    let _gpu = one_at_a_time();
    let Some(w) = weights() else { return };

    let model = HtDemucs::load(&w, "model_0", Config::default()).expect("loading htdemucs");
    let got = model
        .forward(&array("model_input"))
        .expect("the forward pass");

    let want = array("model_output");
    assert_eq!(got.shape(), want.shape(), "shape differs");

    let null = null_db(&got, &want);
    println!("whole segment null: {null:.1} dB");
    // Looser than the per-layer tests on purpose: this is fifty-odd layers of
    // accumulated float32, and the reference is itself only one ordering of
    // those operations. Anything near this is the same model; -20 dB would be
    // a wiring mistake wearing a plausible number.
    assert!(null < -60.0, "the model nulls at only {null:.1} dB");
}
