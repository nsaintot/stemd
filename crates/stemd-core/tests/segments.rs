//! Segments run in any order give the same stems as a whole-track run.
//!
//! Ignored by default: it needs the preset weights.
//!
//! ```text
//! STEMD_MODELS=/dir/holding/htdemucs,htdemucs_ft,stemd_roformer.safetensors \
//! cargo test --release -p stemd-core --test segments -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use stemd_core::{
    Audio, HybridConfig, HybridSeparator, MlxConfig, MlxSeparator, Normalisation, Parts, Precision,
    RoformerSeparator, Separate, Silent, Stems,
};

fn models() -> Option<PathBuf> {
    match std::env::var("STEMD_MODELS") {
        Ok(v) => Some(PathBuf::from(v)),
        Err(_) => {
            eprintln!("skipped: set STEMD_MODELS to run these checks");
            None
        }
    }
}

/// Twenty seconds of stereo tones and noise, ending in a short last segment.
fn track() -> Audio {
    let rate = 44_100;
    let frames = 20 * rate + 12_345;
    let mut state = 0x9e37_79b9_u32;
    let mut noise = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    let data = (0..2)
        .map(|c| {
            (0..frames)
                .map(|i| {
                    let t = i as f32 / rate as f32;
                    let tone = (t * 220.0 * (1.0 + c as f32 * 0.01) * std::f32::consts::TAU).sin();
                    let beat = if (t * 2.0).fract() < 0.05 { 0.8 } else { 0.0 };
                    0.3 * tone + 0.1 * noise() + beat * noise()
                })
                .collect()
        })
        .collect();
    Audio::new(data, rate as u32)
}

fn presets(dir: &Path) -> Vec<(&'static str, Box<dyn Separate>)> {
    vec![
        (
            "Fast",
            Box::new(MlxSeparator::load(dir, "htdemucs", MlxConfig::default()).expect("htdemucs")),
        ),
        (
            "Balanced",
            Box::new(
                HybridSeparator::demucs_specialists(dir, "htdemucs_ft", HybridConfig::default())
                    .expect("htdemucs_ft"),
            ),
        ),
        (
            "Quality",
            Box::new(
                RoformerSeparator::load(dir, "stemd_roformer", Precision::F32, 0.25)
                    .expect("stemd_roformer"),
            ),
        ),
    ]
}

fn excerpt(mix: &Audio, from: usize, to: usize) -> Audio {
    Audio::new(
        mix.data.iter().map(|c| c[from..to].to_vec()).collect(),
        mix.sample_rate,
    )
}

fn named(stems: &Stems) -> Vec<(&'static str, &Audio)> {
    stems
        .shipped
        .iter()
        .chain(&stems.more)
        .map(|(n, a)| (*n, a))
        .collect()
}

/// Every stem of `got` is bit for bit `want` over `from..to`.
fn same_bits(label: &str, got: &Stems, want: &Stems, from: usize, to: usize) {
    let want = named(want);
    let got = named(got);
    assert_eq!(
        got.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        want.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
        "{label}: different stems"
    );
    for ((name, g), (_, w)) in got.iter().zip(&want) {
        for c in 0..w.channels() {
            let differs = g.data[c]
                .iter()
                .zip(&w.data[c][from..to])
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(
                differs, None,
                "{label}: {name} channel {c} differs at the sample index shown"
            );
        }
    }
}

#[test]
#[ignore]
fn segments_in_any_order_match_the_whole_track() {
    let Some(dir) = models() else { return };
    let mix = track();
    let parts = Parts::parse("bass,other,vocals").expect("parts");

    for (label, mut separator) in presets(&dir) {
        let whole = separator
            .separate(&mix, parts, &Silent)
            .expect("separating");

        let mut pass = separator
            .begin(mix.frames(), parts, Normalisation::of(&mix))
            .expect("beginning");
        let n = pass.segments();
        // From the middle out, as a player that loaded the track at a cue would.
        let order: Vec<usize> = (n / 2..n).chain((0..n / 2).rev()).collect();
        let (from, to) = (9 * 44_100, 11 * 44_100);
        let mut checked_early = false;
        for k in order {
            let window = pass.window(k);
            separator
                .run(&mut pass, &excerpt(&mix, window.start, window.end), k)
                .expect("running");
            if !checked_early && pass.is_final(from, to) {
                let range = separator
                    .stems(&pass, &excerpt(&mix, from, to), from, to)
                    .expect("a range");
                same_bits(&format!("{label} {from}..{to}"), &range, &whole, from, to);
                checked_early = true;
            }
        }
        assert!(checked_early, "{label}: the middle never became final");

        let all = separator
            .stems(&pass, &mix, 0, mix.frames())
            .expect("the whole track");
        same_bits(label, &all, &whole, 0, mix.frames());
        println!("{label}: {n} segments out of order match the whole track");
    }
}
