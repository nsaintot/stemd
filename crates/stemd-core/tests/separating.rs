//! The backend on a real track, at both precisions.
//!
//! Ignored by default: it needs the weights and a track, and neither belongs in a repository.
//!
//! ```text
//! STEMD_AB_PCM=track.pcm STEMD_AB_MLX=/path/holding/htdemucs.safetensors \
//! cargo test --release -p stemd-core --test separating -- --ignored --nocapture
//! ```
//!
//! `STEMD_AB_PCM` is raw interleaved stereo 16-bit PCM at 44.1 kHz: the body of a
//! WAV with its header removed, and one of the two formats the server already
//! accepts, so this needs no decoder of its own. The hybrid tests also need
//! `--features research`.

use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "research")]
use stemd_core::stems::remainder;
use stemd_core::{Audio, MlxConfig, MlxSeparator, Parts, PcmFormat, Precision, Separate};
#[cfg(feature = "research")]
use stemd_core::{HybridConfig, HybridSeparator, Silent};

/// Raw interleaved stereo 16-bit PCM at 44.1 kHz.
fn read_pcm(path: &std::path::Path) -> Audio {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    Audio::from_interleaved(&bytes, PcmFormat::S16le, 2, 44100)
        .unwrap_or_else(|e| panic!("{} is not stereo 16-bit PCM: {e}", path.display()))
}

fn env_dir(name: &str) -> Option<PathBuf> {
    match std::env::var(name) {
        Ok(v) => Some(PathBuf::from(v)),
        Err(_) => {
            eprintln!("SKIPPED: set {name}; without it nothing here is checked.");
            None
        }
    }
}

#[test]
#[ignore]
fn a_real_track_separates_at_the_measured_quality_and_speed() {
    let (Some(pcm), Some(dir)) = (env_dir("STEMD_AB_PCM"), env_dir("STEMD_AB_MLX")) else {
        return;
    };

    // The artefact to test, so a locally converted single-model artefact can go through the same
    // harness.
    let artefact = std::env::var("STEMD_AB_MODEL").unwrap_or_else(|_| "htdemucs".into());

    let mix = read_pcm(&pcm);
    let seconds = mix.frames() as f64 / f64::from(mix.sample_rate);
    println!(
        "{artefact}: {seconds:.1} s of audio, {} channels",
        mix.channels()
    );

    let run = |precision: Precision| {
        let mut mlx = MlxSeparator::load(
            &dir,
            &artefact,
            MlxConfig {
                precision,
                ..MlxConfig::default()
            },
        )
        .expect("loading the MLX artefact");

        // Record every distinct fraction a polling client would see, on the run already happening.
        let seen = std::sync::Mutex::new(Vec::<f32>::new());
        let watch = |p: stemd_core::Progress| {
            let mut seen = seen.lock().expect("not poisoned");
            if seen.last() != Some(&p.fraction) {
                seen.push(p.fraction);
            }
        };

        let started = Instant::now();
        let stems = mlx
            .separate(&mix, Parts::DEFAULT, &watch)
            .expect("separating");
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "  {precision:>4}: {elapsed:6.1} s  ({:.1}x realtime, model residual {:.1} dB)",
            seconds / elapsed,
            stems.model_residual_db
        );

        let seen = seen.into_inner().expect("not poisoned");
        let steps: Vec<String> = seen.iter().map(|f| format!("{:.0}%", f * 100.0)).collect();
        assert!(
            seen.windows(2).all(|w| w[1] >= w[0]),
            "progress went backwards, so a client sees work repeat: {steps:?}"
        );

        (stems, seconds / elapsed)
    };

    let (from_half, _) = run(Precision::F16);
    let (stems, realtime) = run(Precision::F32);

    let (mut error, mut signal) = (0.0f64, 0.0f64);
    for ((name, a), (half_name, b)) in stems.shipped.iter().zip(&from_half.shipped) {
        assert_eq!(name, half_name, "the two ship stems in a different order");
        for (ca, cb) in a.data.iter().zip(&b.data) {
            for (x, y) in ca.iter().zip(cb) {
                error += f64::from(x - y).powi(2);
                signal += f64::from(*x).powi(2);
            }
        }
    }
    let null = 10.0 * (error / signal).log10();
    println!("f16 against f32: {null:.1} dB");

    // htdemucs's own residual measures -33.9 dB in TorchScript and in this runtime, at both
    // precisions. A drift of several dB would mean that the weights or the arithmetic around them
    // changed, since the model's own error depends on the model alone.
    if artefact == "htdemucs" {
        assert!(
            stems.model_residual_db < -25.0,
            "the model residual is {:.1} dB, well above where htdemucs sits",
            stems.model_residual_db
        );
    }

    // Half precision must stay well under the model's own error; otherwise it would amount to a
    // different model. It measures -53.9 dB on the shipped stems.
    assert!(
        null < stems.model_residual_db - 5.0,
        "half precision differs by {null:.1} dB, too close to the model's own \
         residual of {:.1} dB",
        stems.model_residual_db
    );

    // Full precision measures 13.9x realtime, so this floor leaves plenty of margin. It catches a
    // regression such as a frame-by-frame inverse transform, which costs forty per cent of the
    // forward pass.
    let floor = if artefact == "htdemucs" { 4.0 } else { 1.0 };
    assert!(
        realtime > floor,
        "separating ran at {realtime:.1}x realtime, far below what was measured"
    );
}

/// The hybrid, BS-RoFormer for vocals and htdemucs_ft's drums specialist for the rest, on a real
/// track. This measures its cost; docs/evaluation.md covers its quality.
///
/// ```text
/// STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with both artefacts> \
/// STEMD_AB_HYBRID=1 cargo test --release -p stemd-core --test separating \
///     -- --ignored --nocapture
/// ```
#[cfg(feature = "research")]
#[test]
#[ignore]
fn the_hybrid_separates_a_real_track() {
    if std::env::var("STEMD_AB_HYBRID").is_err() {
        eprintln!("SKIPPED: set STEMD_AB_HYBRID=1 to run the hybrid; it needs both artefacts.");
        return;
    }
    let (Some(pcm), Some(dir)) = (env_dir("STEMD_AB_PCM"), env_dir("STEMD_AB_MLX")) else {
        return;
    };

    let mix = read_pcm(&pcm);
    let seconds = mix.frames() as f64 / f64::from(mix.sample_rate);

    let vocals = std::env::var("STEMD_AB_VOCALS").unwrap_or_else(|_| "bs_roformer_viperx".into());
    let precision = if std::env::var("STEMD_AB_FULL").is_err() {
        Precision::F16
    } else {
        Precision::F32
    };
    let config = HybridConfig {
        vocals_precision: precision,
        drums_precision: precision,
        ..HybridConfig::default()
    };
    // Either arrangement, through the same harness: RoFormer plus a demucs
    // drums specialist, or both specialists from one demucs bag.
    let mut hybrid = if vocals == "htdemucs_ft" {
        HybridSeparator::demucs_specialists(&dir, "htdemucs_ft", config)
    } else {
        HybridSeparator::roformer_and_demucs(&dir, &vocals, "htdemucs_ft", config)
    }
    .expect("loading both halves");
    println!("precision: {precision}");

    // Record every distinct fraction the bar would show, so the test can see the bar stall at 10%
    // and jump.
    let seen = std::sync::Mutex::new(Vec::<f32>::new());
    let watch = |p: stemd_core::Progress| {
        let mut seen = seen.lock().expect("not poisoned");
        if seen.last() != Some(&p.fraction) {
            seen.push(p.fraction);
        }
    };

    let started = Instant::now();
    let stems = hybrid
        .separate(&mix, Parts::DEFAULT, &watch)
        .expect("separating");
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "hybrid: {elapsed:6.1} s for {seconds:.0} s ({:.1}x realtime)",
        seconds / elapsed
    );

    let seen = seen.into_inner().expect("not poisoned");
    let steps: Vec<String> = seen.iter().map(|f| format!("{:.0}%", f * 100.0)).collect();
    println!("bar: {}", steps.join(" "));
    assert!(
        seen.windows(2).all(|w| w[1] >= w[0]),
        "progress went backwards: {steps:?}"
    );
    assert!(
        seen.len() > 8,
        "the bar only moved {} times over {elapsed:.0} s: {steps:?}",
        seen.len()
    );

    // The parts are built to sum, so the residual is float noise and measures nothing. Asserting
    // that it is tiny still checks the arithmetic: a sign error in the remainder would show up here
    // first.
    println!("sum residual: {:.1} dB", stems.model_residual_db);
    assert!(
        stems.model_residual_db < -100.0,
        "the parts do not sum to the mix: {:.1} dB",
        stems.model_residual_db
    );

    // Peaks are reported, and only the weakest claim is asserted: that some audio came back. A
    // silent stem is not a bug: an instrumental track has no vocals, and the reference agrees to
    // within 7e-6 on the track this usually runs against.
    let mut loudest = 0.0f32;
    for (name, audio) in &stems.shipped {
        assert_eq!(audio.frames(), mix.frames(), "{name} is the wrong length");
        let peak = audio
            .data
            .iter()
            .flat_map(|c| c.iter())
            .fold(0.0f32, |m, v| m.max(v.abs()));
        println!("  {name:<10} peak {peak:.4}");
        loudest = loudest.max(peak);
    }
    assert!(loudest > 1e-3, "every shipped stem came back silent");
}

/// What half precision costs BS-RoFormer, which differs from what it costs htdemucs.
///
/// Every RoFormer null test runs at float32, and the hybrid ships float16. This model has 24
/// attention layers against htdemucs's 5, and half-precision error accumulates with depth, so the
/// same track goes through the same arrangement at both precisions and the vocals are nulled
/// against each other.
///
/// ```text
/// STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with both artefacts> \
/// cargo test --release -p stemd-core --test separating -- --ignored \
///     what_half_precision --nocapture
/// ```
#[cfg(feature = "research")]
#[test]
#[ignore]
fn what_half_precision_costs_the_roformer() {
    let (Some(pcm), Some(dir)) = (env_dir("STEMD_AB_PCM"), env_dir("STEMD_AB_MLX")) else {
        return;
    };

    let mix = read_pcm(&pcm);
    let run = |precision: Precision| {
        let mut sep = HybridSeparator::roformer_and_demucs(
            &dir,
            "bs_roformer_viperx",
            "htdemucs_ft",
            HybridConfig {
                vocals_precision: precision,
                drums_precision: precision,
                ..HybridConfig::default()
            },
        )
        .expect("loading the quality tier");
        let started = Instant::now();
        let stems = sep
            .separate(&mix, Parts::DEFAULT, &Silent)
            .expect("separating");
        (stems, started.elapsed().as_secs_f64())
    };

    let (from_half, half_secs) = run(Precision::F16);
    let (from_full, full_secs) = run(Precision::F32);
    println!(
        "f16 {half_secs:.1} s, f32 {full_secs:.1} s ({:.2}x)",
        full_secs / half_secs
    );

    for (name, a) in &from_full.shipped {
        let b = from_half
            .shipped
            .iter()
            .find(|(other, _)| other == name)
            .map(|(_, audio)| audio)
            .unwrap_or_else(|| panic!("half precision ships no {name}"));
        let (mut error, mut signal) = (0.0f64, 0.0f64);
        for (ca, cb) in a.data.iter().zip(&b.data) {
            for (x, y) in ca.iter().zip(cb) {
                error += f64::from(x - y).powi(2);
                signal += f64::from(*x).powi(2);
            }
        }
        let null = 10.0 * (error / signal).log10();
        println!("  {name:<10} f16 against f32: {null:.1} dB");

        // The tier exists for a 1.86 dB vocals gain over Balanced. Half precision must cost well
        // under that, or Quality would lose the vocals gain it exists for.
        if *name == "vocals" {
            assert!(
                null < -30.0,
                "half precision moves the vocals by {null:.1} dB, which is the \
                 same order as the gain this tier exists for"
            );
        }
    }
}

/// Whether feeding the drums model the instrumental, in place of the mix, reduces the vocals that
/// bleed into the drums. A real track has no ground truth, so this measures how much of the vocals
/// stem sits inside the drums stem, as a least-squares projection.
// Chaining changes little. The vocals come from the identical forward pass in both pipelines, and
// harmonics is `mix - vocals - drums` either way, so only the drums estimate changes, and with it
// the line between drums and harmonics. Whatever bleeds into drums appears inverted in harmonics,
// by the same subtraction.
//
//     STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with both artefacts> \
//     cargo test --release -p stemd-core --test separating -- --ignored \
//         chaining --nocapture
#[cfg(feature = "research")]
#[test]
#[ignore]
fn chaining_the_halves_against_running_them_side_by_side() {
    let (Some(pcm), Some(dir)) = (env_dir("STEMD_AB_PCM"), env_dir("STEMD_AB_MLX")) else {
        return;
    };

    let mix = read_pcm(&pcm);

    // Both tiers have two halves and both can chain, but the question differs. Quality's vocals
    // half is the better model, so it passes on a cleaner instrumental than demucs would;
    // Balanced's vocals half is demucs's own specialist, and chaining there passes one demucs
    // model's mistakes to another. `STEMD_AB_VOCALS=htdemucs_ft` selects the second case.
    let vocals_from = std::env::var("STEMD_AB_VOCALS").unwrap_or_else(|_| "roformer".into());
    println!("vocals half: {vocals_from}");

    let run = |cascade: bool| {
        let config = HybridConfig {
            cascade,
            ..HybridConfig::default()
        };
        let mut sep = if vocals_from == "htdemucs_ft" {
            HybridSeparator::demucs_specialists(&dir, "htdemucs_ft", config)
        } else {
            HybridSeparator::roformer_and_demucs(&dir, "bs_roformer_viperx", "htdemucs_ft", config)
        }
        .expect("loading the tier");
        sep.separate(&mix, Parts::DEFAULT, &Silent)
            .expect("separating")
    };

    let side_by_side = run(false);
    let chained = run(true);

    let stem = |stems: &stemd_core::Stems, want: &str| {
        stems
            .shipped
            .iter()
            .find(|(name, _)| *name == want)
            .map(|(_, audio)| audio.clone())
            .unwrap_or_else(|| panic!("no {want} stem"))
    };
    // Drums is not shipped, since the player rebuilds it, so rebuild it the way the player does;
    // that is also the only way to see it here.
    let derived_drums = |stems: &stemd_core::Stems| {
        let (h, v) = (stem(stems, "harmonics"), stem(stems, "vocals"));
        Audio::new(
            (0..mix.channels())
                .map(|c| {
                    (0..mix.frames())
                        .map(|i| mix.data[c][i] - h.data[c][i] - v.data[c][i])
                        .collect()
                })
                .collect(),
            mix.sample_rate,
        )
    };

    // Compare vocals first: if they differ, the drums comparison below no longer isolates the
    // effect of chaining.
    let (a, b) = (stem(&side_by_side, "vocals"), stem(&chained, "vocals"));
    let (mut error, mut signal) = (0.0f64, 0.0f64);
    for (ca, cb) in a.data.iter().zip(&b.data) {
        for (x, y) in ca.iter().zip(cb) {
            error += f64::from(x - y).powi(2);
            signal += f64::from(*x).powi(2);
        }
    }
    println!(
        "vocals, chained against side-by-side: {:.1} dB",
        10.0 * (error / signal).log10()
    );
    assert!(
        error == 0.0,
        "chaining changed the vocals, which it cannot do: they are the same \
         forward pass over the same audio"
    );

    // How much of the vocal is inside the drums, as the scalar multiple of the vocals stem that
    // best explains it, relative to the drums' own energy, so it reads as "the drums stem is N dB
    // of vocal".
    let bleed = |drums: &Audio, vocals: &Audio| {
        let (mut dot, mut vv, mut dd) = (0.0f64, 0.0f64, 0.0f64);
        for (cd, cv) in drums.data.iter().zip(&vocals.data) {
            for (d, v) in cd.iter().zip(cv) {
                dot += f64::from(*d) * f64::from(*v);
                vv += f64::from(*v).powi(2);
                dd += f64::from(*d).powi(2);
            }
        }
        let alpha = if vv > 0.0 { dot / vv } else { 0.0 };
        (alpha, 10.0 * (alpha.powi(2) * vv / dd).log10())
    };

    for (label, stems) in [("side by side", &side_by_side), ("chained", &chained)] {
        let drums = derived_drums(stems);
        let (alpha, db) = bleed(&drums, &stem(stems, "vocals"));
        println!("  {label:<13} vocal in drums: alpha {alpha:+.4}  ({db:.1} dB of the drums)");
    }

    // Also check whether chaining moved the drums at all. If the two drums estimates null deeply,
    // chaining makes no measurable difference.
    let (a, b) = (derived_drums(&side_by_side), derived_drums(&chained));
    let (mut error, mut signal) = (0.0f64, 0.0f64);
    for (ca, cb) in a.data.iter().zip(&b.data) {
        for (x, y) in ca.iter().zip(cb) {
            error += f64::from(x - y).powi(2);
            signal += f64::from(*x).powi(2);
        }
    }
    println!(
        "drums, chained against side-by-side: {:.1} dB",
        10.0 * (error / signal).log10()
    );
}

/// BS PolarFormer against the viperx BS-RoFormer as the Quality tier's vocals
/// half, on a real track.
///
/// Run it on transient-heavy material, where a weaker vocals model puts ticks on the vocals fader.
///
/// ```text
/// STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with both roformers and htdemucs_ft> \
/// cargo test --release -p stemd-core --test separating -- --ignored \
///     polarformer_against --nocapture
/// ```
#[cfg(feature = "research")]
#[test]
#[ignore]
fn polarformer_against_the_viperx_roformer() {
    let (Some(pcm), Some(dir)) = (env_dir("STEMD_AB_PCM"), env_dir("STEMD_AB_MLX")) else {
        return;
    };
    let mix = read_pcm(&pcm);
    let seconds = mix.frames() as f64 / f64::from(mix.sample_rate);
    println!("{seconds:.1} s of audio");

    let run = |vocals: &str| {
        let mut sep = HybridSeparator::roformer_and_demucs(
            &dir,
            vocals,
            "htdemucs_ft",
            HybridConfig::default(),
        )
        .unwrap_or_else(|e| panic!("loading {vocals}: {e:#}"));
        let started = Instant::now();
        let stems = sep
            .separate(&mix, Parts::DEFAULT, &Silent)
            .expect("separating");
        (stems, started.elapsed().as_secs_f64())
    };

    let (viperx, viperx_secs) = run("bs_roformer_viperx");
    let (polar, polar_secs) = run("bs_polarformer");

    let pick = |stems: &stemd_core::Stems, want: &str| {
        stems
            .shipped
            .iter()
            .find(|(n, _)| *n == want)
            .map(|(_, a)| a.clone())
            .expect("a shipped stem")
    };
    // Shared content as squared cosine similarity, which is symmetric; see
    // `what_each_preset_leaves_in_the_vocals`. It is measured against the drums a player rebuilds,
    // so it shows the tick residue as a player hears it.
    let shared = |a: &Audio, b: &Audio| {
        let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
        for (ca, cb) in a.data.iter().zip(&b.data) {
            for (x, y) in ca.iter().zip(cb) {
                dot += f64::from(*x) * f64::from(*y);
                aa += f64::from(*x).powi(2);
                bb += f64::from(*y).powi(2);
            }
        }
        (10.0 * (dot.powi(2) / (aa * bb)).log10(), aa)
    };

    println!(
        "  {:<12}{:>9}{:>12}{:>20}",
        "vocals half", "seconds", "vocals rms", "shared with drums"
    );
    let mut vocals = Vec::new();
    for (label, stems, elapsed) in [
        ("viperx", &viperx, viperx_secs),
        ("polarformer", &polar, polar_secs),
    ] {
        let (h, v) = (pick(stems, "harmonics"), pick(stems, "vocals"));
        let drums = Audio::new(
            (0..mix.channels())
                .map(|c| {
                    (0..mix.frames())
                        .map(|i| mix.data[c][i] - h.data[c][i] - v.data[c][i])
                        .collect()
                })
                .collect(),
            mix.sample_rate,
        );
        let (ticks, energy) = shared(&v, &drums);
        let rms = (energy / (mix.frames() * mix.channels()) as f64).sqrt();
        println!("  {label:<12}{elapsed:>9.1}{rms:>12.4}{ticks:>17.1} dB");
        vocals.push(v);
    }
    println!(
        "\n  polarformer is {:.2}x faster ({:.1}x realtime against {:.1}x)",
        viperx_secs / polar_secs,
        seconds / polar_secs,
        seconds / viperx_secs
    );

    let (agreement, _) = shared(&vocals[0], &vocals[1]);
    println!("  the two vocals share {agreement:.1} dB of content");

    // What each model left behind. A real track has no ground-truth vocal, but each model's vocals
    // stem is a template for what the other missed: how much of each model's vocals shows up in the
    // other's harmonics. The vocals-against-drums figure above answers a different question.
    println!();
    for (label, stems, probe) in [
        ("viperx", &viperx, &vocals[1]),
        ("polarformer", &polar, &vocals[0]),
    ] {
        let (leak, _) = shared(&pick(stems, "harmonics"), probe);
        println!("  {label:<12} harmonics holds {leak:.1} dB of the other's vocals");
    }

    // Two different vocals models never agree perfectly. The useful bound is the lower one: both
    // separate the same thing, so low agreement means one of them is wrong.
    assert!(
        agreement > -6.0,
        "the two vocals models agree on only {agreement:.1} dB of content, \
         which is too little for two models doing the same job: one is wrong"
    );
}

/// Quality's vocals come from BS-RoFormer and differ from Balanced's. Two architectures make
/// different errors, so their vocals stems must not null.
// A wrong vocals stem has the right length, sums with the others to the mix and sounds like vocals,
// so nothing upstream of the audio would notice if the two tiers returned the same stems. This
// nulls the two vocals stems against each other.
//
//     STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with both artefacts> \
//     cargo test --release -p stemd-core --test separating -- --ignored \
//         the_quality_tier --nocapture
#[cfg(feature = "research")]
#[test]
#[ignore]
fn the_quality_tier_does_not_quietly_ship_balanced_vocals() {
    let (Some(pcm), Some(dir)) = (env_dir("STEMD_AB_PCM"), env_dir("STEMD_AB_MLX")) else {
        return;
    };

    let mix = read_pcm(&pcm);
    let seconds = mix.frames() as f64 / f64::from(mix.sample_rate);
    println!("{seconds:.1} s of audio");

    let stems_of = |mut sep: HybridSeparator| {
        let started = Instant::now();
        let stems = sep
            .separate(&mix, Parts::DEFAULT, &Silent)
            .expect("separating");
        (stems, started.elapsed().as_secs_f64())
    };

    let (quality, quality_secs) = stems_of(
        HybridSeparator::roformer_and_demucs(
            &dir,
            "bs_roformer_viperx",
            "htdemucs_ft",
            HybridConfig::default(),
        )
        .expect("loading the quality tier"),
    );
    let (balanced, balanced_secs) = stems_of(
        HybridSeparator::demucs_specialists(&dir, "htdemucs_ft", HybridConfig::default())
            .expect("loading the balanced tier"),
    );
    println!("quality {quality_secs:.1} s, balanced {balanced_secs:.1} s");

    for (name, a) in &quality.shipped {
        let b = balanced
            .shipped
            .iter()
            .find(|(other, _)| other == name)
            .map(|(_, audio)| audio)
            .unwrap_or_else(|| panic!("balanced ships no {name}"));

        let (mut error, mut signal, mut other) = (0.0f64, 0.0f64, 0.0f64);
        let (mut peak, mut other_peak) = (0.0f32, 0.0f32);
        for (ca, cb) in a.data.iter().zip(&b.data) {
            for (x, y) in ca.iter().zip(cb) {
                error += f64::from(x - y).powi(2);
                signal += f64::from(*x).powi(2);
                other += f64::from(*y).powi(2);
                peak = peak.max(x.abs());
                other_peak = other_peak.max(y.abs());
            }
        }
        let count = (a.frames() * a.channels()) as f64;
        let null = 10.0 * (error / signal).log10();
        println!(
            "  {name:<10} quality peak {peak:.4} rms {:.4} | balanced peak {other_peak:.4} \
             rms {:.4} | null {null:.1} dB",
            (signal / count).sqrt(),
            (other / count).sqrt(),
        );

        if *name == "vocals" {
            // On an instrumental, one tier may return silence and the other not, the largest
            // possible difference; the null then goes positive, and the comparison below reads that
            // correctly. Only if both came back silent is there nothing to tell apart.
            assert!(
                peak.max(other_peak) > 1e-2,
                "neither tier returned any vocals ({peak:.4} and {other_peak:.4} peak), \
                 so this track cannot tell them apart; use one with vocals"
            );
            assert!(
                null > -40.0,
                "the two tiers' vocals agree to {null:.1} dB, which two different \
                 architectures cannot do: Quality is not running the RoFormer"
            );
        }
    }

    // Measure what the player holds. `drums` is not shipped, so measuring it means doing what the
    // client does. Both tiers take drums from the same htdemucs_ft specialist and ship harmonics as
    // the remainder, so this comes back identical: the vocals estimate appears once in `vocals` and
    // once negated in `harmonics`.
    let rebuild = |stems: &stemd_core::Stems| -> Vec<Vec<f32>> {
        let of = |name: &str| {
            stems
                .shipped
                .iter()
                .find(|(stem, _)| *stem == name)
                .map(|(_, audio)| audio)
                .unwrap_or_else(|| panic!("no {name} shipped"))
        };
        let (harmonics, vocals) = (of("harmonics"), of("vocals"));
        (0..mix.channels())
            .map(|c| {
                (0..mix.frames())
                    .map(|i| mix.data[c][i] - harmonics.data[c][i] - vocals.data[c][i])
                    .collect()
            })
            .collect()
    };

    let (a, b) = (rebuild(&quality), rebuild(&balanced));
    let (mut error, mut signal) = (0.0f64, 0.0f64);
    for (ca, cb) in a.iter().zip(&b) {
        for (x, y) in ca.iter().zip(cb) {
            error += f64::from(x - y).powi(2);
            signal += f64::from(*x).powi(2);
        }
    }
    let null = 10.0 * (error / signal).log10();
    println!("  drums (rebuilt by the client) quality vs balanced: {null:.1} dB");
    assert!(
        null < -100.0,
        "the two tiers rebuild different drums ({null:.1} dB), so they differ \
         somewhere other than the vocals estimate"
    );
}

/// `stemd_roformer` against the former Quality tier, BS PolarFormer chained into
/// `htdemucs_ft`, on MUSDB test tracks.
///
/// Prints global SDR per part, averaged over tracks, and seconds per track.
///
/// ```text
/// STEMD_AB_MUSDB=<dir> STEMD_AB_MLX=<dir with bs_polarformer, htdemucs_ft and
/// stemd_roformer> cargo test --release -p stemd-core --test separating \
///     -- --ignored stemd_roformer --nocapture
/// ```
///
/// `STEMD_AB_MUSDB` holds `NN.mix.pcm`, `NN.drums.pcm` and `NN.vocals.pcm`: raw
/// f32le stereo at 44.1 kHz, with the mix the sum of all four stems.
#[cfg(feature = "research")]
#[test]
#[ignore]
fn stemd_roformer_against_quality() {
    let (Some(tracks), Some(dir)) = (env_dir("STEMD_AB_MUSDB"), env_dir("STEMD_AB_MLX")) else {
        return;
    };
    let read = |path: PathBuf| {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Audio::from_interleaved(&bytes, PcmFormat::F32le, 2, 44100).expect("f32le stereo")
    };
    let mut names: Vec<String> = std::fs::read_dir(&tracks)
        .expect("reading the track directory")
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            name.strip_suffix(".mix.pcm").map(str::to_owned)
        })
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no NN.mix.pcm in {}", tracks.display());

    let sdr = |truth: &Audio, est: &Audio| {
        let (mut signal, mut error) = (0.0f64, 0.0f64);
        for (t, e) in truth.data.iter().zip(&est.data) {
            for (x, y) in t.iter().zip(e) {
                signal += f64::from(*x).powi(2);
                error += f64::from(x - y).powi(2);
            }
        }
        10.0 * (signal / error).log10()
    };
    let score = |label: &str, sep: &mut dyn Separate| {
        let (mut sums, mut seconds) = ([0.0f64; 3], 0.0f64);
        for name in &names {
            let mix = read(tracks.join(format!("{name}.mix.pcm")));
            let drums = read(tracks.join(format!("{name}.drums.pcm")));
            let vocals = read(tracks.join(format!("{name}.vocals.pcm")));
            let harmonics = remainder(&mix, [&drums, &vocals]);

            let started = Instant::now();
            let stems = sep
                .separate(&mix, Parts::DEFAULT, &Silent)
                .expect("separating");
            seconds += started.elapsed().as_secs_f64();
            let of = |stem: &str| {
                stems
                    .stem(stem)
                    .unwrap_or_else(|| panic!("no {stem} shipped"))
            };
            let (est_h, est_v) = (of("harmonics"), of("vocals"));
            // What the player does: drums are the mix less the shipped stems.
            let est_d = remainder(&mix, [est_h, est_v]);
            let row = [
                sdr(&drums, &est_d),
                sdr(&harmonics, est_h),
                sdr(&vocals, est_v),
            ];
            println!(
                "  {label:<8} {name}  drums {:6.2}  harmonics {:6.2}  vocals {:6.2}",
                row[0], row[1], row[2]
            );
            for (s, v) in sums.iter_mut().zip(row) {
                *s += v;
            }
        }
        let n = names.len() as f64;
        let mean = sums.map(|s| s / n);
        println!(
            "RESULT {label:<8} drums {:.2} harmonics {:.2} vocals {:.2} mean {:.2}  {:.1} s per track",
            mean[0],
            mean[1],
            mean[2],
            mean.iter().sum::<f64>() / 3.0,
            seconds / n
        );
    };

    let on = stemd_core::Accelerator::detect();
    let roformer = Precision::preferred(stemd_core::Family::Roformer, on);
    let demucs = Precision::preferred(stemd_core::Family::HtDemucs, on);

    let mut stemd = stemd_core::RoformerSeparator::load(
        &dir,
        "stemd_roformer",
        roformer,
        stemd_mlx::apply::DEFAULT_OVERLAP,
    )
    .expect("loading stemd_roformer");
    score("stemd", &mut stemd);
    drop(stemd);

    let mut quality = HybridSeparator::roformer_and_demucs(
        &dir,
        "bs_polarformer",
        "htdemucs_ft",
        HybridConfig {
            vocals_precision: roformer,
            drums_precision: demucs,
            cascade: true,
            ..HybridConfig::default()
        },
    )
    .expect("loading the quality tier");
    score("quality", &mut quality);
}
