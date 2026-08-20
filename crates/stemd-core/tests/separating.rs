//! The backend on a real track, at both precisions.
//!
//! Ignored by default: it needs the weights and a track to run on, neither of
//! which belongs in a repository.
//!
//! ```text
//! STEMD_AB_PCM=track.pcm STEMD_AB_MLX=/path/holding/htdemucs.safetensors \
//! cargo test --release -p stemd-core --test separating -- --ignored --nocapture
//! ```
//!
//! `STEMD_AB_PCM` is raw interleaved stereo 16-bit PCM at 44.1 kHz: the body of a
//! WAV with its header removed, and one of the two formats the server already
//! accepts, so this needs no decoder of its own.

use std::path::PathBuf;
use std::time::Instant;

use stemd_core::{
    Audio, HybridConfig, HybridSeparator, MlxConfig, MlxSeparator, PcmFormat, Precision, Separate,
    Silent,
};

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

    // Which artefact, so a locally converted single-model one can go through
    // the same harness rather than being measured by multiplying and hoping.
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

        //  Every distinct fraction a polling client would have seen, watched on the run
        //  that is happening anyway rather than in a separation of its own.
        let seen = std::sync::Mutex::new(Vec::<f32>::new());
        let watch = |p: stemd_core::Progress| {
            let mut seen = seen.lock().expect("not poisoned");
            if seen.last() != Some(&p.fraction) {
                seen.push(p.fraction);
            }
        };

        let started = Instant::now();
        let stems = mlx.separate(&mix, &watch).expect("separating");
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

    // htdemucs measured -33.9 dB against TorchScript and against this, at both
    // precisions. A drift of several dB would mean the weights or the
    // arithmetic around them moved, not that the track is different: this is
    // the model's own error, which is a property of the model.
    if artefact == "htdemucs" {
        assert!(
            stems.model_residual_db < -25.0,
            "the model residual is {:.1} dB, well above where htdemucs sits",
            stems.model_residual_db
        );
    }

    // Half precision has to stay well under the model's own error, or it is not
    // a precision change, it is a different model. Measured at -53.9 dB on the
    // stems that ship, which is where the TorchScript swap landed too.
    assert!(
        null < stems.model_residual_db - 5.0,
        "half precision differs by {null:.1} dB, too close to the model's own \
         residual of {:.1} dB",
        stems.model_residual_db
    );

    // 13.9x realtime measured in full precision, with plenty of margin here:
    // this is a floor against a regression like the frame-by-frame inverse
    // transform that once cost forty per cent of the forward pass, not a
    // benchmark to defend.
    let floor = if artefact == "htdemucs" { 4.0 } else { 1.0 };
    assert!(
        realtime > floor,
        "separating ran at {realtime:.1}x realtime, far below what was measured"
    );
}

/// The hybrid: BS-RoFormer for vocals, htdemucs_ft's drums specialist for the
/// rest, on a real track. This measures what it costs, not how good it is: see
/// docs/evaluation.md for the latter.
///
/// ```text
/// STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with both artefacts> \
/// STEMD_AB_HYBRID=1 cargo test --release -p stemd-core --test separating \
///     -- --ignored --nocapture
/// ```
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

    // Every distinct fraction the bar would have shown, so "it sat at 10% and
    // jumped" is a thing this test can see.
    let seen = std::sync::Mutex::new(Vec::<f32>::new());
    let watch = |p: stemd_core::Progress| {
        let mut seen = seen.lock().expect("not poisoned");
        if seen.last() != Some(&p.fraction) {
            seen.push(p.fraction);
        }
    };

    let started = Instant::now();
    let stems = hybrid.separate(&mix, &watch).expect("separating");
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

    // The parts are built to sum, so the residual is float noise rather than a
    // measurement. Asserting it is tiny is asserting the arithmetic, which is
    // still worth doing: a sign error in the remainder would show up here and
    // nowhere else until someone listened.
    println!("sum residual: {:.1} dB", stems.model_residual_db);
    assert!(
        stems.model_residual_db < -100.0,
        "the parts do not sum to the mix: {:.1} dB",
        stems.model_residual_db
    );

    // Peaks are reported and only the weakest claim is asserted: that
    // *something* came back. A silent stem is not a bug -- an instrumental
    // track has no vocals, and the reference agrees to within 7e-6 on the one
    // this is usually run against, which cost an hour of looking for a fault
    // that was not there.
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

/// What half precision costs BS-RoFormer, which is not what it costs htdemucs.
///
/// Every RoFormer null test runs at float32 and the hybrid ships float16. This
/// model is 24 attention layers deep against htdemucs's 5, and half-precision
/// error accumulates with depth, so the same track goes through the same
/// arrangement at both precisions with the vocals nulled against each other.
///
/// ```text
/// STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with both artefacts> \
/// cargo test --release -p stemd-core --test separating -- --ignored \
///     what_half_precision --nocapture
/// ```
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
        let stems = sep.separate(&mix, &Silent).expect("separating");
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

        // The tier exists for a 1.86 dB gain on vocals over Balanced. Half
        // precision has to sit well under that or it is spending the gain to
        // buy back the time, which would make the whole tier pointless.
        if *name == "vocals" {
            assert!(
                null < -30.0,
                "half precision moves the vocals by {null:.1} dB, which is the \
                 same order as the gain this tier exists for"
            );
        }
    }
}

/// Chaining the halves: does handing the drums model the instrumental beat
/// handing it the mixture?
///
/// What chaining can move is narrow. Vocals are the identical forward pass in
/// both pipelines, and harmonics is `mix - vocals - drums` either way, so the
/// only thing that changes is the drums estimate, and therefore where the line
/// between drums and harmonics falls.
///
/// There is no ground truth on a real track, so this measures the artefact rather
/// than the score: how much of the vocals stem sits inside the drums stem, as a
/// least-squares projection. Whatever bleeds into drums appears inverted in
/// harmonics, which is the same subtraction.
///
/// ```text
/// STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with both artefacts> \
/// cargo test --release -p stemd-core --test separating -- --ignored \
///     chaining --nocapture
/// ```
#[test]
#[ignore]
fn chaining_the_halves_against_running_them_side_by_side() {
    let (Some(pcm), Some(dir)) = (env_dir("STEMD_AB_PCM"), env_dir("STEMD_AB_MLX")) else {
        return;
    };

    let mix = read_pcm(&pcm);

    // Both tiers are two halves and both can chain, but they are not the same
    // question. Quality's vocals half is the better model, so what it hands on
    // is a cleaner instrumental than demucs would make; Balanced's vocals half
    // is demucs's own specialist, and chaining there feeds one demucs model's
    // mistakes to another. `STEMD_AB_VOCALS=htdemucs_ft` asks the second.
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
        sep.separate(&mix, &Silent).expect("separating")
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
    // Drums is not shipped -- the player rebuilds it -- so rebuild it the same
    // way the player does, which is also the only way to see it from here.
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

    // Vocals first, because if these differ at all the rest of the comparison
    // is measuring something other than what it claims to.
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

    // How much of the vocal is inside the drums, as the scalar multiple of the
    // vocals stem that best explains it. Reported against the drums' own
    // energy, so it is "the drums stem is N dB of vocal".
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

    // And whether it moved the drums at all. If the two drums estimates null
    // deeply, the whole idea is moot regardless of which is better.
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
/// MUSDB scores the two the same and torch times PolarFormer faster, and neither
/// of those is this: MUSDB is pop and rock, and torch is not the runtime that
/// ships. Transient-heavy material is the case worth choosing, because that is
/// where a vocals model that is slightly wrong puts ticks on the vocals fader.
///
/// ```text
/// STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with both roformers and htdemucs_ft> \
/// cargo test --release -p stemd-core --test separating -- --ignored \
///     polarformer_against --nocapture
/// ```
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
        let stems = sep.separate(&mix, &Silent).expect("separating");
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
    // Shared content, as squared cosine similarity -- symmetric, see
    // `what_each_preset_leaves_in_the_vocals`. Against the drums a player
    // rebuilds, so this is the tick residue as it is actually met.
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

    //  What each model left behind. There is no ground-truth vocal on a real track,
    //  but each model's vocals stem is a template for what the other one missed: how
    //  much of each model's vocals shows up in the other's harmonics. The
    //  vocals-against-drums figure above asks a different question.
    println!();
    for (label, stems, probe) in [
        ("viperx", &viperx, &vocals[1]),
        ("polarformer", &polar, &vocals[0]),
    ] {
        let (leak, _) = shared(&pick(stems, "harmonics"), probe);
        println!("  {label:<12} harmonics holds {leak:.1} dB of the other's vocals");
    }

    // Two vocals models cannot agree perfectly, and if they did it would mean
    // one of them was not running. The useful bound is the other way: they are
    // both meant to be separating the same thing, so a *low* agreement would
    // say one of them is wrong rather than merely different.
    assert!(
        agreement > -6.0,
        "the two vocals models agree on only {agreement:.1} dB of content, \
         which is too little for two models doing the same job: one is wrong"
    );
}

/// Quality's vocals must actually be BS-RoFormer's, not Balanced's under a
/// different name.
///
/// Everything upstream of the audio agrees that they are, and none of it would
/// notice if the two tiers returned the same thing: a wrong vocals stem has the
/// right length, sums with the others to the mix, and sounds like vocals. So this
/// asks the audio, nulling the two vocals stems against each other. Two different
/// architectures are wrong differently and cannot agree closely.
///
/// ```text
/// STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with both artefacts> \
/// cargo test --release -p stemd-core --test separating -- --ignored \
///     the_quality_tier --nocapture
/// ```
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
        let stems = sep.separate(&mix, &Silent).expect("separating");
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
            //  Only if both came back silent is there nothing to tell apart. One silent and
            //  the other not is the loudest possible answer, and it is what an instrumental
            //  gives. The null goes positive there: the difference has more energy than the
            //  stem it is measured against, which the comparison below reads correctly.
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

    //  What the player actually holds. `drums` is not shipped, so measuring it means
    //  doing what the client does. Both tiers take drums from the same htdemucs_ft
    //  specialist and ship harmonics as the remainder, so this comes back identical:
    //  the vocals estimate appears once in `vocals` and once negated in `harmonics`.
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
