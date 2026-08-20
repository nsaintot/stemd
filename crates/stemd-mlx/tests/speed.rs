//! Where the time goes.
//!
//! The Python implementation separates a two-minute track in about 6.7 s on this
//! machine, one segment at a time, so a Rust port had no structural excuse to be
//! slower. When it was, the question was which half: the model on one segment, or
//! the segmenting and overlap-add around it. Both had something in them.
//!
//! Ignored: these need the weights and they are measurements, not assertions.
//! `STEMD_MLX_WEIGHTS=... cargo test --release -p stemd-mlx --test speed --
//! --ignored --nocapture`

mod common;

use std::time::Instant;

use common::{null_db, one_at_a_time, roformer_weights, weights};
use mlx_rs::Array;
use stemd_mlx::Precision;
use stemd_mlx::apply::{DEFAULT_OVERLAP, over_track};
use stemd_mlx::htdemucs::{Config, HtDemucs};

/// A deterministic, non-trivial signal. Silence would let a lazy graph skip
/// work that real audio makes it do.
fn noise(samples: usize) -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..samples)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 8_388_608.0 - 1.0
        })
        .collect()
}

#[test]
#[ignore]
fn one_segment_through_the_model() {
    let _gpu = one_at_a_time();
    let Some(w) = weights() else { return };

    let config = Config::default();
    let (segment, rate) = (config.training_length(), config.sample_rate);
    let model = HtDemucs::load(&w, "model_0", config).expect("loading htdemucs");

    let input = Array::from_slice(&noise(2 * segment as usize), &[1, 2, segment]);

    // The first pass builds kernels and allocates; it is not the number.
    let warm = model.forward(&input).expect("warm-up");
    mlx_rs::transforms::eval([&warm]).expect("evaluating");

    let mut best = f64::MAX;
    for _ in 0..3 {
        let started = Instant::now();
        let out = model.forward(&input).expect("forward");
        mlx_rs::transforms::eval([&out]).expect("evaluating");
        best = best.min(started.elapsed().as_secs_f64());
    }
    println!(
        "one segment ({:.1} s of audio): {best:.3} s",
        f64::from(segment) / f64::from(rate)
    );
}

/// Both halves of the half-precision trade, on the same segment.
///
/// Speed is only half the question. The other half is what it costs, and the
/// bar is not "sounds fine": htdemucs's own residual: how far its four sources
/// miss summing back to the mix: is -33.9 dB, and swapping the runtime from
/// TorchScript to this one measured -54 dB. A precision change has to land
/// under that same bar to be a precision change rather than a different model.
#[test]
#[ignore]
fn what_half_precision_costs_and_buys() {
    let _gpu = one_at_a_time();
    let Some(w) = weights() else { return };

    let segment = Config::default().training_length();
    let input = Array::from_slice(&noise(2 * segment as usize), &[1, 2, segment]);

    let time = |model: &HtDemucs| -> (Array, f64) {
        let warm = model.forward(&input).expect("warm-up");
        mlx_rs::transforms::eval([&warm]).expect("evaluating");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let started = Instant::now();
            let out = model.forward(&input).expect("forward");
            mlx_rs::transforms::eval([&out]).expect("evaluating");
            best = best.min(started.elapsed().as_secs_f64());
        }
        (warm, best)
    };

    let full = HtDemucs::load(&w, "model_0", Config::default()).expect("loading f32");
    let (from_full, full_secs) = time(&full);

    let half_weights = w.cast(Precision::F16).expect("casting the weights");
    let half = HtDemucs::load(
        &half_weights,
        "model_0",
        Config {
            precision: Precision::F16,
            ..Config::default()
        },
    )
    .expect("loading f16");
    let (from_half, half_secs) = time(&half);

    let null = null_db(&from_half, &from_full);
    println!(
        "f32: {full_secs:.3} s   f16: {half_secs:.3} s   ({:.2}x)",
        full_secs / half_secs
    );
    println!("f16 against f32: {null:.1} dB");
}

#[test]
#[ignore]
fn a_whole_track_through_the_segmenting() {
    let _gpu = one_at_a_time();
    let Some(w) = weights() else { return };

    let config = Config::default();
    let rate = config.sample_rate;
    let model = HtDemucs::load(&w, "model_0", config).expect("loading htdemucs");

    let seconds = 120;
    let frames = seconds * rate;
    let input = Array::from_slice(&noise(2 * frames as usize), &[1, 2, frames]);

    let mut segments = 0usize;
    let started = Instant::now();
    let out = over_track(&model, &input, DEFAULT_OVERLAP, &mut |_, total| {
        segments = total;
        Ok(())
    })
    .expect("separating");
    mlx_rs::transforms::eval([&out]).expect("evaluating");
    let elapsed = started.elapsed().as_secs_f64();

    println!(
        "{seconds} s over {segments} segments: {elapsed:.2} s \
         ({:.1}x realtime, {:.3} s per segment)",
        f64::from(seconds) / elapsed,
        elapsed / segments as f64
    );
}

/// Where BS-RoFormer's time goes, on one full chunk.
///
/// ```text
/// spectrogram    0.001 s
/// + band split   0.010 s
/// transformer    4.716 s
/// mask           0.075 s
/// whole forward  4.341 s
/// ```
///
/// The band split and the mask estimators are sixty-two small matmuls each and
/// look like launch overhead; together they are 85 ms of 4.3 s. The cost is the
/// transformer stack, and it is arithmetic: twelve blocks of two transformers over
/// 801 time steps and 62 bands at width 512 is roughly 8.5 TFLOP for eight seconds
/// of audio, against htdemucs's fifteen times less.
///
/// Those figures are the rotary 512-wide artefact. The shape of the answer is what
/// they are kept for; the seconds move with whichever artefact
/// `STEMD_MLX_ROFORMER` points at.
///
/// `whole forward` is smaller than `transformer` alone. A stage timed alone, on a
/// warm pre-computed input with nothing else in flight, is not that stage inside a
/// whole forward. Deriving the tail by subtraction gave -0.46 s here and 2.57 s on
/// CUDA off the same code. The tail is measured now, through [`BsRoformer::apply`],
/// and what is left over is printed as a residual.
#[test]
#[ignore]
fn where_roformer_spends_its_time() {
    let _gpu = one_at_a_time();
    let Some(w) = roformer_weights() else { return };

    //  Asked of the tensors, exactly as the null tests do it. A hardcoded `default()`
    //  outlived the artefact it described: a 512-wide config against a 256-wide file
    //  fails to load everywhere.
    //
    //  The cast is the other half: `precision` alone only says what the activations
    //  start as, and mlx promotes, so f32 weights meeting it compute in f32. `embed`
    //  does not cast and `forward` does, so timing `transform` on what `embed` returns
    //  measures a precision this model never runs at.
    let precision = Precision::F16;
    let dtype = mlx_rs::Dtype::Float16;
    let config = stemd_mlx::roformer::Config {
        precision,
        ..stemd_mlx::roformer::Config::of(&w)
    };
    let (chunk, rate) = (config.chunk, config.sample_rate);
    let cast = w.cast(precision).expect("casting the artefact");
    let model = stemd_mlx::roformer::BsRoformer::load(&cast, config).expect("loading");
    let input = Array::from_slice(&noise(2 * chunk as usize), &[1, 2, chunk]);

    let best = |label: &str, run: &dyn Fn() -> Array| -> f64 {
        let warm = run();
        mlx_rs::transforms::eval([&warm]).expect("warm-up");
        let mut best = f64::MAX;
        for _ in 0..3 {
            let started = Instant::now();
            let out = run();
            mlx_rs::transforms::eval([&out]).expect("evaluating");
            best = best.min(started.elapsed().as_secs_f64());
        }
        println!("  {label:<14} {best:.3} s");
        best
    };

    let spectrogram = best("spectrogram", &|| {
        model.spectrogram(&input).expect("stft").0
    });
    // Cast to what `forward` runs at, so `transform` below is timed on the same
    // dtype the model actually gives it.
    let embedded = model
        .embed(&input)
        .expect("embed")
        .as_dtype(dtype)
        .expect("cast");
    let embed = best("+ band split", &|| {
        model
            .embed(&input)
            .expect("embed")
            .as_dtype(dtype)
            .expect("cast")
    });
    let transformed = model.transform(&embedded).expect("transform");
    let transform = best("transformer", &|| {
        model.transform(&embedded).expect("transform")
    });
    let masked = model.mask(&transformed).expect("mask");
    let mask = best("mask", &|| model.mask(&transformed).expect("mask"));

    // Measured, not subtracted. This is the whole reason `apply` is a method:
    // the tail used to be `whole - embed - transform - mask`, which reported
    // 2.57 s on one machine and -0.46 s on another off the same code.
    let (real, imag) = model.spectrogram(&input).expect("stft");
    let tail = best("tail (mask x + istft)", &|| {
        model.apply(&masked, &real, &imag, chunk).expect("apply")
    });
    let whole = best("whole forward", &|| model.forward(&input).expect("forward"));

    println!("\n  band split alone ~{:.3} s", embed - spectrogram);

    // Every stage above is timed on a warm, pre-computed input with nothing else
    // in flight, and the whole forward is not. The residual is what that costs,
    // and it is a diagnostic rather than a stage: it says how far the
    // decomposition can be trusted, and nothing about where any time went.
    //
    // `embed` already contains `spectrogram`, so it is not added twice.
    let residual = whole - (embed + transform + mask + tail);
    println!(
        "  stages {:.3} s against a whole forward of {whole:.3} s, residual {residual:+.3} s \
         ({:+.0}%)",
        embed + transform + mask + tail,
        100.0 * residual / whole
    );
    assert!(
        residual > -0.1 * whole,
        "the stages sum to more than the whole forward they are stages of \
         ({:.3} s against {whole:.3} s). Timing each one alone and adding them \
         up does not describe this model, so no line above is a finding.",
        embed + transform + mask + tail
    );
    if residual > 0.1 * whole {
        println!(
            "  ^ the stages account for {:.0}% of the forward and the rest is \
             unattributed. Do NOT quote a line above as a share of the whole, and \
             do not name the residual after whichever stage happens to sit last \
             in the pipeline -- that is exactly how a 2 ms tail came to be \
             published as 71% of this model.",
            100.0 * (whole - residual) / whole
        );
    }
    println!(
        "  {:.1} s of audio in {whole:.3} s = {:.1}x realtime for this half alone",
        f64::from(chunk) / f64::from(rate),
        f64::from(chunk) / f64::from(rate) / whole
    );
}
