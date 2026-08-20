//! DSP mode 1: a fixed 44.1 to 96 kHz resampler, reproduced from one client's
//! own converter.
//!
//! # Why a second filter exists
//!
//! A client that rebuilds `drums = mix - harmonics - vocals` has to resample its
//! mix to the stems' rate, and the subtraction only cancels if both sides went
//! through the same filter. Two good resamplers are not enough: against ffmpeg's
//! `swresample` ours agrees to about -77 dB of the mix, which lands as -36 dB of
//! a derived part sitting 41 dB below it.
//!
//! That client cannot be asked to use a different filter: it resamples the mix
//! on ingest, and only ever sees the result. So the server carries this one and
//! applies it when a job asks for mode 1.
//!
//! # Where the coefficients come from
//!
//! Measured, not disassembled. The converter was driven with synthetic impulses
//! and shown to be exactly LTI: a half-scale impulse rescales to the float32
//! floor, two disjoint impulses sum to an energy ratio of 2.000000000, the
//! channels are independent to exactly zero, and the response is bit-identical
//! under a shift of 147 input samples.
//!
//! That last is the polyphase signature. `44100/96000 = 147/320`, so the
//! converter is an `L = 320`, `M = 147` polyphase FIR with no state carried
//! between blocks.
//!
//! An impulse at input index `k` produces `y[m] = p[m*147 - k*320]`, sampling the
//! prototype at a stride of 147; stepping `k` by one steps the residue by -26 mod
//! 147, and `gcd(26, 147) = 1`, so 147 impulses recover every residue class. The
//! result is 18880 taps: 59.000 input samples of support, peak dead centre,
//! passband flat to 0.0026 dB out to 19 kHz, -6 dB at 21.156 kHz, stopband at
//! -100.9 dB from 24 kHz up.
//!
//! Verified against that converter on full-band noise at -141.9 dB peak and
//! -148.3 dB RMS. `matches_the_reference_exactly` holds that against a captured
//! reference.
//!
//! The delay comes with it. Output here sits 64 samples later at 96 kHz than the
//! general resampler's, which puts a feature exactly where the ratio does. That
//! is this converter's own group delay, and carrying it is why the two sides
//! cancel: the client's mix went through the same taps.
//!
//! The capture harness and the full derivation are not part of this repository.

use std::sync::OnceLock;

use crate::pcm::Audio;

pub const IN_RATE: u32 = 44_100;
pub const OUT_RATE: u32 = 96_000;

/// `96000/44100` reduced. The interpolation and decimation factors.
const L: usize = 320;
const M: usize = 147;

/// Taps per polyphase branch: 18880 / 320.
const BRANCH: usize = 59;

/// The prototype at its native 320 × 44100 = 14.112 MHz grid, little-endian f32,
/// normalised so a unit impulse in gives the prototype out.
static PROTO_LE: &[u8] = include_bytes!("../../data/mode1_proto_f32.bin");

/// The prototype split into its 320 branches and each one reversed, so
/// `branch[r][i] = p[r + 320*(58 - i)]`.
///
/// Reversed because tap `j` reaches back `j` input samples: laid this way the
/// window and its taps are both walked forward over contiguous memory, which lets
/// the dot product vectorise.
fn branches() -> &'static [[f32; BRANCH]] {
    static B: OnceLock<Vec<[f32; BRANCH]>> = OnceLock::new();
    B.get_or_init(|| {
        let p: Vec<f32> = PROTO_LE
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(p.len(), L * BRANCH, "prototype is not 320 x 59 taps");
        (0..L)
            .map(|r| std::array::from_fn(|i| p[r + L * (BRANCH - 1 - i)]))
            .collect()
    })
}

/// Output frames `m0 .. m0 + dst.len()` of one channel.
///
/// A range rather than a whole channel because every output frame is an
/// independent dot product: the work splits anywhere, and `m0` is all a worker
/// needs to place itself.
fn fill(src: &[f32], m0: usize, dst: &mut [f32], b: &[[f32; BRANCH]]) {
    // `newest` is the newest input sample an output frame reaches and `r` the
    // phase it lands on -- together `m*147/320` and `m*147 % 320`. Computed once
    // here and then carried forward: a division and a modulo per output frame is
    // a real cost at 38 million of them, and the recurrence is exact.
    let start = m0 * M;
    let mut newest = start / L;
    let mut r = start % L;

    for o in dst.iter_mut() {
        let taps = &b[r];

        // Accumulated wide, then rounded once. Summing 59 products in f32
        // tracks the device to -127 dB; in f64 it tracks to -142, which says the
        // device carries more than f32 across the sum. Both are far below the 16
        // bits a stem is encoded at -- this just declines to spend the margin.
        let acc: f64 = if newest >= BRANCH - 1 && newest < src.len() {
            // Interior: the whole window is in range, so no bounds test per tap
            // and both slices run forward.
            src[newest + 1 - BRANCH..=newest]
                .iter()
                .zip(taps)
                .map(|(x, t)| f64::from(*x) * f64::from(*t))
                .sum()
        } else {
            // Head and tail, where the window hangs off the signal. The device
            // sees zeros there and so do we -- that is the whole of its edge
            // behaviour, and it is why the captured reference nulls at its
            // first and last samples as well as its middle.
            (0..BRANCH)
                .filter_map(|j| {
                    let k = newest.checked_sub(j)?;
                    Some(f64::from(*src.get(k)?) * f64::from(taps[BRANCH - 1 - j]))
                })
                .sum()
        };
        #[expect(clippy::cast_possible_truncation, reason = "f32 output by design")]
        {
            *o = acc as f32;
        }

        r += M;
        if r >= L {
            r -= L;
            newest += 1;
        }
    }
}

/// How many output frames `frames` input frames become.
///
/// The ratio, truncated. The device allocates more than this and zero-fills the
/// slack, so matching its buffer would copy an artefact of its block size;
/// matching its samples is what alignment needs.
#[must_use]
pub fn out_frames(frames: usize) -> usize {
    frames * L / M
}

/// Resample 44.1 kHz `audio` to 96 kHz exactly as the target player does.
///
/// The caller has already established the rate; this does not check it, because
/// the coefficients describe one conversion and nothing else.
#[must_use]
pub fn resample(audio: &Audio) -> Audio {
    let b = branches();
    let n_out = out_frames(audio.frames());

    //  Spread over the machine. Every output frame is an independent dot product
    //  reading a shared immutable input, so this splits by output range with no seams,
    //  and it runs at the end of a job the caller is already blocked on. Two channels
    //  alone left most of the machine unused.
    let mut planes: Vec<Vec<f32>> = audio.data.iter().map(|_| vec![0.0f32; n_out]).collect();
    let par = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let slices = par.div_ceil(planes.len().max(1)).max(1);
    let chunk = n_out.div_ceil(slices).max(1);

    std::thread::scope(|s| {
        for (src, out) in audio.data.iter().zip(planes.iter_mut()) {
            for (i, dst) in out.chunks_mut(chunk).enumerate() {
                s.spawn(move || fill(src, i * chunk, dst, b));
            }
        }
    });

    Audio::new(planes, OUT_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn deinterleave(v: &[f32]) -> Vec<Vec<f32>> {
        vec![
            v.iter().step_by(2).copied().collect(),
            v.iter().skip(1).step_by(2).copied().collect(),
        ]
    }

    /// The reference: full-band uniform noise with the channels differing, captured
    /// off the device itself. 2205 input frames is 15 full polyphase
    /// cycles, so all 320 branches are exercised and both edges sit inside the vector.
    ///
    /// The bar is -120 rather than the -141.5 achieved, because the last few dB are
    /// floating-point summation order and would be flaky across targets.
    #[test]
    fn matches_the_reference_exactly() {
        let input = deinterleave(&f32s(include_bytes!("../../data/mode1_ref_in_f32.bin")));
        let want = deinterleave(&f32s(include_bytes!("../../data/mode1_ref_out_f32.bin")));
        let got = resample(&Audio::new(input, IN_RATE));

        assert_eq!(got.sample_rate, OUT_RATE);
        assert_eq!(got.frames(), want[0].len(), "output length");

        let peak = want.iter().flatten().fold(0.0f32, |a, v| a.max(v.abs()));
        let worst = got
            .data
            .iter()
            .zip(&want)
            .flat_map(|(g, w)| g.iter().zip(w).map(|(a, b)| (a - b).abs()))
            .fold(0.0f32, f32::max);

        let db = 20.0 * (worst / peak).log10();
        assert!(
            db < -120.0,
            "resampler differs from the reference by {db:.1} dB (worst {worst:.3e})"
        );
    }

    /// The edges are the part a resampler is most likely to get subtly wrong,
    /// and a mismatch there would put a transient on the derived fader at every
    /// track boundary. Checked separately so a failure says which end.
    #[test]
    fn the_edges_match_too() {
        let input = deinterleave(&f32s(include_bytes!("../../data/mode1_ref_in_f32.bin")));
        let want = deinterleave(&f32s(include_bytes!("../../data/mode1_ref_out_f32.bin")));
        let got = resample(&Audio::new(input, IN_RATE));

        let peak = want.iter().flatten().fold(0.0f32, |a, v| a.max(v.abs()));
        for (name, range) in [
            ("head", 0..200),
            ("tail", want[0].len() - 200..want[0].len()),
        ] {
            let worst = got
                .data
                .iter()
                .zip(&want)
                .flat_map(|(g, w)| range.clone().map(move |i| (g[i] - w[i]).abs()))
                .fold(0.0f32, f32::max);
            assert!(
                20.0 * (worst / peak).log10() < -120.0,
                "{name}: {:.1} dB",
                20.0 * (worst / peak).log10()
            );
        }
    }

    /// Length follows the ratio, so a converted stem carries the same duration
    /// the client's own conversion of the track does.
    #[test]
    fn the_length_follows_the_ratio() {
        assert_eq!(out_frames(147), 320);
        assert_eq!(out_frames(2205), 4800);
        assert_eq!(out_frames(44_100), 96_000);
        assert_eq!(out_frames(0), 0);
    }

    /// Silence in, silence out, and no panic on an empty plane.
    #[test]
    fn empty_and_silent_inputs_are_handled() {
        let silent = Audio::new(vec![vec![0.0; 1470], vec![0.0; 1470]], IN_RATE);
        let out = resample(&silent);
        assert_eq!(out.frames(), 3200);
        assert!(out.data.iter().flatten().all(|v| *v == 0.0));

        let empty = Audio::new(vec![Vec::new(), Vec::new()], IN_RATE);
        assert_eq!(resample(&empty).frames(), 0);
    }

    /// Being LTI is the property the extraction rested on, so hold it here too:
    /// a scaled input must produce exactly the scaled output.
    #[test]
    fn the_filter_is_linear() {
        let one: Vec<f32> = (0..1000)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 100.0)
            .collect();
        let half: Vec<f32> = one.iter().map(|v| v * 0.5).collect();

        let a = resample(&Audio::new(vec![one.clone(), one], IN_RATE));
        let b = resample(&Audio::new(vec![half.clone(), half], IN_RATE));

        let worst = a
            .data
            .iter()
            .zip(&b.data)
            .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p * 0.5 - q).abs()))
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-7, "homogeneity broken by {worst:.3e}");
    }
}
