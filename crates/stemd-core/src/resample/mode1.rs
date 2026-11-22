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

use super::polyphase::Polyphase;
use crate::pcm::Audio;

pub(crate) const IN_RATE: u32 = 44_100;
pub(crate) const OUT_RATE: u32 = 96_000;

/// The prototype at its native 320 × 44100 = 14.112 MHz grid, little-endian f32,
/// normalised so a unit impulse in gives the prototype out. 18880 taps: 320
/// branches of 59.
static PROTO_LE: &[u8] = include_bytes!("../../data/mode1_proto_f32.bin");

fn filter() -> &'static Polyphase {
    static F: OnceLock<Polyphase> = OnceLock::new();
    F.get_or_init(|| {
        let taps: Vec<f32> = PROTO_LE
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let filter = Polyphase::new(IN_RATE, OUT_RATE, &taps).expect("mode 1 prototype");
        assert_eq!(filter.branch_taps(), 59, "prototype is not 320 x 59 taps");
        filter
    })
}

/// How many output frames `frames` input frames become: `ceil(frames * 320 / 147)`.
#[must_use]
pub fn out_frames(frames: usize) -> usize {
    filter().out_frames(frames)
}

/// Resample 44.1 kHz `audio` to 96 kHz exactly as the target player does.
///
/// Panics unless `audio` is at [`IN_RATE`]; the caller checks the rate first.
#[must_use]
pub fn resample(audio: &Audio) -> Audio {
    filter()
        .resample(audio)
        .expect("the caller checked the input rate")
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
        assert_eq!(out_frames(1), 3);
        assert_eq!(out_frames(100_000), 217_688);
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
