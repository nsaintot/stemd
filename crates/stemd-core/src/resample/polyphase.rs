//! A stateless polyphase FIR converting one fixed rate pair.
//!
//! With `L / M = out_rate / in_rate` in lowest terms and a prototype `p` of
//! `L * B` taps, output frame `m` is
//!
//! ```text
//! y[m] = sum over k of x[k] * p[m*M - k*L]
//! ```
//!
//! where `p` is zero outside `0 .. L*B` and `x` is zero outside the input. The
//! output has `ceil(frames * L / M)` frames. Taps are used as given, so a
//! filter with unity passband gain has taps summing to `L`. No delay is
//! compensated: the filter's group delay is part of its output.
//!
//! Each output frame reads `B` input samples, at most [`Polyphase::MAX_BRANCH`].

use anyhow::{Result, ensure};

use crate::pcm::Audio;

pub struct Polyphase {
    in_rate: u32,
    out_rate: u32,
    l: usize,
    m: usize,
    /// Taps per branch, `B`.
    branch: usize,
    /// `L` branches of `B` reversed taps: `branches[r*B + i] = p[r + L*(B - 1 - i)]`.
    branches: Vec<f32>,
}

impl std::fmt::Debug for Polyphase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Polyphase")
            .field("in_rate", &self.in_rate)
            .field("out_rate", &self.out_rate)
            .field("taps", &self.taps())
            .finish_non_exhaustive()
    }
}

const fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl Polyphase {
    /// Most input samples one output frame reads. A session converts a range
    /// with 8192 samples of context, which must cover it.
    pub const MAX_BRANCH: usize = 1024;

    /// Most taps in one prototype.
    pub const MAX_TAPS: usize = 1 << 20;

    /// A filter converting `in_rate` to `out_rate` through the prototype `taps`.
    pub fn new(in_rate: u32, out_rate: u32, taps: &[f32]) -> Result<Self> {
        ensure!(
            in_rate > 0 && out_rate > 0,
            "rates must be non-zero, got {in_rate} to {out_rate}"
        );
        ensure!(
            in_rate != out_rate,
            "{in_rate} to {out_rate} is not a conversion"
        );
        let g = gcd(in_rate as usize, out_rate as usize);
        let (l, m) = (out_rate as usize / g, in_rate as usize / g);
        ensure!(
            !taps.is_empty() && taps.len().is_multiple_of(l),
            "{in_rate} to {out_rate} Hz needs a multiple of {l} taps, got {}",
            taps.len()
        );
        ensure!(
            taps.len() <= Self::MAX_TAPS,
            "{} taps is over the limit of {}",
            taps.len(),
            Self::MAX_TAPS
        );
        let branch = taps.len() / l;
        ensure!(
            branch <= Self::MAX_BRANCH,
            "{branch} taps per branch is over the limit of {}",
            Self::MAX_BRANCH
        );
        ensure!(taps.iter().all(|t| t.is_finite()), "taps must be finite");
        let mut branches = Vec::with_capacity(taps.len());
        for r in 0..l {
            branches.extend((0..branch).map(|i| taps[r + l * (branch - 1 - i)]));
        }
        Ok(Self {
            in_rate,
            out_rate,
            l,
            m,
            branch,
            branches,
        })
    }

    pub const fn in_rate(&self) -> u32 {
        self.in_rate
    }

    pub const fn out_rate(&self) -> u32 {
        self.out_rate
    }

    pub const fn taps(&self) -> usize {
        self.l * self.branch
    }

    pub const fn branch_taps(&self) -> usize {
        self.branch
    }

    /// Output frames for `frames` input frames: `ceil(frames * L / M)`.
    pub(crate) const fn out_frames(&self, frames: usize) -> usize {
        (frames * self.l).div_ceil(self.m)
    }

    /// Convert `audio`, which must be at [`Self::in_rate`].
    pub fn resample(&self, audio: &Audio) -> Result<Audio> {
        ensure!(
            audio.sample_rate == self.in_rate,
            "this filter converts {} to {} Hz, not {}",
            self.in_rate,
            self.out_rate,
            audio.sample_rate
        );
        let n_out = self.out_frames(audio.frames());

        //  Every output frame is an independent dot product over a shared input,
        //  so the work splits by output range with no seams.
        let mut planes: Vec<Vec<f32>> = audio.data.iter().map(|_| vec![0.0f32; n_out]).collect();
        let par = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        let slices = par.div_ceil(planes.len().max(1)).max(1);
        let chunk = n_out.div_ceil(slices).max(1);

        std::thread::scope(|s| {
            for (src, out) in audio.data.iter().zip(planes.iter_mut()) {
                for (i, dst) in out.chunks_mut(chunk).enumerate() {
                    s.spawn(move || self.fill(src, i * chunk, dst));
                }
            }
        });

        Ok(Audio::new(planes, self.out_rate))
    }

    /// Output frames `m0 .. m0 + dst.len()` of one channel.
    fn fill(&self, src: &[f32], m0: usize, dst: &mut [f32]) {
        let (l, m, b) = (self.l, self.m, self.branch);
        // `newest` is the newest input sample frame `m` reads and `r` its
        // branch: `m*M / L` and `m*M % L`, carried forward instead of divided
        // out per frame. Each frame advances them by `M / L` and `M % L`.
        let start = m0 * m;
        let mut newest = start / l;
        let mut r = start % l;
        let (skip, step) = (m / l, m % l);

        for o in dst.iter_mut() {
            let taps = &self.branches[r * b..(r + 1) * b];

            // Accumulated in f64 and rounded once.
            let acc: f64 = if newest + 1 >= b && newest < src.len() {
                src[newest + 1 - b..=newest]
                    .iter()
                    .zip(taps)
                    .map(|(x, t)| f64::from(*x) * f64::from(*t))
                    .sum()
            } else {
                // The window hangs off an edge of the signal, where the input
                // is zero.
                (0..b)
                    .filter_map(|j| {
                        let k = newest.checked_sub(j)?;
                        Some(f64::from(*src.get(k)?) * f64::from(taps[b - 1 - j]))
                    })
                    .sum()
            };
            #[expect(clippy::cast_possible_truncation, reason = "f32 output by design")]
            {
                *o = acc as f32;
            }

            newest += skip;
            r += step;
            if r >= l {
                r -= l;
                newest += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition evaluated term by term, for comparison.
    fn direct(taps: &[f32], l: usize, m: usize, x: &[f32]) -> Vec<f32> {
        let n_out = (x.len() * l).div_ceil(m);
        (0..n_out)
            .map(|o| {
                let mut acc = 0.0f64;
                for (k, xk) in x.iter().enumerate() {
                    let idx = (o * m) as i64 - (k * l) as i64;
                    if idx >= 0 && (idx as usize) < taps.len() {
                        acc += f64::from(*xk) * f64::from(taps[idx as usize]);
                    }
                }
                acc as f32
            })
            .collect()
    }

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn matches_the_definition_for_up_and_down_conversion() {
        for (from, to) in [(44_100u32, 96_000u32), (44_100, 48_000), (44_100, 24_000)] {
            let g = gcd(from as usize, to as usize);
            let (l, m) = (to as usize / g, from as usize / g);
            let taps = noise(l * 7, to);
            let x = noise(600, from);
            let poly = Polyphase::new(from, to, &taps).unwrap();
            let got = poly.resample(&Audio::new(vec![x.clone()], from)).unwrap();
            let want = direct(&taps, l, m, &x);
            assert_eq!(got.data[0].len(), want.len(), "{from} to {to}: length");
            let worst = got.data[0]
                .iter()
                .zip(&want)
                .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
            assert!(worst < 1e-6, "{from} to {to}: off by {worst:.3e}");
        }
    }

    #[test]
    fn refuses_a_prototype_it_cannot_split() {
        // 44.1 to 96 kHz is L = 320.
        assert!(Polyphase::new(44_100, 96_000, &[0.0; 319]).is_err());
        assert!(Polyphase::new(44_100, 96_000, &[]).is_err());
        assert!(Polyphase::new(44_100, 44_100, &[0.0; 320]).is_err());
        assert!(Polyphase::new(0, 96_000, &[0.0; 320]).is_err());
        let mut nan = vec![0.0; 320];
        nan[3] = f32::NAN;
        assert!(Polyphase::new(44_100, 96_000, &nan).is_err());
        let long = vec![0.0; 320 * (Polyphase::MAX_BRANCH + 1)];
        assert!(Polyphase::new(44_100, 96_000, &long).is_err());
    }

    #[test]
    fn refuses_audio_at_another_rate() {
        let poly = Polyphase::new(44_100, 96_000, &[1.0; 320]).unwrap();
        let audio = Audio::new(vec![vec![0.0; 100]], 48_000);
        assert!(poly.resample(&audio).is_err());
    }

    #[test]
    fn splits_across_threads_without_seams() {
        let taps = noise(320 * 59, 1);
        let x = noise(44_100, 2);
        let poly = Polyphase::new(44_100, 96_000, &taps).unwrap();
        let whole = poly.resample(&Audio::new(vec![x.clone()], 44_100)).unwrap();
        let mut one = vec![0.0f32; poly.out_frames(x.len())];
        poly.fill(&x, 0, &mut one);
        assert_eq!(whole.data[0], one);
    }
}
