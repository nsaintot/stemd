//! The spectrogram demucs works on, which is not quite a textbook STFT.
//!
//! Two layers, and they stay distinct. [`Stft`] is the ordinary transform:
//! periodic Hann, centred, unnormalised, the convention `torch.stft` uses.
//! [`Spectral`] pads by three half-hops, drops the top frequency bin, and keeps
//! only the frames corresponding to the input, so a round trip is lossy by
//! construction.
//!
//! Every constant here is dictated by the traced model.

use anyhow::{Result, bail};
use mlx_rs::Array;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;

/// `ceil(a / b)` for positive integers. `i32::div_ceil` is still unstable.
const fn div_ceil(a: i32, b: i32) -> i32 {
    (a + b - 1) / b
}

/// Window and transform size for every demucs v4 artefact.
pub const N_FFT: i32 = 4096;

/// Distance between frames. `n_fft / 4`.
pub const HOP: i32 = 1024;

/// A periodic Hann window of length `n`.
///
/// Periodic, not symmetric: the window is `0.5 - 0.5*cos(2*pi*i/n)` over `n`
/// points rather than `n - 1`, which is what makes overlapping frames sum flat.
/// `torch.hann_window` defaults to the same.
fn hann(n: i32) -> Array {
    let values: Vec<f32> = (0..n)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
        .collect();
    Array::from_slice(&values, &[n])
}

/// Reflect padding along the last axis.
///
/// mlx-rs offers `PadMode::Constant` and `PadMode::Edge` and nothing else, so this
/// is built from gathers. Mirrors without repeating the edge sample: `[1,2,3,4]`
/// padded by two is `[3,2,1,2,3,4,3,2]`, which is what `torch` does.
pub fn reflect_pad(x: &Array, left: i32, right: i32) -> Result<Array> {
    let n = *x.shape().last().expect("an array has a last axis");
    if left >= n || right >= n {
        bail!("reflect padding of ({left}, {right}) needs more than {n} samples");
    }
    let mut parts = Vec::with_capacity(3);
    if left > 0 {
        let idx = Array::from_iter((1..=left).rev(), &[left]);
        parts.push(x.take_axis(&idx, -1)?);
    }
    parts.push(x.clone());
    if right > 0 {
        let idx = Array::from_iter((n - 1 - right..n - 1).rev(), &[right]);
        parts.push(x.take_axis(&idx, -1)?);
    }
    Ok(ops::concatenate_axis(&parts, -1)?)
}

/// The ordinary transform, over the last axis.
pub struct Stft {
    n_fft: i32,
    hop: i32,
    window: Array,
}

impl Stft {
    pub fn new(n_fft: i32, hop: i32) -> Self {
        Self {
            n_fft,
            hop,
            window: hann(n_fft),
        }
    }

    /// `[..., T] -> [..., F, N]`, with `F = n_fft/2 + 1`.
    ///
    /// Framing is a strided view rather than a loop over frames: a four-minute
    /// track is ten thousand frames, and ten thousand slices would dominate the
    /// transform they were meant to feed.
    pub fn forward(&self, x: &Array, centred: bool) -> Result<Array> {
        let x = if centred {
            reflect_pad(x, self.n_fft / 2, self.n_fft / 2)?
        } else {
            x.clone()
        };
        let shape = x.shape().to_vec();
        let samples = *shape.last().expect("an array has a last axis");
        if samples < self.n_fft {
            bail!(
                "{samples} samples is fewer than one {}-point frame",
                self.n_fft
            );
        }
        let frames = (samples - self.n_fft) / self.hop + 1;

        // Flattened to [rows, T] so the stride arithmetic has one leading axis
        // to step over, whatever the caller's batch and channel layout was.
        let rows: i32 = shape[..shape.len() - 1].iter().product();
        let flat = x.reshape(&[rows, samples])?;
        let framed = ops::as_strided(
            &flat,
            &[rows, frames, self.n_fft][..],
            &[i64::from(samples), i64::from(self.hop), 1][..],
            0,
        )?;

        let windowed = ops::multiply(&framed, &self.window)?;
        let spectrum = mlx_rs::fft::rfft(&windowed, None, -1)?;

        // Back to the caller's layout, with frequency before time as demucs
        // expects: [..., F, N].
        let bins = self.n_fft / 2 + 1;
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(frames);
        out_shape.push(bins);
        let spectrum = spectrum.reshape(&out_shape)?;
        let last = out_shape.len() as i32 - 1;
        Ok(ops::swap_axes(&spectrum, last - 1, last)?)
    }

    /// `[..., F, N] -> [..., length]`, undoing [`Self::forward`].
    ///
    /// Overlap-add with the usual window-square normalisation, which is
    /// accumulated rather than assumed flat: at the ends, and wherever frames
    /// have been dropped or added, it is not.
    pub fn inverse(&self, z: &Array, length: i32, centred: bool) -> Result<Array> {
        let shape = z.shape().to_vec();
        let (bins, frames) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        let rows: i32 = shape[..shape.len() - 2].iter().product();

        let flat = z.reshape(&[rows, bins, frames])?;
        let flat = ops::swap_axes(&flat, -1, -2)?; // [rows, frames, bins]
        let frames_time = mlx_rs::fft::irfft(&flat, self.n_fft, -1)?;
        let windowed = ops::multiply(&frames_time, &self.window)?;

        let pad = if centred { self.n_fft / 2 } else { 0 };
        let width = length + 2 * pad;
        let square = ops::multiply(&self.window, &self.window)?;
        let squares = ops::broadcast_to(
            &square.reshape(&[1, 1, self.n_fft])?,
            &[1, frames, self.n_fft][..],
        )?;

        let out = overlap_add(&windowed, self.hop, width)?;
        let norm = overlap_add(&squares, self.hop, width)?;
        let out = out.index((.., pad..pad + length));
        let norm = norm.index((.., pad..pad + length));
        let out = ops::divide(&out, &ops::maximum(&norm, Array::from_f32(1e-11))?)?;

        let mut out_shape = shape[..shape.len() - 2].to_vec();
        out_shape.push(length);
        Ok(out.reshape(&out_shape)?)
    }
}

/// The model's spectrogram: [`Stft`] plus demucs's own padding and trimming.
pub struct Spectral {
    stft: Stft,
    hop: i32,
}

impl Spectral {
    pub fn new() -> Self {
        Self {
            stft: Stft::new(N_FFT, HOP),
            hop: HOP,
        }
    }

    /// `[B, C, T] -> [B, C, F-1, ceil(T/hop)]`, complex.
    ///
    /// The top frequency bin is dropped and the first two frames are skipped,
    /// both because the traced model expects exactly that shape. The result is
    /// not invertible on its own: see [`Self::inverse`].
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let samples = *x.shape().last().expect("an array has a last axis");
        let frames = div_ceil(samples, self.hop);
        let pad = self.hop / 2 * 3;
        let x = reflect_pad(x, pad, pad + frames * self.hop - samples)?;

        let z = self.stft.forward(&x, true)?;
        let bins = z.shape()[z.ndim() - 2];
        // Drop the Nyquist bin, then keep the frames covering the input.
        let z = z.index((.., .., 0..bins - 1, 2..2 + frames));
        Ok(z)
    }

    /// `[..., F-1, N] -> [..., length]`, undoing [`Self::forward`]'s trimming first: a
    /// zero bin back on top, and two frames of padding at each end.
    ///
    /// Rank-agnostic because it is called with both `[B, C, F, N]` and
    /// `[B, S, C, F, N]`.
    pub fn inverse(&self, z: &Array, length: i32) -> Result<Array> {
        let mut widths = vec![(0, 0); z.ndim()];
        let n = widths.len();
        widths[n - 2] = (0, 1); // the Nyquist bin `forward` dropped
        widths[n - 1] = (2, 2); // the frames it skipped at each end
        let z = ops::pad(z, &widths[..], Array::from_f32(0.0), None)?;

        let pad = self.hop / 2 * 3;
        let full = self.hop * div_ceil(length, self.hop) + 2 * pad;
        let x = self.istft(&z, full)?;
        slice_last(&x, pad, pad + length)
    }

    /// Inverse transform by overlap-add, with the usual window-square
    /// normalisation so overlapping frames do not sum to a ripple.
    ///
    /// The window-square sum is accumulated rather than assumed flat: with the
    /// frames this drops and re-pads at the ends, it is not.
    fn istft(&self, z: &Array, length: i32) -> Result<Array> {
        if N_FFT % self.hop != 0 {
            bail!("a hop of {} does not divide {N_FFT} evenly", self.hop);
        }
        let shape = z.shape().to_vec();
        let (bins, frames) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        let rows: i32 = shape[..shape.len() - 2].iter().product();

        let flat = z.reshape(&[rows, bins, frames])?;
        let flat = ops::swap_axes(&flat, -1, -2)?; // [rows, frames, bins]
        let frames_time = mlx_rs::fft::irfft(&real_dc(&flat)?, N_FFT, -1)?;
        let windowed = ops::multiply(&frames_time, &self.stft.window)?;

        let extent = (frames - 1) * self.hop + N_FFT;
        let padded = (length + N_FFT).max(extent);
        let win_sq = ops::multiply(&self.stft.window, &self.stft.window)?;
        let squares = ops::broadcast_to(&win_sq.reshape(&[1, 1, N_FFT])?, &[1, frames, N_FFT][..])?;

        let out = overlap_add(&windowed, self.hop, padded)?;
        let norm = overlap_add(&squares, self.hop, padded)?;

        // The centred transform padded by n_fft/2 at the front; drop it.
        let out = out.index((.., N_FFT / 2..N_FFT / 2 + length));
        let norm = norm.index((.., N_FFT / 2..N_FFT / 2 + length));
        let out = ops::divide(&out, &ops::maximum(&norm, Array::from_f32(1e-11))?)?;

        let mut out_shape = shape[..shape.len() - 2].to_vec();
        out_shape.push(length);
        Ok(out.reshape(&out_shape)?)
    }
}

/// Sum `[rows, frames, n_fft]` back into `[rows, width]`, each frame `hop`
/// further along than the last.
///
/// A handful of reshapes, not a loop. Frames `ceil(n_fft / hop)` apart cannot
/// overlap, so gathering every k-th one and laying them end to end places them all
/// at once, and the whole overlap-add is k of those summed at one-hop offsets.
///
/// End to end is only right when the hop divides the window. demucs hops a quarter
/// of a 4096-point window and it does; RoFormer hops 441 of 2048 and it does not,
/// leaving a 157-sample gap between frames five apart. Widening each frame's slot
/// to `k * hop` restores the spacing and costs nothing when the slot is already
/// the window.
///
/// The obvious spelling, walking the frames and placing each into a zeroed buffer,
/// allocates a whole-segment tensor per frame, which measured at forty per cent of
/// the model's entire forward pass.
fn overlap_add(frames_time: &Array, hop: i32, width: i32) -> Result<Array> {
    let s = frames_time.shape().to_vec();
    let (rows, frames, n_fft) = (s[0], s[1], s[2]);
    let groups = div_ceil(n_fft, hop);
    let slot = groups * hop;

    let framed = if slot > n_fft {
        ops::pad(
            frames_time,
            &[(0, 0), (0, 0), (0, slot - n_fft)][..],
            Array::from_f32(0.0),
            None,
        )?
    } else {
        frames_time.clone()
    };
    let count_in = |start: i32| div_ceil(frames - start, groups);

    // The slots of the last group can run past the audio when the hop does not
    // divide the window, so the buffer is sized to hold them and trimmed after.
    let mut buffer = width;
    for start in 0..groups {
        buffer = buffer.max(start * hop + count_in(start) * slot);
    }

    let mut total: Option<Array> = None;
    for start in 0..groups {
        let taken: Vec<i32> = (start..frames).step_by(groups as usize).collect();
        let Ok(count) = i32::try_from(taken.len()) else {
            bail!("{} frames is more than this can index", taken.len());
        };
        if count == 0 {
            continue;
        }
        let idx = Array::from_slice(&taken, &[count]);
        let joined = framed.take_axis(&idx, 1)?.reshape(&[rows, count * slot])?;

        let left = start * hop;
        let right = buffer - left - count * slot;
        let placed = ops::pad(
            &joined,
            &[(0, 0), (left, right)][..],
            Array::from_f32(0.0),
            None,
        )?;
        total = Some(match total {
            Some(t) => ops::add(&t, &placed)?,
            None => placed,
        });
    }
    let total = total.ok_or_else(|| anyhow::anyhow!("there are no frames to invert"))?;
    Ok(total.index((.., 0..width)))
}

/// `x[..., start..end]`, whatever the rank.
fn slice_last(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx = Array::from_iter(start..end, &[end - start]);
    Ok(x.take_axis(&idx, -1)?)
}

/// Drop the phase of the DC bin, which the transform is defined to ignore.
///
/// `irfft` reads `n/2 + 1` complex bins and writes `n` real samples, so the
/// imaginary parts of DC and Nyquist are two degrees of freedom the output cannot
/// carry. A correct inverse discards them. MLX's CUDA kernel does not, and folds
/// them into the samples instead.
///
/// The affected shapes look like kernel selection rather than a threshold:
/// monotonic in the batch for a fixed `n`, and not monotonic in `n`. `n = 8192` is
/// wrong at a single row while `n = 16384` is right up to 32, and `n = 512` is
/// right everywhere. Both models sit inside the region, htdemucs at `n = 4096`
/// from 64 rows and RoFormer at `n = 2048` from 1024. CPU and Metal are outside it
/// everywhere.
///
/// The decoder's mask does not arrive with a real DC, so this is not
/// hypothetical. Measured by `a_whole_track_matches_the_reference`:
///
/// ```text
///          CUDA      Metal
///   before  -56.3    -124.8
///   after  -121.4    -123.9
/// ```
///
/// Metal's 0.9 dB is second-order rounding, below the level the two answers
/// already differ by.
///
/// Nyquist needs no such care: `inverse` pads that bin in whole, and zero.
fn real_dc(z: &Array) -> Result<Array> {
    let bins = z.shape()[z.ndim() - 1];
    let mut selector = vec![0.0f32; bins as usize];
    selector[0] = 1.0;
    let selector = Array::from_slice(&selector, &[bins]);

    //  Subtracting the component rather than rebuilding the array around a real DC.
    //  Both spellings measure identically, and this one is exact by construction in
    //  both directions: at bin 0 the imaginary part cancels with itself, and
    //  everywhere else the subtrahend is zero.
    let i = Array::from_complex(mlx_rs::complex64::new(0.0, 1.0));
    let phase = ops::multiply(&ops::multiply(&ops::imag(z)?, &selector)?, &i)?;
    Ok(ops::subtract(z, &phase)?)
}

impl Default for Spectral {
    fn default() -> Self {
        Self::new()
    }
}
