//! Sample-rate conversion, on the way in and on the way out.
//!
//! Separation always runs at the model's own rate. An upload at any other rate is
//! converted before the model sees it, and the stems are converted again to
//! whatever the job asked for, so neither rate changes what the model saw.
//!
//! # Reconstructing at a converted rate
//!
//! Only the two shipped stems cross the wire, so a client rebuilding
//! `mix - harmonics - vocals` must resample its own mix to the stems' rate first.
//! That recovers the same signal, model residual included, because resampling is
//! linear: converting the parts and converting the mix commute.
//! `a_client_rebuilds_the_same_derived_part_at_any_rate` holds it at better than
//! -100 dB.
//!
//! It is only exact if the client's filter matches this one. Against ffmpeg's
//! `swresample` the two agree to about -77 dB of the mix, which on a derived part
//! sitting 41 dB below the mix showed up as -36 dB of that part.
//!
//! # DSP modes
//!
//! [`DspMode::General`], mode 0, is the default and handles any rate pair.
//!
//! A client that cannot match its filter to ours can ask for a mode that matches
//! ours to its. [`DspMode::Mode1`] is one such: it reproduces a particular
//! client's own 44.1 to 96 kHz converter, which turns the -36 dB above into
//! -101 dB. It is that one rate pair and nothing else, and it is never selected
//! unless a job names it.

pub mod mode1;

use std::fmt;
use std::str::FromStr;

use anyhow::{Context, Result};
use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Resampler};
use serde::{Deserialize, Serialize};

use crate::pcm::Audio;

/// Frames per processing chunk. Large enough that the FFT overhead is amortised,
/// small enough that the intermediate buffers stay modest for a long track.
const CHUNK: usize = 4096;

/// Which filter a conversion runs through.
///
/// Numbered rather than named. Mode 0 is a resampler; the rest are copies of
/// somebody else's, and exist so that both sides of a client's subtraction can
/// have gone through the same one. Naming them after what they sound like would
/// suggest a quality ladder, which this is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DspMode {
    /// General-purpose, any rate pair. The default.
    #[default]
    General,
    /// 44.1 to 96 kHz only, and an error on any other pair. See [`mode1`].
    Mode1,
}

impl DspMode {
    pub const ALL: [Self; 2] = [Self::General, Self::Mode1];

    pub const fn id(self) -> u8 {
        match self {
            Self::General => 0,
            Self::Mode1 => 1,
        }
    }

    /// The rate pair this mode converts, or `None` when it converts any.
    pub const fn only_pair(self) -> Option<(u32, u32)> {
        match self {
            Self::General => None,
            Self::Mode1 => Some((mode1::IN_RATE, mode1::OUT_RATE)),
        }
    }

    /// The numbered mode covering exactly this conversion, if there is one.
    ///
    /// For telling a client it is converting a pair somebody bothered to
    /// reproduce a filter for, without deciding for it that it wants that
    /// filter: most callers do not, and the ones that do have to say so.
    pub fn for_pair(from: u32, to: u32) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.only_pair() == Some((from, to)))
    }

    /// The ids, for an error message that cannot drift from [`Self::ALL`].
    fn listed() -> String {
        Self::ALL
            .iter()
            .map(|mode| mode.id().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unknown dsp_mode {asked:?}, expected one of {}", DspMode::listed())]
pub struct UnknownDspMode {
    asked: String,
}

impl TryFrom<u8> for DspMode {
    type Error = UnknownDspMode;

    fn try_from(id: u8) -> Result<Self, Self::Error> {
        Self::ALL
            .into_iter()
            .find(|mode| mode.id() == id)
            .ok_or_else(|| UnknownDspMode {
                asked: id.to_string(),
            })
    }
}

impl FromStr for DspMode {
    type Err = UnknownDspMode;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        trimmed
            .parse::<u8>()
            .ok()
            .and_then(|id| Self::try_from(id).ok())
            .ok_or_else(|| UnknownDspMode {
                asked: trimmed.to_owned(),
            })
    }
}

impl fmt::Display for DspMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id())
    }
}

/// Output sample rates a client may ask for.
///
/// A closed set rather than an arbitrary integer: each one is a rate real
/// hardware runs at, and refusing the rest keeps a typo from silently pitching a
/// track. [`Self::Hz44100`] is the model's own rate and costs no conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputRate {
    /// 24 kHz. Half-band; useful where bandwidth matters more than headroom.
    #[serde(rename = "24000")]
    Hz24000,
    /// 44.1 kHz: the model's own rate, and the default. No conversion runs.
    #[default]
    #[serde(rename = "44100")]
    Hz44100,
    #[serde(rename = "48000")]
    Hz48000,
    #[serde(rename = "96000")]
    Hz96000,
}

impl OutputRate {
    pub const ALL: [Self; 4] = [Self::Hz24000, Self::Hz44100, Self::Hz48000, Self::Hz96000];

    pub const fn hz(self) -> u32 {
        match self {
            Self::Hz24000 => 24_000,
            Self::Hz44100 => 44_100,
            Self::Hz48000 => 48_000,
            Self::Hz96000 => 96_000,
        }
    }

    /// The rate matching `hz`, if it is one this server offers.
    pub fn from_hz(hz: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|rate| rate.hz() == hz)
    }
}

impl fmt::Display for OutputRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.hz())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("unsupported output sample rate {0:?}, expected one of 24000, 44100, 48000, 96000")]
pub struct UnsupportedRate(String);

impl FromStr for OutputRate {
    type Err = UnsupportedRate;

    /// Accepts the rate in Hz, and the kHz spellings a person would type.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "24000" | "24k" | "24" => Ok(Self::Hz24000),
            "44100" | "44.1k" | "44.1" => Ok(Self::Hz44100),
            "48000" | "48k" | "48" => Ok(Self::Hz48000),
            "96000" | "96k" | "96" => Ok(Self::Hz96000),
            other => Err(UnsupportedRate(other.to_owned())),
        }
    }
}

/// Convert `audio` to `target` Hz with the general resampler, or hand it back
/// untouched when it is already there.
pub fn to_rate(audio: &Audio, target: u32) -> Result<Audio> {
    to_rate_with(audio, target, DspMode::General)
}

/// Convert `audio` to `target` Hz through `mode`'s filter.
///
/// Returning the input unchanged at a matching rate keeps the default path free: a
/// client that does not ask for a rate pays nothing, and the native path's
/// bit-exactness is not spent on a round trip through a filter. A mode that only
/// covers one pair is an error on any other, rather than a silent fall back to
/// the general resampler, which would hand the client a filter it cannot match.
pub fn to_rate_with(audio: &Audio, target: u32, mode: DspMode) -> Result<Audio> {
    if audio.sample_rate == target || audio.frames() == 0 {
        return Ok(audio.clone());
    }
    anyhow::ensure!(target > 0, "target sample rate must be non-zero");

    match mode {
        DspMode::General => {}
        DspMode::Mode1 => {
            anyhow::ensure!(
                audio.sample_rate == mode1::IN_RATE && target == mode1::OUT_RATE,
                "dsp mode 1 converts {} to {} Hz, not {} to {target}",
                mode1::IN_RATE,
                mode1::OUT_RATE,
                audio.sample_rate
            );
            return Ok(mode1::resample(audio));
        }
    }

    let channels = audio.channels();
    anyhow::ensure!(channels > 0, "cannot resample zero channels");

    let frames = audio.frames();
    let mut resampler = Fft::<f32>::new(
        audio.sample_rate as usize,
        target as usize,
        CHUNK,
        channels,
        FixedSync::Input,
    )
    .with_context(|| format!("preparing {} -> {target} Hz", audio.sample_rate))?;

    let capacity = resampler.process_all_needed_output_len(frames);
    let mut planes = vec![vec![0.0f32; capacity]; channels];

    let input = SequentialSliceOfVecs::new(&audio.data, channels, frames)
        .map_err(|e| anyhow::anyhow!("wrapping the input: {e}"))?;
    let written = {
        let mut output = SequentialSliceOfVecs::new_mut(&mut planes, channels, capacity)
            .map_err(|e| anyhow::anyhow!("wrapping the output: {e}"))?;
        let (_, written) = resampler
            .process_all_into_buffer(&input, &mut output, frames, None)
            .with_context(|| format!("resampling {} -> {target} Hz", audio.sample_rate))?;
        written
    };

    // `capacity` allows for the resampler's own latency; only the leading
    // `written` frames are signal.
    for plane in &mut planes {
        plane.truncate(written);
    }
    Ok(Audio::new(planes, target))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sine at `hz`, which a correct resampler reproduces at the same
    /// frequency and amplitude in the new rate.
    fn tone(hz: f32, rate: u32, secs: f32) -> Audio {
        let frames = (rate as f32 * secs) as usize;
        let plane: Vec<f32> = (0..frames)
            .map(|i| (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * 0.5)
            .collect();
        Audio::new(vec![plane.clone(), plane], rate)
    }

    /// Dominant frequency, by picking the strongest bin of a naive DFT over a
    /// window well inside the clip (so the resampler's edges do not skew it).
    fn dominant_hz(audio: &Audio) -> f32 {
        let n = 8192.min(audio.frames());
        let start = (audio.frames() - n) / 2;
        let samples = &audio.data[0][start..start + n];
        let mut best = (0.0f32, 0usize);
        for k in 1..n / 2 {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (i, s) in samples.iter().enumerate() {
                let a = -std::f32::consts::TAU * k as f32 * i as f32 / n as f32;
                re += s * a.cos();
                im += s * a.sin();
            }
            let mag = re * re + im * im;
            if mag > best.0 {
                best = (mag, k);
            }
        }
        best.1 as f32 * audio.sample_rate as f32 / n as f32
    }

    #[test]
    fn the_rate_set_round_trips_through_its_strings() {
        for rate in OutputRate::ALL {
            assert_eq!(rate.to_string().parse::<OutputRate>().unwrap(), rate);
            assert_eq!(OutputRate::from_hz(rate.hz()), Some(rate));
        }
        assert_eq!("44.1k".parse::<OutputRate>().unwrap(), OutputRate::Hz44100);
        assert_eq!("96".parse::<OutputRate>().unwrap(), OutputRate::Hz96000);
        assert_eq!(OutputRate::default(), OutputRate::Hz44100);
    }

    #[test]
    fn an_unsupported_rate_is_refused_rather_than_approximated() {
        // 22050 is a real rate, just not one this server offers; silently
        // picking the nearest would pitch the track.
        assert!("22050".parse::<OutputRate>().is_err());
        assert!("".parse::<OutputRate>().is_err());
        assert!(OutputRate::from_hz(22_050).is_none());
    }

    #[test]
    fn the_native_rate_returns_the_samples_untouched() {
        let audio = tone(440.0, 44_100, 0.5);
        let out = to_rate(&audio, 44_100).unwrap();
        assert_eq!(out.data, audio.data, "the default path must not filter");
        assert_eq!(out.sample_rate, 44_100);
    }

    #[test]
    fn every_offered_rate_produces_the_expected_geometry() {
        let audio = tone(440.0, 44_100, 1.0);
        for rate in OutputRate::ALL {
            let out = to_rate(&audio, rate.hz()).unwrap();
            assert_eq!(out.sample_rate, rate.hz());
            assert_eq!(out.channels(), 2, "{rate} lost a channel");
            // Duration is preserved, within a few frames of resampler edge.
            let want = audio.duration_secs();
            assert!(
                (out.duration_secs() - want).abs() < 0.01,
                "{rate}: {:.4}s vs {want:.4}s",
                out.duration_secs()
            );
        }
    }

    #[test]
    fn a_tone_keeps_its_pitch_through_every_conversion() {
        // The failure this catches is a wrong ratio, which shifts pitch: the
        // one resampling bug that is obvious by ear and invisible in geometry.
        let audio = tone(1000.0, 44_100, 1.0);
        for rate in OutputRate::ALL {
            let out = to_rate(&audio, rate.hz()).unwrap();
            let found = dominant_hz(&out);
            assert!(
                (found - 1000.0).abs() < 15.0,
                "{rate}: tone moved to {found:.0} Hz"
            );
        }
    }

    #[test]
    fn a_tone_keeps_its_level_through_every_conversion() {
        let audio = tone(1000.0, 44_100, 1.0);
        for rate in OutputRate::ALL {
            let out = to_rate(&audio, rate.hz()).unwrap();
            let mid = out.frames() / 4..out.frames() * 3 / 4;
            let peak = out.data[0][mid].iter().fold(0.0f32, |a, v| a.max(v.abs()));
            assert!((peak - 0.5).abs() < 0.05, "{rate}: peak {peak:.3}");
        }
    }

    #[test]
    fn empty_audio_is_not_an_error() {
        let silence = Audio::new(vec![Vec::new(), Vec::new()], 44_100);
        let out = to_rate(&silence, 48_000).unwrap();
        assert_eq!(out.frames(), 0);
    }

    /// The reconciliation a client performs at a converted rate.
    ///
    /// It resamples its own mix, subtracts the two stems it was sent, and must end up
    /// with the same `drums` it would have had at the native rate:
    /// `R(mix - h - v) == R(mix) - R(h) - R(v)`, which holds because resampling is
    /// linear.
    #[test]
    fn a_client_rebuilds_the_same_derived_part_at_any_rate() {
        // Parts that deliberately do NOT sum to the mix, so the model residual
        // is real and has to survive the round trip onto the derived fader.
        let mix = tone(220.0, 44_100, 1.0);
        let harmonics = Audio::new(
            mix.data
                .iter()
                .map(|c| c.iter().map(|v| v * 0.31).collect())
                .collect(),
            44_100,
        );
        let vocals = tone(660.0, 44_100, 1.0);

        // What the server would compute at the native rate.
        let native_derived = Audio::new(
            (0..mix.channels())
                .map(|ch| {
                    (0..mix.frames())
                        .map(|i| mix.data[ch][i] - harmonics.data[ch][i] - vocals.data[ch][i])
                        .collect()
                })
                .collect(),
            44_100,
        );

        for rate in OutputRate::ALL {
            let want = to_rate(&native_derived, rate.hz()).unwrap();

            // What the client computes: convert the mix, subtract the two
            // stems it was sent (converted independently, by the server).
            let mix_r = to_rate(&mix, rate.hz()).unwrap();
            let h_r = to_rate(&harmonics, rate.hz()).unwrap();
            let v_r = to_rate(&vocals, rate.hz()).unwrap();

            let n = [want.frames(), mix_r.frames(), h_r.frames(), v_r.frames()]
                .into_iter()
                .min()
                .unwrap();
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for ch in 0..want.channels() {
                for i in 0..n {
                    let rebuilt = mix_r.data[ch][i] - h_r.data[ch][i] - v_r.data[ch][i];
                    num += f64::from(want.data[ch][i] - rebuilt).powi(2);
                    den += f64::from(want.data[ch][i]).powi(2);
                }
            }
            let null_db = 10.0 * (num / den).log10();
            println!("  {rate:>5} Hz: client's rebuilt drums matches at {null_db:.1} dB");
            assert!(
                null_db < -100.0,
                "{rate}: the rebuilt derived part differs by {null_db:.1} dB"
            );
        }
    }

    /// The property the reconciliation above rests on: resampling is linear, so
    /// parts that sum to the mix at the native rate still sum to the resampled
    /// mix after three independent conversions.
    #[test]
    fn independently_resampled_parts_still_sum_to_the_resampled_mix() {
        let mix = tone(440.0, 44_100, 1.0);
        // A split that sums to the mix exactly, as `derived_part` produces.
        let a = tone(440.0, 44_100, 1.0);
        let b = Audio::new(
            a.data
                .iter()
                .map(|c| c.iter().map(|v| v * 0.3).collect())
                .collect(),
            44_100,
        );
        let c = Audio::new(
            mix.data
                .iter()
                .enumerate()
                .map(|(ch, plane)| {
                    plane
                        .iter()
                        .enumerate()
                        .map(|(i, v)| v - b.data[ch][i])
                        .collect()
                })
                .collect(),
            44_100,
        );

        for rate in OutputRate::ALL {
            let rm = to_rate(&mix, rate.hz()).unwrap();
            let rb = to_rate(&b, rate.hz()).unwrap();
            let rc = to_rate(&c, rate.hz()).unwrap();

            let n = rm.frames().min(rb.frames()).min(rc.frames());
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for ch in 0..rm.channels() {
                for i in 0..n {
                    let sum = rb.data[ch][i] + rc.data[ch][i];
                    num += f64::from(rm.data[ch][i] - sum).powi(2);
                    den += f64::from(rm.data[ch][i]).powi(2);
                }
            }
            let null_db = 10.0 * (num / den).log10();
            println!("  {rate:>5} Hz: parts sum to the resampled mix at {null_db:.1} dB");
            assert!(
                null_db < -100.0,
                "{rate}: linearity lost, null only {null_db:.1} dB"
            );
        }
        let _ = a;
    }

    #[test]
    fn the_dsp_modes_round_trip_through_their_ids() {
        for mode in DspMode::ALL {
            assert_eq!(mode.to_string().parse::<DspMode>().unwrap(), mode);
            assert_eq!(DspMode::try_from(mode.id()).unwrap(), mode);
        }
        assert_eq!(DspMode::default(), DspMode::General);
        assert_eq!(DspMode::for_pair(44_100, 96_000), Some(DspMode::Mode1));
        assert_eq!(DspMode::for_pair(44_100, 48_000), None);
        assert_eq!(DspMode::for_pair(96_000, 44_100), None);
        for bad in ["2", "-1", "", "one"] {
            assert!(bad.parse::<DspMode>().is_err(), "{bad} was accepted");
        }
    }

    /// A mode that covers one pair refuses every other, so a client asking for a
    /// filter it can match is never quietly handed a different one. A conversion
    /// that does not run is not a filter, so the native rate stays free.
    #[test]
    fn mode_1_converts_only_its_own_pair() {
        let at_44 = tone(440.0, 44_100, 0.2);
        assert!(to_rate_with(&at_44, 96_000, DspMode::Mode1).is_ok());
        assert!(to_rate_with(&at_44, 48_000, DspMode::Mode1).is_err());
        assert!(to_rate_with(&at_44, 44_100, DspMode::Mode1).is_ok());

        let at_48 = tone(440.0, 48_000, 0.2);
        assert!(to_rate_with(&at_48, 96_000, DspMode::Mode1).is_err());
    }

    /// The two modes are different filters, not the same one behind two names,
    /// and the general one is what a job gets when it does not ask.
    #[test]
    fn mode_1_is_a_different_filter_from_the_general_one() {
        let mix = tone(440.0, 44_100, 0.5);
        let general = to_rate(&mix, 96_000).unwrap();
        let one = to_rate_with(&mix, 96_000, DspMode::Mode1).unwrap();

        assert_eq!(general.sample_rate, one.sample_rate);
        let n = general.frames().min(one.frames());
        let worst = general.data[0][..n]
            .iter()
            .zip(&one.data[0][..n])
            .fold(0.0f32, |a, (g, m)| a.max((g - m).abs()));
        assert!(worst > 1e-6, "the two modes agree to {worst:.3e}");

        let default = to_rate_with(&mix, 96_000, DspMode::default()).unwrap();
        assert_eq!(default.data, general.data);
    }

    /// Where a feature lands after conversion. Not a detail a client can absorb:
    /// a stem late against the track is late under the fader.
    ///
    /// The general resampler is aligned, so an impulse comes out where the ratio
    /// puts it. Mode 1 is 64 output samples later at 96 kHz, which is the
    /// converter's own group delay (18880 taps, peak centred, so 29.5 input
    /// samples) and is the point of the mode rather than a defect in it: the
    /// client's mix went through the same filter and carries the same delay.
    #[test]
    fn a_conversion_puts_a_feature_where_its_own_filter_puts_it() {
        let at = 22_050;
        let mut plane = vec![0.0f32; 44_100];
        plane[at] = 1.0;
        let src = Audio::new(vec![plane.clone(), plane], 44_100);

        let peak = |audio: &Audio| {
            audio.data[0]
                .iter()
                .enumerate()
                .fold((0usize, 0.0f32), |best, (i, v)| {
                    if v.abs() > best.1 { (i, v.abs()) } else { best }
                })
                .0
        };

        for rate in OutputRate::ALL {
            let out = to_rate(&src, rate.hz()).unwrap();
            let want = (at as f64 * f64::from(rate.hz()) / 44_100.0).round() as usize;
            assert_eq!(
                peak(&out),
                want,
                "{rate}: the general filter is not aligned"
            );
        }

        let one = to_rate_with(&src, 96_000, DspMode::Mode1).unwrap();
        let want = (at as f64 * 96_000.0 / 44_100.0).round() as usize;
        assert_eq!(peak(&one), want + 64, "mode 1's group delay moved");
    }

    /// The two modes are not interchangeable on the one path that matters, and
    /// this is what it costs to get it wrong.
    ///
    /// A client that rebuilds `drums = mix - harmonics - vocals` resamples its
    /// own mix with its own filter. Serve it stems through the same filter and
    /// the subtraction cancels. Serve it stems through the general one, which is
    /// aligned where the other carries 64 samples of group delay, and the parts
    /// do not line up: at 1.5 kHz that offset is a third of a cycle, so the
    /// vocals are not removed from the mix, they are roughly doubled in it.
    ///
    /// Which is why `dsp_mode` is opt-in on the wire and a client that needs it
    /// has to say so on every job.
    #[test]
    fn a_client_that_does_not_ask_for_its_own_filter_gets_no_cancellation() {
        let rate = 44_100;
        let secs = 2.0;
        let part = |hz: f32, amp: f32| {
            let plane: Vec<f32> = (0..(rate as f32 * secs) as usize)
                .map(|i| (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * amp)
                .collect();
            Audio::new(vec![plane.clone(), plane], rate)
        };
        // Three parts that sum to the mix exactly, as the server's split does.
        let harmonics = part(220.0, 0.30);
        let vocals = part(1_500.0, 0.25);
        let drums = part(80.0, 0.20);
        let mix = Audio::new(
            (0..2)
                .map(|ch| {
                    (0..drums.frames())
                        .map(|i| harmonics.data[ch][i] + vocals.data[ch][i] + drums.data[ch][i])
                        .collect()
                })
                .collect(),
            rate,
        );

        // The client's side is fixed: its own mix, through its own filter.
        let mix96 = to_rate_with(&mix, 96_000, DspMode::Mode1).unwrap();
        let want = to_rate_with(&drums, 96_000, DspMode::Mode1).unwrap();

        let rebuild = |mode: DspMode| {
            let h = to_rate_with(&harmonics, 96_000, mode).unwrap();
            let v = to_rate_with(&vocals, 96_000, mode).unwrap();
            let n = [mix96.frames(), h.frames(), v.frames(), want.frames()]
                .into_iter()
                .min()
                .unwrap();
            // The filters ramp at both ends; the question is the steady state.
            let (lo, hi) = (n / 8, n - n / 8);
            let (mut err, mut sig) = (0.0f64, 0.0f64);
            for i in lo..hi {
                let got = mix96.data[0][i] - h.data[0][i] - v.data[0][i];
                err += f64::from(got - want.data[0][i]).powi(2);
                sig += f64::from(want.data[0][i]).powi(2);
            }
            10.0 * (err / sig).log10()
        };

        let matched = rebuild(DspMode::Mode1);
        let general = rebuild(DspMode::General);
        println!("  same filter both sides: {matched:.1} dB");
        println!("  general on the stems:   {general:.1} dB");
        assert!(
            matched < -100.0,
            "the matched filter should cancel: {matched:.1} dB"
        );
        // Not a near miss. The error is louder than the part it replaces, which
        // is why this is audible as the vocals coming back rather than as noise.
        assert!(
            general > 0.0,
            "the general filter should not cancel: {general:.1} dB"
        );
    }
}
