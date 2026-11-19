//! Time left on a separation, plus 15%.
//!
//! With the server's median segment time, the estimate is the segments left
//! times that median, or times this job's own time per segment over the last
//! [`RECENT_STEPS`] segments when that is slower, such as when session
//! segments run in between. The time into the running segment is taken off, up
//! to one segment.
//!
//! Before a segment has run on the loaded model, the estimate is the time left
//! at the slower of two rates of the progress fraction: the average since the
//! window first saw the job separating, and the rate over the last
//! [`RECENT_STEPS`] steps. Both are measured to the last step, and the time
//! since it is taken off, up to one average step: a stalled step holds the
//! estimate.
//!
//! The shown value is recomputed every [`RECALC_SECS`] and counts down in
//! between. A new estimate below the countdown replaces it; one above replaces
//! it only when it is [`RISE`] times the countdown and [`RISE_SECS`] more.

use std::collections::VecDeque;

use stemd_core::{Progress, Stage};

/// Seconds a job must have been watched before an estimate shows.
const SETTLE_SECS: f64 = 3.0;

/// Progress a job must have made while watched before an estimate shows.
const SETTLE_FRACTION: f32 = 0.02;

/// Progress steps the recent rate spans.
const RECENT_STEPS: usize = 4;

/// Factor applied to the estimate.
const MARGIN: f64 = 1.15;

/// Seconds between two estimates, the 5 s the shortest text rounds to.
const RECALC_SECS: f64 = 5.0;

/// How far a new estimate must exceed the countdown to replace it, as a factor
/// and in seconds.
const RISE: f64 = 1.2;
const RISE_SECS: f64 = 10.0;

/// Follows one job at a time, named by `key`, such as a job number.
#[derive(Default)]
pub struct Eta {
    job: Option<u64>,
    /// Time and fraction when the job was first seen separating.
    first: (f64, f32),
    /// Time and fraction at each of the last steps, oldest first.
    steps: VecDeque<(f64, f32)>,
    /// Time and segments done at each of the last steps, oldest first.
    segments: VecDeque<(f64, u32)>,
    /// When the shown value was computed, and the value.
    shown: Option<(f64, f64)>,
}

impl Eta {
    /// Seconds left on job `key` at `progress`, `now` being the window's clock
    /// and `segment_secs` the server's median segment time. `None` outside
    /// [`Stage::Separating`]. Starts over when `key` changes.
    pub fn update(
        &mut self,
        key: u64,
        now: f64,
        progress: &Progress,
        segment_secs: Option<f64>,
    ) -> Option<f64> {
        let fraction = progress.fraction;
        if progress.stage != Stage::Separating {
            *self = Self::default();
            return None;
        }
        if self.job != Some(key) {
            *self = Self {
                job: Some(key),
                first: (now, fraction),
                steps: VecDeque::from([(now, fraction)]),
                segments: VecDeque::from([(now, progress.completed)]),
                shown: None,
            };
        } else if self.steps.back().is_some_and(|(_, last)| fraction > *last) {
            self.steps.push_back((now, fraction));
            if self.steps.len() > RECENT_STEPS + 1 {
                self.steps.pop_front();
            }
            self.segments.push_back((now, progress.completed));
            if self.segments.len() > RECENT_STEPS + 1 {
                self.segments.pop_front();
            }
        }
        if let Some((at, secs)) = self.shown
            && now - at < RECALC_SECS
        {
            return Some(secs);
        }
        let estimate = match segment_secs.filter(|secs| *secs > 0.0) {
            Some(median) => by_segments(progress, median, self.segments.make_contiguous(), now),
            None => remaining(self.first, self.steps.make_contiguous(), now)?,
        };
        let secs = match self.shown {
            Some((at, secs)) => next_shown(secs - (now - at), estimate),
            None => estimate,
        };
        self.shown = Some((now, secs));
        Some(secs)
    }
}

/// The value to show next, from the countdown on the last one and a new
/// estimate.
fn next_shown(countdown: f64, estimate: f64) -> f64 {
    let countdown = countdown.max(0.0);
    let strong = estimate >= countdown * RISE && estimate - countdown >= RISE_SECS;
    if estimate <= countdown || strong {
        estimate
    } else {
        countdown
    }
}

/// Seconds left at time `now` from the segments left in `progress`, at
/// `median` seconds each or at the pace over `segments`, the time and segments
/// done at each of the last steps, when that pace is slower.
fn by_segments(progress: &Progress, median: f64, segments: &[(f64, u32)], now: f64) -> f64 {
    let left = progress.total.saturating_sub(progress.completed);
    let pace = match (segments.first(), segments.last()) {
        (Some(&(from_at, from)), Some(&(last_at, last)))
            if segments.len() > RECENT_STEPS && last > from =>
        {
            (last_at - from_at) / f64::from(last - from)
        }
        _ => 0.0,
    };
    let each = median.max(pace);
    let into = segments.last().map_or(0.0, |&(at, _)| now - at);
    ((f64::from(left) * each - into.min(each)) * MARGIN).max(0.0)
}

/// Seconds left at time `now`, from the time and fraction when the job was
/// first seen, and at each of the last steps, oldest first.
pub fn remaining(first: (f64, f32), steps: &[(f64, f32)], now: f64) -> Option<f64> {
    let (&(from_at, from), &(last_at, last)) = (steps.first()?, steps.last()?);
    if now - first.0 < SETTLE_SECS || last - first.1 < SETTLE_FRACTION {
        return None;
    }
    let rate = |at: f64, from: f32| {
        let span = last_at - at;
        (span > 0.0).then(|| f64::from(last - from) / span)
    };
    let average = rate(first.0, first.1)?;
    let slowest = rate(from_at, from).map_or(average, |recent| recent.min(average));
    let step = (last_at - from_at) / (steps.len() - 1).max(1) as f64;
    let left = f64::from(1.0 - last) / slowest - (now - last_at).min(step);
    (slowest > 0.0).then(|| (left * MARGIN).max(0.0))
}

/// `about 45 s left`, `about 1 min 20 s left`, or `about 12 min left`. Rounded
/// up to 5 s under a minute, 10 s under 10 min, and whole minutes above.
pub fn left_text(secs: f64) -> String {
    let up = |step: f64| (secs.max(0.0) / step).ceil().max(1.0) * step;
    if up(5.0) < 60.0 {
        return format!("about {} s left", up(5.0));
    }
    let total = if secs < 600.0 { up(10.0) } else { up(60.0) };
    format!("about {} left", duration_text(total))
}

/// `45 s`, `1 min 20 s`, or `2 min`, to the nearest second.
pub fn duration_text(secs: f64) -> String {
    let total = secs.max(0.0).round() as u64;
    match (total / 60, total % 60) {
        (0, s) => format!("{s} s"),
        (min, 0) => format!("{min} min"),
        (min, s) => format!("{min} min {s} s"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn separating(fraction: f32) -> Progress {
        Progress {
            fraction,
            ..Progress::new(Stage::Separating)
        }
    }

    #[test]
    fn the_estimate_waits_for_progress() {
        let steps = [(0.0, 0.1), (1.0, 0.3)];
        assert_eq!(remaining((0.0, 0.1), &steps, 1.0), None, "too soon");
        let steps = [(0.0, 0.1), (10.0, 0.11)];
        assert_eq!(
            remaining((0.0, 0.1), &steps, 10.0),
            None,
            "too little progress"
        );
    }

    #[test]
    fn the_estimate_takes_the_slower_rate_plus_the_margin() {
        let close = |got: Option<f64>, want: f64| {
            let got = got.unwrap();
            assert!((got - want * MARGIN).abs() < 1e-3, "{got}, not {want}");
        };
        // 0.25 in 10 s on average; the last steps went at the same rate.
        close(
            remaining((0.0, 0.25), &[(4.0, 0.35), (10.0, 0.5)], 10.0),
            20.0,
        );
        // The last step slowed to 0.05 in 5 s: the recent rate wins.
        close(
            remaining((0.0, 0.25), &[(5.0, 0.45), (10.0, 0.5)], 10.0),
            50.0,
        );
        // The last step sped up: the average wins.
        close(
            remaining((0.0, 0.25), &[(9.0, 0.3), (10.0, 0.5)], 10.0),
            20.0,
        );
        // Between steps the estimate counts down.
        close(
            remaining((0.0, 0.25), &[(4.0, 0.35), (10.0, 0.5)], 12.0),
            18.0,
        );
    }

    #[test]
    fn with_the_median_the_estimate_shows_at_once() {
        let mut eta = Eta::default();
        let progress = Progress::counted(Stage::Separating, 2, 10);
        let secs = eta.update(1, 0.0, &progress, Some(3.0)).unwrap();
        assert!((secs - 24.0 * MARGIN).abs() < 1e-9, "{secs}");
    }

    #[test]
    fn the_estimate_counts_segments_left_at_the_median_or_a_slower_pace() {
        let close = |got: f64, want: f64| {
            assert!((got - want * MARGIN).abs() < 1e-9, "{got}, not {want}");
        };
        let at = |completed| Progress::counted(Stage::Separating, completed, 10);
        // 8 segments of 3 s, 2 s into the running one.
        close(by_segments(&at(2), 3.0, &[(0.0, 2)], 2.0), 22.0);
        // At most one segment is taken off for the running one.
        close(by_segments(&at(2), 3.0, &[(0.0, 2)], 10.0), 21.0);
        // 4 s per segment over the last 4 segments: the slower pace wins.
        let paced = [(0.0, 0), (4.0, 1), (8.0, 2), (12.0, 3), (16.0, 4)];
        close(by_segments(&at(4), 2.0, &paced, 16.0), 24.0);
        // A pace over fewer segments is not used.
        close(by_segments(&at(1), 2.0, &[(0.0, 0), (8.0, 1)], 8.0), 18.0);
        // The last segment is done.
        close(by_segments(&at(10), 2.0, &[(0.0, 10)], 1.0), 0.0);
    }

    #[test]
    fn a_stalled_step_holds_the_estimate() {
        let steps = [(0.0, 0.1), (4.0, 0.2), (8.0, 0.3)];
        let at = |now| remaining((0.0, 0.1), &steps, now).unwrap();
        assert!(at(10.0) > at(12.0), "counts down within a step");
        assert_eq!(at(12.0), at(60.0), "holds past one step");
    }

    #[test]
    fn the_shown_value_holds_for_5_s() {
        let mut eta = Eta::default();
        assert_eq!(eta.update(1, 0.0, &separating(0.1), None), None);
        let first = eta.update(1, 10.0, &separating(0.5), None).unwrap();
        assert_eq!(eta.update(1, 12.0, &separating(0.7), None), Some(first));
        assert_eq!(eta.update(1, 14.9, &separating(0.8), None), Some(first));
        assert_ne!(eta.update(1, 15.0, &separating(0.8), None), Some(first));
    }

    #[test]
    fn the_countdown_rises_only_past_both_thresholds() {
        assert_eq!(next_shown(40.0, 30.0), 30.0, "a drop is taken");
        assert_eq!(next_shown(40.0, 47.0), 40.0, "a small rise is not");
        assert_eq!(next_shown(20.0, 28.0), 20.0, "nor 8 s more");
        assert_eq!(next_shown(100.0, 115.0), 100.0, "nor 1.15 times");
        assert_eq!(next_shown(40.0, 50.0), 50.0, "1.25 times and 10 s more");
        assert_eq!(next_shown(-3.0, 4.0), 0.0, "an overrun stays at the end");
    }

    #[test]
    fn a_steady_job_counts_down_without_rising() {
        // Chunks of 0.04 every 4 s, read every 0.25 s: the estimate swings
        // with the time since the last chunk.
        let mut eta = Eta::default();
        let mut shown = Vec::new();
        for tick in 0..=400 {
            let t = f64::from(tick) * 0.25;
            let fraction = 0.1 + 0.04 * (t / 4.0).floor() as f32;
            if let Some(secs) = eta.update(1, t, &separating(fraction.min(0.9)), None) {
                shown.push(secs);
            }
        }
        for pair in shown.windows(2) {
            assert!(pair[1] <= pair[0], "rose from {} to {}", pair[0], pair[1]);
        }
        let (start, end) = (shown[0], shown[shown.len() - 1]);
        assert!(end < start / 4.0, "stuck at {end} from {start}");
    }

    #[test]
    fn only_a_separation_has_an_estimate() {
        let mut eta = Eta::default();
        assert_eq!(eta.update(1, 0.0, &separating(0.1), None), None);
        assert!(eta.update(1, 10.0, &separating(0.5), None).is_some());
        for stage in [Stage::Queued, Stage::Analysing, Stage::Writing] {
            assert_eq!(
                eta.update(1, 11.0, &Progress::new(stage), None),
                None,
                "{stage:?}"
            );
        }
        assert_eq!(
            eta.update(1, 12.0, &separating(0.6), None),
            None,
            "restarted"
        );
    }

    #[test]
    fn a_new_job_restarts_the_clock() {
        let mut eta = Eta::default();
        assert_eq!(eta.update(1, 0.0, &separating(0.0), None), None);
        assert!(eta.update(1, 10.0, &separating(0.5), None).is_some());
        assert_eq!(
            eta.update(2, 11.0, &separating(0.5), None),
            None,
            "kept job 1's clock"
        );
        assert!(eta.update(2, 21.0, &separating(0.75), None).is_some());
    }

    #[test]
    fn the_time_left_reads_in_seconds_then_minutes() {
        assert_eq!(left_text(0.4), "about 5 s left");
        assert_eq!(left_text(37.0), "about 40 s left");
        assert_eq!(left_text(57.0), "about 1 min left");
        assert_eq!(left_text(61.0), "about 1 min 10 s left");
        assert_eq!(left_text(80.0), "about 1 min 20 s left");
        assert_eq!(left_text(120.0), "about 2 min left");
        assert_eq!(left_text(599.0), "about 10 min left");
        assert_eq!(left_text(601.0), "about 11 min left");
    }

    #[test]
    fn a_duration_reads_in_minutes_past_60_s() {
        assert_eq!(duration_text(42.4), "42 s");
        assert_eq!(duration_text(60.0), "1 min");
        assert_eq!(duration_text(124.24), "2 min 4 s");
    }
}
