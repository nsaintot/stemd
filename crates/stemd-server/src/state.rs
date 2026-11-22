//! The services every handler, the window and the shutdown path share.

use std::sync::Arc;

use stemd_core::PcmFormat;

use crate::cache::Cache;
use crate::disk::DiskFull;
use crate::endpoint::MoveError;
use crate::filters::Filters;
use crate::jobs::Jobs;
use crate::logbuf::LogBuffer;
use crate::queue::{Queue, QueueFull, QueuedWork};
use crate::settings::SettingsFile;

pub struct AppState {
    pub jobs: Arc<Jobs>,
    /// The one cache of separated stems; the same as the sessions'.
    pub cache: Arc<Cache>,
    pub queue: Arc<Queue>,
    pub switcher: Arc<crate::switch::Switcher>,
    pub logs: LogBuffer,
    /// Files dropped on the window, and what became of them.
    pub drops: Arc<crate::drops::Drops>,
    /// What a request that names no output format or rate gets. Live: the window
    /// changes these while the server is running.
    pub settings: Arc<SettingsFile>,
    /// Whether the API answers, and its mDNS advertisement. Shared so every
    /// exit path can withdraw the advertisement: macOS skips Drop on Cmd-Q.
    pub serving: crate::serving::Serving,
    /// Conversion filters clients uploaded.
    pub filters: Arc<Filters>,
    /// Empty the cache at quit, from `--ephemeral`.
    pub ephemeral: bool,
    /// Where the API listens.
    pub endpoint: crate::endpoint::Endpoint,
}

/// Why a claimed job was not queued.
#[derive(Debug, thiserror::Error)]
pub enum Refused {
    #[error(transparent)]
    Disk(#[from] DiskFull),
    #[error(transparent)]
    Full(#[from] QueueFull),
}

impl AppState {
    /// Queue `work` and return its position, or fail its job when the cache volume or the queue is full.
    pub fn queue_claimed(&self, work: QueuedWork) -> Result<usize, Refused> {
        let job = Arc::clone(&work.job);
        // Stems are written to the cache before they are served, and a full disk
        // also leaves the system no room to swap.
        let queued = crate::disk::ensure_room(self.cache.root())
            .map_err(Refused::from)
            .and_then(|()| self.queue.submit(work).map_err(Refused::from));
        // A job left at Queued would hold its key, and later requests for
        // the track would join it and wait forever.
        if let Err(err) = &queued {
            job.fail(err.to_string());
        }
        queued
    }

    /// Cancel job `id` and drop it from the table. Returns whether the worker was running it.
    pub fn stop_job(&self, id: &str) -> bool {
        let cancelled = self.queue.cancel(id);
        self.jobs.remove(id);
        cancelled
    }
    /// Longest track accepted: the saved length, never past what this machine's memory holds.
    pub fn max_track_secs(&self) -> f64 {
        self.settings
            .get()
            .max_track_secs(crate::disk::longest_track_minutes())
    }

    /// Turn the API on or off and save the choice. Turning it off cancels the API jobs, whose
    /// clients can no longer collect them; returns how many it cancelled.
    pub fn set_serving(&self, on: bool) -> usize {
        self.serving.set(on, &self.queue.info().model);
        self.settings.set_serving(on);
        if on {
            return 0;
        }
        let cancelled = self.queue.cancel_api_jobs();
        if cancelled > 0 {
            tracing::info!("cancelled {cancelled} API job(s)");
        }
        cancelled
    }

    /// Delete every track no session is playing, on a thread of its own.
    pub fn empty_cache(&self) {
        let cache = Arc::clone(&self.cache);
        std::thread::spawn(move || {
            let (tracks, freed) = cache.clear();
            tracing::info!(
                "cache emptied: {tracks} tracks, {:.0} MB",
                freed as f64 / 1e6
            );
        });
    }

    /// Apply the saved cache size and age on a thread of its own, then pass `done` the tracks
    /// and bytes deleted.
    pub fn apply_cache_limits(&self, done: impl FnOnce(usize, u64) + Send + 'static) {
        let (max_bytes, whole_max_age) = self.settings.get().cache_limits();
        let cache = Arc::clone(&self.cache);
        std::thread::spawn(move || {
            let (tracks, freed) = cache.set_limits(max_bytes, whole_max_age);
            done(tracks, freed);
        });
    }

    /// Largest request body to buffer: the longest track at the widest format.
    pub fn max_body_bytes(&self) -> usize {
        max_body_bytes(
            self.max_track_secs(),
            highest_rate(),
            self.queue.info().channels,
        )
    }
}

impl AppState {
    /// Serve on `port` from now on: rebind, advertise it, and save it.
    pub fn move_to_port(&self, port: u16) -> Result<(), MoveError> {
        if port == self.endpoint.port() {
            return Ok(());
        }
        self.endpoint.move_to(port)?;
        self.serving.move_to(port, &self.queue.info().model);
        self.settings.set_port(port);
        tracing::info!("listening on http://{}", self.endpoint.addr());
        Ok(())
    }
}

/// Bytes the longest permitted track can occupy, at the widest sample format the
/// API accepts. `f32le`, the larger of the two. This bounds what axum buffers
/// before the handler runs, so it has to be an upper bound.
///
/// Taken at the highest rate the API names rather than the model's, since an
/// upload arrives at whatever rate the client holds and is converted here. Above
/// that rate the body is refused by length; the duration limit still applies
/// underneath it, so this only decides how far past 44.1 kHz an upload may go.
pub fn max_body_bytes(max_track_secs: f64, sample_rate: u32, channels: usize) -> usize {
    let per_second =
        f64::from(sample_rate) * channels as f64 * PcmFormat::F32le.bytes_per_sample() as f64;
    (max_track_secs * per_second) as usize
}

/// The highest rate `output_sample_rate` names, and the highest an upload can
/// carry within the body limit.
pub fn highest_rate() -> u32 {
    stemd_core::OutputRate::ALL
        .iter()
        .map(|rate| rate.hz())
        .max()
        .unwrap_or(96_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turning_the_server_off_and_on_is_saved() {
        let state = crate::testkit::state();
        assert_eq!(state.set_serving(false), 0);
        assert!(!state.serving.is_on());
        assert!(!state.settings.get().serving);
        assert_eq!(state.set_serving(true), 0);
        assert!(state.serving.is_on());
        assert!(state.settings.get().serving);
    }

    #[test]
    fn emptying_the_cache_deletes_its_tracks_in_the_background() {
        let state = crate::testkit::state();
        let dir = state.cache.verified_dir("model", "type", "track");
        state
            .cache
            .write_verified(&dir, "vocals", 0, "flac", &[0; 10])
            .expect("write");
        assert_eq!(state.cache.stats().tracks, 1);
        state.empty_cache();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while state.cache.stats().tracks > 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "the cache was not emptied"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    #[test]
    fn applying_the_cache_limits_reports_what_it_deleted() {
        let state = crate::testkit::state();
        let (sent, received) = std::sync::mpsc::channel();
        state.apply_cache_limits(move |tracks, freed| sent.send((tracks, freed)).unwrap());
        let got = received
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("done was called");
        assert_eq!(got, (0, 0));
    }

    #[test]
    fn the_body_limit_admits_the_longest_track_at_every_rate_it_names() {
        // The two limits are one number seen twice; the body limit must never
        // reject audio the duration gate would have allowed, or a legal upload
        // fails before the handler can explain why. An upload arrives at
        // whatever rate the client holds, so it has to hold at each of them.
        let secs = 600.0;
        let limit = max_body_bytes(secs, highest_rate(), 2);

        for rate in stemd_core::OutputRate::ALL {
            let f32_track = (secs * f64::from(rate.hz()) * 2.0 * 4.0) as usize;
            let s16_track = (secs * f64::from(rate.hz()) * 2.0 * 2.0) as usize;
            assert!(limit >= f32_track, "{rate}: {limit} < {f32_track}");
            assert!(limit >= s16_track);
        }

        // And one second more at the highest of them must not fit, or the gate
        // is the only thing holding the line.
        let widest = f64::from(highest_rate()) * 2.0 * 4.0;
        assert!(limit < ((secs + 1.0) * widest) as usize);
    }

    #[test]
    fn the_body_limit_follows_the_rate_and_channel_count() {
        assert_eq!(
            max_body_bytes(600.0, 48000, 2),
            max_body_bytes(600.0, 44100, 2) * 48000 / 44100
        );
        assert_eq!(
            max_body_bytes(600.0, 44100, 4),
            max_body_bytes(600.0, 44100, 2) * 2
        );
    }
}
