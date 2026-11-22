//! The command-line flags.

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;
use stemd_core::Precision;

#[derive(Parser, Debug)]
#[command(name = "stemd", about, version)]
pub struct Args {
    /// Directory holding the model artefacts.
    #[arg(long, default_value = "models")]
    pub models: PathBuf,

    /// Weights file inside --models, without the extension.
    ///
    /// Overrides the saved model for this run and leaves the saved choice unchanged. Unset,
    /// the server loads the model last selected in the window. Unlike the window, this
    /// accepts an artefact that is not a preset.
    #[arg(long)]
    pub demucs_model: Option<String>,

    /// Run every model at float32.
    ///
    /// By default each model runs at the precision that suits it on this backend. Half
    /// precision is about 1.2x faster and differs from full precision by -54 dB on the
    /// shipped stems, 20 dB below the model's own residual. Use this flag if half precision
    /// sounds wrong on some material.
    #[arg(long)]
    pub full_precision: bool,

    /// Fraction of a demucs segment shared with its neighbour.
    #[arg(long, default_value_t = 0.25)]
    pub overlap: f32,

    /// Address to bind, such as `127.0.0.1:8420`.
    ///
    /// Overrides the saved port for this run. Unset binds every interface on the
    /// saved port, 8420 by default.
    #[arg(long)]
    pub bind: Option<SocketAddr>,

    /// Where separated stems are kept, across restarts. Defaults to a `stemd`
    /// directory under the platform's cache directory.
    ///
    /// Point it at a directory stemd owns: the cache deletes what it no longer
    /// needs.
    #[arg(long, alias = crate::migrate::PIECES_DIR)]
    pub cache_dir: Option<PathBuf>,

    /// How stems are written when a request does not say: `flac`, `s16le` or `f32le`.
    ///
    /// FLAC carries the same 16-bit samples as `s16le` in roughly half the bytes.
    /// `f32le` doubles the size and exists for an exact null with no quantisation.
    /// Overrides the saved default for this run.
    #[arg(long)]
    pub output_format: Option<String>,

    /// Rate stems are converted to when a request does not say: 24000, 44100,
    /// 48000 or 96000. 44100 is the model's own rate and runs no conversion.
    ///
    /// Overrides the saved default for this run. Unset uses it.
    #[arg(long)]
    pub output_sample_rate: Option<String>,

    /// Where preferences the window changes are kept. Defaults to
    /// `settings.json` in the platform's per-user data directory.
    #[arg(long)]
    pub settings: Option<PathBuf>,

    /// Jobs that may wait for the worker before submissions are refused with 429.
    /// Separation runs one job at a time on purpose; see queue/mod.rs.
    #[arg(long, default_value_t = 16)]
    pub queue_depth: usize,

    /// Seconds a session may sit with nothing to run and no request from its
    /// client before it is ended and its memory freed.
    #[arg(long, default_value_t = 30)]
    pub session_idle_secs: u64,

    /// Sessions that may be open at once. Each holds its track and its stems in
    /// memory: about 0.9 GB for a six-minute track on Fast.
    #[arg(long, default_value_t = 4)]
    pub max_sessions: usize,

    /// Disk the cache may occupy, in gigabytes, and never more than the free disk minus 2
    /// GB. Past this, partial tracks are deleted oldest first, then whole tracks by last
    /// use. A six-minute track takes about 40 MB as 44.1 kHz FLAC with two parts and the
    /// derived one.
    ///
    /// Overrides the saved setting, 8 by default, for this run.
    #[arg(long, alias = "pieces-max-gb")]
    pub cache_max_gb: Option<f64>,

    /// Hours a partial track, with some stem not stored whole, is kept unused.
    #[arg(long, default_value_t = 24.0)]
    pub partial_max_hours: f64,

    /// Days a whole track is kept unused; 0 keeps it until the cache is full.
    ///
    /// Overrides the saved setting, 30 by default, for this run.
    #[arg(long)]
    pub whole_max_days: Option<f64>,

    /// Empty the cache at start and at quit, for a borrowed machine.
    #[arg(long)]
    pub ephemeral: bool,

    /// Seconds a finished job's result is kept while some of its stems have not been
    /// fetched. Each stem fetch restarts the count.
    #[arg(long, default_value_t = 300)]
    pub unfetched_ttl: u64,

    /// Longest track accepted, in minutes. This is the only size limit.
    ///
    /// Peak memory is roughly 1.8 GB plus 155 MB per audio-minute, so the track length
    /// bounds it. The HTTP body limit is derived from this value, since the upload is raw
    /// PCM.
    ///
    /// Overrides the saved setting, 10 by default, for this run. Capped at the
    /// longest track that fits in three quarters of the machine's memory; 0 uses
    /// that cap.
    #[arg(long)]
    pub max_track_minutes: Option<f64>,

    /// mDNS instance name clients will see.
    #[arg(long, default_value = "stemd")]
    pub instance: String,

    /// Do not advertise over mDNS. Clients then need the address out of band.
    #[arg(long)]
    pub no_mdns: bool,

    /// Never download a model. A missing model fails the start, so an air-gapped machine
    /// stays offline.
    #[arg(long)]
    pub offline: bool,

    /// Run without a window. The window is the default because the app ships as a bundle,
    /// where stdout is not visible and the window's log view shows the log.
    #[arg(long)]
    pub headless: bool,

    /// Download the CUDA runtime, about 1.2 GB, beside this executable and exit. Windows
    /// only. Use it on a machine with an NVIDIA card that currently runs on its CPU; the
    /// startup log says when that is the case.
    #[arg(long)]
    pub install_cuda: bool,
}

impl Args {
    /// Parse the process arguments, dropping the process serial number Finder
    /// passes when launching a bundle: clap would reject it as an unknown flag.
    pub fn from_process() -> Self {
        Self::parse_from(std::env::args_os().filter(|a| !a.to_string_lossy().starts_with("-psn_")))
    }

    /// A precision forced for every model, or `None` to let each model choose.
    ///
    /// The right precision depends on the model as much as on the backend, and a chained preset
    /// runs two architectures that need different precisions. See [`crate::precision::Precisions`].
    pub const fn forced_precision(&self) -> Option<Precision> {
        if self.full_precision {
            Some(Precision::F32)
        } else {
            None
        }
    }

    pub fn unfetched_ttl(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.unfetched_ttl)
    }

    /// The cache root `--cache-dir` names, or [`crate::cache::default_dir`].
    pub fn cache_root(&self) -> anyhow::Result<std::path::PathBuf> {
        match self.cache_dir.clone() {
            Some(dir) => Ok(dir),
            None => crate::cache::default_dir(),
        }
    }

    /// The cache's deletion rules, with the size and whole-track age `settings` holds.
    pub fn cache_rules(&self, settings: &crate::settings::Settings) -> crate::cache::Rules {
        let (max_bytes, whole_max_age) = settings.cache_limits();
        crate::cache::Rules {
            max_bytes,
            partial_max_age: crate::settings::saturating_secs(self.partial_max_hours * 3600.0),
            whole_max_age,
            unfetched_ttl: self.unfetched_ttl(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_partial_age_past_what_a_duration_holds_keeps_partial_tracks_forever() {
        let args = Args::parse_from(["stemd", "--partial-max-hours", "inf"]);
        let rules = args.cache_rules(&crate::settings::Settings::default());
        assert_eq!(rules.partial_max_age, std::time::Duration::MAX);
    }
}
