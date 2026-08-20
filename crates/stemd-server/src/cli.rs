//! Command-line surface.

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
    /// Overrides the saved model for this run without replacing it: a flag is
    /// how you try something, and the window is how you choose. Unset loads
    /// whatever the window last selected. Unlike the window, this accepts an
    /// artefact that is not one of the presets.
    #[arg(long)]
    pub demucs_model: Option<String>,

    /// Run the model at float32 instead of float16.
    ///
    /// Half precision is the default: 1.2x faster, and -54 dB against full precision
    /// on the stems that ship, which is 20 dB below the model's own residual. This is
    /// the way back if that trade ever looks wrong on some material.
    #[arg(long)]
    pub full_precision: bool,

    /// Fraction of a demucs segment shared with its neighbour.
    #[arg(long, default_value_t = 0.25)]
    pub overlap: f32,

    /// Address to bind.
    #[arg(long, default_value = "0.0.0.0:8420")]
    pub bind: SocketAddr,

    /// Where separated stems are kept. Defaults to a `stemd` directory under the
    /// platform's cache directory.
    ///
    /// Its contents are deleted at every start, so point it at a directory
    /// stemd owns rather than one that holds anything else.
    #[arg(long)]
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

    /// Jobs that may wait for the worker before submissions are refused with
    /// 429. Separation is serialised on purpose: see queue.rs.
    #[arg(long, default_value_t = 16)]
    pub queue_depth: usize,

    /// Disk the separated stems may occupy, in gigabytes. Past this the least
    /// recently used tracks are dropped. A five-minute stereo track costs about
    /// 106 MB at s16le and 212 MB at f32le.
    #[arg(long, default_value_t = 4.0)]
    pub cache_max_gb: f64,

    /// Seconds a separation nobody pulled must sit idle before it becomes the
    /// first thing given up for space. It is not a delete timer: nothing is
    /// evicted while the stem cache is under half full, however long this is.
    /// The clock restarts on every stem fetched, so a client part-way through
    /// collecting is never cut off.
    #[arg(long, default_value_t = 300)]
    pub unfetched_ttl: u64,

    /// Longest track accepted, which is the only size limit there is.
    ///
    /// Peak memory is roughly 1.8 GB plus 155 MB per audio-minute, so length is what
    /// bounds it. The HTTP body limit is derived from this rather than configured
    /// separately: the upload is raw PCM, so a second knob could only disagree.
    #[arg(long, default_value_t = 10.0)]
    pub max_track_minutes: f64,

    /// mDNS instance name clients will see.
    #[arg(long, default_value = "stemd")]
    pub instance: String,

    /// Do not advertise over mDNS. Clients then need the address out of band.
    #[arg(long)]
    pub no_mdns: bool,

    /// Never reach the network for the model. Fails instead of downloading, so
    /// a machine that is meant to be air-gapped stays that way.
    #[arg(long)]
    pub offline: bool,

    /// Run without a window. The window is the default: this ships as an .app,
    /// where stdout goes nowhere and the log view is the only way to see
    /// anything.
    #[arg(long)]
    pub headless: bool,

    /// Download the CUDA runtime beside this executable and exit. Windows only,
    /// about 1.2 GB, and only worth doing on a machine with an NVIDIA card that
    /// is currently running on its CPU. The startup log says when that is the
    /// case. See [`crate::cuda`].
    #[arg(long)]
    pub install_cuda: bool,
}

impl Args {
    /// Parse the process arguments, dropping the process serial number Finder
    /// passes when launching a bundle: clap would reject it as an unknown flag.
    pub fn from_process() -> Self {
        Self::parse_from(std::env::args_os().filter(|a| !a.to_string_lossy().starts_with("-psn_")))
    }

    /// A precision forced for every model, or `None` to let each be asked what suits
    /// it.
    ///
    /// What precision a model wants depends on the model as much as the backend, and a
    /// chained preset runs two architectures that disagree. See
    /// [`crate::precision::Precisions`].
    pub const fn forced_precision(&self) -> Option<Precision> {
        if self.full_precision {
            Some(Precision::F32)
        } else {
            None
        }
    }

    pub fn max_track_secs(&self) -> f64 {
        self.max_track_minutes * 60.0
    }

    pub fn unfetched_ttl(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.unfetched_ttl)
    }

    pub fn cache_max_bytes(&self) -> u64 {
        const BYTES_PER_GB: f64 = 1e9;
        (self.cache_max_gb.max(0.0) * BYTES_PER_GB) as u64
    }
}
