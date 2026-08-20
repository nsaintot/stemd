//! HTTP client for a stemd server.
//!
//! The response types are declared here rather than shared with the server:
//! depending on `stemd-core` would pull the whole model stack into a client that
//! is HTTP plus audio I/O. They mirror `stemd_server::jobs` and must be kept in
//! step with it.

use std::io::Read;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::discovery;

/// How long to wait between polls of a running job.
const POLL_INTERVAL: Duration = Duration::from_millis(300);

#[derive(Debug, Deserialize)]
pub struct Health {
    pub model: String,
    #[serde(default = "default_rate")]
    pub sample_rate: u32,
    /// Stems the server transfers. The remaining part is rebuilt client-side.
    #[serde(default)]
    pub stems: Vec<String>,
    #[serde(default = "unbounded")]
    pub max_track_seconds: f64,
    /// Output rates this server offers. Empty when it is too old to say.
    #[serde(default)]
    pub output_sample_rates: Vec<u32>,
    /// DSP modes this server offers. Empty when it is too old to say.
    #[serde(default)]
    pub dsp_modes: Vec<u8>,
}

const fn default_rate() -> u32 {
    44100
}

const fn unbounded() -> f64 {
    f64::INFINITY
}

#[derive(Debug, Deserialize)]
pub struct StemRef {
    pub name: String,
    /// Path on this server, relative to its base URL.
    pub url: String,
    /// Scale already applied to the samples. Multiply by `1.0 / gain` to
    /// restore the original level.
    #[serde(default = "unity")]
    pub gain: f32,
}

const fn unity() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
pub struct JobResult {
    /// Rate the stems are at, which is the requested output rate rather than
    /// the rate of the mix that was uploaded.
    pub sample_rate: u32,
    pub format: String,
    pub stems: Vec<StemRef>,
    pub model_residual_db: f64,
    pub separation_secs: f64,
    pub realtime_factor: f64,
}

#[derive(Debug, Deserialize)]
struct Progress {
    stage: String,
    #[serde(default)]
    completed: u64,
    #[serde(default)]
    total: u64,
    #[serde(default)]
    fraction: f64,
}

#[derive(Debug, Deserialize)]
struct JobView {
    progress: Progress,
    #[serde(default)]
    result: Option<JobResult>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Created {
    id: String,
}

/// How the samples are encoded in each direction.
#[derive(Debug, Clone, Copy)]
pub struct Formats {
    /// Encoding of the uploaded PCM.
    pub upload: &'static str,
    /// Container the stems come back in.
    pub download: &'static str,
    /// Rate the server converts the stems to, when not its native one.
    pub rate: Option<u32>,
    /// Ask the server for the derived part rather than rebuilding it.
    pub derived: bool,
    /// Filter the server converts the stems with, when not its default.
    pub dsp: Option<u8>,
}

impl Formats {
    /// Stems come back as FLAC by default: the same 16-bit samples, half the
    /// wire. `--f32` swaps both directions to float for an exact null.
    pub const fn new(f32_everywhere: bool) -> Self {
        if f32_everywhere {
            Self {
                upload: "f32le",
                download: "f32le",
                rate: None,
                derived: false,
                dsp: None,
            }
        } else {
            Self {
                upload: "s16le",
                download: "flac",
                rate: None,
                derived: false,
                dsp: None,
            }
        }
    }

    #[must_use]
    pub const fn at_rate(mut self, rate: Option<u32>) -> Self {
        self.rate = rate;
        self
    }

    #[must_use]
    pub const fn with_derived(mut self, derived: bool) -> Self {
        self.derived = derived;
        self
    }

    #[must_use]
    pub const fn with_dsp_mode(mut self, dsp: Option<u8>) -> Self {
        self.dsp = dsp;
        self
    }
}

pub struct Server {
    base: String,
}

impl Server {
    /// Connect to `host`, discovering one over mDNS when it is not given.
    pub fn connect(host: Option<String>, discover_timeout: Duration) -> Result<Self> {
        let host = match host {
            Some(host) => host,
            None => discovery::discover(discover_timeout)?,
        };
        Ok(Self {
            base: format!("http://{host}"),
        })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn health(&self) -> Result<Health> {
        get_json(&format!("{}/v1/health", self.base))
    }

    /// Upload the mix and return the new job's id.
    pub fn submit(&self, body: &[u8], sample_rate: u32, formats: Formats) -> Result<String> {
        let mut url = format!(
            "{}/v1/jobs?sample_rate={sample_rate}&channels=2&format={}&output_format={}",
            self.base, formats.upload, formats.download,
        );
        if let Some(rate) = formats.rate {
            url.push_str(&format!("&output_sample_rate={rate}"));
        }
        if formats.derived {
            url.push_str("&include_derived=true");
        }
        if let Some(mode) = formats.dsp {
            url.push_str(&format!("&dsp_mode={mode}"));
        }
        let response = ureq::post(&url)
            .header("content-type", "application/octet-stream")
            .send(body)
            .map_err(|e| anyhow::anyhow!("submitting job: {e}"))?;
        let created: Created = serde_json::from_str(&read_body(response)?)?;
        Ok(created.id)
    }

    /// Poll until the job reaches a terminal stage, reporting progress changes
    /// to `on_progress` as they happen.
    pub fn wait(&self, id: &str, mut on_progress: impl FnMut(&str, f64)) -> Result<JobResult> {
        let url = format!("{}/v1/jobs/{id}", self.base);
        let mut last = String::new();
        loop {
            let job: JobView = get_json(&url)?;
            let line = describe(&job.progress);
            if line != last {
                on_progress(&line, job.progress.fraction);
                last = line;
            }

            match job.progress.stage.as_str() {
                "done" => {
                    return job
                        .result
                        .context("server reported done with no result attached");
                }
                "failed" => bail!(
                    "server reported failure: {}",
                    job.error.as_deref().unwrap_or("unknown")
                ),
                // Not reachable from here, this client never cancels, and a job
                // it is waiting on holds a reference that stops anyone else's
                // DELETE from stopping it. Handled anyway, because the arm below
                // would wait forever on a terminal stage it does not recognise.
                "cancelled" => bail!("the job was cancelled"),
                _ => std::thread::sleep(POLL_INTERVAL),
            }
        }
    }

    /// Download one stem, whose `url` is relative to this server.
    pub fn fetch_stem(&self, stem: &StemRef) -> Result<Vec<u8>> {
        get_bytes(&format!("{}{}", self.base, stem.url))
    }
}

fn describe(progress: &Progress) -> String {
    match progress.total {
        0 => progress.stage.clone(),
        total => format!("{} {}/{total}", progress.stage, progress.completed),
    }
}

fn get_json<T: serde::de::DeserializeOwned>(url: &str) -> Result<T> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    let body = read_body(response)?;
    serde_json::from_str(&body).with_context(|| format!("parsing the response from {url}"))
}

fn get_bytes(url: &str) -> Result<Vec<u8>> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    let mut buf = Vec::new();
    response.into_body().into_reader().read_to_end(&mut buf)?;
    Ok(buf)
}

fn read_body(response: ureq::http::Response<ureq::Body>) -> Result<String> {
    let mut body = String::new();
    response
        .into_body()
        .into_reader()
        .read_to_string(&mut body)?;
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_countable_stage_shows_its_counter() {
        let progress = Progress {
            stage: "separating".into(),
            completed: 3,
            total: 12,
            fraction: 0.25,
        };
        assert_eq!(describe(&progress), "separating 3/12");
    }

    #[test]
    fn an_uncountable_stage_is_just_its_name() {
        let progress = Progress {
            stage: "queued".into(),
            completed: 0,
            total: 0,
            fraction: 0.0,
        };
        assert_eq!(describe(&progress), "queued");
    }

    #[test]
    fn a_health_response_survives_fields_this_client_ignores() {
        // The server sends a good deal more than this; a client that broke on
        // an unknown field would break on every server-side addition.
        let health: Health = serde_json::from_str(
            r#"{"model":"htdemucs","sample_rate":44100,"stems":["harmonics","vocals"],
                "max_track_seconds":600.0,"backend":"demucs","cached_tracks":3}"#,
        )
        .unwrap();
        assert_eq!(health.model, "htdemucs");
        assert_eq!(health.stems.len(), 2);
        assert!((health.max_track_seconds - 600.0).abs() < 1e-9);
    }

    #[test]
    fn a_health_response_missing_optional_fields_falls_back() {
        let health: Health = serde_json::from_str(r#"{"model":"x"}"#).unwrap();
        assert_eq!(health.sample_rate, 44100);
        assert!(health.stems.is_empty());
        assert!(
            health.max_track_seconds.is_infinite(),
            "an unstated limit must not reject every track"
        );
    }

    #[test]
    fn formats_pair_the_wire_encodings() {
        assert_eq!(
            Formats::new(false).rate,
            None,
            "the default asks for no conversion"
        );
        assert_eq!(Formats::new(false).at_rate(Some(48_000)).rate, Some(48_000));
        assert_eq!(Formats::new(false).upload, "s16le");
        assert_eq!(
            Formats::new(false).download,
            "flac",
            "the default must not spend twice the wire on float"
        );
        let exact = Formats::new(true);
        assert_eq!((exact.upload, exact.download), ("f32le", "f32le"));
    }
}
