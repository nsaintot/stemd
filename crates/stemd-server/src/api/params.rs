//! The query string of a submission or session, checked against the loaded model.

use axum::body::Bytes;
use axum::http::StatusCode;
use serde::Deserialize;
use stemd_core::{DspMode, OutputRate, Parts, PcmFormat, StemFormat};

use crate::filters::{Dsp, Filters};
use crate::output::Output;
use crate::state::AppState;

use super::error::ApiError;

#[derive(Debug, Deserialize)]
pub(crate) struct CreateParams {
    #[serde(default = "default_rate")]
    pub(crate) sample_rate: u32,
    #[serde(default = "default_channels")]
    pub(crate) channels: usize,
    /// Encoding of the uploaded body.
    #[serde(default)]
    pub(crate) format: Option<String>,
    /// Container for the stems returned: `flac`, `s16le` or `f32le`. Defaults to
    /// `default_output.format` from `/v2/health`.
    #[serde(default)]
    pub(crate) output_format: Option<String>,
    /// Rate the stems are converted to: 24000, 44100, 48000 or 96000. Defaults
    /// to `default_output.sample_rate` from `/v2/health`.
    #[serde(default)]
    pub(crate) output_sample_rate: Option<String>,
    /// Ship the derived part as well. Off by default: a client holding the mix
    /// can rebuild it for free, and not sending it keeps the transfer at two
    /// stems.
    #[serde(default)]
    pub(crate) include_derived: Option<bool>,
    /// Comma-separated parts to ship, such as `bass,other,vocals`. Defaults to
    /// `default_output.parts` from `/v2/health`. They must not overlap, and must leave out
    /// one part or none: that part is the one `include_derived` adds.
    #[serde(default)]
    pub(crate) parts: Option<String>,
    /// Filter the conversion to `output_sample_rate` runs through. `0`, the
    /// default, converts any pair. See `dsp_modes` in `/v2/health`.
    #[serde(default)]
    pub(crate) dsp_mode: Option<String>,
    /// Id of an uploaded filter the conversion runs through instead. See
    /// `PUT /v2/filters/{id}`.
    #[serde(default)]
    pub(crate) dsp_filter: Option<String>,
    /// The client's key for the file the audio came from, recorded against the track id.
    #[serde(default)]
    pub(crate) file: Option<String>,
}

pub(crate) const fn default_rate() -> u32 {
    44100
}

pub(crate) const fn default_channels() -> usize {
    2
}

/// Duration of an upload, from its length. Exact for raw PCM, and available
/// without decoding it.
pub(super) fn track_seconds(bytes: usize, params: &CreateParams, format: PcmFormat) -> f64 {
    let stride = params.channels * format.bytes_per_sample();
    if stride == 0 || params.sample_rate == 0 {
        return 0.0;
    }
    (bytes / stride) as f64 / f64::from(params.sample_rate)
}

/// Whether `dsp` covers the conversion this job will actually run.
///
/// A mode that covers one rate pair is refused on any other rather than ignored:
/// a client asks for one because it has to match that filter, and quietly giving
/// it a different one is the failure it was trying to avoid. The uploaded rate
/// does not enter into it, since the conversion starts from the model's rate
/// whatever arrived.
pub(super) fn check_dsp(dsp: &Dsp, model_rate: u32, out_rate: OutputRate) -> Result<(), String> {
    let name = match dsp {
        Dsp::Mode(mode) => format!("dsp_mode {mode}"),
        Dsp::Fir(fir) => format!("dsp_filter {}", fir.id),
    };
    match dsp.only_pair() {
        Some((from, to)) if from != model_rate || to != out_rate.hz() => Err(format!(
            "{name} converts {from} to {to} Hz; this job converts {model_rate} to {} Hz",
            out_rate.hz()
        )),
        _ => Ok(()),
    }
}

/// Log once per run that a client still uses a numbered mode.
pub(super) fn warn_numbered_mode_once(mode: DspMode) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            "a client asked for dsp_mode {mode}, which is deprecated; \
             it should upload its filter and send dsp_filter instead"
        );
    }
}

/// The filter a request names with `dsp_mode` or `dsp_filter`.
pub(super) fn parse_dsp(params: &CreateParams, filters: &Filters) -> Result<Dsp, ApiError> {
    let mode: DspMode = match params.dsp_mode.as_deref() {
        Some(s) => parse_param(s)?,
        None => DspMode::default(),
    };
    let Some(id) = params.dsp_filter.as_deref() else {
        if mode != DspMode::General {
            warn_numbered_mode_once(mode);
        }
        return Ok(mode.into());
    };
    if mode != DspMode::General {
        return Err(ApiError::bad_request(
            "dsp_mode and dsp_filter both name a filter; send one",
        ));
    }
    if !crate::filters::is_id(id) {
        return Err(ApiError::bad_request(format!(
            "{id:?} is not a filter id: expected 64 lowercase hex digits"
        )));
    }
    filters.get(id).map(Dsp::Fir).ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            format!("no filter {id}; upload it with PUT /v2/filters/{id}"),
        )
        .with_code("unknown_filter")
    })
}

/// A submission whose query string has been parsed and checked against the body.
#[derive(Clone)]
pub(super) struct Submission {
    pub(super) in_format: PcmFormat,
    pub(super) output: Output,
    pub(super) channels: usize,
    /// Rate the body is at, which the model's need not match.
    pub(super) sample_rate: u32,
    /// Rate the model runs at, read once here so the whole request works from
    /// one answer even if the window switches models underneath it.
    pub(super) model_rate: u32,
}

/// The upload format and the output a request asks for, checked against the
/// loaded model.
pub(crate) fn parse_output(
    params: &CreateParams,
    state: &AppState,
) -> Result<(PcmFormat, Output, u32), ApiError> {
    let in_format: PcmFormat = parse_param(params.format.as_deref().unwrap_or("s16le"))?;
    // Read once: the window can change these between two requests, and a job
    // whose format and rate came from different instants is not one anybody
    // asked for.
    let defaults = state.settings.get();
    let out_format = match params.output_format.as_deref() {
        Some(s) => parse_param(s)?,
        None => defaults.format,
    };
    let out_rate = match params.output_sample_rate.as_deref() {
        Some(s) => parse_param(s)?,
        None => defaults.rate,
    };
    let dsp = parse_dsp(params, &state.filters)?;

    let info = state.queue.info();
    let model_rate = info.sample_rate;
    let parts = parse_parts(params, &info)?;
    check_dsp(&dsp, model_rate, out_rate).map_err(ApiError::bad_request)?;
    note_general_resampler(&dsp, model_rate, out_rate);
    check_carries(out_format, out_rate)?;

    Ok((
        in_format,
        Output {
            format: out_format,
            rate: out_rate,
            derived: params.include_derived.unwrap_or(false),
            dsp,
            parts,
        },
        model_rate,
    ))
}

/// A query parameter parsed as `T`, or a 400 with the parse error.
fn parse_param<T: std::str::FromStr>(s: &str) -> Result<T, ApiError>
where
    T::Err: std::fmt::Display,
{
    s.parse().map_err(|e| ApiError::bad_request(format!("{e}")))
}

/// The parts a request names, or the default, each one the loaded model offers.
fn parse_parts(params: &CreateParams, info: &stemd_core::BackendInfo) -> Result<Parts, ApiError> {
    let parts = match params.parts.as_deref() {
        Some(list) => Parts::parse(list).map_err(ApiError::bad_request)?,
        None => Parts::DEFAULT,
    };
    if let Some(missing) = parts.names().find(|p| !info.parts.iter().any(|q| q == p)) {
        return Err(ApiError::bad_request(format!(
            "{} cannot produce {missing}; it offers {:?}",
            info.model, info.parts
        )));
    }
    Ok(parts)
}

/// Log a job that converts with the general resampler where a numbered mode covers the pair.
fn note_general_resampler(dsp: &Dsp, model_rate: u32, out_rate: OutputRate) {
    // Said out loud, because the way this goes wrong is inaudible from here
    // and unmistakable at the other end. A client rebuilding the derived
    // part subtracts these stems from a mix it resampled itself, and the two
    // filters have to be the same one: mode 1 carries 64 samples of group
    // delay that the general resampler does not, which at 1.5 kHz is a third
    // of a cycle, so the vocals come back rather than cancelling. Not an
    // error, because plenty of clients ask for a rate and subtract nothing.
    if dsp.is_general()
        && let Some(other) = DspMode::for_pair(model_rate, out_rate.hz())
    {
        tracing::info!(
            "converting {model_rate} to {} Hz with the general resampler; \
                 dsp_mode {other} is this server's copy of one client's own \
                 filter for that pair, and this job did not ask for it",
            out_rate.hz()
        );
    }
}

/// A 400 unless `out_format` has a mode at `out_rate`.
fn check_carries(out_format: StemFormat, out_rate: OutputRate) -> Result<(), ApiError> {
    // Refused rather than quietly honoured at a rate the format can carry:
    // a client that asked for 96 kHz mp3 and got 48 has no way to notice.
    if !out_format.carries(out_rate) {
        return Err(ApiError::bad_request(format!(
            "{out_format} has no {} Hz mode; it carries {}",
            out_rate.hz(),
            out_format
                .rates()
                .map(|r| r.hz().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(())
}

impl Submission {
    /// Reject anything malformed before the upload is hashed or decoded.
    pub(super) fn parse(
        params: &CreateParams,
        body: &Bytes,
        state: &AppState,
    ) -> Result<Self, ApiError> {
        let (in_format, output, model_rate) = parse_output(params, state)?;

        if body.is_empty() {
            return Err(ApiError::bad_request("empty payload"));
        }
        // Validated before they are used as divisors below.
        if params.channels == 0 {
            return Err(ApiError::bad_request("channels must be at least 1"));
        }
        if params.sample_rate == 0 {
            return Err(ApiError::bad_request("sample_rate must be at least 1"));
        }

        // Length is exact from the byte count, so an over-long track is refused
        // before it is hashed or decoded rather than after it has been paid for.
        let secs = track_seconds(body.len(), params, in_format);
        let max_secs = state.max_track_secs();
        if secs > max_secs {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "track is {:.1} minutes; this server accepts up to {:.0}",
                    secs / 60.0,
                    max_secs / 60.0
                ),
            ));
        }

        Ok(Self {
            in_format,
            output,
            channels: params.channels,
            sample_rate: params.sample_rate,
            model_rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(channels: usize, sample_rate: u32) -> CreateParams {
        CreateParams {
            sample_rate,
            channels,
            format: None,
            output_format: None,
            output_sample_rate: None,
            include_derived: None,
            parts: None,
            dsp_mode: None,
            dsp_filter: None,
            file: None,
        }
    }

    #[test]
    fn track_length_is_exact_from_the_byte_count() {
        // 10 s of 44.1k stereo s16 = 44100 * 10 * 2 * 2 bytes.
        let bytes = 44100 * 10 * 2 * 2;
        let secs = track_seconds(bytes, &params(2, 44100), PcmFormat::S16le);
        assert!((secs - 10.0).abs() < 1e-9, "got {secs}");

        // The same audio as f32 is twice the bytes and the same duration.
        let secs = track_seconds(bytes * 2, &params(2, 44100), PcmFormat::F32le);
        assert!((secs - 10.0).abs() < 1e-9, "got {secs}");
    }

    /// Mode 0 fits every job. Mode 1 fits exactly one, and says which when it
    /// does not, so a client that asked for the wrong output rate is told rather
    /// than handed the general filter under the name it asked for.
    #[test]
    fn a_numbered_dsp_mode_is_refused_outside_its_own_rate_pair() {
        for rate in OutputRate::ALL {
            assert!(check_dsp(&DspMode::General.into(), 44_100, rate).is_ok());
        }

        assert!(check_dsp(&DspMode::Mode1.into(), 44_100, OutputRate::Hz96000).is_ok());

        let err = check_dsp(&DspMode::Mode1.into(), 44_100, OutputRate::Hz48000)
            .expect_err("48 kHz is not mode 1's pair");
        assert!(err.contains("44100 to 96000"), "{err}");
        assert!(err.contains("44100 to 48000"), "{err}");

        // A model at another rate takes mode 1 out of reach entirely, and the
        // message has to say so rather than blaming the output rate.
        let err = check_dsp(&DspMode::Mode1.into(), 48_000, OutputRate::Hz96000)
            .expect_err("mode 1 starts at 44.1 kHz");
        assert!(err.contains("48000 to 96000"), "{err}");
    }

    #[test]
    fn a_job_names_an_uploaded_filter_by_its_id() {
        let dir = std::env::temp_dir().join(format!("stemd-api-filters-{}", std::process::id()));
        let filters = Filters::open(dir, 4).unwrap();
        let body = crate::filters::encode(44_100, 96_000, &[1.0; 320]);
        let id = crate::filters::id_of(&body);

        let mut p = params(2, 44_100);
        p.dsp_filter = Some(id.clone());
        let err = parse_dsp(&p, &filters).unwrap_err();
        assert_eq!(err.status, StatusCode::NOT_FOUND);
        assert_eq!(err.code, Some("unknown_filter"));

        filters.put(&id, &body).unwrap();
        let dsp = parse_dsp(&p, &filters).unwrap();
        assert_eq!(dsp.to_string(), format!("fir:{id}"));
        assert!(check_dsp(&dsp, 44_100, OutputRate::Hz96000).is_ok());
        let err = check_dsp(&dsp, 44_100, OutputRate::Hz48000).unwrap_err();
        assert!(err.contains(&format!("dsp_filter {id}")), "{err}");

        p.dsp_mode = Some("1".into());
        assert_eq!(
            parse_dsp(&p, &filters).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );

        p.dsp_mode = None;
        p.dsp_filter = Some("not-a-hash".into());
        assert_eq!(
            parse_dsp(&p, &filters).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn a_zero_stride_cannot_divide_by_zero() {
        // Both come straight off the query string, so both are attacker-chosen.
        assert_eq!(
            track_seconds(1024, &params(0, 44100), PcmFormat::S16le),
            0.0
        );
        assert_eq!(track_seconds(1024, &params(2, 0), PcmFormat::S16le), 0.0);
    }
}
