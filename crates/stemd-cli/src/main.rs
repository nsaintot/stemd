//! stemd-cli: send a track to a stemd server and write the stems out.
//!
//! Exists so the separation can be judged by ear on real music.
//!
//! ```text
//! discovery  find a server over mDNS
//! client     submit the mix, poll, download the stems
//! decode     read the input file, and the FLAC stems that come back
//! output     assemble the parts and write them
//! ```

mod client;
mod discovery;
mod output;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;

use crate::client::{Formats, Health, JobResult, Server};
use crate::output::{Encoding, Part, Planar};
use stemd_audio::Decoded;

/// The part the client rebuilds from the mix; stem names come from `/v2/health`.
const DERIVED: &str = "drums";

#[derive(Parser, Debug)]
#[command(name = "stemd-cli", about, version)]
struct Args {
    /// Audio file to separate (wav, mp3, flac, m4a...).
    input: PathBuf,

    /// Server as host:port. Discovered over mDNS when omitted.
    #[arg(long)]
    host: Option<String>,

    /// Where to write the stems. Defaults to <input stem>-stems/.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Upload and receive float samples instead of 16-bit. Doubles the
    /// transfer; only worth it to see the exact reconstruction null.
    #[arg(long)]
    f32: bool,

    /// Write float32 wavs instead of FLAC. Unscaled and four times the size;
    /// use it when the output is going into a DAW and headroom matters more
    /// than disk.
    #[arg(long)]
    wav_f32: bool,

    /// Ask the server to convert the stems to this rate in Hz (24000, 44100,
    /// 48000, 96000). Defaults to the model's own rate.
    ///
    /// The reconstruction below only nulls at the native rate: the mix is not
    /// resampled with it, so `drums` is derived from buffers at two rates.
    #[arg(long)]
    output_sample_rate: Option<u32>,

    /// Ask the server to send the derived part instead of rebuilding it here.
    ///
    /// Costs a third of the transfer again. Worth it when the mix is not at the
    /// stems' rate, since rebuilding then needs the mix resampled to match.
    #[arg(long)]
    include_derived: bool,

    /// Parts to ask for, comma-separated, such as `bass,other,vocals`. Defaults
    /// to the server's `harmonics,vocals`. Whatever the list leaves out is
    /// rebuilt here from the mix; `/v2/health` lists what a server offers.
    #[arg(long)]
    parts: Option<String>,

    /// Filter the server converts the stems with. 0, the default, converts any
    /// rate pair; a numbered mode covers one pair and is refused on any other.
    ///
    /// Deprecated: use --dsp-filter. Only ask for one if you are subtracting the
    /// stems from a mix converted by that same filter. `/v2/health` lists what a
    /// server offers.
    #[arg(long, conflicts_with = "dsp_filter")]
    dsp_mode: Option<u8>,

    /// Filter file (`.fir`) the server converts the stems with. Uploaded first
    /// if the server does not hold it. Needs --output-sample-rate set to the
    /// filter's output rate; see `dsp_filter` in docs/api.md for the format.
    #[arg(long)]
    dsp_filter: Option<PathBuf>,

    /// Seconds to wait for mDNS discovery.
    #[arg(long, default_value_t = 5)]
    discover_timeout: u64,
}

impl Args {
    /// Where the stems go, defaulting to a directory beside the input.
    fn out_dir(&self) -> PathBuf {
        self.out.clone().unwrap_or_else(|| {
            let stem = self
                .input
                .file_stem()
                .map_or_else(|| "track".into(), |s| s.to_string_lossy().into_owned());
            PathBuf::from(format!("{stem}-stems"))
        })
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let server = Server::connect(
        args.host.clone(),
        Duration::from_secs(args.discover_timeout),
    )?;
    println!("server       : {}", server.base());
    let health = server.health()?;
    report_server(&health);

    check_rate(&args, &health)?;
    let filter = upload_filter(&args, &server)?;
    let audio = load_input(&args, &health)?;
    let formats = Formats::new(args.f32)
        .at_rate(args.output_sample_rate)
        .with_derived(args.include_derived)
        .with_parts(args.parts.as_deref())
        .with_dsp_mode(args.dsp_mode)
        .with_dsp_filter(filter.as_deref());

    let started = Instant::now();
    let result = separate(&server, &audio, formats)?;
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "separated    : {:.2}s server-side ({:.1}x realtime)",
        result.separation_secs, result.realtime_factor,
    );

    let parts = assemble(&server, &audio, &result, args.f32)?;
    write_parts(&args, &result, &parts)?;
    report_reconstruction(&audio, &parts, &result, elapsed);
    Ok(())
}

/// Refuse a rate or a mode this server does not offer, before the upload is
/// spent.
///
/// Each check is skipped when the server does not advertise the field at all, so
/// an older one is left to answer for itself rather than being second-guessed
/// here.
fn check_rate(args: &Args, health: &Health) -> Result<()> {
    if let Some(rate) = args.output_sample_rate
        && !health.output_sample_rates.is_empty()
        && !health.output_sample_rates.contains(&rate)
    {
        bail!(
            "this server converts to {} Hz, not {rate}",
            health
                .output_sample_rates
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" / ")
        );
    }

    if let Some(mode) = args.dsp_mode
        && !health.dsp_modes.is_empty()
        && !health.dsp_modes.contains(&mode)
    {
        bail!(
            "this server offers dsp mode {}, not {mode}",
            health
                .dsp_modes
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(" / ")
        );
    }
    Ok(())
}

/// Make sure the server holds the `--dsp-filter` file, and return its id.
fn upload_filter(args: &Args, server: &Server) -> Result<Option<String>> {
    let Some(path) = &args.dsp_filter else {
        return Ok(None);
    };
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let (filter, uploaded) = server.ensure_filter(&bytes)?;
    if args.output_sample_rate != Some(filter.out_rate) {
        bail!(
            "{} converts to {} Hz; pass --output-sample-rate {}",
            path.display(),
            filter.out_rate,
            filter.out_rate
        );
    }
    println!(
        "filter       : {} ({} to {} Hz, {} taps), {}",
        &filter.id[..12],
        filter.in_rate,
        filter.out_rate,
        filter.taps,
        if uploaded {
            "uploaded"
        } else {
            "already on the server"
        }
    );
    Ok(Some(filter.id))
}

fn report_server(health: &Health) {
    println!(
        "model        : {} ({} Hz)",
        health.model, health.sample_rate
    );
    println!(
        "transfers    : {} stems [{}] by default — `{}` is rebuilt here",
        health.default_output.parts.len(),
        health.default_output.parts.join(", "),
        health.default_output.derived.as_deref().unwrap_or(DERIVED),
    );
}

/// Decode the input and refuse anything this server cannot take.
///
/// Refused locally rather than spending the upload to be told to take it back.
/// The limit comes from the server so the two cannot disagree. The rate does not:
/// the server converts whatever arrives to the rate its model runs at.
fn load_input(args: &Args, health: &Health) -> Result<Decoded> {
    print!("decoding     : {} ... ", args.input.display());
    let audio = stemd_audio::decode(&args.input)?;
    println!(
        "{:.1}s, {} ch, {} Hz",
        audio.duration_secs(),
        audio.data.len(),
        audio.sample_rate
    );

    if audio.duration_secs() > health.max_track_seconds {
        bail!(
            "track is {:.1} minutes; this server accepts up to {:.0}. \
             Separation memory grows with length, so longer tracks are refused \
             rather than risked — trim it or raise --max-track-minutes.",
            audio.duration_secs() / 60.0,
            health.max_track_seconds / 60.0
        );
    }
    if audio.sample_rate != health.sample_rate {
        println!(
            "note         : {} Hz, so the server converts it to the model's {} Hz",
            audio.sample_rate, health.sample_rate
        );
    }
    Ok(audio)
}

/// Upload the mix and wait for the stems.
fn separate(server: &Server, audio: &Decoded, formats: Formats<'_>) -> Result<JobResult> {
    let body = audio.to_interleaved(formats.upload == "f32le");
    println!(
        "uploading    : {:.1} MB as {}, stems back as {}",
        body.len() as f64 / 1e6,
        formats.upload,
        formats.download,
    );

    let id = server.submit(&body, audio.sample_rate, formats)?;
    server.wait(&id, |stage, fraction| {
        println!("  {stage:<24} {:>3}%", (fraction * 100.0).round() as i64);
    })
}

/// Download the shipped stems and rebuild the derived part from the mix.
///
/// Deriving it here is exactly what a player does, so these files are what the
/// three faders control, but only two of them came off the wire, and the labels
/// say which.
fn assemble(
    server: &Server,
    audio: &Decoded,
    result: &JobResult,
    f32_wire: bool,
) -> Result<Vec<Part>> {
    let mut downloaded: HashMap<&str, Planar> = HashMap::new();
    let mut parts = Vec::with_capacity(result.stems.len() + 1);

    for stem in &result.stems {
        let raw = server.fetch_stem(stem)?;
        // Each stem carries its own scale so a quiet one keeps its bits.
        let samples = if result.output.format == "flac" {
            rescale(
                stemd_audio::decode_flac(raw)?.data,
                DECODED_S16_TO_STEM * inverse(stem.gain),
            )
        } else {
            deinterleave(&raw, f32_wire, stem.gain)
        };
        downloaded.insert(stem.name.as_str(), samples.clone());
        parts.push(Part {
            origin: "downloaded",
            name: stem.name.clone(),
            samples,
        });
    }

    // Nothing left to rebuild, or already sent because it was asked for.
    let Some(derived_name) = result.output.derived.as_deref() else {
        return Ok(parts);
    };
    if parts.iter().any(|part| part.name == derived_name) {
        return Ok(parts);
    }

    // `mix - stems` only lines up when both sides share a rate. Rebuilding the
    // derived part at a converted rate means resampling this mix with the same
    // filter the server used, which this client does not carry, so it is left
    // out rather than computed from buffers that do not correspond. Pass
    // --include-derived to have the server send it instead.
    if result.output.sample_rate != audio.sample_rate {
        println!(
            "note         : stems at {} Hz, so `{derived_name}` is not rebuilt — pass \
             --include-derived to have the server send it",
            result.output.sample_rate
        );
        return Ok(parts);
    }

    let mut derived = audio.data.clone();
    for stem in &result.stems {
        if let Some(samples) = downloaded.get(stem.name.as_str()) {
            derived = output::subtract(&derived, samples);
        }
    }
    parts.push(Part {
        origin: "rebuilt here",
        name: derived_name.to_owned(),
        samples: derived,
    });

    Ok(parts)
}

fn write_parts(args: &Args, result: &JobResult, parts: &[Part]) -> Result<()> {
    let dir = args.out_dir();
    std::fs::create_dir_all(&dir)?;

    let encoding = Encoding::choose(parts, args.wav_f32);
    if let Encoding::Flac { gain } = encoding
        && gain < 1.0
    {
        println!(
            "headroom     : scaled by {gain:.3} to fit 16-bit \
             (use --wav-f32 for the unscaled output)"
        );
    }

    for part in parts {
        let file = output::write(&dir, part, result.output.sample_rate, encoding)?;
        println!(
            "  {:<12} : {file:<14} {:.1} dBFS",
            part.origin,
            output::rms_dbfs(&part.samples)
        );
    }
    Ok(())
}

/// The claim the whole topology rests on: the parts add back up to the track.
/// Checked against the mix we decoded rather than asserted.
fn report_reconstruction(audio: &Decoded, parts: &[Part], result: &JobResult, elapsed: f64) {
    // Only when every part is present and at the mix's rate; otherwise the
    // number would describe two buffers that do not line up.
    if result.output.sample_rate == audio.sample_rate {
        let names: Vec<&str> = parts.iter().map(|p| p.name.as_str()).collect();
        println!(
            "reconstruct  : {:.1} dB  ({} vs the original)",
            output::reconstruction_null_db(&audio.data, parts),
            names.join("+")
        );
    }
    match (result.model_residual_db, result.output.derived.as_deref()) {
        (Some(resid), Some(derived)) => println!(
            "model resid  : {resid:.1} dB (the model's own error — it all lands on `{derived}`)",
        ),
        (Some(resid), None) => println!(
            "model resid  : {resid:.1} dB (the model's own error — no part is left to absorb it)",
        ),
        (None, _) => println!("model resid  : unknown (stems stored by a session)"),
    }
    println!("total        : {elapsed:.1}s end to end");
}

/// Stems are quantised as `round(x × 32767)`; the FLAC decoder reads an `i16` as `v / 32768`.
const DECODED_S16_TO_STEM: f32 = 32768.0 / 32767.0;

/// Multiply every sample by `scale`.
fn rescale(data: Planar, scale: f32) -> Planar {
    data.into_iter()
        .map(|ch| ch.into_iter().map(|v| v * scale).collect())
        .collect()
}

/// Split an interleaved stereo payload into planes, undoing the transfer gain.
fn deinterleave(raw: &[u8], f32_format: bool, gain: f32) -> Planar {
    let inv = inverse(gain);
    let mut out = vec![Vec::new(), Vec::new()];
    if f32_format {
        for (i, chunk) in raw.chunks_exact(4).enumerate() {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            out[i % 2].push(v * inv);
        }
    } else {
        for (i, chunk) in raw.chunks_exact(2).enumerate() {
            let v = f32::from(i16::from_le_bytes([chunk[0], chunk[1]])) / 32767.0;
            out[i % 2].push(v * inv);
        }
    }
    out
}

/// `1.0 / gain`, treating a nonsensical gain as unity rather than dividing by
/// zero and filling a stem with infinities.
fn inverse(gain: f32) -> f32 {
    if gain > 0.0 { 1.0 / gain } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_gain_does_not_produce_infinities() {
        assert_eq!(inverse(0.0), 1.0);
        assert_eq!(inverse(-1.0), 1.0);
        assert!((inverse(0.5) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn deinterleaving_undoes_the_transfer_gain() {
        // Two frames of stereo s16 at half scale, sent with gain 0.5.
        let mut raw = Vec::new();
        for v in [16384i16, -16384, 8192, -8192] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let planes = deinterleave(&raw, false, 0.5);
        assert_eq!(planes.len(), 2);
        assert!((planes[0][0] - 1.0).abs() < 1e-3, "{}", planes[0][0]);
        assert!((planes[1][0] + 1.0).abs() < 1e-3);
    }

    #[test]
    fn rescaling_restores_the_original_level() {
        let restored = rescale(vec![vec![0.25, -0.5]], inverse(0.5));
        assert!((restored[0][0] - 0.5).abs() < 1e-6);
        assert!((restored[0][1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_full_scale_16_bit_stem_sample_reads_as_one() {
        let raw = i16::MAX.to_le_bytes();
        assert_eq!(deinterleave(&raw, false, 1.0)[0][0], 1.0);
        let decoded = f32::from(i16::MAX) / 32768.0;
        let restored = rescale(vec![vec![decoded]], DECODED_S16_TO_STEM);
        assert!((restored[0][0] - 1.0).abs() < 1e-7, "{}", restored[0][0]);
    }

    #[test]
    fn the_output_directory_defaults_beside_the_input() {
        let args = Args::parse_from(["stemd-cli", "/music/Some Track.flac"]);
        assert_eq!(args.out_dir(), PathBuf::from("Some Track-stems"));
    }

    #[test]
    fn an_explicit_output_directory_wins() {
        let args = Args::parse_from(["stemd-cli", "in.wav", "--out", "/tmp/here"]);
        assert_eq!(args.out_dir(), PathBuf::from("/tmp/here"));
    }
}
