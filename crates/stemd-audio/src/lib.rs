//! Decode an audio file to planar f32.
//!
//! Shared by both entrypoints: the window decodes what is dropped on it, and the
//! command-line client decodes what it is pointed at. An AAC file carries encoder
//! delay that has to come off the front, or every stem is offset against the mix
//! and the client's reconstruction stops nulling.

use std::path::Path;

use anyhow::{Context, Result, bail};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub struct Decoded {
    /// `data[channel][sample]`
    pub data: Vec<Vec<f32>>,
    pub sample_rate: u32,
}

impl Decoded {
    pub fn frames(&self) -> usize {
        self.data.first().map_or(0, Vec::len)
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / f64::from(self.sample_rate)
    }

    /// Interleave to s16le or f32le for upload.
    pub fn to_interleaved(&self, f32_output: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for i in 0..self.frames() {
            for ch in &self.data {
                if f32_output {
                    out.extend_from_slice(&ch[i].to_le_bytes());
                } else {
                    let q = (ch[i].clamp(-1.0, 1.0) * 32767.0).round() as i16;
                    out.extend_from_slice(&q.to_le_bytes());
                }
            }
        }
        out
    }
}

pub fn decode(path: &Path) -> Result<Decoded> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut decoded = decode_stream(mss, &hint, &path.display().to_string())?;

    if let Some(skip) = mp4_encoder_delay(path)
        && skip > 0
        && decoded.data[0].len() > skip
    {
        eprintln!("note: dropping {skip} priming samples declared by the mp4 edit list");
        for ch in &mut decoded.data {
            ch.drain(..skip);
        }
    }
    Ok(decoded)
}

/// Decode a FLAC stem held in memory.
pub fn decode_flac(bytes: Vec<u8>) -> Result<Decoded> {
    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(bytes)), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("flac");
    decode_stream(mss, &hint, "the flac stem")
}

fn decode_stream(mss: MediaSourceStream, hint: &Hint, label: &str) -> Result<Decoded> {
    let probed = symphonia::default::get_probe()
        .format(
            hint,
            mss,
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &MetadataOptions::default(),
        )
        .with_context(|| format!("probing {label}"))?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .context("file contains no default audio track")?;
    let track_id = track.id;
    let declared_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("no decoder for this codec")?;

    let mut sink = PlanarSink::new(declared_rate);
    while let Some(packet) = next_packet(&mut *format)? {
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio) => sink.append(audio),
            // A corrupt packet is skipped rather than failing the whole file.
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decoding"),
        }
    }

    sink.finish(label)
}

/// The next packet, or `None` at end of stream.
///
/// Symphonia signals the end as an `UnexpectedEof` io error rather than a
/// dedicated variant, so it is translated here instead of at the call site.
fn next_packet(
    format: &mut dyn symphonia::core::formats::FormatReader,
) -> Result<Option<symphonia::core::formats::Packet>> {
    match format.next_packet() {
        Ok(packet) => Ok(Some(packet)),
        Err(symphonia::core::errors::Error::IoError(e))
            if e.kind() == std::io::ErrorKind::UnexpectedEof =>
        {
            Ok(None)
        }
        Err(e) => Err(e).context("reading packet"),
    }
}

/// Accumulates decoded buffers into planar channels.
///
/// The channel count is not known until the first decoded buffer arrives, so it
/// is established there rather than up front.
struct PlanarSink {
    channels: Vec<Vec<f32>>,
    sample_rate: u32,
    interleaved: Option<SampleBuffer<f32>>,
}

impl PlanarSink {
    const fn new(declared_rate: u32) -> Self {
        Self {
            channels: Vec::new(),
            sample_rate: declared_rate,
            interleaved: None,
        }
    }

    fn append(&mut self, audio: symphonia::core::audio::AudioBufferRef<'_>) {
        let spec = *audio.spec();
        if self.sample_rate == 0 {
            self.sample_rate = spec.rate;
        }
        if self.channels.is_empty() {
            self.channels = vec![Vec::new(); spec.channels.count()];
        }

        let buffer = self
            .interleaved
            .get_or_insert_with(|| SampleBuffer::<f32>::new(audio.capacity() as u64, spec));
        buffer.copy_interleaved_ref(audio);

        let n = self.channels.len();
        for (i, sample) in buffer.samples().iter().enumerate() {
            self.channels[i % n].push(*sample);
        }
    }

    /// Fold to the stereo pair the model requires.
    fn finish(self, label: &str) -> Result<Decoded> {
        let mut channels = self.channels;
        if channels.is_empty() || channels[0].is_empty() {
            bail!("decoded no audio from {label}");
        }

        let data = match channels.len() {
            1 => vec![channels[0].clone(), channels[0].clone()],
            2 => channels,
            n => {
                eprintln!("note: {n} channels, keeping the first two");
                channels.truncate(2);
                channels
            }
        };

        Ok(Decoded {
            data,
            sample_rate: self.sample_rate,
        })
    }
}

/// Leading samples an mp4/m4a edit list says to skip, i.e. AAC encoder delay.
///
/// Symphonia's `enable_gapless` only understands the iTunes `iTunSMPB` tag, and a
/// file can declare the same delay through an `elst` edit list instead, as
/// some exporters do. Missed, the decode carries about 2048 samples (46 ms) of
/// priming that every other player strips, which puts the stems late against the
/// original.
///
/// A shallow, allocation-free scan rather than a dependency: a malformed box must
/// return `None` rather than fail the decode.
fn mp4_encoder_delay(path: &Path) -> Option<usize> {
    let data = std::fs::read(path).ok()?;
    // `elst` is inside moov/trak/edts, always near the head of the file.
    elst_media_time(&data[..data.len().min(1 << 22)])
}

fn elst_media_time(hay: &[u8]) -> Option<usize> {
    let mut at = 0usize;
    while let Some(found) = find(&hay[at..], b"elst") {
        let start = at + found;
        // version(1) flags(3) entry_count(4), then the first entry.
        let body = hay.get(start + 4..start + 24)?;
        let version = body[0];
        let count = u32::from_be_bytes(body[4..8].try_into().ok()?);
        if count == 0 {
            at = start + 4;
            continue;
        }
        let media_time = if version == 1 {
            i64::from_be_bytes(hay.get(start + 20..start + 28)?.try_into().ok()?)
        } else {
            i64::from(i32::from_be_bytes(body[12..16].try_into().ok()?))
        };
        // -1 means an empty edit (silent lead-in), which is not encoder delay.
        return usize::try_from(media_time).ok();
    }
    None
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::elst_media_time;

    /// version 0 box: 4 size + 4 type + 1 version + 3 flags + 4 count,
    /// then per entry 4 duration + 4 media_time + 4 rate.
    fn elst_v0(count: u32, media_time: i32) -> Vec<u8> {
        let mut b = vec![0u8; 4];
        b.extend_from_slice(b"elst");
        b.push(0);
        b.extend_from_slice(&[0, 0, 0]);
        b.extend_from_slice(&count.to_be_bytes());
        b.extend_from_slice(&1000u32.to_be_bytes());
        b.extend_from_slice(&media_time.to_be_bytes());
        b.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        b
    }

    #[test]
    fn reads_the_delay_from_a_version_0_edit_list() {
        assert_eq!(elst_media_time(&elst_v0(1, 2048)), Some(2048));
    }

    #[test]
    fn an_empty_edit_is_not_encoder_delay() {
        // media_time -1 marks a silent lead-in, not priming to discard.
        assert_eq!(elst_media_time(&elst_v0(1, -1)), None);
    }

    #[test]
    fn no_edit_list_means_no_delay() {
        assert_eq!(elst_media_time(b"no boxes here at all, just bytes"), None);
    }

    #[test]
    fn a_truncated_box_does_not_panic() {
        let full = elst_v0(1, 2048);
        for cut in 0..full.len() {
            let _ = elst_media_time(&full[..cut]);
        }
    }
}
