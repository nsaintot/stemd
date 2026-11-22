//! Stores a finished job's stems whole in the cache.

use std::sync::Arc;

use anyhow::{Context, Result};
use stemd_core::{Audio, Stems};

use super::QueuedWork;
use crate::cache::{Cache, Consolidation, Separation, TrackInfo, type_dir};
use crate::ident::track_id;
use crate::jobs::Entry;
use crate::output::Output;

/// The parts that `output.parts` names, and the derived part (`mix` minus those parts), each at the
/// output rate.
pub(super) fn to_output_rate(
    mix: &Audio,
    stems: &Stems,
    output: &Output,
) -> Result<Vec<(&'static str, Audio)>> {
    let mut parts: Vec<(&'static str, &Audio)> = Vec::new();
    for name in output.parts.names() {
        let audio = stems
            .stem(name)
            .with_context(|| format!("this separation has no {name}"))?;
        parts.push((name, audio));
    }
    let derived = output.parts.derived().map(|name| {
        (
            name,
            stemd_core::stems::remainder(mix, parts.iter().map(|(_, audio)| *audio)),
        )
    });
    let parts: Vec<(&'static str, &Audio)> = parts
        .into_iter()
        .chain(derived.iter().map(|(name, audio)| (*name, audio)))
        .collect();
    // One thread per part; see `is_converting_three_stems_at_once_worth_it`.
    std::thread::scope(|scope| {
        let running: Vec<_> = parts
            .iter()
            .map(|(name, audio)| scope.spawn(|| one_rate(name, audio, output)))
            .collect();
        running
            .into_iter()
            .map(|thread| {
                thread
                    .join()
                    .unwrap_or_else(|p| std::panic::resume_unwind(p))
            })
            .collect()
    })
}

fn one_rate(name: &'static str, audio: &Audio, output: &Output) -> Result<(&'static str, Audio)> {
    let began = std::time::Instant::now();
    let converted = output
        .dsp
        .convert(audio, output.rate.hz())
        .with_context(|| format!("converting {name} to {} Hz", output.rate))?;
    // Timed because the format decides whether conversion or encoding dominates.
    tracing::debug!(
        "{name}: {} -> {} Hz in {:.2?}",
        audio.sample_rate,
        output.rate,
        began.elapsed()
    );
    Ok((name, converted))
}

/// Store a finished job's stems whole at the track's gain, and return the result, held.
pub(super) fn store_whole(
    cache: &Cache,
    work: &QueuedWork,
    stems: &Stems,
    stride: usize,
    separation: Separation,
) -> Result<Arc<Entry>> {
    let (mix, output) = (&work.mix, &work.output);
    let converted = to_output_rate(mix, stems, output)?;
    for (stem, audio) in &converted {
        audio
            .peak()
            .with_context(|| format!("the separated {stem} is not finite"))?;
    }
    let id = track_id(mix);
    let gain = output.gain(mix.peak().ok());
    let name = type_dir(output, gain);
    let dir = cache.verified_dir(&work.cache_id, &name, &id);
    let _writing = cache.writing(&dir);
    let frames = mix.frames();
    let info = TrackInfo {
        frames,
        sample_rate: mix.sample_rate,
        channels: mix.channels(),
        stride,
        pieces: frames.div_ceil(stride),
    };
    cache.write_info(&dir, &info)?;
    let task = Consolidation::new(dir.clone(), output, info);
    let expected = info.frames_at(output.rate.hz());
    // One thread per stem, so storing takes as long as the slowest encode.
    let encoded: Vec<Result<Encoded>> = std::thread::scope(|scope| {
        let running: Vec<_> = converted
            .iter()
            .map(|(stem, audio)| {
                scope.spawn(move || {
                    let audio = fit(audio, expected);
                    let (bytes, clipped) =
                        crate::whole::encode(&audio, output.format, gain, false)?;
                    Ok((*stem, bytes, clipped))
                })
            })
            .collect();
        running
            .into_iter()
            .map(|t| t.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
            .collect()
    });
    for stored in encoded {
        let (stem, bytes, clipped) = stored?;
        cache.consolidate(&task, stem, &bytes, expected, clipped)?;
    }
    cache.write_separation(&dir, &separation)?;
    let entry = Entry::stored(&dir, output, gain, mix.channels(), Some(separation))
        .context("the stored stems are incomplete")?;
    let entry = Arc::new(entry);
    cache.hold(&entry);
    tracing::info!(
        "track {id}: stored {} stems whole in {name}",
        converted.len()
    );
    Ok(entry)
}

/// A stem's name, its encoded bytes, and the number of samples clipped, which is absent for
/// `f32le`.
type Encoded<'a> = (&'a str, Vec<u8>, Option<usize>);

/// `audio` cut or padded with silence to `frames`.
fn fit(audio: &Audio, frames: usize) -> std::borrow::Cow<'_, Audio> {
    if audio.frames() == frames {
        return std::borrow::Cow::Borrowed(audio);
    }
    let data = audio
        .data
        .iter()
        .map(|channel| {
            let mut channel = channel[..channel.len().min(frames)].to_vec();
            channel.resize(frames, 0.0);
            channel
        })
        .collect();
    std::borrow::Cow::Owned(Audio::new(data, audio.sample_rate))
}
