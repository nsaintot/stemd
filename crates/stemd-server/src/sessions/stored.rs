//! What a session reads from and writes to the stem cache.

use std::ops::Range;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use stemd_core::Audio;

use crate::cache::{Cache, Consolidation, TrackInfo, type_dir};
use crate::output::{Output, output_span};
use crate::whole::Whole;

use super::view::spans_of;
use super::{Excerpt, RESAMPLE_CONTEXT, Session, Sessions, Source, StemView, Track};

impl Sessions {
    /// The track directory of `session`'s output, once its track id is verified or claimed.
    pub fn track_dir(&self, session: &Session) -> Option<PathBuf> {
        let track = session.track();
        let type_dir = type_dir(&session.output, session.gain);
        Some(
            self.cache
                .verified_dir(&track.cache_id, &type_dir, &track.stored_id()?),
        )
    }

    /// The pieces of `session`'s output stored under its track's id.
    pub fn cached_for(&self, session: &Session) -> Vec<usize> {
        let Some(dir) = self.track_dir(session) else {
            return Vec::new();
        };
        Cache::cached(
            &dir,
            &session.output.served(),
            session.output.format.extension(),
            session.track().spec.grid.len(),
        )
    }

    /// Once `track` is hashed, encode and write the complete pieces near `touched`.
    pub fn store_pieces(&self, track: &Track, touched: Range<usize>) {
        let Some(id) = track.verified() else {
            return;
        };
        let cache = &self.cache;
        // Pieces are only a cache: on a nearly full disk, skip them and let the job go on.
        if cache.disk_low() {
            tracing::debug!(track = %id, "disk nearly full, not writing pieces");
            return;
        }
        let info = TrackInfo::from(&track.spec);
        let first = touched.start.saturating_sub(RESAMPLE_CONTEXT) / info.stride;
        let last = (touched.end + RESAMPLE_CONTEXT)
            .div_ceil(info.stride)
            .min(info.pieces);

        let mut outputs: Vec<(String, Output, f32)> = Vec::new();
        for session in self.of_track(track) {
            let name = type_dir(&session.output, session.gain);
            if !outputs.iter().any(|(n, _, _)| *n == name) {
                let mut output = session.output.clone();
                output.derived = true;
                outputs.push((name, output, session.gain));
            }
        }
        for (name, output, gain) in outputs {
            let converts = output.rate.hz() != track.spec.sample_rate;
            let dir = cache.verified_dir(&track.cache_id, &name, &id);
            if let Err(err) = cache.write_info(&dir, &info) {
                tracing::warn!(track = %id, "describing {name}: {err:#}");
                continue;
            }
            for k in first..last {
                if !track.piece_writable(k, &name, converts) {
                    continue;
                }
                let piece = track.piece(k);
                let written = output.stored_parts().into_iter().try_for_each(|stem| {
                    let excerpt = track.read(&output, stem, piece.start, piece.end)?;
                    let bytes = stemd_core::encode_stem(&excerpt.audio, output.format, gain)?;
                    cache.write_verified(&dir, stem, k, output.format.extension(), &bytes)
                });
                match written {
                    Ok(()) => track.mark_written(k, &name),
                    Err(err) => tracing::warn!(track = %id, "piece {k}: {err:#}"),
                }
            }
            let task = Consolidation::new(dir, &output, info);
            if Cache::needs_consolidation(&task) {
                cache.consolidate_later(task);
            }
        }
    }

    /// Part `name` over `from..to` at the output rate and the transfer scale, from memory or disk.
    pub fn read_stored(
        &self,
        session: &Session,
        name: &str,
        from: usize,
        to: usize,
    ) -> Result<Excerpt> {
        session.ensure_open()?;
        let output = &session.output;
        if !output.served().contains(&name) {
            bail!("stem: expected one of {:?}, got {name}", output.served());
        }
        let track = session.track();
        if from >= to || to > track.spec.frames {
            bail!(
                "{from}..{to} is not a range of this {}-sample track",
                track.spec.frames
            );
        }
        let whole = self.whole(session, name);
        let runs = self.runs(session, name, whole.is_some(), from, to)?;
        let (model, out) = (
            u64::from(track.spec.sample_rate),
            u64::from(output.rate.hz()),
        );
        let span = output_span(&(from..to), model, out);
        let (start, frames) = (span.start, span.len());
        let mut data = vec![Vec::with_capacity(frames); track.spec.channels];
        for (run, source) in runs {
            let span = output_span(&run, model, out);
            let len = span.len();
            let audio = self.read_run(session, name, whole.as_ref(), &run, source, span)?;
            if audio.frames() != len {
                bail!(
                    "samples {}..{}: expected {len} samples, got {}",
                    run.start,
                    run.end,
                    audio.frames()
                );
            }
            for (dst, src) in data.iter_mut().zip(audio.data) {
                dst.extend(src);
            }
        }
        Ok(Excerpt {
            audio: Audio::new(data, output.rate.hz()),
            from: start,
        })
    }

    /// The chunks of `from..to` and where each one is read from, with consecutive memory chunks or
    /// whole-file chunks merged.
    fn runs(
        &self,
        session: &Session,
        name: &str,
        whole: bool,
        from: usize,
        to: usize,
    ) -> Result<Vec<(Range<usize>, Source)>> {
        let (track, output) = (session.track(), &session.output);
        let stride = track.spec.grid.stride().unsigned_abs() as usize;
        let dir = self.track_dir(session);
        let lookup = |k: usize| {
            if whole {
                return Some(Source::Whole);
            }
            Cache::lookup(dir.as_ref()?, name, k, output.format.extension()).map(Source::Piece)
        };

        // Read a chunk from memory when it is ready with the context a rate conversion needs,
        // otherwise from disk, and as a last resort from memory with less context.
        let converts = output.rate.hz() != track.spec.sample_rate;
        let mut runs: Vec<(Range<usize>, Source)> = Vec::new();
        for k in from / stride..to.div_ceil(stride) {
            let piece = track.piece(k);
            let chunk = piece.start.max(from)..piece.end.min(to);
            let context = if converts {
                chunk.start.saturating_sub(RESAMPLE_CONTEXT)
                    ..(chunk.end + RESAMPLE_CONTEXT).min(track.spec.frames)
            } else {
                chunk.clone()
            };
            let source = if track.ready_covers(&context) {
                Source::Memory
            } else if let Some(source) = lookup(k) {
                source
            } else if track.ready_covers(&chunk) {
                Source::Memory
            } else {
                bail!(
                    "samples {}..{} are neither ready nor stored",
                    chunk.start,
                    chunk.end
                );
            };
            match (runs.last_mut(), source) {
                (Some((run, Source::Memory)), Source::Memory)
                | (Some((run, Source::Whole)), Source::Whole) => run.end = chunk.end,
                (_, source) => runs.push((chunk, source)),
            }
        }
        Ok(runs)
    }

    /// The output-rate samples `span` of `name`, within the part of the track that `run` covers,
    /// read from `source`.
    fn read_run(
        &self,
        session: &Session,
        name: &str,
        whole: Option<&Whole>,
        run: &Range<usize>,
        source: Source,
        span: Range<usize>,
    ) -> Result<Audio> {
        let (track, output) = (session.track(), &session.output);
        let cut = |whole: &Whole| {
            crate::whole::read(
                whole,
                output.format,
                track.spec.channels,
                output.rate.hz(),
                span.start,
                span.end,
            )
        };
        match source {
            Source::Memory => {
                let mut audio = track.read(output, name, run.start, run.end)?.audio;
                for v in audio.data.iter_mut().flatten() {
                    *v *= session.gain;
                }
                Ok(audio)
            }
            Source::Whole => cut(whole.context("the whole file went away")?),
            Source::Piece(path) => match std::fs::read(&path) {
                Ok(bytes) => {
                    let stride = track.spec.grid.stride().unsigned_abs() as usize;
                    let piece = track.piece(run.start / stride);
                    let (model, out) = (
                        u64::from(track.spec.sample_rate),
                        u64::from(output.rate.hz()),
                    );
                    let piece = output_span(&piece, model, out);
                    let (piece_at, piece_len) = (piece.start, piece.len());
                    let audio = crate::whole::decode(
                        bytes,
                        output.format,
                        track.spec.channels,
                        output.rate.hz(),
                    )
                    .with_context(|| format!("decoding {}", path.display()))?;
                    if audio.frames() != piece_len {
                        bail!(
                            "{}: expected {piece_len} samples, got {}",
                            path.display(),
                            audio.frames()
                        );
                    }
                    Ok(audio.excerpt(span.start - piece_at, span.end - piece_at))
                }
                // The stem was stored whole after the lookup.
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => cut(&self
                    .whole(session, name)
                    .with_context(|| format!("{} went away", path.display()))?),
                Err(err) => Err(err).with_context(|| format!("reading {}", path.display())),
            },
        }
    }

    /// The whole file of `name` for `session`'s output, when it holds the whole track.
    pub fn whole(&self, session: &Session, name: &str) -> Option<Whole> {
        if session.closed().is_some() || !session.output.served().contains(&name) {
            return None;
        }
        let dir = self.track_dir(session)?;
        let info = TrackInfo::from(&session.track().spec);
        let whole = info.find_whole(&dir, name, &session.output)?;
        Cache::touch(&dir);
        Some(whole)
    }

    /// Store `bytes`, the whole of `name` just encoded for `session`, in place
    /// of its pieces. Does nothing for a track that is not shared.
    pub fn consolidate(
        &self,
        session: &Session,
        name: &str,
        bytes: &[u8],
        frames: usize,
        clipped: Option<usize>,
    ) -> Result<()> {
        let track = session.track();
        let (Some(dir), Some(id)) = (self.track_dir(session), track.verified()) else {
            return Ok(());
        };
        let cache = &self.cache;
        if cache.disk_low() {
            return Ok(());
        }
        let task = Consolidation::new(dir, &session.output, TrackInfo::from(&track.spec));
        cache.consolidate(&task, name, bytes, frames, clipped)?;
        tracing::info!(track = %id, "stored {name} whole");
        Ok(())
    }

    /// How each stem that `session` serves is stored.
    pub fn stems_for(&self, session: &Session) -> Vec<StemView> {
        use crate::output::{Storage, sources};
        let track = session.track();
        let dir = self.track_dir(session);
        let extension = session.output.format.extension();
        session
            .output
            .served()
            .into_iter()
            .map(|name| {
                let mut stem = StemView {
                    name,
                    sources: sources(name),
                    stored: Storage::None,
                    bytes: None,
                    pieces: Vec::new(),
                };
                if let Some(whole) = self.whole(session, name) {
                    stem.stored = Storage::Whole;
                    stem.bytes = std::fs::metadata(&whole.path).map(|m| m.len()).ok();
                } else if let Some(dir) = &dir {
                    let stored = Cache::stem_pieces(dir, name, extension, track.spec.grid.len());
                    if !stored.is_empty() {
                        stem.stored = Storage::Pieces;
                        stem.pieces = spans_of(&stored);
                    }
                }
                stem
            })
            .collect()
    }
}
