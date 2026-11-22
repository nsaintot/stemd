//! One track's mix, upload progress and separated pieces, shared by its sessions.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;
use stemd_core::{Audio, Grid, Normalisation, Parts, Stems};

use crate::cache::TrackInfo;
use crate::ident::track_id;
use crate::output::{Output, output_index};

use super::{Excerpt, RESAMPLE_CONTEXT, Spans, gcd};

/// A track's fixed properties, set when it is opened.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackSpec {
    pub frames: usize,
    pub channels: usize,
    /// The model's rate.
    pub sample_rate: u32,
    pub grid: Grid,
    /// True when the model needs the whole mix's normalisation.
    pub normalises: bool,
    /// The parts to separate, which decide which models run.
    pub parts: Parts,
}

impl TrackSpec {
    /// The bytes a track holds from the moment it opens: the mix and one f32 buffer per part.
    pub fn memory(&self) -> u64 {
        let buffers = 1 + self.parts.names().count();
        (buffers * self.channels * self.frames * std::mem::size_of::<f32>()) as u64
    }
}

impl From<&TrackSpec> for TrackInfo {
    fn from(spec: &TrackSpec) -> Self {
        Self {
            frames: spec.frames,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            stride: spec.grid.stride().unsigned_abs() as usize,
            pieces: spec.grid.len(),
        }
    }
}

/// The outcome of checking the whole upload once it has arrived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The samples hash to this id, and any statistics the client sent match.
    Verified(String),
    /// The client's mean and std do not match the mix, so the stems differ from a whole-track run
    /// and nothing of this track is shared.
    Unpublishable(String),
    /// The samples hash to this id, which differs from the id the track was opened with.
    Mismatch(String),
}

pub(super) struct TrackState {
    /// The mix, shared once the whole upload is in, so it can be hashed outside the lock.
    pub(super) mix: Arc<Audio>,
    pub(super) arrived: Spans,
    pub(super) norm: Option<Normalisation>,
    /// True when `norm` came from a client and has not been checked.
    pub(super) norm_from_client: bool,
    pub(super) stems: Vec<(&'static str, Audio)>,
    pub(super) ready: Spans,
    pub(super) done: Vec<bool>,
    pub(super) verdict: Option<Verdict>,
    /// Pieces already written, by output type.
    pub(super) written: HashSet<(String, usize)>,
    pub(super) closed: Option<String>,
}

pub struct Track {
    /// Identifies the track while it is open; it stays the same when sessions join.
    pub uid: u64,
    /// The cache id of the model that separates it, such as `htdemucs-22c7f2c9-1f2e3d4c`.
    pub cache_id: String,
    pub spec: TrackSpec,
    /// The id that the open named with `track=`. Until the upload hashes to it, it is trusted only
    /// for reading stored pieces.
    pub claim: Option<String>,
    pub(super) state: Mutex<TrackState>,
}

impl Track {
    pub(super) fn new(
        uid: u64,
        cache_id: String,
        spec: TrackSpec,
        norm: Option<Normalisation>,
        claim: Option<String>,
    ) -> Self {
        let silent = || {
            Audio::new(
                vec![vec![0.0; spec.frames]; spec.channels],
                spec.sample_rate,
            )
        };
        Self {
            uid,
            cache_id,
            claim,
            state: Mutex::new(TrackState {
                mix: Arc::new(silent()),
                arrived: Spans::default(),
                norm: norm.filter(|_| spec.normalises),
                norm_from_client: norm.is_some() && spec.normalises,
                stems: spec.parts.names().map(|name| (name, silent())).collect(),
                ready: Spans::default(),
                done: vec![false; spec.grid.len()],
                verdict: None,
                written: HashSet::new(),
                closed: None,
            }),
            spec,
        }
    }

    pub(super) fn check_open(state: &TrackState) -> Result<()> {
        match &state.closed {
            Some(reason) => bail!("this track is closed: {reason}"),
            None => Ok(()),
        }
    }

    /// The samples that piece `k` covers.
    pub fn piece(&self, k: usize) -> Range<usize> {
        let stride = self.spec.grid.stride().unsigned_abs() as usize;
        crate::output::piece_span(k, stride, self.spec.frames)
    }

    /// Place `audio` at track sample `offset`.
    ///
    /// Samples that another session already uploaded must match bit for bit. Returns the verdict
    /// when this upload completes the track.
    pub fn upload(&self, offset: usize, audio: &Audio) -> Result<Option<Verdict>, UploadError> {
        let mut state = self.state.lock();
        Self::check_open(&state).map_err(UploadError::Closed)?;
        let end = offset
            .checked_add(audio.frames())
            .filter(|&end| end <= self.spec.frames)
            .ok_or_else(|| {
                UploadError::Invalid(format!(
                    "{} samples at {offset} run past the track's {}",
                    audio.frames(),
                    self.spec.frames
                ))
            })?;
        for seen in state.arrived.overlaps(&(offset..end)) {
            for (have, given) in state.mix.data.iter().zip(&audio.data) {
                let given = &given[seen.start - offset..seen.end - offset];
                if have[seen.clone()]
                    .iter()
                    .zip(given)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    return Err(UploadError::Differs(seen.start, seen.end));
                }
            }
        }
        if state.arrived.covers(&(0..self.spec.frames)) {
            return Ok(None);
        }
        let mix = Arc::make_mut(&mut state.mix);
        for (dst, src) in mix.data.iter_mut().zip(&audio.data) {
            dst[offset..end].copy_from_slice(src);
        }
        state.arrived.insert(offset..end);
        if !state.arrived.covers(&(0..self.spec.frames)) {
            return Ok(None);
        }

        // The whole track is here: settle its statistics and its identity.
        let mix = Arc::clone(&state.mix);
        drop(state);
        let computed = Normalisation::of(&mix);
        let actual = track_id(&mix);
        let mut state = self.state.lock();
        let client_norm_differs = self.spec.normalises
            && state.norm_from_client
            && state.norm.is_some_and(|n| {
                n.mean().to_bits() != computed.mean().to_bits()
                    || n.std().to_bits() != computed.std().to_bits()
            });
        if self.spec.normalises && state.norm.is_none() {
            state.norm = Some(computed);
        }
        let verdict = if self.claim.as_ref().is_some_and(|claim| *claim != actual) {
            Verdict::Mismatch(actual)
        } else if client_norm_differs {
            Verdict::Unpublishable(actual)
        } else {
            Verdict::Verified(actual)
        };
        state.verdict = Some(verdict.clone());
        Ok(Some(verdict))
    }

    /// The track id, once the whole upload has hashed to it and may be shared.
    pub fn verified(&self) -> Option<String> {
        match &self.state.lock().verdict {
            Some(Verdict::Verified(id)) => Some(id.clone()),
            _ => None,
        }
    }

    /// The id that stored pieces are read under: the verified id, or the claim until the upload is
    /// hashed.
    pub fn stored_id(&self) -> Option<String> {
        match &self.state.lock().verdict {
            Some(Verdict::Verified(id)) => Some(id.clone()),
            Some(Verdict::Mismatch(_)) => None,
            None | Some(Verdict::Unpublishable(_)) => self.claim.clone(),
        }
    }

    /// True when segment `k` can run now.
    pub(super) fn runnable(&self, state: &TrackState, k: usize) -> bool {
        let window = self.spec.grid.window(k);
        let input = window.from.unsigned_abs() as usize..window.to.unsigned_abs() as usize;
        state.closed.is_none()
            && !state.done[k]
            && state.arrived.covers(&input)
            && (!self.spec.normalises || state.norm.is_some())
    }

    pub fn is_done(&self, k: usize) -> bool {
        self.state.lock().done.get(k).copied().unwrap_or(false)
    }

    /// Segment `k`'s input and the normalisation to run it with.
    pub fn input(&self, k: usize) -> Result<(Audio, Normalisation)> {
        let state = self.state.lock();
        Self::check_open(&state)?;
        let window = self.spec.grid.window(k);
        let (from, to) = (
            window.from.unsigned_abs() as usize,
            window.to.unsigned_abs() as usize,
        );
        // Models trained on raw audio ignore the normalisation.
        let norm = state.norm.unwrap_or_else(|| Normalisation::new(0.0, 1.0));
        Ok((state.mix.excerpt(from, to), norm))
    }

    /// The mix over `from..to`.
    pub fn mix(&self, from: usize, to: usize) -> Audio {
        self.state.lock().mix.excerpt(from, to)
    }

    /// Record that segment `k` has run.
    pub fn done(&self, k: usize) {
        if let Some(d) = self.state.lock().done.get_mut(k) {
            *d = true;
        }
    }

    /// Copy the final stems over `from..to` into the track.
    pub fn store(&self, from: usize, to: usize, stems: &Stems) -> Result<()> {
        let mut state = self.state.lock();
        for (name, buffer) in &mut state.stems {
            let audio = stems
                .stem(name)
                .with_context(|| format!("this separation has no {name}"))?;
            for (dst, src) in buffer.data.iter_mut().zip(&audio.data) {
                dst[from..to].copy_from_slice(src);
            }
        }
        state.ready.insert(from..to);
        Ok(())
    }

    pub fn close(&self, reason: impl Into<String>) {
        let mut state = self.state.lock();
        if state.closed.is_none() {
            state.closed = Some(reason.into());
        }
    }

    pub fn closed(&self) -> Option<String> {
        self.state.lock().closed.clone()
    }

    /// True when piece `k` at `output` can be written: the piece and the context a rate conversion
    /// needs are ready, and the piece has not been written yet.
    pub fn piece_writable(&self, k: usize, type_dir: &str, converts: bool) -> bool {
        let state = self.state.lock();
        if state.closed.is_some() || state.written.contains(&(type_dir.to_owned(), k)) {
            return false;
        }
        let piece = self.piece(k);
        let needed = if converts {
            piece.start.saturating_sub(RESAMPLE_CONTEXT)
                ..(piece.end + RESAMPLE_CONTEXT).min(self.spec.frames)
        } else {
            piece
        };
        state.ready.covers(&needed)
    }

    /// True when every sample of `range` is ready in memory.
    pub fn ready_covers(&self, range: &Range<usize>) -> bool {
        self.state.lock().ready.covers(range)
    }

    /// True when every sample of the track is ready or in one of the `stored` pieces.
    pub fn complete_with(&self, stored: &[usize]) -> bool {
        let mut spans = self.state.lock().ready.clone();
        for &k in stored {
            spans.insert(self.piece(k));
        }
        spans.covers(&(0..self.spec.frames))
    }

    pub fn mark_written(&self, k: usize, type_dir: &str) {
        self.state.lock().written.insert((type_dir.to_owned(), k));
    }

    /// Part `name` over `from..to` at `output`'s rate.
    ///
    /// `from..to` are track samples at the model's rate and must be ready. At another rate, the
    /// range is converted with up to [`RESAMPLE_CONTEXT`] samples on each side, taken from the
    /// ready samples around it.
    pub fn read(&self, output: &Output, name: &str, from: usize, to: usize) -> Result<Excerpt> {
        let (audio, start) = {
            let state = self.state.lock();
            Self::check_open(&state)?;
            if from >= to || !state.ready.covers(&(from..to)) {
                bail!("samples {from}..{to} are not ready");
            }
            if !output.served().contains(&name) {
                bail!("stem: expected one of {:?}, got {name}", output.served());
            }
            let model = self.spec.sample_rate;
            let out = output.rate.hz();
            let (start, end, ready) = if model == out {
                (from, to, from)
            } else {
                let span = state.ready.span_of(from).cloned().unwrap_or(from..to);
                let step = (model / gcd(model, out)) as usize;
                // Starting at a multiple of `step` keeps the output grid aligned; samples that are
                // not ready read as silence.
                let start = from.saturating_sub(RESAMPLE_CONTEXT).max(span.start) / step * step;
                (start, (to + RESAMPLE_CONTEXT).min(span.end), span.start)
            };
            let mut audio = Self::part(&state, name, start, end);
            let unready = ready.saturating_sub(start);
            for channel in &mut audio.data {
                channel[..unready].fill(0.0);
            }
            (audio, start)
        };

        let model = u64::from(self.spec.sample_rate);
        let out = u64::from(output.rate.hz());
        if model == out {
            return Ok(Excerpt { audio, from });
        }
        let converted = output.dsp.convert(&audio, output.rate.hz())?;
        let offset = output_index(start, model, out);
        let a = output_index(from, model, out) - offset;
        let b = (output_index(to, model, out) - offset).min(converted.frames());
        Ok(Excerpt {
            audio: converted.excerpt(a.min(b), b),
            from: output_index(from, model, out),
        })
    }

    /// Part `name` over `from..to` at the model's rate, the derived one included.
    pub(super) fn part(state: &TrackState, name: &str, from: usize, to: usize) -> Audio {
        if let Some((_, audio)) = state.stems.iter().find(|(n, _)| *n == name) {
            return audio.excerpt(from, to);
        }
        let parts: Vec<Audio> = state
            .stems
            .iter()
            .map(|(_, a)| a.excerpt(from, to))
            .collect();
        stemd_core::stems::remainder(&state.mix.excerpt(from, to), parts.iter())
    }

    /// The pieces ready in memory or in `stored`, the samples uploaded, and the samples ready or
    /// stored.
    pub(super) fn counts(&self, stored: &HashSet<usize>) -> (usize, usize, usize) {
        let state = self.state.lock();
        let mut covered = state.ready.clone();
        for &k in stored {
            covered.insert(self.piece(k));
        }
        let ready = (0..self.spec.grid.len())
            .filter(|&k| covered.covers(&self.piece(k)))
            .count();
        (ready, state.arrived.total(), covered.total())
    }

    pub(super) fn summary(&self) -> TrackSummary {
        let state = self.state.lock();
        TrackSummary {
            arrived: state.arrived.as_pairs(),
            ready: state.ready.as_pairs(),
            needs_stats: self.spec.normalises && state.norm.is_none(),
            closed: state.closed.clone(),
        }
    }
}

pub(super) struct TrackSummary {
    pub(super) arrived: Vec<[usize; 2]>,
    pub(super) ready: Vec<[usize; 2]>,
    pub(super) needs_stats: bool,
    pub(super) closed: Option<String>,
}

/// Why an upload was refused.
#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    #[error("{0}")]
    Invalid(String),
    #[error("samples {0}..{1} differ from what was already uploaded for this track")]
    Differs(usize, usize),
    #[error("{0:#}")]
    Closed(anyhow::Error),
}
