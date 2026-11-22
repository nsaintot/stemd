//! What the cache, the running jobs and the open sessions hold for one track and output.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Serialize;
use stemd_core::StemFormat;

use crate::cache::{Cache, Consolidation, TrackInfo, type_dir, type_prefix};
use crate::jobs::{Entry, Jobs, MixShape, job_key};
use crate::output::{Output, OutputView, Storage, sources};
use crate::sessions::Sessions;
use crate::whole::Whole;

/// What the server holds for one track and output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Held {
    /// Every stem is stored whole.
    Stored,
    /// Every stem is stored whole in another container of the same samples.
    Derivable,
    /// Every piece is stored, and the whole files are being written.
    Consolidating,
    /// A job for this output is running; `job` is its id.
    Separating,
    /// The mix is in memory: a session can join it without an upload.
    Open,
    /// Some pieces are stored, and the mix is not in memory.
    Partial,
}

impl Held {
    /// The name this state has in JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stored => "stored",
            Self::Derivable => "derivable",
            Self::Consolidating => "consolidating",
            Self::Separating => "separating",
            Self::Open => "open",
            Self::Partial => "partial",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StemState {
    pub name: &'static str,
    /// The sources the stem holds, see [`crate::output::sources`].
    pub sources: u16,
    pub stored: Storage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
}

/// The answer to `GET /v2/separations/{id}`.
#[derive(Debug, Clone, Serialize)]
pub struct Lookup {
    pub track: String,
    pub model_id: String,
    /// Track length at the model's rate.
    pub frames: usize,
    pub sample_rate: u32,
    pub channels: usize,
    pub output: OutputView,
    pub state: Held,
    /// The stored output a `derivable` one is converted from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<OutputView>,
    /// The running job, when `state` is `separating`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job: Option<String>,
    pub stems: Vec<StemState>,
}

/// A track directory of one output at one gain.
#[derive(Debug, Clone)]
pub struct TypeDir {
    pub dir: PathBuf,
    pub output: Output,
    pub gain: f32,
    pub info: TrackInfo,
}

impl TypeDir {
    /// The whole file of `stem`, when it holds the whole track.
    pub fn whole(&self, stem: &str) -> Option<Whole> {
        self.info.find_whole(&self.dir, stem, &self.output)
    }

    fn all_whole(&self) -> bool {
        self.output
            .stored_parts()
            .iter()
            .all(|s| self.whole(s).is_some())
    }

    fn consolidation(&self) -> Consolidation {
        Consolidation::new(self.dir.clone(), &self.output, self.info)
    }

    fn storage(&self, stem: &str) -> (Storage, Option<u64>) {
        if let Some(whole) = self.whole(stem) {
            let bytes = std::fs::metadata(&whole.path).map(|m| m.len()).ok();
            return (Storage::Whole, bytes);
        }
        let any_piece = std::fs::read_dir(self.dir.join(stem))
            .map(|mut entries| entries.next().is_some())
            .unwrap_or(false);
        let storage = if any_piece {
            Storage::Pieces
        } else {
            Storage::None
        };
        (storage, None)
    }

    /// True when any stem has a piece or a whole file.
    fn holds_any(&self) -> bool {
        self.output
            .stored_parts()
            .iter()
            .any(|s| self.storage(s).0 != Storage::None)
    }
}

/// `output`'s track directories for `id` at any gain: all whole, then all pieces, then the rest.
fn type_dirs(cache: &Cache, model_id: &str, output: &Output, id: &str) -> Vec<TypeDir> {
    let prefix = type_prefix(output);
    let mut out: Vec<TypeDir> = cache
        .types(model_id)
        .into_iter()
        .filter_map(|name| {
            let gain: f32 = name.strip_prefix(&prefix)?.parse().ok()?;
            let dir = cache.verified_dir(model_id, &name, id);
            let info = Cache::info(&dir)?;
            Some(TypeDir {
                dir,
                output: output.clone(),
                gain,
                info,
            })
        })
        .collect();
    out.sort_by_key(|f| {
        if f.all_whole() {
            0
        } else if Cache::needs_consolidation(&f.consolidation()) {
            1
        } else {
            2
        }
    });
    out
}

/// The outputs whose whole files convert to `output`'s samples, each with the gain it gives.
fn sources_of(output: &Output) -> Vec<(Output, Option<f32>)> {
    if output.format.has_headroom() {
        return Vec::new();
    }
    let with = |format: StemFormat| Output {
        format,
        ..output.clone()
    };
    let mut out: Vec<(Output, Option<f32>)> =
        [StemFormat::Flac, StemFormat::Wav, StemFormat::Pcm16]
            .into_iter()
            .filter(|f| *f != output.format)
            .map(|f| (with(f), None))
            .collect();
    // A 16-bit read of `pcm32` samples, as a session without `peak` makes it.
    out.push((with(StemFormat::Pcm32), Some(output.gain(None))));
    out
}

/// A track directory of `output` for `id` with every stem stored whole, at any gain.
pub fn whole_dir(cache: &Cache, model_id: &str, output: &Output, id: &str) -> Option<TypeDir> {
    let best = type_dirs(cache, model_id, output, id)
        .into_iter()
        .find(TypeDir::all_whole)?;
    Cache::touch(&best.dir);
    Some(best)
}

/// A stored output that converts to `output`, and the gain the result has.
pub fn derivable(
    cache: &Cache,
    model_id: &str,
    output: &Output,
    id: &str,
) -> Option<(TypeDir, f32)> {
    sources_of(output).into_iter().find_map(|(source, gain)| {
        let found = type_dirs(cache, model_id, &source, id)
            .into_iter()
            .find(TypeDir::all_whole)?;
        let gain = gain.unwrap_or(found.gain);
        Some((found, gain))
    })
}

/// A track directory of `output` for `id` that holds a piece or a whole stem, at `gain` when one does.
pub fn claimable(
    cache: &Cache,
    model_id: &str,
    output: &Output,
    id: &str,
    gain: f32,
) -> Option<TypeDir> {
    let own = cache.verified_dir(model_id, &type_dir(output, gain), id);
    let mut found: Vec<TypeDir> = type_dirs(cache, model_id, output, id)
        .into_iter()
        .filter(TypeDir::holds_any)
        .collect();
    let at = found.iter().position(|f| f.dir == own).unwrap_or(0);
    (at < found.len()).then(|| found.swap_remove(at))
}

/// What the server holds for `id` at `output`, queueing consolidation once every piece is stored.
pub fn lookup(
    cache: &Arc<Cache>,
    sessions: &Sessions,
    jobs: &Jobs,
    model_id: &str,
    output: &Output,
    id: &str,
) -> Option<Lookup> {
    let served = output.served();
    let stem_states = |found: Option<&TypeDir>| -> Vec<StemState> {
        served
            .iter()
            .map(|&name| {
                let (stored, bytes) = found.map_or((Storage::None, None), |f| f.storage(name));
                StemState {
                    name,
                    sources: sources(name),
                    stored,
                    bytes,
                }
            })
            .collect()
    };
    let answer = |state: Held, found: Option<&TypeDir>, mix: MixShape, gain: Option<f32>| Lookup {
        track: id.to_owned(),
        model_id: model_id.to_owned(),
        frames: mix.frames,
        sample_rate: mix.sample_rate,
        channels: mix.channels,
        output: OutputView::of(output, gain),
        state,
        from: None,
        job: None,
        stems: stem_states(found),
    };

    let exact = type_dirs(cache, model_id, output, id);
    if let Some(best) = exact.first() {
        Cache::touch(&best.dir);
        if best.all_whole() {
            return Some(answer(
                Held::Stored,
                Some(best),
                best.info.into(),
                Some(best.gain),
            ));
        }
        let task = best.consolidation();
        if Cache::needs_consolidation(&task) && cache.consolidate_later(task) {
            return Some(answer(
                Held::Consolidating,
                Some(best),
                best.info.into(),
                Some(best.gain),
            ));
        }
    }
    if let Some((source, gain)) = derivable(cache, model_id, output, id) {
        Cache::touch(&source.dir);
        let mut lookup = answer(Held::Derivable, None, source.info.into(), Some(gain));
        lookup.from = Some(OutputView::of(&source.output, Some(source.gain)));
        return Some(lookup);
    }
    if let Some(job) = jobs.running(&job_key(id, output, model_id)) {
        let mut lookup = answer(Held::Separating, exact.first(), job.mix, None);
        lookup.job = Some(job.id.clone());
        return Some(lookup);
    }
    if let Some(track) = sessions.open_track(id, output.parts) {
        let info = TrackInfo::from(&track.spec);
        return Some(answer(Held::Open, exact.first(), info.into(), None));
    }
    let best = exact.iter().find(|f| f.holds_any())?;
    Some(answer(
        Held::Partial,
        Some(best),
        best.info.into(),
        Some(best.gain),
    ))
}

/// Convert `source` into `output` for track `id`, skipping stems already stored.
pub fn derive(
    cache: &Cache,
    model_id: &str,
    id: &str,
    source: &TypeDir,
    output: &Output,
    gain: f32,
) -> Result<TypeDir> {
    let dir = cache.verified_dir(model_id, &type_dir(output, gain), id);
    // A second request for the same output waits here, then finds the stems stored.
    let converting = cache.converting(&dir);
    let _held = converting.lock();
    cache.write_info(&dir, &source.info)?;
    let target = TypeDir {
        dir,
        output: output.clone(),
        gain,
        info: source.info,
    };
    let rate = output.rate.hz();
    for stem in output.stored_parts() {
        if target.whole(stem).is_some() {
            continue;
        }
        let whole = source
            .whole(stem)
            .with_context(|| format!("{stem} is not stored whole"))?;
        let bytes = std::fs::read(&whole.path)
            .with_context(|| format!("reading {}", whole.path.display()))?;
        let audio = crate::whole::decode(bytes, source.output.format, source.info.channels, rate)
            .with_context(|| format!("decoding {}", whole.path.display()))?;
        let (bytes, clipped) = if source.output.format.has_headroom() {
            crate::whole::encode(&audio, output.format, gain, false)?
        } else {
            (
                stemd_core::encode_stem(&audio, output.format, 1.0)?,
                whole.meta.clipped,
            )
        };
        cache.consolidate(
            &target.consolidation(),
            stem,
            &bytes,
            audio.frames(),
            clipped,
        )?;
    }
    tracing::info!(track = %id, "converted {} to {}", source.output.format, output.format);
    Ok(target)
}

/// The job result for track `id` at `output` when the cache holds every stem
/// whole, at any gain, held for the client to fetch.
pub fn cached_entry(
    cache: &Arc<Cache>,
    cache_id: &str,
    output: &Output,
    id: &str,
) -> Option<Arc<Entry>> {
    let found = whole_dir(cache, cache_id, output, id)?;
    let _writing = cache.writing(&found.dir);
    let entry = Arc::new(Entry::stored(
        &found.dir,
        output,
        found.gain,
        found.info.channels,
        Cache::separation(&found.dir),
    )?);
    cache.hold(&entry);
    entry.present().then_some(entry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_state_names_itself_as_its_json_does() {
        use Held::*;
        for held in [Stored, Derivable, Consolidating, Separating, Open, Partial] {
            assert_eq!(serde_json::to_value(held).unwrap(), held.as_str());
        }
    }
}
