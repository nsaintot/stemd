//! Files dropped on the window.
//!
//! A drop is the same job a `POST /v1/jobs` makes: hashed the same way, claimed in
//! the same store, separated by the same worker. Dropping a track a deck already
//! asked for joins that separation, and dropping the same file twice is a cache
//! hit. What differs is the two ends: the audio comes from a file rather than a
//! socket, and the stems are copied to a folder rather than served.
//!
//! Local output always includes the derived part. Over the wire it is left out
//! because a client holding the mix can rebuild it for free; a folder of stems has
//! no such context.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use stemd_core::{Audio, DspMode, PcmFormat, Progress, Stage};

use crate::api::AppState;
use crate::cache::{Entry, Ingredients, Output};
use crate::queue::QueuedWork;

/// Recent drops kept for the window's list. Older ones fall off the end.
pub const REMEMBERED: usize = 12;

/// What a stems folder is called, after the track. Named once because the
/// window's delete control checks it before removing anything recursively.
const STEMS_SUFFIX: &str = "-stems";

/// How often the watcher rechecks a job it is waiting on.
///
/// The window repaints four times a second, so anything finer than this is
/// invisible. It only paces the copy-out; progress itself comes from the job.
const POLL: std::time::Duration = std::time::Duration::from_millis(100);

/// Extensions offered to whoever is dropping. Not a gate: symphonia probes the
/// content, but a drop of a text file should be refused before it is decoded.
///
/// The window's file picker is built from this same list, so what can be dropped
/// and what can be chosen cannot drift apart.
pub const AUDIO_EXTENSIONS: [&str; 9] = [
    "wav", "flac", "mp3", "m4a", "aac", "aif", "aiff", "mp4", "ogg",
];

pub fn looks_like_audio(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| AUDIO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// Where one drop has got to.
#[derive(Debug, Clone)]
pub enum State {
    Reading,
    Working(Progress),
    Done(Finished),
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Finished {
    pub dir: PathBuf,
    pub parts: Vec<String>,
    /// What the separation cost when it ran, which is preserved across cache
    /// hits, so this is what it cost, not what this drop waited.
    pub secs: f64,
    /// The stems were already on disk and nothing was separated.
    pub cached: bool,
}

/// One dropped file, from the moment it lands until its stems are written.
pub struct Dropped {
    pub source: PathBuf,
    /// File name without the extension, which is what the window shows.
    pub title: String,
    state: Mutex<State>,
}

impl Dropped {
    pub fn state(&self) -> State {
        self.state.lock().clone()
    }

    /// A finished drop, for tests that need a list without running a model.
    /// `state` is private to this module, so the window's own tests cannot build
    /// one of these for themselves.
    #[cfg(test)]
    pub fn finished(title: &str, dir: &Path, parts: usize) -> Arc<Self> {
        Arc::new(Self {
            source: dir.join(format!("{title}.wav")),
            title: title.to_owned(),
            state: Mutex::new(State::Done(Finished {
                dir: dir.to_path_buf(),
                parts: (0..parts).map(|i| format!("stem {i}")).collect(),
                secs: 1.0,
                cached: false,
            })),
        })
    }

    fn set(&self, stage: State) {
        *self.state.lock() = stage;
    }

    pub fn is_running(&self) -> bool {
        matches!(self.state(), State::Reading | State::Working(_))
    }
}

/// The window's list of drops, newest first.
#[derive(Default)]
pub struct Drops {
    recent: Mutex<Vec<Arc<Dropped>>>,
}

impl Drops {
    /// Take a file and start working on it. Returns the entry the window draws.
    ///
    /// Everything after this happens on its own thread: decoding a five-minute
    /// track takes long enough to drop frames, and the window has to keep
    /// painting the progress of the job this creates.
    pub fn accept(self: &Arc<Self>, state: &Arc<AppState>, path: PathBuf) -> Arc<Dropped> {
        let dropped = Arc::new(Dropped {
            title: title_of(&path),
            source: path,
            state: Mutex::new(State::Reading),
        });

        {
            let mut recent = self.recent.lock();
            recent.insert(0, Arc::clone(&dropped));
            recent.truncate(REMEMBERED);
        }

        let state = Arc::clone(state);
        let started = Arc::clone(&dropped);
        std::thread::Builder::new()
            .name("stemd-drop".into())
            .spawn(move || {
                if let Err(err) = run(&state, &started) {
                    tracing::error!("{}: {err:#}", started.title);
                    started.set(State::Failed(format!("{err:#}")));
                }
            })
            .expect("spawning the drop worker");

        dropped
    }

    pub fn recent(&self) -> Vec<Arc<Dropped>> {
        self.recent.lock().clone()
    }

    /// Take a drop out of the list, leaving its stems on disk.
    ///
    /// By identity rather than by name: two files with the same title in
    /// different folders are two drops, and dropping one track twice makes two
    /// rows that must be dismissable separately.
    pub fn forget(&self, item: &Arc<Dropped>) {
        self.recent.lock().retain(|other| !Arc::ptr_eq(other, item));
    }

    /// Forget a drop and delete the stems it wrote.
    ///
    /// Deleted outright rather than moved to the Trash, which would need Finder
    /// automation and a permission prompt inside an `.app`. The separation is still in
    /// the cache, so dropping the same file again writes the folder back without
    /// running the model.
    pub fn discard(&self, item: &Arc<Dropped>) {
        if let State::Done(finished) = item.state() {
            remove_stems(&finished.dir);
        }
        self.forget(item);
    }
}

/// Delete a folder this wrote, and nothing else.
///
/// The name check is not defensive programming for its own sake: this is a
/// recursive delete of a path taken from a struct field, and the cost of the
/// check is one string comparison against the cost of being wrong.
fn remove_stems(dir: &Path) {
    if !dir
        .file_name()
        .is_some_and(|name| name.to_string_lossy().ends_with(STEMS_SUFFIX))
    {
        tracing::warn!("refusing to delete {}: not a stems folder", dir.display());
        return;
    }
    match std::fs::remove_dir_all(dir) {
        Ok(()) => tracing::info!("deleted {}", dir.display()),
        // Already gone is the outcome that was wanted.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => tracing::warn!("could not delete {}: {err}", dir.display()),
    }
}

/// Decode, separate, and write the stems out.
fn run(state: &Arc<AppState>, item: &Arc<Dropped>) -> Result<()> {
    let mix = read(state, &item.source)?;
    let settings = state.settings.get();
    let output = Output {
        format: settings.format,
        rate: settings.rate,
        derived: true,
        // A drop writes files, it does not feed a client's subtraction.
        dsp: DspMode::General,
    };

    let (entry, cached) = separate(state, item, &mix, output)?;
    let finished = write_out(&item.source, &entry, cached)?;
    tracing::info!(
        "{}: {} parts written to {}",
        item.title,
        finished.parts.len(),
        finished.dir.display()
    );
    item.set(State::Done(finished));
    Ok(())
}

/// Decode the file, and bring it to the rate the model works at.
///
/// The command-line client refuses anything that is not already the model's rate.
/// A drop has no such conversation available, and the resampler that runs on the
/// way out is the same one.
fn read(state: &AppState, path: &Path) -> Result<Audio> {
    let decoded = stemd_audio::decode(path)?;
    let audio = Audio::new(decoded.data, decoded.sample_rate);

    let secs = audio.duration_secs();
    if secs > state.max_track_secs {
        anyhow::bail!(
            "track is {:.1} minutes; this server accepts up to {:.0}. Separation \
             memory grows with length, so longer tracks are refused rather than \
             risked",
            secs / 60.0,
            state.max_track_secs / 60.0
        );
    }

    let wanted = state.queue.info().sample_rate;
    if audio.sample_rate == wanted {
        return Ok(audio);
    }
    tracing::info!(
        "{}: {} Hz, resampling to the model's {} Hz",
        path.display(),
        audio.sample_rate,
        wanted
    );
    stemd_core::resample::to_rate(&audio, wanted).context("resampling the input")
}

/// Put the mix through the same queue an upload uses, and wait for the stems.
fn separate(
    state: &Arc<AppState>,
    item: &Arc<Dropped>,
    mix: &Audio,
    output: Output,
) -> Result<(Arc<Entry>, bool)> {
    // Hashed from the same bytes an upload would have carried, so a drop and a
    // deck asking for the same track land on the same cache entry.
    let pcm = mix.to_interleaved(PcmFormat::S16le);
    let key = crate::cache::key(&Ingredients {
        pcm: &pcm,
        sample_rate: mix.sample_rate,
        channels: mix.channels(),
        in_format: PcmFormat::S16le,
        out_format: output.format,
        out_rate: output.rate,
        include_derived: output.derived,
        dsp: output.dsp,
        model: &state.switcher.identity(),
    });
    std::mem::drop(pcm);

    if let Some(entry) = state.cache.get(&key) {
        // The row in the window already says "already separated"; this is only
        // for reading afterwards.
        tracing::debug!("{}: already separated, reusing it", item.title);
        return Ok((entry, true));
    }

    // Claim before enqueuing, exactly as the HTTP path does: it is the same
    // store, so a drop can join a separation a deck already started.
    let job = match state.store.claim(&key) {
        Ok(job) => {
            state.queue.submit(QueuedWork {
                job: Arc::clone(&job),
                mix: mix.clone(),
                output,
            })?;
            job
        }
        Err(existing) => {
            tracing::debug!("{}: joining {}, already under way", item.title, existing.id);
            existing.join();
            existing
        }
    };

    loop {
        let view = job.view();
        match view.progress.stage {
            Stage::Done => {
                let entry = job.entry.lock().clone();
                return Ok((
                    entry.context("the job finished with no stems attached")?,
                    false,
                ));
            }
            Stage::Failed => {
                anyhow::bail!(
                    "{}",
                    view.error.unwrap_or_else(|| "separation failed".into())
                )
            }
            Stage::Cancelled => anyhow::bail!("stopped"),
            _ => {
                item.set(State::Working(view.progress));
                std::thread::sleep(POLL);
            }
        }
    }
}

/// Copy the stems out of the cache into a folder beside the source file.
///
/// Copied rather than moved: the entry stays in the cache, so a deck asking for
/// the same track over the network still gets it without separating again.
fn write_out(source: &Path, entry: &Entry, cached: bool) -> Result<Finished> {
    let dir = output_dir(source);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let mut parts = Vec::new();
    let mut scaled = Vec::new();
    for stem in &entry.stems {
        let name = stem
            .path
            .file_name()
            .map_or_else(|| stem.name.clone(), |n| n.to_string_lossy().into_owned());
        std::fs::copy(&stem.path, dir.join(&name))
            .with_context(|| format!("copying {} into {}", name, dir.display()))?;
        // Tells the cache this entry was collected, so the rule that reaps
        // separations nobody pulled leaves it alone.
        entry.mark_fetched(&stem.name);
        if stem.gain != 1.0 {
            scaled.push((stem.name.clone(), stem.gain));
        }
        parts.push(stem.name.clone());
    }

    if !scaled.is_empty() {
        write_gains(&dir, &scaled)?;
    }

    Ok(Finished {
        dir,
        parts,
        secs: entry.separation_secs,
        cached,
    })
}

/// `<track>-stems/` beside the file that was dropped, matching what the
/// command-line client writes by default.
fn output_dir(source: &Path) -> PathBuf {
    let parent = source.parent().unwrap_or(Path::new("."));
    parent.join(format!("{}{STEMS_SUFFIX}", title_of(source)))
}

/// Record the scale a part was written at, when it is not unity.
///
/// A stem that peaks past full scale cannot be stored in a 16-bit container
/// without clipping, so it is written quieter. That is the right file, but it
/// means the parts no longer sum back to the mix, and nothing in a folder of
/// audio can say so. This can.
fn write_gains(dir: &Path, scaled: &[(String, f32)]) -> Result<()> {
    let gains: serde_json::Map<String, serde_json::Value> = scaled
        .iter()
        .map(|(name, gain)| (name.clone(), serde_json::json!(f64::from(*gain))))
        .collect();
    let note = serde_json::json!({
        "note": "These parts were scaled to fit a 16-bit container. Divide by the \
                 gain to recover the separated level; until you do, they will not \
                 sum back to the mix.",
        "gains": gains,
    });
    let path = dir.join("stems.json");
    std::fs::write(&path, serde_json::to_string_pretty(&note)?)
        .with_context(|| format!("writing {}", path.display()))?;
    tracing::info!(
        "{} parts were scaled to fit; noted in stems.json",
        scaled.len()
    );
    Ok(())
}

fn title_of(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "track".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_audio_is_worth_decoding() {
        for good in ["a.wav", "A.WAV", "b.flac", "c.mp3", "d.m4a", "e.aiff"] {
            assert!(looks_like_audio(Path::new(good)), "{good}");
        }
        for bad in ["notes.txt", "cover.jpg", "no-extension", "track.wav.zip"] {
            assert!(!looks_like_audio(Path::new(bad)), "{bad}");
        }
    }

    /// The folder goes beside the file, not into the working directory: an
    /// `.app` is launched with its cwd set to `/`.
    #[test]
    fn stems_land_beside_the_track() {
        assert_eq!(
            output_dir(Path::new("/Users/dj/Music/Track One.flac")),
            Path::new("/Users/dj/Music/Track One-stems")
        );
    }

    fn finished_drop(dir: &Path) -> Arc<Dropped> {
        Dropped::finished("track", dir, 1)
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("stemd-drops-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// The cross deletes a directory recursively, so what it will delete is
    /// worth a test rather than a reading of the code that produces the path.
    #[test]
    fn discarding_takes_the_stems_and_the_row() {
        let root = scratch("discard");
        let stems = root.join("track-stems");
        std::fs::create_dir_all(&stems).unwrap();
        std::fs::write(stems.join("vocals.flac"), b"not really a flac").unwrap();

        let drops = Arc::new(Drops::default());
        let item = finished_drop(&stems);
        drops.recent.lock().push(Arc::clone(&item));

        drops.discard(&item);
        assert!(!stems.exists(), "the stems folder survived");
        assert!(drops.recent().is_empty(), "the row survived");
    }

    /// And nothing else. `Finished::dir` is always a folder this wrote, but a
    /// recursive delete is not the place to rely on that staying true.
    #[test]
    fn nothing_but_a_stems_folder_is_ever_deleted() {
        let root = scratch("guard");
        let precious = root.join("Music");
        std::fs::create_dir_all(&precious).unwrap();
        std::fs::write(precious.join("keep me.wav"), b"...").unwrap();

        let drops = Arc::new(Drops::default());
        let item = finished_drop(&precious);
        drops.recent.lock().push(Arc::clone(&item));

        drops.discard(&item);
        assert!(
            precious.join("keep me.wav").exists(),
            "deleted the wrong thing"
        );
        // The row still goes: it is the list entry the cross was clicked on.
        assert!(drops.recent().is_empty());
    }

    /// Two drops of files with the same name in different folders are two rows,
    /// and dismissing one must not take the other.
    #[test]
    fn forgetting_is_by_identity_not_by_title() {
        let drops = Arc::new(Drops::default());
        let first = finished_drop(Path::new("/a/track-stems"));
        let second = finished_drop(Path::new("/b/track-stems"));
        drops.recent.lock().push(Arc::clone(&first));
        drops.recent.lock().push(Arc::clone(&second));

        drops.forget(&first);
        let left = drops.recent();
        assert_eq!(left.len(), 1);
        assert!(Arc::ptr_eq(&left[0], &second));
    }

    #[test]
    fn a_file_with_no_parent_still_gets_a_folder() {
        assert_eq!(output_dir(Path::new("track.wav")), Path::new("track-stems"));
    }
}
