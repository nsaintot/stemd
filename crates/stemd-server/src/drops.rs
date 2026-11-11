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
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use parking_lot::{Condvar, Mutex};
use stemd_core::{Audio, DspMode, PART_NAMES, PcmFormat, Progress, Stage};

use crate::api::AppState;
use crate::cache::{Entry, Ingredients, Output};
use crate::jobs::Job;
use crate::queue::QueuedWork;
use crate::settings::DropParts;

/// Where a drop records the scale of parts written below unity.
const GAINS_FILE: &str = "stems.json";

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
    /// The job this drop waits on, once claimed. Whoever takes it releases it.
    job: Mutex<Option<Arc<Job>>>,
    /// Set by [`Drops::stop`].
    stopped: AtomicBool,
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
            job: Mutex::new(None),
            stopped: AtomicBool::new(false),
        })
    }

    /// A drop part way through, for the window's tests.
    #[cfg(test)]
    pub fn working(title: &str, progress: Progress) -> Arc<Self> {
        Arc::new(Self {
            source: PathBuf::from(format!("/Music/{title}.wav")),
            title: title.to_owned(),
            state: Mutex::new(State::Working(progress)),
            job: Mutex::new(None),
            stopped: AtomicBool::new(false),
        })
    }

    fn set(&self, stage: State) {
        *self.state.lock() = stage;
    }

    pub fn is_running(&self) -> bool {
        matches!(self.state(), State::Reading | State::Working(_))
    }

    fn stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// Let go of the job, and cancel it if nobody else waits on it. Returns
    /// whether this drop was the last waiter.
    fn release(&self, state: &AppState) -> bool {
        let Some(job) = self.job.lock().take() else {
            return false;
        };
        if job.release() > 0 {
            return false;
        }
        if !job.view().progress.stage.is_terminal() {
            state.queue.cancel(&job.id);
            state.store.remove(&job.id);
        }
        true
    }
}

/// The window's list of drops, newest first.
#[derive(Default)]
pub struct Drops {
    recent: Mutex<Vec<Arc<Dropped>>>,
    turns: Turnstile,
}

/// Queue positions in drop order: a short track decodes faster than a long one
/// dropped before it, and must still be queued after it.
#[derive(Default)]
struct Turnstile {
    next: Mutex<Line>,
    moved: Condvar,
}

#[derive(Default)]
struct Line {
    issued: u64,
    serving: u64,
}

impl Turnstile {
    fn ticket(&self) -> u64 {
        let mut line = self.next.lock();
        line.issued += 1;
        line.issued - 1
    }
}

/// One drop's place in line. [`Turn::wait`] takes the turn and [`Turn::pass`]
/// hands it on; dropping a turn that was never passed does both, so a drop
/// that fails or panics cannot hold up the ones behind it.
struct Turn<'a> {
    turns: &'a Turnstile,
    ticket: u64,
    waited: bool,
    passed: bool,
}

impl<'a> Turn<'a> {
    const fn new(turns: &'a Turnstile, ticket: u64) -> Self {
        Self {
            turns,
            ticket,
            waited: false,
            passed: false,
        }
    }

    /// Block until every earlier drop has been queued or has failed.
    fn wait(&mut self) {
        if self.waited {
            return;
        }
        let mut line = self.turns.next.lock();
        while line.serving != self.ticket {
            self.turns.moved.wait(&mut line);
        }
        self.waited = true;
    }

    /// Let the next drop go.
    fn pass(&mut self) {
        if self.passed {
            return;
        }
        self.wait();
        self.turns.next.lock().serving += 1;
        self.turns.moved.notify_all();
        self.passed = true;
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.pass();
    }
}

impl Drops {
    /// Insert a drop at the top, replacing any with the same source to avoid duplicates.
    fn remember(&self, dropped: Arc<Dropped>) {
        let mut recent = self.recent.lock();
        recent.retain(|other| other.source != dropped.source);
        recent.insert(0, dropped);
        recent.truncate(REMEMBERED);
    }

    /// Take a file and start working on it. Returns the entry the window draws.
    ///
    /// Everything after this happens on its own thread: decoding a five-minute
    /// track takes long enough to drop frames, and the window has to keep
    /// painting the progress of the job this creates.
    pub fn accept(self: &Arc<Self>, state: &Arc<AppState>, path: PathBuf) -> Arc<Dropped> {
        // A second drop of a track still in progress would write the same folder.
        if let Some(running) = self
            .recent
            .lock()
            .iter()
            .find(|d| d.source == path && d.is_running())
        {
            return Arc::clone(running);
        }
        let dropped = Arc::new(Dropped {
            title: title_of(&path),
            source: path,
            state: Mutex::new(State::Reading),
            job: Mutex::new(None),
            stopped: AtomicBool::new(false),
        });

        self.remember(Arc::clone(&dropped));

        let state = Arc::clone(state);
        let drops = Arc::clone(self);
        let ticket = self.turns.ticket();
        let started = Arc::clone(&dropped);
        let spawned = std::thread::Builder::new()
            .name("stemd-drop".into())
            .spawn(move || {
                let mut turn = Turn::new(&drops.turns, ticket);
                let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(&state, &started, &mut turn)
                }));
                let failure = match ran {
                    Ok(Ok(())) => None,
                    Ok(Err(err)) => Some(format!("{err:#}")),
                    Err(payload) => Some(format!(
                        "the drop panicked: {}",
                        crate::panics::message(&*payload)
                    )),
                };
                // Marked failed before the turn is handed on, which may wait.
                if let Some(why) = failure {
                    if started.stopped() {
                        tracing::info!("{}: stopped", started.title);
                    } else {
                        tracing::error!("{}: {why}", started.title);
                    }
                    started.set(State::Failed(why));
                }
                turn.pass();
            });
        if let Err(err) = spawned {
            // Hand the turn on, or every later drop waits for it.
            std::mem::drop(Turn::new(&self.turns, ticket));
            tracing::error!("{}: could not start: {err}", dropped.title);
            dropped.set(State::Failed(format!("could not start: {err}")));
        }

        dropped
    }

    pub fn recent(&self) -> Vec<Arc<Dropped>> {
        self.recent.lock().clone()
    }

    /// Take a drop out of the list, leaving its stems on disk.
    ///
    /// By identity rather than by name: two files with the same title in
    /// different folders are two drops.
    pub fn forget(&self, item: &Arc<Dropped>) {
        self.recent.lock().retain(|other| !Arc::ptr_eq(other, item));
    }

    /// Stop a drop that is still running and take it out of the list. Its job
    /// is cancelled unless an API client also waits on it.
    pub fn stop(&self, state: &AppState, item: &Arc<Dropped>) {
        item.stopped.store(true, Ordering::SeqCst);
        item.release(state);
        self.forget(item);
    }

    /// Forget a drop and delete the stems it wrote.
    ///
    /// The folder is deleted outright. Moving it to the Trash needs Finder
    /// automation and a permission prompt inside an `.app`.
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
fn run(state: &Arc<AppState>, item: &Arc<Dropped>, turn: &mut Turn) -> Result<()> {
    let mix = read(state, &item.source)?;
    if item.stopped() {
        anyhow::bail!("stopped");
    }
    let settings = state.settings.get();
    let output = Output {
        format: settings.format,
        rate: settings.rate,
        derived: true,
        // A drop writes files, it does not feed a client's subtraction.
        dsp: DspMode::General,
        parts: settings.drop_parts.request(),
    };

    let (entry, origin) = separate(state, item, mix, output, turn)?;
    let written = write_out(
        &item.source,
        &entry,
        matches!(origin, Origin::Cached),
        settings.drop_parts,
    );
    // The folder is the drop's only copy. The last drop waiting on the job
    // deletes the entry, unless an API client holds the job or fetched the
    // entry. Kept when writing failed, so a second drop needs no separation.
    if matches!(origin, Origin::Queued) && item.release(state) && written.is_ok() {
        state.cache.discard_unshared(&entry);
    }
    let finished = written?;
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
    // The only mix in the server that is not built from a shape this process
    // chose: the upload path lays its channels out from one frame count in
    // `Audio::from_interleaved`, while this one is whatever came back off a
    // file. See `Audio::checked`.
    let audio = Audio::checked(decoded.data, decoded.sample_rate)
        .with_context(|| format!("decoding {}", path.display()))?;

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

/// Where a drop's cache entry came from.
enum Origin {
    /// Already in the cache, made by an earlier request.
    Cached,
    /// Separated by a job this drop started or joined, held in `Dropped::job`.
    /// The job counts every drop and API client waiting on it.
    Queued,
}

/// Put the mix through the same queue an upload uses, and wait for the stems.
fn separate(
    state: &Arc<AppState>,
    item: &Arc<Dropped>,
    mix: Audio,
    output: Output,
    turn: &mut Turn,
) -> Result<(Arc<Entry>, Origin)> {
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
        parts: output.parts,
        dsp: output.dsp,
        model: &state.switcher.identity(),
    });
    std::mem::drop(pcm);

    // Ahead of the turn: a cache hit queues nothing, so it need not wait for
    // the drops before it. It still passes its turn when it ends.
    if let Some(entry) = state.cache.get(&key) {
        // The row in the window already says "already separated"; this is only
        // for reading afterwards.
        tracing::debug!("{}: already separated, reusing it", item.title);
        return Ok((entry, Origin::Cached));
    }

    turn.wait();
    if item.stopped() {
        anyhow::bail!("stopped");
    }

    // Claim before enqueuing, exactly as the HTTP path does: it is the same
    // store, so a drop can join a separation a deck already started.
    let job = match state.store.claim(&key) {
        Ok(job) => {
            // The stems pass through the cache first. The folder beside the
            // track is checked for the stems' own size once they exist.
            let queued = crate::disk::ensure_room(state.cache.root())
                .map_err(anyhow::Error::from)
                .and_then(|()| {
                    state
                        .queue
                        .submit(QueuedWork {
                            job: Arc::clone(&job),
                            mix,
                            output,
                            model_id: state.switcher.published_identity(),
                            from_api: false,
                        })
                        .map_err(anyhow::Error::from)
                });
            // A job left at Queued would hold its key, and later requests for
            // the track would join it and wait forever.
            if let Err(err) = queued {
                job.fail(format!("{err:#}"));
                return Err(err);
            }
            job
        }
        // `claim` counted this drop as a waiter.
        Err(existing) => {
            tracing::debug!("{}: joining {}, already under way", item.title, existing.id);
            existing
        }
    };
    turn.pass();
    *item.job.lock() = Some(Arc::clone(&job));
    let origin = Origin::Queued;

    loop {
        // `stop` releases the job itself unless it came before the job was stored.
        if item.stopped() {
            item.release(state);
            anyhow::bail!("stopped");
        }
        let view = job.view();
        match view.progress.stage {
            Stage::Done => {
                let entry = job.entry.lock().clone();
                return Ok((
                    entry.context("the job finished with no stems attached")?,
                    origin,
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
fn write_out(source: &Path, entry: &Entry, cached: bool, wanted: DropParts) -> Result<Finished> {
    let dir = output_dir(source);
    let needed = entry
        .stems
        .iter()
        .filter(|stem| wanted.has(&stem.name))
        .map(|stem| stem.bytes)
        .sum();
    crate::disk::ensure_room_for(&dir, needed)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    clear_parts(&dir)?;

    let mut parts = Vec::new();
    let mut scaled = Vec::new();
    for stem in &entry.stems {
        // Not ticked: count it as fetched so the reaper treats the entry as consumed.
        if !wanted.has(&stem.name) {
            entry.mark_fetched(&stem.name);
            continue;
        }
        let name = stem
            .path
            .file_name()
            .map_or_else(|| stem.name.clone(), |n| n.to_string_lossy().into_owned());
        if let Err(err) = std::fs::copy(&stem.path, dir.join(&name)) {
            // Half a set of stems reads as a finished one.
            let _ = clear_parts(&dir);
            return Err(err).with_context(|| format!("copying {} into {}", name, dir.display()));
        }
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

/// Remove the part files and gains note an earlier drop of this track wrote.
fn clear_parts(dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let ours = PART_NAMES.contains(&stem) || path.file_name() == Some(GAINS_FILE.as_ref());
        if ours && path.is_file() {
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        }
    }
    Ok(())
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
    let path = dir.join(GAINS_FILE);
    std::fs::write(&path, serde_json::to_string_pretty(&note)?)
        .with_context(|| format!("writing {}", path.display()))?;
    tracing::info!(
        "{} parts were scaled to fit; noted in {GAINS_FILE}",
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

    /// Drops reach the queue in the order they were dropped, whichever
    /// finishes decoding first.
    #[test]
    fn drops_take_their_turn_in_drop_order() {
        let turns = Arc::new(Turnstile::default());
        let tickets: Vec<u64> = (0..4).map(|_| turns.ticket()).collect();
        let order = Arc::new(Mutex::new(Vec::new()));
        // Started last ticket first, as if the later drops decoded faster.
        let threads: Vec<_> = tickets
            .into_iter()
            .rev()
            .map(|ticket| {
                let turns = Arc::clone(&turns);
                let order = Arc::clone(&order);
                std::thread::spawn(move || {
                    let mut turn = Turn::new(&turns, ticket);
                    turn.wait();
                    order.lock().push(ticket);
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(*order.lock(), vec![0, 1, 2, 3]);
    }

    /// A drop that fails before reaching the queue still hands on its turn.
    #[test]
    fn a_failed_drop_does_not_hold_up_the_next() {
        let turns = Turnstile::default();
        let (first, second) = (turns.ticket(), turns.ticket());
        std::mem::drop(Turn::new(&turns, first));
        let mut next = Turn::new(&turns, second);
        next.wait();
        next.pass();
    }

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
        drops.remember(Arc::clone(&item));

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
        drops.remember(Arc::clone(&item));

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
        drops.remember(Arc::clone(&first));
        drops.remember(Arc::clone(&second));

        drops.forget(&first);
        let left = drops.recent();
        assert_eq!(left.len(), 1);
        assert!(Arc::ptr_eq(&left[0], &second));
    }

    #[test]
    fn a_file_with_no_parent_still_gets_a_folder() {
        assert_eq!(output_dir(Path::new("track.wav")), Path::new("track-stems"));
    }

    /// Re-dropping a track already in the list moves it to the top instead of
    /// adding a second row for it; this is also what keeps two rows from ever
    /// sharing the window's per-row egui id, which is derived from the source.
    #[test]
    fn re_dropping_a_track_replaces_its_row_instead_of_duplicating() {
        let drops = Drops::default();
        let other = finished_drop(Path::new("/b"));
        drops.remember(Arc::clone(&other));
        let first_run = finished_drop(Path::new("/a"));
        drops.remember(Arc::clone(&first_run));
        assert_eq!(drops.recent().len(), 2);

        let second_run = finished_drop(Path::new("/a"));
        drops.remember(Arc::clone(&second_run));

        let recent = drops.recent();
        assert_eq!(
            recent.len(),
            2,
            "dropping the same track again duplicated its row"
        );
        assert!(
            Arc::ptr_eq(&recent[0], &second_run),
            "the re-run did not move to the top"
        );
        assert!(Arc::ptr_eq(&recent[1], &other), "an unrelated row moved");
    }
}
