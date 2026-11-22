//! Files dropped on the window.
//!
//! A drop becomes the same job that `POST /v2/jobs` makes, with the same track id, job table,
//! worker and cache. Dropping a track that a deck already asked for joins that separation, and
//! dropping the same file twice is a cache hit. The two ends differ: the audio comes from a file,
//! and the stems are written to a folder.
//!
//! Local output always includes the derived part. An API client holding the mix can rebuild that
//! part, but a folder of stems has no mix to rebuild it from.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use anyhow::{Context, Result};
use parking_lot::{Condvar, Mutex};
use sha2::{Digest, Sha256};
use stemd_core::{Audio, DspMode, PART_NAMES, Progress, Stage};

use crate::jobs::{CachedStem, Entry, Job};
use crate::output::Output;
use crate::queue::QueuedWork;
use crate::settings::{DropParts, Settings};
use crate::state::AppState;

/// Where a drop records the scale of parts written below unity.
const GAINS_FILE: &str = "stems.json";

/// The number of recent drops kept for the window's list; older drops fall off the end.
pub const REMEMBERED: usize = 12;

/// The number of finished drops kept in memory to answer a repeat drop without separating. It is
/// independent of [`REMEMBERED`]: removing a row from the list keeps its entry here.
pub const HISTORY: usize = 25;

/// The suffix of a stems folder's name, after the track's name. It is defined once because the
/// window's delete control checks it before deleting anything recursively.
const STEMS_SUFFIX: &str = "-stems";

/// How often the watcher rechecks a job it is waiting on.
///
/// The window repaints four times a second, so a shorter interval would not show. It paces only the
/// copy-out; progress comes from the job itself.
const POLL: std::time::Duration = std::time::Duration::from_millis(100);

/// The file extensions a drop accepts. Symphonia probes the content anyway, but a text file should
/// be refused before it is decoded.
///
/// The window's file picker is built from this same list, so the files that can be dropped and the
/// files that can be chosen always match.
pub const AUDIO_EXTENSIONS: [&str; 9] = [
    "wav", "flac", "mp3", "m4a", "aac", "aif", "aiff", "mp4", "ogg",
];

pub fn looks_like_audio(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| AUDIO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// The state of one drop.
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
    /// The time the separation took when it ran. Cache hits keep this value, so it is not the time
    /// this drop waited.
    pub secs: f64,
    /// The stems were already on disk and nothing was separated.
    pub cached: bool,
    /// Every file written: the parts and the gains note, if any.
    pub files: Vec<PathBuf>,
}

/// One dropped file, from the moment it lands until its stems are written.
pub struct Dropped {
    pub source: PathBuf,
    /// The file name without the extension, as the window shows it.
    pub title: String,
    state: Mutex<State>,
    /// The job this drop waits on, once claimed. Whoever takes the job out of this field must
    /// release it.
    job: Mutex<Option<Arc<Job>>>,
    /// Set by [`Drops::stop`].
    stopped: AtomicBool,
}

impl Dropped {
    pub fn state(&self) -> State {
        self.state.lock().clone()
    }

    /// A finished drop, for tests that need a list without running a model. `state` is private to
    /// this module, so the window's tests cannot build one themselves.
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
                files: Vec::new(),
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
            state.stop_job(&job.id);
        }
        true
    }
}

/// The window's list of drops, newest first.
#[derive(Default)]
pub struct Drops {
    recent: Mutex<Vec<Arc<Dropped>>>,
    turns: Turnstile,
    history: History,
}

/// The last [`HISTORY`] finished drops, newest first.
#[derive(Default)]
struct History(Mutex<Vec<Past>>);

/// A finished drop and the state of its files when it finished.
#[derive(Clone)]
struct Past {
    /// See [`history_key`].
    key: String,
    finished: Finished,
    files: Vec<Stamp>,
}

/// A file's length and modification time.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Stamp {
    path: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
}

impl Stamp {
    /// `None` when `path` is not a file.
    fn of(path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(path)
            .ok()
            .filter(std::fs::Metadata::is_file)?;
        Some(Self {
            path: path.to_path_buf(),
            len: meta.len(),
            modified: meta.modified().ok(),
        })
    }
}

impl History {
    /// Keep `finished` under `key`, replacing an older entry with the same key.
    fn record(&self, key: String, finished: &Finished) {
        let Some(files) = finished.files.iter().map(|f| Stamp::of(f)).collect() else {
            return;
        };
        let mut past = self.0.lock();
        past.retain(|p| p.key != key);
        past.insert(
            0,
            Past {
                key,
                finished: finished.clone(),
                files,
            },
        );
        past.truncate(HISTORY);
    }

    /// The drop finished under `key`, if each of its files is still there and
    /// unchanged. An entry with a missing or changed file is removed.
    fn find(&self, key: &str) -> Option<Finished> {
        let mut past = self.0.lock();
        let at = past.iter().position(|p| p.key == key)?;
        if past[at]
            .files
            .iter()
            .all(|f| Stamp::of(&f.path).as_ref() == Some(f))
        {
            return Some(past[at].finished.clone());
        }
        past.remove(at);
        None
    }
}

/// The SHA-256 of the file's bytes, the output settings and the model, which together decide a
/// drop's result. The path and the title play no part.
fn history_key(source: &Path, settings: &Settings, model: &str) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(source)?;
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    for field in [
        format!("{:?}", settings.format),
        format!("{:?}", settings.rate),
        format!("{:?}", settings.drop_parts),
        model.to_owned(),
    ] {
        hasher.update(u32::try_from(field.len()).unwrap_or(u32::MAX).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
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
    /// Decoding and separation run on their own thread: decoding a five-minute track takes long
    /// enough to drop frames, and the window must keep painting the job's progress.
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
                    run(&state, &drops.history, &started, &mut turn)
                }));
                let failure = match ran {
                    Ok(Ok(())) => None,
                    Ok(Err(err)) => Some(format!("{err:#}")),
                    Err(payload) => Some(format!(
                        "the drop panicked: {}",
                        crate::panics::message(&*payload)
                    )),
                };
                // Mark the drop failed before handing on the turn, which may wait.
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
    /// The drop is matched by identity, since two files with the same title in different folders
    /// are two drops.
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
    /// The folder is deleted at once. Moving it to the Trash would need Finder automation and a
    /// permission prompt inside an `.app`.
    pub fn discard(&self, item: &Arc<Dropped>) {
        if let State::Done(finished) = item.state() {
            remove_stems(&finished.dir);
        }
        self.forget(item);
    }
}

/// Delete a folder this wrote, and nothing else.
///
/// The name check costs one string comparison and guards a recursive delete of a path taken from a
/// struct field.
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
        // A folder that is already gone is fine.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => tracing::warn!("could not delete {}: {err}", dir.display()),
    }
}

/// Decode, separate, and write the stems out. A track finished before with the
/// same settings and model, whose files are unchanged, is shown as it was.
fn run(
    state: &Arc<AppState>,
    history: &History,
    item: &Arc<Dropped>,
    turn: &mut Turn,
) -> Result<()> {
    let settings = state.settings.get();
    let key = history_key(&item.source, &settings, &state.switcher.identity())
        .inspect_err(|err| tracing::debug!("{}: not hashed: {err}", item.title))
        .ok();
    if let Some(finished) = key.as_deref().and_then(|key| history.find(key)) {
        tracing::info!(
            "{}: already separated into {}",
            item.title,
            finished.dir.display()
        );
        if let Some(key) = key {
            history.record(key, &finished);
        }
        item.set(State::Done(Finished {
            cached: true,
            ..finished
        }));
        return Ok(());
    }
    let mix = read(state, &item.source)?;
    if item.stopped() {
        anyhow::bail!("stopped");
    }
    let output = Output {
        format: settings.format,
        rate: settings.rate,
        derived: true,
        // A drop writes files and feeds no client's subtraction, so the general resampler suffices.
        dsp: DspMode::General.into(),
        parts: settings.drop_parts.request(),
    };

    let (entry, origin) = separate(state, item, mix, output, turn)?;
    let written = write_out(
        &item.source,
        &entry,
        matches!(origin, Origin::Cached),
        settings.drop_parts,
    );
    if matches!(origin, Origin::Queued) {
        item.release(state);
    }
    let finished = written?;
    tracing::info!(
        "{}: {} parts written to {}",
        item.title,
        finished.parts.len(),
        finished.dir.display()
    );
    if let Some(key) = key {
        history.record(key, &finished);
    }
    item.set(State::Done(finished));
    Ok(())
}

/// Decode the file, and bring it to the rate the model works at.
///
/// The command-line client refuses any rate other than the model's, but a drop cannot ask the user,
/// so it converts with the same resampler that runs on the way out.
fn read(state: &AppState, path: &Path) -> Result<Audio> {
    let decoded = stemd_audio::decode(path)?;
    // This is the only mix whose shape comes from a decoded file; the upload path builds its
    // channels from one frame count in `Audio::from_interleaved`. See `Audio::checked`.
    let audio = Audio::checked(decoded.data, decoded.sample_rate)
        .with_context(|| format!("decoding {}", path.display()))?;

    let secs = audio.duration_secs();
    let max_secs = state.max_track_secs();
    if secs > max_secs {
        anyhow::bail!(
            "track is {:.1} minutes; this server accepts up to {:.0}. Separation \
             memory grows with length, so longer tracks are refused rather than \
             risked",
            secs / 60.0,
            max_secs / 60.0
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
    // The same track id an upload of these samples gets, so a drop and a deck
    // asking for the same track land on the same stems.
    let cache_id = state.switcher.cache_id();
    let id = crate::ident::track_id(&mix);
    let key = crate::jobs::job_key(&id, &output, &cache_id);

    // Check the cache before taking the turn: a cache hit queues nothing, so it need not wait for
    // earlier drops. It still passes its turn when it ends.
    if let Some(entry) = crate::held::cached_entry(&state.cache, &cache_id, &output, &id) {
        // The window's row already says "already separated"; this line is for the log.
        tracing::debug!("{}: already separated, reusing it", item.title);
        return Ok((entry, Origin::Cached));
    }

    turn.wait();
    if item.stopped() {
        anyhow::bail!("stopped");
    }

    // Claim before enqueuing, as the HTTP path does: it is the same table, so a drop can join a
    // separation that a deck already started.
    let job = match state.jobs.claim(&key, crate::jobs::MixShape::of(&mix)) {
        Ok(job) => {
            // The stems go into the cache first. Free space in the folder next to the track is
            // checked against the stems' size once they exist.
            state.queue_claimed(QueuedWork {
                job: Arc::clone(&job),
                mix,
                output,
                model_id: cache_id,
                from_api: false,
            })?;
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
    Ok((wait_for(state, item, &job)?, Origin::Queued))
}

/// Follow `job` on `item`'s row until it ends, and return its stems.
fn wait_for(
    state: &Arc<AppState>,
    item: &Arc<Dropped>,
    job: &Arc<crate::jobs::Job>,
) -> Result<Arc<Entry>> {
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
                return entry.context("the job finished with no stems attached");
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
    let mut files = Vec::new();
    let mut scaled = Vec::new();
    for stem in &entry.stems {
        // A part that is not ticked counts as fetched, so the cache stops holding it.
        if !wanted.has(&stem.name) {
            entry.mark_fetched(&stem.name);
            continue;
        }
        let name = stem
            .path
            .file_name()
            .map_or_else(|| stem.name.clone(), |n| n.to_string_lossy().into_owned());
        let gain = match write_part(entry, stem, &dir.join(&name)) {
            Ok(gain) => gain,
            Err(err) => {
                // Remove the parts already written: a partial folder looks finished.
                let _ = clear_parts(&dir);
                return Err(err)
                    .with_context(|| format!("writing {} into {}", name, dir.display()));
            }
        };
        entry.mark_fetched(&stem.name);
        if gain != 1.0 {
            scaled.push((stem.name.clone(), gain));
        }
        parts.push(stem.name.clone());
        files.push(dir.join(&name));
    }

    if !scaled.is_empty() {
        write_gains(&dir, &scaled)?;
        files.push(dir.join(GAINS_FILE));
    }

    Ok(Finished {
        dir,
        parts,
        secs: entry.separation_secs,
        cached,
        files,
    })
}

/// Write `stem` to `path` at unity, or scaled down so it does not clip, and return the scale.
fn write_part(entry: &Entry, stem: &CachedStem, path: &Path) -> Result<f32> {
    let format = entry.output.format;
    if entry.gain == 1.0 || format == stemd_core::StemFormat::Mp3 {
        std::fs::copy(&stem.path, path)?;
        return Ok(entry.gain);
    }
    let bytes = std::fs::read(&stem.path)?;
    let mut audio = crate::whole::decode(bytes, format, entry.channels, entry.output.rate.hz())?;
    let restore = 1.0 / entry.gain;
    for v in audio.data.iter_mut().flatten() {
        *v *= restore;
    }
    let peak = audio
        .data
        .iter()
        .flatten()
        .fold(0f32, |m, v| m.max(v.abs()));
    let gain = if peak <= 1.0 { 1.0 } else { 1.0 / peak };
    std::fs::write(path, stemd_core::encode_stem(&audio, format, gain)?)?;
    Ok(gain)
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
/// A stem that peaks past full scale cannot be stored in a 16-bit container without clipping, so it
/// is written quieter. The parts then no longer sum to the mix, and this file records the scale so
/// a reader can undo it.
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

    /// The folder goes next to the file, because an `.app` is launched with its working directory
    /// set to `/`.
    #[test]
    fn a_part_stored_below_unity_is_written_at_unity() {
        let cache = crate::testkit::cache(1 << 30, std::time::Duration::from_secs(60));
        let output = crate::testkit::output();
        let dir = cache.verified_dir("model", &crate::cache::type_dir(&output, 0.5), "track");
        let task = crate::cache::Consolidation::at_44100(dir.clone(), output.format, 3);
        let stem = Audio::new(vec![vec![0.6, -0.3, 0.0], vec![0.1, 0.2, -0.6]], 44100);
        for name in ["harmonics", "vocals"] {
            let bytes = stemd_core::encode_stem(&stem, output.format, 0.5).expect("encode");
            cache
                .consolidate(&task, name, &bytes, 3, Some(0))
                .expect("store");
        }
        let entry = Entry::stored(&dir, &output, 0.5, 2, None).expect("whole");
        let path = std::env::temp_dir().join(format!("stemd-part-{}.raw", std::process::id()));
        let gain = write_part(&entry, &entry.stems[0], &path).expect("written");
        assert_eq!(gain, 1.0);
        let written =
            crate::whole::decode(std::fs::read(&path).unwrap(), output.format, 2, 44100).unwrap();
        for (got, want) in written
            .data
            .iter()
            .flatten()
            .zip(stem.data.iter().flatten())
        {
            assert!((got - want).abs() < 1e-4, "{got} for {want}");
        }
        let _ = std::fs::remove_file(path);
    }

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
        crate::scratch::dir(&format!("drops-{name}"))
    }

    /// The cross deletes a directory recursively, so this test checks which directory it deletes.
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

    /// The cross deletes nothing else. `Finished::dir` is always a folder this module wrote, and
    /// the delete checks that again.
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

    /// Two drops of files with the same name in different folders are two rows, and dismissing one
    /// does not remove the other.
    #[test]
    fn forgetting_a_drop_keeps_another_with_the_same_title() {
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
    fn dropping_a_track_again_replaces_its_row() {
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

    fn finished_in(dir: &Path, names: &[&str]) -> Finished {
        let files: Vec<PathBuf> = names.iter().map(|n| dir.join(n)).collect();
        for file in &files {
            std::fs::write(file, b"stem").unwrap();
        }
        Finished {
            dir: dir.to_path_buf(),
            parts: names.iter().map(|n| (*n).to_owned()).collect(),
            secs: 12.0,
            cached: false,
            files,
        }
    }

    #[test]
    fn a_repeat_drop_finds_its_result_while_the_files_are_unchanged() {
        let dir = scratch("history");
        let history = History::default();
        history.record(
            "a".into(),
            &finished_in(&dir, &["vocals.flac", "drums.flac"]),
        );
        assert_eq!(history.find("a").unwrap().dir, dir);
        assert!(history.find("b").is_none(), "another key");

        std::fs::write(dir.join("drums.flac"), b"edited stem").unwrap();
        assert!(history.find("a").is_none(), "a changed file");

        history.record("a".into(), &finished_in(&dir, &["vocals.flac"]));
        std::fs::remove_file(dir.join("vocals.flac")).unwrap();
        assert!(history.find("a").is_none(), "a missing file");
    }

    #[test]
    fn the_history_keeps_the_last_25() {
        let dir = scratch("history-cap");
        let history = History::default();
        let finished = finished_in(&dir, &["vocals.flac"]);
        for n in 0..=HISTORY {
            history.record(n.to_string(), &finished);
        }
        assert!(history.find("0").is_none(), "the oldest fell off");
        assert!(history.find("1").is_some());
    }

    #[test]
    fn a_track_is_known_by_its_bytes_and_settings() {
        let dir = scratch("history-key");
        let (a, b, c) = (dir.join("a.wav"), dir.join("b.wav"), dir.join("c.wav"));
        std::fs::write(&a, b"same audio").unwrap();
        std::fs::write(&b, b"same audio").unwrap();
        std::fs::write(&c, b"other audio").unwrap();
        let settings = Settings::default();
        let key = |path: &Path, settings: &Settings, model: &str| {
            history_key(path, settings, model).unwrap()
        };
        assert_eq!(key(&a, &settings, "m"), key(&b, &settings, "m"), "a copy");
        assert_ne!(key(&a, &settings, "m"), key(&c, &settings, "m"));
        assert_ne!(key(&a, &settings, "m"), key(&a, &settings, "n"), "a model");
        let parts = Settings {
            drop_parts: settings.drop_parts.toggle("vocals"),
            ..settings
        };
        assert_ne!(key(&a, &settings, "m"), key(&a, &parts, "m"), "the parts");
    }

    /// Wait up to 20 s for `item` to finish or fail.
    fn settled(item: &Dropped) -> State {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let state = item.state();
            if matches!(state, State::Done(_) | State::Failed(_)) {
                return state;
            }
            assert!(std::time::Instant::now() < deadline, "the drop never ended");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    #[test]
    fn a_dropped_track_is_separated_beside_itself_and_found_again_unchanged() {
        let state = crate::testkit::state();
        let source = scratch("pipeline").join("song.wav");
        let data = (0..2)
            .map(|c| {
                (0..44_100)
                    .map(|i| ((i * 7 + c) % 200) as f32 / 400.0 - 0.25)
                    .collect()
            })
            .collect();
        let mix = Audio::new(data, stemd_core::toy::RATE);
        let wav = stemd_core::encode_stem(&mix, stemd_core::StemFormat::Wav, 1.0).expect("a wav");
        std::fs::write(&source, wav).unwrap();

        let first = state.drops.accept(&state, source.clone());
        let State::Done(finished) = settled(&first) else {
            panic!("{:?}", first.state());
        };
        assert!(!finished.cached);
        assert_eq!(finished.dir, output_dir(&source));
        assert!(!finished.parts.is_empty());
        for file in &finished.files {
            assert!(file.is_file(), "{} was not written", file.display());
        }
        assert!(first.job.lock().is_none(), "the job was not released");

        let again = state.drops.accept(&state, source);
        let State::Done(repeat) = settled(&again) else {
            panic!("{:?}", again.state());
        };
        assert!(repeat.cached);
        assert_eq!(repeat.dir, finished.dir);
        assert_eq!(state.drops.recent().len(), 1, "the row was replaced");
    }

    #[test]
    fn a_drop_that_does_not_decode_fails() {
        let state = crate::testkit::state();
        let source = scratch("pipeline-bad").join("broken.wav");
        std::fs::write(&source, b"not audio").unwrap();
        let dropped = state.drops.accept(&state, source);
        assert!(matches!(settled(&dropped), State::Failed(_)));
        assert!(!dropped.is_running());
    }
}
