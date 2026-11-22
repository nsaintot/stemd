//! Session tests that run a toy model through a queue, the cache and the session HTTP helpers.

use std::sync::Arc;

use anyhow::Result;
use parking_lot::Mutex;
use stemd_core::{Audio, Separate};

use crate::cache::{Cache, type_dir};
use crate::output::Output;
use crate::queue::POLL_INTERVAL;
use crate::queue::{BuildSeparator, Queue, QueuedWork};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use stemd_core::Stage;

const KEY: &str = "0123456789abcdef01234567";
use crate::sessions::plan::Region;
use crate::sessions::{Request, Session, TrackSpec, WindowRequest};
use crate::testkit::cache;
use std::time::Duration;
use stemd_core::toy::Toy;
use stemd_core::{DspMode, Normalisation, OutputRate, Parts, PcmFormat, Silent, StemFormat};

/// MLX runs on one thread at a time; each queue here has its own worker.
static GPU: Mutex<()> = Mutex::new(());

const SEGMENT: i32 = 4410;

fn track(frames: usize, seed: u32) -> Audio {
    let mut state = seed;
    let data = (0..2)
        .map(|_| {
            (0..frames)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect()
        })
        .collect();
    Audio::new(data, stemd_core::toy::RATE)
}

fn output(format: StemFormat) -> Output {
    Output {
        format,
        rate: OutputRate::default(),
        derived: true,
        dsp: DspMode::default().into(),
        parts: Parts::DEFAULT,
    }
}

fn start(toy: Toy, cache: Arc<Cache>) -> Queue {
    let build: BuildSeparator = Box::new(move || Ok(Box::new(toy) as Box<dyn Separate>));
    Queue::start(build, 8, cache).expect("the toy builds").0
}

/// A cache no other test touches.
fn fresh_cache() -> Arc<Cache> {
    cache(1 << 30, Duration::from_secs(60))
}

/// A cache whose stems are never stored whole in the background.
fn held_cache() -> Arc<Cache> {
    let cache = fresh_cache();
    cache.set_gate(Box::new(|| false));
    cache
}

fn open_as(
    queue: &Queue,
    frames: usize,
    norm: Option<Normalisation>,
    format: StemFormat,
) -> Arc<Session> {
    let grid = queue
        .segmenting()
        .expect("the toy segments")
        .grid(frames)
        .expect("a grid");
    queue
        .sessions()
        .open(Request {
            output: output(format),
            in_format: PcmFormat::F32le,
            track: TrackSpec {
                frames,
                channels: 2,
                sample_rate: stemd_core::toy::RATE,
                grid,
                normalises: queue.normalises(),
                parts: Parts::DEFAULT,
            },
            norm,
            peak: None,
            model_id: "toy-00000000".into(),
            claim: None,
        })
        .expect("room for a session")
}

fn open(queue: &Queue, frames: usize, norm: Option<Normalisation>) -> Arc<Session> {
    open_as(queue, frames, norm, StemFormat::Pcm32)
}

fn whole(session: &Session) -> WindowRequest {
    WindowRequest {
        regions: vec![Region {
            at: 0,
            ahead: session.track().spec.frames,
            behind: 0,
            priority: 0,
        }],
    }
}

fn ready_within(session: &Session, from: usize, to: usize, within: Duration) -> bool {
    let deadline = std::time::Instant::now() + within;
    while std::time::Instant::now() < deadline {
        if session.read("vocals", from, to).is_ok() {
            return true;
        }
        thread::sleep(POLL_INTERVAL);
    }
    false
}

fn bits(audio: &Audio) -> Vec<u32> {
    audio.data.iter().flatten().map(|v| v.to_bits()).collect()
}

fn upload(queue: &Queue, session: &Session, offset: usize, audio: &Audio) {
    queue
        .sessions()
        .upload(session, offset, audio)
        .expect("uploading");
}

#[test]
fn a_session_uploaded_backwards_matches_a_whole_track_run() {
    let _gpu = GPU.lock();
    let frames = 31_337;
    let mix = track(frames, 1);
    let mut direct = Toy::new(SEGMENT, 0.25);
    let want = direct
        .separate(&mix, Parts::DEFAULT, &Silent)
        .expect("separating");

    let queue = start(Toy::new(SEGMENT, 0.25), fresh_cache());
    let session = open(&queue, frames, None);
    session.set_window(&whole(&session)).expect("open");
    // Last piece first, so segments wait on their input and on the whole
    // mix's statistics.
    for at in (0..frames)
        .step_by(10_000)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let to = (at + 10_000).min(frames);
        upload(&queue, &session, at, &mix.excerpt(at, to));
        queue.wake();
    }
    assert!(ready_within(&session, 0, frames, Duration::from_secs(20)));

    for name in ["harmonics", "vocals"] {
        let got = session.read(name, 0, frames).expect("ready").audio;
        let want = want.stem(name).expect("a shipped stem");
        assert_eq!(bits(&got), bits(want), "{name} differs");
    }
    let derived = session.read("drums", 0, frames).expect("ready").audio;
    assert_eq!(
        bits(&derived),
        bits(&want.derived_part(&mix)),
        "drums differs"
    );
}

#[test]
fn nothing_runs_on_a_normalising_model_until_the_mix_is_whole() {
    let _gpu = GPU.lock();
    let frames = 40_000;
    let mix = track(frames, 2);
    let toy = Toy::new(SEGMENT, 0.25);
    let runs = Arc::clone(&toy.runs);
    let queue = start(toy, fresh_cache());
    let session = open(&queue, frames, None);
    session.set_window(&whole(&session)).expect("open");
    upload(&queue, &session, 0, &mix.excerpt(0, 30_000));
    queue.wake();
    thread::sleep(Duration::from_millis(200));
    assert_eq!(runs.load(Ordering::SeqCst), 0, "ran without the statistics");

    upload(&queue, &session, 30_000, &mix.excerpt(30_000, frames));
    queue.wake();
    assert!(ready_within(&session, 0, frames, Duration::from_secs(20)));
}

#[test]
fn with_the_clients_statistics_a_segment_runs_once_its_own_input_is_in() {
    let _gpu = GPU.lock();
    let frames = 400_000;
    let mix = track(frames, 3);
    let queue = start(Toy::new(SEGMENT, 0.25), fresh_cache());
    let session = open(&queue, frames, Some(Normalisation::of(&mix)));
    let needle = 200_000;
    session
        .set_window(&WindowRequest {
            regions: vec![Region {
                at: needle,
                ahead: 10_000,
                behind: 0,
                priority: 0,
            }],
        })
        .expect("open");
    // Upload only the region around the cue.
    upload(&queue, &session, 190_000, &mix.excerpt(190_000, 220_000));
    queue.wake();
    assert!(ready_within(
        &session,
        needle,
        needle + 5_000,
        Duration::from_secs(20)
    ));
    assert!(session.read("vocals", 0, 1_000).is_err());
}

#[test]
fn high_priority_segments_run_between_the_segments_of_a_job() {
    let _gpu = GPU.lock();
    let mut toy = Toy::new(SEGMENT, 0.25);
    toy.delay = Duration::from_millis(20);
    let queue = start(toy, fresh_cache());

    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let job = jobs.create("long".into());
    queue
        .submit(QueuedWork {
            job: Arc::clone(&job),
            // About eighty segments, so well over a second of work.
            mix: track(80 * 3307, 4),
            output: output(StemFormat::Pcm32),
            model_id: "toy-00000000".into(),
            from_api: true,
        })
        .expect("room");
    thread::sleep(Duration::from_millis(100));

    let frames = 20_000;
    let mix = track(frames, 5);
    let session = open(&queue, frames, Some(Normalisation::of(&mix)));
    upload(&queue, &session, 0, &mix);
    session.set_window(&whole(&session)).expect("open");
    queue.wake();

    assert!(ready_within(&session, 0, frames, Duration::from_secs(20)));
    assert!(
        queue.running().is_some(),
        "the session waited for the whole job"
    );
    queue.cancel(&job.id);
}

#[test]
fn client_jobs_list_the_running_one_then_the_queue_and_skip_drops() {
    let _gpu = GPU.lock();
    let mut toy = Toy::new(SEGMENT, 0.25);
    toy.delay = Duration::from_millis(20);
    let queue = start(toy, fresh_cache());

    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let submit = |key: &str, from_api: bool| {
        let job = jobs.create(key.into());
        queue
            .submit(QueuedWork {
                job: Arc::clone(&job),
                mix: track(40 * 3307, 8),
                output: output(StemFormat::Pcm32),
                model_id: "toy-00000000".into(),
                from_api,
            })
            .expect("room");
        job
    };
    let first = submit("first", true);
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while queue.running().is_none() {
        assert!(
            std::time::Instant::now() < deadline,
            "the worker never took the first job"
        );
        thread::sleep(POLL_INTERVAL);
    }
    let dropped = submit("dropped", false);
    let second = submit("second", true);

    let listed: Vec<(u64, bool, Stage)> = queue
        .client_jobs()
        .into_iter()
        .map(|j| (j.number, j.running, j.progress.stage))
        .collect();
    assert_eq!(listed.len(), 2, "the drop is listed: {listed:?}");
    assert_eq!((listed[0].0, listed[0].1), (1, true));
    assert_eq!(listed[1], (2, false, Stage::Queued));
    for job in [&first, &dropped, &second] {
        queue.cancel(&job.id);
    }
}

#[test]
fn turning_the_server_off_cancels_api_jobs_and_keeps_drops() {
    let _gpu = GPU.lock();
    let mut toy = Toy::new(SEGMENT, 0.25);
    toy.delay = Duration::from_millis(20);
    let queue = start(toy, fresh_cache());

    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let submit = |key: &str, from_api: bool| {
        let job = jobs.create(key.into());
        queue
            .submit(QueuedWork {
                job: Arc::clone(&job),
                mix: track(40 * 3307, 8),
                output: output(StemFormat::Pcm32),
                model_id: "toy-00000000".into(),
                from_api,
            })
            .expect("room");
        job
    };
    let running = submit("running", true);
    thread::sleep(Duration::from_millis(100));
    let waiting = submit("waiting", true);
    let dropped = submit("dropped", false);

    assert_eq!(queue.cancel_api_jobs(), 2);
    assert_eq!(waiting.view().progress.stage, Stage::Cancelled);
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while running.view().progress.stage != Stage::Cancelled {
        assert!(
            std::time::Instant::now() < deadline,
            "the running job kept going"
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert_ne!(dropped.view().progress.stage, Stage::Cancelled);
    queue.cancel(&dropped.id);
}

#[test]
fn normal_priority_segments_wait_for_waiting_jobs() {
    let _gpu = GPU.lock();
    let mut toy = Toy::new(SEGMENT, 0.25);
    toy.delay = Duration::from_millis(10);
    let queue = start(toy, fresh_cache());

    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let job = jobs.create("long".into());
    queue
        .submit(QueuedWork {
            job: Arc::clone(&job),
            mix: track(40 * 3307, 6),
            output: output(StemFormat::Pcm32),
            model_id: "toy-00000000".into(),
            from_api: true,
        })
        .expect("room");
    thread::sleep(Duration::from_millis(50));

    let frames = 20_000;
    let mix = track(frames, 7);
    let session = open(&queue, frames, Some(Normalisation::of(&mix)));
    upload(&queue, &session, 0, &mix);
    let mut request = whole(&session);
    request.regions[0].priority = crate::sessions::plan::BEFORE_JOBS;
    session.set_window(&request).expect("open");
    queue.wake();

    assert!(ready_within(&session, 0, frames, Duration::from_secs(20)));
    assert_eq!(
        job.view().progress.stage,
        Stage::Done,
        "a normal-priority segment ran ahead of the job"
    );
}

#[test]
fn a_model_switch_closes_every_session() {
    let _gpu = GPU.lock();
    let queue = start(Toy::new(SEGMENT, 0.25), fresh_cache());
    let session = open(&queue, 20_000, None);
    queue
        .install(Box::new(|| {
            Ok(Box::new(Toy::new(SEGMENT, 0.25)) as Box<dyn Separate>)
        }))
        .expect("installing");
    assert!(session.closed().is_some());
    assert!(queue.sessions().is_empty());
}

#[test]
fn a_second_deck_on_the_same_audio_joins_the_first_once_its_upload_completes() {
    let _gpu = GPU.lock();
    let frames = 60_000;
    let mix = track(frames, 8);
    let toy = Toy::new(SEGMENT, 0.25);
    let runs = Arc::clone(&toy.runs);
    let queue = start(toy, fresh_cache());

    let a = open_as(&queue, frames, None, StemFormat::Pcm32);
    a.set_window(&whole(&a)).expect("open");
    upload(&queue, &a, 0, &mix);
    queue.wake();
    assert!(ready_within(&a, 0, frames, Duration::from_secs(20)));
    let segments = a.track().spec.grid.len();
    assert_eq!(runs.load(Ordering::SeqCst), segments);

    let b = open_as(&queue, frames, None, StemFormat::Flac);
    upload(&queue, &b, 0, &mix);
    assert!(Arc::ptr_eq(&a.track(), &b.track()), "b did not join a");
    b.set_window(&whole(&b)).expect("open");
    queue.wake();
    assert!(ready_within(&b, 0, frames, Duration::from_secs(1)));
    assert_eq!(runs.load(Ordering::SeqCst), segments, "b separated again");
    assert_eq!(
        a.track().verified().as_deref(),
        Some(crate::ident::track_id(&mix).as_str())
    );
}

#[test]
fn a_later_session_on_the_same_audio_finds_its_pieces_on_disk() {
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 9);
    let id = crate::ident::track_id(&mix);
    let toy = Toy::new(SEGMENT, 0.25);
    let runs = Arc::clone(&toy.runs);
    let cache = fresh_cache();
    let queue = start(toy, Arc::clone(&cache));

    let first = open_as(&queue, frames, None, StemFormat::Flac);
    first.set_window(&whole(&first)).expect("open");
    upload(&queue, &first, 0, &mix);
    queue.wake();
    assert!(ready_within(&first, 0, frames, Duration::from_secs(20)));
    let count = first.track().spec.grid.len();
    let type_name = type_dir(&first.output, first.gain);
    let stems = ["harmonics", "vocals", "drums"];
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let dir = cache.verified_dir("toy-00000000", &type_name, &id);
    while Cache::cached(&dir, &stems, "flac", count).len() < count {
        assert!(
            std::time::Instant::now() < deadline,
            "the pieces were never written"
        );
        thread::sleep(POLL_INTERVAL);
    }
    queue.sessions().remove(&first.id, "ended by the client");
    let ran = runs.load(Ordering::SeqCst);

    // A new session on the same audio, once the first has gone.
    let second = open_as(&queue, frames, None, StemFormat::Flac);
    second.set_window(&whole(&second)).expect("open");
    upload(&queue, &second, 0, &mix);
    queue.wake();
    assert_eq!(queue.sessions().cached_for(&second).len(), count);
    thread::sleep(Duration::from_millis(100));
    assert_eq!(
        runs.load(Ordering::SeqCst),
        ran,
        "cached pieces were separated again"
    );
}

/// Wait until every piece of `session`'s output is on disk.
fn stored_within(queue: &Queue, session: &Session, within: Duration) -> bool {
    let deadline = std::time::Instant::now() + within;
    let count = session.track().spec.grid.len();
    while std::time::Instant::now() < deadline {
        if queue.sessions().cached_for(session).len() == count {
            return true;
        }
        thread::sleep(POLL_INTERVAL);
    }
    false
}

/// The status, `x-stemd-*` headers and body of a whole-track read.
fn whole_file(queue: &Queue, session: &Session, name: &str) -> (u16, Vec<String>, Vec<u8>) {
    let frames = session.track().spec.frames;
    parts_of(crate::session_api::encode_range(
        queue.sessions(),
        session,
        name,
        0,
        frames,
    ))
}

/// The same as [`whole_file`], sent from the stored whole file.
fn sent_whole(queue: &Queue, session: &Session, name: &str) -> (u16, Vec<String>, Vec<u8>) {
    let whole = queue.sessions().whole(session, name).expect("a whole file");
    let response = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime")
        .block_on(crate::stem_http::send_whole(
            session.output.format,
            session.gain,
            whole,
        ));
    parts_of(response)
}

fn parts_of(
    response: Result<axum::response::Response, crate::api::ApiError>,
) -> (u16, Vec<String>, Vec<u8>) {
    let response = match response {
        Ok(response) => response,
        Err(err) => axum::response::IntoResponse::into_response(err),
    };
    let status = response.status().as_u16();
    let headers = response
        .headers()
        .iter()
        .filter(|(k, _)| k.as_str().starts_with("x-stemd-"))
        .map(|(k, v)| format!("{k}: {}", v.to_str().unwrap_or("")))
        .collect();
    let body = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime")
        .block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
        .expect("the body");
    (status, headers, body.to_vec())
}

fn open_at(queue: &Queue, frames: usize, format: StemFormat, rate: OutputRate) -> Arc<Session> {
    let grid = queue
        .segmenting()
        .expect("the toy segments")
        .grid(frames)
        .expect("a grid");
    let mut output = output(format);
    output.rate = rate;
    queue
        .sessions()
        .open(Request {
            output,
            in_format: PcmFormat::F32le,
            track: TrackSpec {
                frames,
                channels: 2,
                sample_rate: stemd_core::toy::RATE,
                grid,
                normalises: queue.normalises(),
                parts: Parts::DEFAULT,
            },
            norm: None,
            peak: None,
            model_id: "toy-00000000".into(),
            claim: None,
        })
        .expect("room for a session")
}

/// Replace the whole files under `dir` with the pieces copied to `backup`.
fn restore_pieces(dir: &std::path::Path, backup: &std::path::Path, format: StemFormat) {
    for stem in ["harmonics", "vocals", "drums"] {
        let _ = std::fs::remove_file(crate::whole::file(dir, stem, format.extension()));
        let _ = std::fs::remove_file(crate::whole::meta_file(dir, stem));
        let _ = std::fs::remove_dir_all(dir.join(stem));
        std::fs::create_dir_all(dir.join(stem)).expect("a stem directory");
        for entry in std::fs::read_dir(backup.join(stem)).expect("a backup") {
            let entry = entry.expect("an entry");
            std::fs::copy(entry.path(), dir.join(stem).join(entry.file_name()))
                .expect("copying a piece");
        }
    }
}

/// The whole-track file read four ways: from a fresh session, from one with every piece stored,
/// from one with half of them stored, and from one with the stem stored whole.
fn four_ways(format: StemFormat, rate: OutputRate, seed: u32) -> [(u16, Vec<String>, Vec<u8>); 4] {
    let frames = 40_000;
    let mix = track(frames, seed);
    let cache = held_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let stems = ["harmonics", "vocals", "drums"];

    let fresh = open_at(&queue, frames, format, rate);
    fresh.set_window(&whole(&fresh)).expect("open");
    upload(&queue, &fresh, 0, &mix);
    queue.wake();
    assert!(ready_within(&fresh, 0, frames, Duration::from_secs(20)));
    assert!(stored_within(&queue, &fresh, Duration::from_secs(10)));
    let dir = cache.verified_dir(
        "toy-00000000",
        &type_dir(&fresh.output, fresh.gain),
        &crate::ident::track_id(&mix),
    );
    let backup = dir.with_extension("backup");
    for stem in stems {
        std::fs::create_dir_all(backup.join(stem)).expect("a backup");
        for entry in std::fs::read_dir(dir.join(stem)).expect("pieces") {
            let entry = entry.expect("an entry");
            std::fs::copy(entry.path(), backup.join(stem).join(entry.file_name()))
                .expect("copying a piece");
        }
    }
    let (from, to) = (frames / 3, frames / 3 + 7000);
    let fresh_range = parts_of(crate::session_api::encode_range(
        queue.sessions(),
        &fresh,
        "vocals",
        from,
        to,
    ));
    let fresh_file = whole_file(&queue, &fresh, "vocals");
    assert!(
        !dir.join("vocals").exists(),
        "the pieces stayed beside the whole file"
    );
    queue.sessions().remove(&fresh.id, "ended by the client");

    let whole_session = open_at(&queue, frames, format, rate);
    upload(&queue, &whole_session, 0, &mix);
    let view = queue.view_of(&whole_session);
    assert!(view.complete, "the whole file does not count as stored");
    let sent = sent_whole(&queue, &whole_session, "vocals");
    if rate == OutputRate::default() {
        let cut = parts_of(crate::session_api::encode_range(
            queue.sessions(),
            &whole_session,
            "vocals",
            from,
            to,
        ));
        assert_eq!(
            cut, fresh_range,
            "{format}: a range cut from the whole file"
        );
    }
    assert_eq!(
        whole_file(&queue, &whole_session, "vocals"),
        sent,
        "a whole read decoded from the file differs from the file"
    );
    queue
        .sessions()
        .remove(&whole_session.id, "ended by the client");

    restore_pieces(&dir, &backup, format);
    let stored = open_at(&queue, frames, format, rate);
    upload(&queue, &stored, 0, &mix);
    let view = queue.view_of(&stored);
    assert!(view.ready.is_empty(), "stored pieces were separated");
    assert!(view.complete, "every piece is stored");
    let stored_file = whole_file(&queue, &stored, "vocals");
    queue.sessions().remove(&stored.id, "ended by the client");

    // Drop the second half of the pieces, so that half separates again.
    restore_pieces(&dir, &backup, format);
    let count = fresh.track().spec.grid.len();
    for k in count / 2..count {
        for stem in stems {
            let file = dir
                .join(stem)
                .join(format!("{k:04}.{}", format.extension()));
            std::fs::remove_file(&file).expect("a stored piece");
        }
    }
    let partly = open_at(&queue, frames, format, rate);
    upload(&queue, &partly, 0, &mix);
    assert!(
        !partly
            .track()
            .complete_with(&queue.sessions().cached_for(&partly))
    );
    let (status, _, body) = whole_file(&queue, &partly, "vocals");
    assert_eq!(status, 409, "{}", String::from_utf8_lossy(&body));
    partly.set_window(&whole(&partly)).expect("open");
    queue.wake();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while !partly
        .track()
        .complete_with(&queue.sessions().cached_for(&partly))
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the missing half never ran"
        );
        thread::sleep(POLL_INTERVAL);
    }
    let summary = queue.view_of(&partly);
    assert!(summary.complete);
    assert_ne!(
        summary.ready,
        vec![[0, frames]],
        "the stored half was separated"
    );
    let partly_file = whole_file(&queue, &partly, "vocals");
    [fresh_file, stored_file, partly_file, sent]
}

#[test]
fn a_whole_track_read_is_the_same_file_from_memory_and_from_stored_pieces() {
    let _gpu = GPU.lock();
    for format in [
        StemFormat::Flac,
        StemFormat::Wav,
        StemFormat::Pcm16,
        StemFormat::Pcm32,
    ] {
        let [fresh, stored, partly, whole] = four_ways(format, OutputRate::default(), 11);
        assert_eq!(fresh.0, 200, "{format}");
        assert_eq!(stored, fresh, "{format}: all stored");
        assert_eq!(partly, fresh, "{format}: partly stored");
        assert_eq!(whole, fresh, "{format}: stored whole");
    }
}

#[test]
fn at_another_rate_stored_pieces_differ_by_at_most_one_lsb() {
    let _gpu = GPU.lock();
    let samples = |body: &[u8]| -> Vec<i16> {
        body.chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect()
    };
    for rate in [OutputRate::Hz48000, OutputRate::Hz96000] {
        let [fresh, stored, partly, whole] = four_ways(StemFormat::Pcm16, rate, 13);
        assert_eq!(whole, fresh, "{rate}: stored whole");
        let want = samples(&fresh.2);
        for (label, got) in [("all stored", &stored), ("partly stored", &partly)] {
            assert_eq!(got.1, fresh.1, "{rate}, {label}: headers");
            let got = samples(&got.2);
            assert_eq!(got.len(), want.len(), "{rate}, {label}: length");
            let diffs: Vec<i32> = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
                .collect();
            // The resampler's rounding where a piece was converted on its own.
            let max = diffs.iter().max().copied().unwrap_or(0);
            let count = diffs.iter().filter(|d| **d > 0).count();
            assert!(max <= 1, "{rate}, {label}: off by {max} LSB");
            assert!(
                count * 1000 < diffs.len(),
                "{rate}, {label}: {count} samples differ"
            );
        }
    }
}

#[test]
fn a_piece_deleted_during_a_session_is_separated_again_on_the_next_window() {
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 14);
    let cache = held_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));

    let first = open_as(&queue, frames, None, StemFormat::Flac);
    first.set_window(&whole(&first)).expect("open");
    upload(&queue, &first, 0, &mix);
    queue.wake();
    assert!(stored_within(&queue, &first, Duration::from_secs(20)));
    let want = stemd_core::encode_stem(
        &first.read("vocals", 0, frames).expect("ready").audio,
        StemFormat::Flac,
        first.gain,
    )
    .expect("encoding");
    queue.sessions().remove(&first.id, "ended by the client");

    let second = open_as(&queue, frames, None, StemFormat::Flac);
    upload(&queue, &second, 0, &mix);
    let dir = cache.verified_dir(
        "toy-00000000",
        &type_dir(&second.output, second.gain),
        &crate::ident::track_id(&mix),
    );
    for stem in ["harmonics", "vocals", "drums"] {
        std::fs::remove_file(dir.join(stem).join("0003.flac")).expect("a stored piece");
    }
    let view = queue.view_of(&second);
    assert!(!view.complete);
    assert_eq!(whole_file(&queue, &second, "vocals").0, 409);

    queue
        .sessions()
        .set_window(&second, &whole(&second))
        .expect("open");
    queue.wake();
    assert!(stored_within(&queue, &second, Duration::from_secs(20)));
    assert_eq!(whole_file(&queue, &second, "vocals").2, want);
}

#[test]
fn a_track_with_every_piece_stored_is_stored_whole_once_the_gate_opens() {
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 15);
    let cache = fresh_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let open = Arc::new(AtomicBool::new(false));
    let gate = Arc::clone(&open);
    cache.set_gate(Box::new(move || gate.load(Ordering::SeqCst)));
    let stems = ["harmonics", "vocals", "drums"];

    let first = open_as(&queue, frames, None, StemFormat::Flac);
    first.set_window(&whole(&first)).expect("open");
    upload(&queue, &first, 0, &mix);
    queue.wake();
    assert!(stored_within(&queue, &first, Duration::from_secs(20)));
    let range = |session: &Session| {
        parts_of(crate::session_api::encode_range(
            queue.sessions(),
            session,
            "vocals",
            frames / 3,
            frames / 3 + 5000,
        ))
    };
    let from_memory = range(&first);
    let dir = cache.verified_dir(
        "toy-00000000",
        &type_dir(&first.output, first.gain),
        &crate::ident::track_id(&mix),
    );
    thread::sleep(Duration::from_millis(300));
    assert!(
        stems.iter().all(|s| dir.join(s).is_dir()),
        "stored whole while the gate was shut"
    );
    assert!(cache.is_consolidating(&dir));

    open.store(true, Ordering::SeqCst);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while stems
        .iter()
        .any(|s| crate::whole::find(&dir, s, "flac").is_none())
    {
        assert!(std::time::Instant::now() < deadline, "never stored whole");
        thread::sleep(POLL_INTERVAL);
    }
    while cache.is_consolidating(&dir) {
        thread::sleep(POLL_INTERVAL);
    }
    for stem in stems {
        assert!(!dir.join(stem).exists(), "{stem} kept its pieces");
    }
    queue.sessions().remove(&first.id, "ended by the client");

    let second = open_as(&queue, frames, None, StemFormat::Flac);
    upload(&queue, &second, 0, &mix);
    let view = queue.view_of(&second);
    assert!(view.complete);
    assert!(view.ready.is_empty());
    assert_eq!(
        range(&second),
        from_memory,
        "a range cut from the whole file"
    );
    assert_eq!(
        whole_file(&queue, &second, "vocals"),
        sent_whole(&queue, &second, "vocals")
    );
}

/// Wait until every stem of `output` for `mix` is stored whole.
fn whole_within(cache: &Cache, mix: &Audio, output: &Output, gain: f32) -> std::path::PathBuf {
    let dir = cache.verified_dir(
        "toy-00000000",
        &type_dir(output, gain),
        &crate::ident::track_id(mix),
    );
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while output
        .stored_parts()
        .iter()
        .any(|s| crate::whole::find(&dir, s, output.format.extension()).is_none())
        || cache.is_consolidating(&dir)
    {
        assert!(std::time::Instant::now() < deadline, "never stored whole");
        thread::sleep(POLL_INTERVAL);
    }
    dir
}

#[test]
fn a_lookup_follows_a_track_from_open_to_stored() {
    use crate::held::{Held, lookup};
    use crate::output::Storage;
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 16);
    let id = crate::ident::track_id(&mix);
    let cache = fresh_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let open = Arc::new(AtomicBool::new(false));
    let gate = Arc::clone(&open);
    cache.set_gate(Box::new(move || gate.load(Ordering::SeqCst)));
    let flac = output(StemFormat::Flac);
    let held = || lookup(&cache, queue.sessions(), &jobs, "toy-00000000", &flac, &id);
    assert!(held().is_none());

    let session = open_as(&queue, frames, None, StemFormat::Flac);
    upload(&queue, &session, 0, &mix);
    let now = held().expect("open");
    assert_eq!(now.state, Held::Open);
    assert_eq!(now.frames, frames);
    assert_eq!(now.output.gain, None);

    session.set_window(&whole(&session)).expect("open");
    queue.wake();
    assert!(stored_within(&queue, &session, Duration::from_secs(20)));
    assert_eq!(held().expect("stored").state, Held::Consolidating);

    open.store(true, Ordering::SeqCst);
    let dir = whole_within(&cache, &mix, &flac, session.gain);
    let now = held().expect("stored");
    assert_eq!(now.state, Held::Stored);
    assert_eq!(now.output.gain, Some(session.gain));
    assert_eq!(
        now.stems
            .iter()
            .map(|s| (s.name, s.sources))
            .collect::<Vec<_>>(),
        [("harmonics", 6), ("vocals", 8), ("drums", 1)]
    );
    for stem in &now.stems {
        assert_eq!(stem.stored, Storage::Whole);
        let len = std::fs::metadata(dir.join(format!("{}.flac", stem.name)))
            .expect("a whole file")
            .len();
        assert_eq!(stem.bytes, Some(len));
    }

    queue.sessions().remove(&session.id, "ended by the client");
    std::fs::remove_file(dir.join("vocals.flac")).expect("a whole file");
    let now = held().expect("partial");
    assert_eq!(now.state, Held::Partial);
    assert_eq!(now.stems[1].stored, Storage::None);
}

#[test]
fn a_track_uploaded_but_never_separated_is_unknown_once_its_session_ends() {
    use crate::held::lookup;
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 23);
    let id = crate::ident::track_id(&mix);
    let cache = held_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let flac = output(StemFormat::Flac);
    let session = open_as(&queue, frames, None, StemFormat::Flac);
    upload(&queue, &session, 0, &mix);
    queue.sessions().remove(&session.id, "ended by the client");
    assert!(lookup(&cache, queue.sessions(), &jobs, "toy-00000000", &flac, &id).is_none());
}

#[test]
fn a_track_that_fails_to_be_stored_whole_reads_as_partial_until_the_retry() {
    use crate::held::{Held, lookup};
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 24);
    let id = crate::ident::track_id(&mix);
    let cache = fresh_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let open = Arc::new(AtomicBool::new(false));
    let gate = Arc::clone(&open);
    cache.set_gate(Box::new(move || gate.load(Ordering::SeqCst)));
    let flac = output(StemFormat::Flac);
    let session = open_as(&queue, frames, None, StemFormat::Flac);
    session.set_window(&whole(&session)).expect("open");
    upload(&queue, &session, 0, &mix);
    queue.wake();
    assert!(stored_within(&queue, &session, Duration::from_secs(20)));
    queue.sessions().remove(&session.id, "ended by the client");

    let dir = cache.verified_dir("toy-00000000", &type_dir(&flac, session.gain), &id);
    let piece = std::fs::read_dir(dir.join("harmonics"))
        .expect("pieces")
        .flatten()
        .next()
        .expect("a piece")
        .path();
    std::fs::write(&piece, b"not flac").expect("writing");
    open.store(true, Ordering::SeqCst);
    while cache.is_consolidating(&dir) {
        thread::sleep(POLL_INTERVAL);
    }
    assert!(cache.consolidation_failed(&dir));
    let held = lookup(&cache, queue.sessions(), &jobs, "toy-00000000", &flac, &id);
    assert_eq!(held.expect("partial").state, Held::Partial);
    assert!(!cache.is_consolidating(&dir), "queued again");
}

#[test]
fn a_stored_output_converts_to_another_container_of_the_same_samples() {
    use crate::held::{Held, derive, lookup};
    let _gpu = GPU.lock();
    let frames = 30_000;
    let cache = fresh_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let find = |out: &Output, mix: &Audio| {
        lookup(
            &cache,
            queue.sessions(),
            &jobs,
            "toy-00000000",
            out,
            &crate::ident::track_id(mix),
        )
    };
    let derive_from = |out: &Output, mix: &Audio, source: &Output, gain: f32| {
        let id = crate::ident::track_id(mix);
        let dir = cache.verified_dir("toy-00000000", &type_dir(source, gain), &id);
        let source = crate::held::TypeDir {
            info: Cache::info(&dir).expect("a track.json"),
            dir,
            output: source.clone(),
            gain,
        };
        let to_gain = find(out, mix)
            .expect("derivable")
            .output
            .gain
            .expect("a gain");
        derive(&cache, "toy-00000000", &id, &source, out, to_gain).expect("derived")
    };

    // flac to wav: the same 16-bit samples in another container.
    let mix = track(frames, 17);
    let session = open_as(&queue, frames, None, StemFormat::Flac);
    session.set_window(&whole(&session)).expect("open");
    upload(&queue, &session, 0, &mix);
    queue.wake();
    let flac = output(StemFormat::Flac);
    let flac_dir = whole_within(&cache, &mix, &flac, session.gain);
    let wav = output(StemFormat::Wav);
    let now = find(&wav, &mix).expect("derivable");
    assert_eq!(now.state, Held::Derivable);
    assert_eq!(now.from.expect("a source").format, "flac");
    let wav_dir = derive_from(&wav, &mix, &flac, session.gain).dir;
    assert_eq!(find(&wav, &mix).expect("stored").state, Held::Stored);
    for stem in ["harmonics", "vocals", "drums"] {
        let decoded = |dir: &std::path::Path, format: StemFormat| {
            let path = dir.join(format!("{stem}.{}", format.extension()));
            let bytes = std::fs::read(path).expect("a whole file");
            bits(&crate::whole::decode(bytes, format, 2, 44_100).expect("decodes"))
        };
        assert_eq!(
            decoded(&wav_dir, StemFormat::Wav),
            decoded(&flac_dir, StemFormat::Flac),
            "{stem}"
        );
    }
    queue.sessions().remove(&session.id, "ended by the client");

    // f32le to flac: a 16-bit read at the gain a session without `peak` uses.
    let mix = track(frames, 18);
    let session = open_as(&queue, frames, None, StemFormat::Pcm32);
    session.set_window(&whole(&session)).expect("open");
    upload(&queue, &session, 0, &mix);
    queue.wake();
    let pcm32 = output(StemFormat::Pcm32);
    whole_within(&cache, &mix, &pcm32, 1.0);
    let to_gain = flac.gain(None);
    let want = stemd_core::encode_stem(
        &session.read("vocals", 0, frames).expect("ready").audio,
        StemFormat::Flac,
        to_gain,
    )
    .expect("encoding");
    let flac_dir = derive_from(&flac, &mix, &pcm32, 1.0).dir;
    assert_eq!(
        std::fs::read(flac_dir.join("vocals.flac")).expect("a whole file"),
        want
    );
    assert!(
        find(&output(StemFormat::Pcm32), &track(frames, 19)).is_none(),
        "nothing converts to f32le"
    );
}

#[test]
fn a_view_shows_how_each_stem_is_stored_and_changes_its_rev() {
    use crate::output::Storage;
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 20);
    let queue = start(Toy::new(SEGMENT, 0.25), held_cache());
    let session = open_as(&queue, frames, None, StemFormat::Flac);
    let mut changes = queue.sessions().subscribe();
    changes.borrow_and_update();
    let before = queue.view_of(&session);
    assert_eq!(before.rev, queue.view_of(&session).rev);
    assert_eq!(before.track.id, None);
    assert!(before.stems.iter().all(|s| s.stored == Storage::None));

    session.set_window(&whole(&session)).expect("open");
    upload(&queue, &session, 0, &mix);
    queue.wake();
    assert!(stored_within(&queue, &session, Duration::from_secs(20)));
    assert!(changes.has_changed().expect("a sender"));
    let count = session.track().spec.grid.len();
    let view = queue.view_of(&session);
    assert_ne!(view.rev, before.rev);
    assert_eq!(view.track.id, Some(crate::ident::track_id(&mix)));
    for stem in &view.stems {
        assert_eq!(stem.stored, Storage::Pieces, "{}", stem.name);
        assert_eq!(stem.pieces, vec![[0, count]]);
    }

    assert_eq!(whole_file(&queue, &session, "vocals").0, 200);
    let view = queue.view_of(&session);
    let vocals = view
        .stems
        .iter()
        .find(|s| s.name == "vocals")
        .expect("vocals");
    assert_eq!(vocals.stored, Storage::Whole);
    assert!(vocals.bytes.is_some_and(|b| b > 0));
    assert!(vocals.pieces.is_empty());
    let harmonics = &view.stems[0];
    assert_eq!(harmonics.stored, Storage::Pieces);
}

#[test]
fn a_session_joins_an_open_track_without_an_upload() {
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 21);
    let toy = Toy::new(SEGMENT, 0.25);
    let runs = Arc::clone(&toy.runs);
    let queue = start(toy, held_cache());
    let first = open_as(&queue, frames, None, StemFormat::Flac);
    upload(&queue, &first, 0, &mix);
    let id = crate::ident::track_id(&mix);
    let track = queue
        .sessions()
        .open_track(&id, Parts::DEFAULT)
        .expect("an open track");

    let second = queue
        .sessions()
        .join(output(StemFormat::Wav), PcmFormat::F32le, None, track)
        .expect("room");
    assert_eq!(second.track().uid, first.track().uid);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    second.set_window(&whole(&second)).expect("open");
    queue.wake();
    assert!(ready_within(&first, 0, frames, Duration::from_secs(20)));
    assert_eq!(
        runs.load(Ordering::SeqCst),
        second.track().spec.grid.len(),
        "each segment ran once"
    );
    assert!(
        queue
            .sessions()
            .open_track("000000000000000000000000", Parts::DEFAULT)
            .is_none()
    );
}

#[test]
fn a_join_after_the_last_session_left_finds_the_track_gone() {
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 22);
    let queue = start(Toy::new(SEGMENT, 0.25), held_cache());
    let first = open(&queue, frames, None);
    upload(&queue, &first, 0, &mix);
    let id = crate::ident::track_id(&mix);
    let track = queue
        .sessions()
        .open_track(&id, Parts::DEFAULT)
        .expect("an open track");

    assert!(queue.sessions().remove(&first.id, "ended by the client"));
    let joined = queue
        .sessions()
        .join(output(StemFormat::Pcm32), PcmFormat::F32le, None, track);
    assert!(
        matches!(joined, Err(crate::sessions::JoinError::Gone)),
        "the join found the track open"
    );
    assert!(queue.sessions().is_empty());
}

#[test]
fn session_segments_set_the_segment_time() {
    let _gpu = GPU.lock();
    let frames = 30_000;
    let queue = start(Toy::new(SEGMENT, 0.25), fresh_cache());
    let session = open(&queue, frames, None);
    session.set_window(&whole(&session)).expect("open");
    upload(&queue, &session, 0, &track(frames, 22));
    queue.wake();
    assert!(ready_within(&session, 0, frames, Duration::from_secs(20)));
    assert!(queue.view_of(&session).grid.segment_run_seconds.is_some());
}

#[test]
fn a_whole_track_read_before_any_separation_is_refused() {
    let _gpu = GPU.lock();
    let queue = start(Toy::new(SEGMENT, 0.25), fresh_cache());
    let frames = 20_000;
    let session = open_at(&queue, frames, StemFormat::Flac, OutputRate::default());
    upload(&queue, &session, 0, &track(frames, 12));
    let (status, _, body) = whole_file(&queue, &session, "vocals");
    assert_eq!(status, 409);
    assert!(
        String::from_utf8_lossy(&body).contains("neither ready nor stored"),
        "{}",
        String::from_utf8_lossy(&body)
    );
}

#[test]
fn a_job_leaves_its_pieces_for_sessions() {
    let _gpu = GPU.lock();
    let cache = fresh_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let frames = 25_000;
    let mix = track(frames, 10);
    let id = crate::ident::track_id(&mix);
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let job = jobs.create("for-pieces".into());
    queue
        .submit(QueuedWork {
            job: Arc::clone(&job),
            mix: mix.clone(),
            output: output(StemFormat::Flac),
            model_id: "toy-00000000".into(),
            from_api: true,
        })
        .expect("room");
    let name = type_dir(
        &output(StemFormat::Flac),
        output(StemFormat::Flac).gain(mix.peak().ok()),
    );
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let dir = cache.verified_dir("toy-00000000", &name, &id);
    let stems = ["harmonics", "vocals", "drums"];
    // A whole file is published before its pieces are removed.
    let stored =
        |stem: &&str| crate::whole::find(&dir, stem, "flac").is_some() && !dir.join(stem).exists();
    while !stems.iter().all(stored) {
        assert!(
            std::time::Instant::now() < deadline,
            "the job stored no whole stems"
        );
        thread::sleep(POLL_INTERVAL);
    }
    for stem in ["harmonics", "vocals", "drums"] {
        assert!(
            crate::whole::find(&dir, stem, "flac").is_some(),
            "{stem} is not stored whole"
        );
        assert!(!dir.join(stem).exists(), "{stem} kept its pieces");
    }
}

#[test]
fn a_jobs_stems_answer_a_later_job_after_a_restart() {
    let _gpu = GPU.lock();
    let cache = fresh_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let mix = track(25_000, 11);
    let id = crate::ident::track_id(&mix);
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let job = jobs.create("first".into());
    queue
        .submit(QueuedWork {
            job: Arc::clone(&job),
            mix,
            output: output(StemFormat::Flac),
            model_id: "toy-00000000".into(),
            from_api: true,
        })
        .expect("room");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while !job.progress.lock().stage.is_terminal() {
        assert!(std::time::Instant::now() < deadline, "the job never ended");
        thread::sleep(POLL_INTERVAL);
    }
    let first = job.entry.lock().clone().expect("stems");
    assert!(first.model_residual_db.is_some());

    let reopened = Cache::open(cache.root().to_owned(), 1 << 30).expect("reopen");
    let again =
        crate::held::cached_entry(&reopened, "toy-00000000", &output(StemFormat::Flac), &id)
            .expect("found after a restart");
    assert_eq!(again.dir, first.dir);
    assert_eq!(again.separation_secs, first.separation_secs);
    assert_eq!(again.model_residual_db, first.model_residual_db);
    let names: Vec<&str> = again.stems.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["harmonics", "vocals", "drums"]);
    assert!(again.present());
}

#[test]
fn a_session_records_its_file_key_once_the_upload_is_hashed() {
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 25);
    let cache = held_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let session = open(&queue, frames, None);
    session.set_file(Some(KEY.into()));
    upload(&queue, &session, 0, &mix.excerpt(0, frames / 2));
    assert_eq!(cache.file_track(KEY), None, "recorded before the hash");
    upload(
        &queue,
        &session,
        frames / 2,
        &mix.excerpt(frames / 2, frames),
    );
    assert_eq!(cache.file_track(KEY), Some(crate::ident::track_id(&mix)));
}

#[test]
fn a_stored_read_comes_from_pieces_and_refuses_a_short_or_missing_one() {
    let _gpu = GPU.lock();
    let frames = 30_000;
    let mix = track(frames, 21);
    let id = crate::ident::track_id(&mix);
    let cache = held_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));

    let first = open_as(&queue, frames, None, StemFormat::Flac);
    first.set_window(&whole(&first)).expect("open");
    upload(&queue, &first, 0, &mix);
    queue.wake();
    assert!(ready_within(&first, 0, frames, Duration::from_secs(20)));
    assert!(stored_within(&queue, &first, Duration::from_secs(10)));
    let dir = cache.verified_dir("toy-00000000", &type_dir(&first.output, first.gain), &id);
    queue.sessions().remove(&first.id, "ended by the client");

    // Uploaded with no window, so nothing is separated in memory.
    let second = open_as(&queue, frames, None, StemFormat::Flac);
    upload(&queue, &second, 0, &mix);
    let read = queue
        .sessions()
        .read_stored(&second, "vocals", 0, frames)
        .expect("read from the pieces");
    assert_eq!(read.audio.frames(), frames);

    let piece = Cache::lookup(&dir, "vocals", 0, "flac").expect("piece 0");
    let bytes = std::fs::read(&piece).unwrap();
    let other = stemd_core::encode_stem(&mix.excerpt(0, 100), StemFormat::Flac, 1.0).unwrap();
    std::fs::write(&piece, other).unwrap();
    let short = queue
        .sessions()
        .read_stored(&second, "vocals", 0, 10)
        .err()
        .expect("a short piece is refused");
    assert!(short.to_string().contains("samples, got"), "{short:#}");

    std::fs::remove_file(&piece).unwrap();
    let missing = queue
        .sessions()
        .read_stored(&second, "vocals", 0, 10)
        .err()
        .expect("a missing piece is refused");
    assert!(
        missing.to_string().contains("neither ready nor stored"),
        "{missing:#}"
    );
    std::fs::write(&piece, bytes).unwrap();
}
