//! Session tests of what the cache stores: pieces, whole files, lookups and conversions.

use super::*;

/// The same as [`whole_file`], sent from the stored whole file.
fn sent_whole(queue: &Queue, session: &Session, name: &str) -> (u16, Vec<String>, Vec<u8>) {
    let whole = queue.sessions().whole(session, name).expect("a whole file");
    let response = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime")
        .block_on(crate::api::stem::send_whole(
            session.output.format,
            session.gain,
            whole,
        ));
    parts_of(response)
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

/// A whole-track response: the status, the headers and the body.
type WholeFile = (u16, Vec<String>, Vec<u8>);

/// The stems the toy model writes.
const STEMS: [&str; 3] = ["harmonics", "vocals", "drums"];

/// The whole-track file read four ways: from a fresh session, from one with every piece stored,
/// from one with half of them stored, and from one with the stem stored whole.
fn four_ways(format: StemFormat, rate: OutputRate, seed: u32) -> [WholeFile; 4] {
    let frames = 40_000;
    let mix = track(frames, seed);
    let cache = held_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let reads = Reads {
        queue: &queue,
        mix: &mix,
        frames,
        format,
        rate,
    };

    let fresh = reads.open();
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
    let backup = back_up(&dir);
    let range = (frames / 3, frames / 3 + 7000);
    let fresh_range = parts_of(crate::api::sessions::encode_range(
        queue.sessions(),
        &fresh,
        "vocals",
        range.0,
        range.1,
    ));
    let fresh_file = whole_file(&queue, &fresh, "vocals");
    assert!(
        !dir.join("vocals").exists(),
        "the pieces stayed beside the whole file"
    );
    queue.sessions().remove(&fresh.id, "ended by the client");

    let sent = reads.read_stored_whole(range, &fresh_range);
    restore_pieces(&dir, &backup, format);
    let stored_file = reads.read_stored_pieces();
    // Drop the second half of the pieces, so that half separates again.
    restore_pieces(&dir, &backup, format);
    let partly_file = reads.read_half_the_pieces(&dir, fresh.track().spec.grid.len());
    [fresh_file, stored_file, partly_file, sent]
}

/// Copy the pieces of each stem in `dir` to a backup beside it. Returns the backup.
fn back_up(dir: &std::path::Path) -> std::path::PathBuf {
    let backup = dir.with_extension("backup");
    for stem in STEMS {
        std::fs::create_dir_all(backup.join(stem)).expect("a backup");
        for entry in std::fs::read_dir(dir.join(stem)).expect("pieces") {
            let entry = entry.expect("an entry");
            std::fs::copy(entry.path(), backup.join(stem).join(entry.file_name()))
                .expect("copying a piece");
        }
    }
    backup
}

/// The track [`four_ways`] reads, and the queue it reads it from.
struct Reads<'a> {
    queue: &'a Queue,
    mix: &'a Audio,
    frames: usize,
    format: StemFormat,
    rate: OutputRate,
}

impl Reads<'_> {
    fn open(&self) -> Arc<Session> {
        open_at(self.queue, self.frames, self.format, self.rate)
    }

    /// The whole file sent from the stem stored whole, checked against `fresh_range`.
    fn read_stored_whole(&self, (from, to): (usize, usize), fresh_range: &WholeFile) -> WholeFile {
        let queue = self.queue;
        let whole_session = self.open();
        upload(queue, &whole_session, 0, self.mix);
        let view = queue.view_of(&whole_session);
        assert!(view.complete, "the whole file does not count as stored");
        let sent = sent_whole(queue, &whole_session, "vocals");
        if self.rate == OutputRate::default() {
            let cut = parts_of(crate::api::sessions::encode_range(
                queue.sessions(),
                &whole_session,
                "vocals",
                from,
                to,
            ));
            assert_eq!(
                &cut, fresh_range,
                "{}: a range cut from the whole file",
                self.format
            );
        }
        assert_eq!(
            whole_file(queue, &whole_session, "vocals"),
            sent,
            "a whole read decoded from the file differs from the file"
        );
        queue
            .sessions()
            .remove(&whole_session.id, "ended by the client");
        sent
    }

    /// The whole file read from a session with every piece stored.
    fn read_stored_pieces(&self) -> WholeFile {
        let queue = self.queue;
        let stored = self.open();
        upload(queue, &stored, 0, self.mix);
        let view = queue.view_of(&stored);
        assert!(view.ready.is_empty(), "stored pieces were separated");
        assert!(view.complete, "every piece is stored");
        let stored_file = whole_file(queue, &stored, "vocals");
        queue.sessions().remove(&stored.id, "ended by the client");
        stored_file
    }

    /// The whole file read once the second half of the `count` pieces in `dir` separates again.
    fn read_half_the_pieces(&self, dir: &std::path::Path, count: usize) -> WholeFile {
        let (queue, format, frames) = (self.queue, self.format, self.frames);
        for k in count / 2..count {
            for stem in STEMS {
                let file = dir
                    .join(stem)
                    .join(format!("{k:04}.{}", format.extension()));
                std::fs::remove_file(&file).expect("a stored piece");
            }
        }
        let partly = self.open();
        upload(queue, &partly, 0, self.mix);
        assert!(
            !partly
                .track()
                .complete_with(&queue.sessions().cached_for(&partly))
        );
        let (status, _, body) = whole_file(queue, &partly, "vocals");
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
        whole_file(queue, &partly, "vocals")
    }
}

#[test]
fn a_whole_track_read_is_the_same_file_from_memory_and_read_stored_pieces() {
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
        parts_of(crate::api::sessions::encode_range(
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

/// Looks up and converts stored outputs of the toy model.
struct Converter<'a> {
    cache: &'a Arc<Cache>,
    queue: &'a Queue,
    jobs: &'a crate::jobs::Jobs,
}

impl Converter<'_> {
    fn find(&self, out: &Output, mix: &Audio) -> Option<crate::held::Lookup> {
        crate::held::lookup(
            self.cache,
            self.queue.sessions(),
            self.jobs,
            "toy-00000000",
            out,
            &crate::ident::track_id(mix),
        )
    }

    /// Convert the stored `source` output of `mix`, at `gain`, to `out`.
    fn derive_from(
        &self,
        out: &Output,
        mix: &Audio,
        source: &Output,
        gain: f32,
    ) -> crate::held::TypeDir {
        let id = crate::ident::track_id(mix);
        let dir = self
            .cache
            .verified_dir("toy-00000000", &type_dir(source, gain), &id);
        let source = crate::held::TypeDir {
            info: Cache::info(&dir).expect("a track.json"),
            dir,
            output: source.clone(),
            gain,
        };
        let to_gain = self
            .find(out, mix)
            .expect("derivable")
            .output
            .gain
            .expect("a gain");
        crate::held::derive(self.cache, "toy-00000000", &id, &source, out, to_gain)
            .expect("derived")
    }
}

#[test]
fn a_stored_output_converts_to_another_container_of_the_same_samples() {
    use crate::held::Held;
    let _gpu = GPU.lock();
    let frames = 30_000;
    let cache = fresh_cache();
    let queue = start(Toy::new(SEGMENT, 0.25), Arc::clone(&cache));
    let jobs = crate::jobs::Jobs::new(Duration::from_secs(60));
    let converter = Converter {
        cache: &cache,
        queue: &queue,
        jobs: &jobs,
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
    let now = converter.find(&wav, &mix).expect("derivable");
    assert_eq!(now.state, Held::Derivable);
    assert_eq!(now.from.expect("a source").format, "flac");
    let wav_dir = converter.derive_from(&wav, &mix, &flac, session.gain).dir;
    assert_eq!(
        converter.find(&wav, &mix).expect("stored").state,
        Held::Stored
    );
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
    let flac_dir = converter.derive_from(&flac, &mix, &pcm32, 1.0).dir;
    assert_eq!(
        std::fs::read(flac_dir.join("vocals.flac")).expect("a whole file"),
        want
    );
    assert!(
        converter
            .find(&output(StemFormat::Pcm32), &track(frames, 19))
            .is_none(),
        "nothing converts to f32le"
    );
}
