use super::evict::track_dirs;
use super::*;
use crate::output::Output;
use stemd_core::{OutputRate, Parts, StemFormat};

#[test]
fn a_second_lock_on_the_same_root_fails() {
    let root = crate::scratch::Dir::new("lock-test");
    let held = lock(&root).unwrap();
    let err = lock(&root).unwrap_err();
    assert!(err.to_string().contains("already running"), "{err:#}");
    assert!(root.join(LOCK_FILE).exists());
    drop(held);
    lock(&root).unwrap();
}

#[test]
fn a_file_key_seen_with_two_tracks_names_neither() {
    let root = crate::scratch::Dir::new("files-test");
    let cache = Cache::open(root.to_path_buf(), 1 << 20).expect("open");
    let (key, a, b) = ("aa".repeat(12), "11".repeat(12), "22".repeat(12));
    assert_eq!(cache.file_track(&key), None);
    cache.record_file(&key, &a).expect("recorded");
    cache.record_file(&key, &a).expect("recorded again");
    assert_eq!(cache.file_track(&key), Some(a.clone()));
    cache.record_file(&key, &b).expect("recorded");
    assert_eq!(cache.file_track(&key), None, "ambiguous");
    cache.record_file(&key, &a).expect("recorded");
    assert_eq!(cache.file_track(&key), None, "stays ambiguous");
    let reopened = Cache::open(root.to_path_buf(), 1 << 20).expect("reopen");
    assert_eq!(reopened.file_track(&key), None);
    assert!(
        track_dirs(&root).is_empty(),
        "aliases are not track directories"
    );
}

#[test]
fn the_type_directory_names_every_setting_that_changes_the_bytes() {
    let mut output = Output {
        format: StemFormat::Flac,
        rate: OutputRate::default(),
        derived: false,
        dsp: stemd_core::DspMode::General.into(),
        parts: Parts::DEFAULT,
    };
    assert_eq!(
        type_dir(&output, std::f32::consts::FRAC_1_SQRT_2),
        "flac-44100-harmonics.vocals-g0.70710677"
    );
    output.dsp = stemd_core::DspMode::Mode1.into();
    output.rate = OutputRate::Hz96000;
    assert_eq!(
        type_dir(&output, 1.0),
        "flac-96000-dsp1-harmonics.vocals-g1"
    );
    output.derived = true;
    assert_eq!(
        type_dir(&output, 1.0),
        "flac-96000-dsp1-harmonics.vocals-g1",
        "the derived part is one more file, not another type"
    );

    let body = crate::filters::encode(44_100, 96_000, &[1.0; 320]);
    let id = crate::filters::id_of(&body);
    output.dsp = crate::filters::Dsp::Fir(std::sync::Arc::new(crate::filters::Fir {
        id: id.clone(),
        poly: crate::filters::parse(&body).unwrap(),
    }));
    assert_eq!(
        type_dir(&output, 1.0),
        format!("flac-96000-fir{id}-harmonics.vocals-g1")
    );
}

fn root() -> crate::scratch::Dir {
    crate::scratch::Dir::new("pieces-unit")
}

/// A cache of `max_bytes` in a fresh directory, deleted with the cache.
fn open(max_bytes: u64) -> Arc<Cache> {
    open_with(Rules::sized(max_bytes))
}

fn open_with(rules: Rules) -> Arc<Cache> {
    let root = root();
    let cache = Cache::open_with(root.to_path_buf(), rules).expect("open");
    cache.keep(root);
    cache
}

#[test]
fn rewriting_a_track_file_counts_only_the_change_in_size() {
    let cache = open(1 << 20);
    let dir = cache.verified_dir("model", "type", "track");
    let on_disk = || -> u64 { files(cache.root()).iter().map(|(_, size, _)| size).sum() };
    let info = TrackInfo {
        frames: 10,
        sample_rate: 44_100,
        channels: 2,
        stride: 10,
        pieces: 1,
    };
    cache.write_info(&dir, &info).expect("info");
    let longer = TrackInfo {
        frames: 1_000_000,
        stride: 1_000_000,
        ..info
    };
    cache.write_info(&dir, &longer).expect("info again");
    let separation = Separation {
        model_residual_db: -1.0,
        separation_secs: 2.0,
    };
    cache
        .write_separation(&dir, &separation)
        .expect("separation");
    let other = Separation {
        model_residual_db: -12.345_678,
        ..separation
    };
    cache
        .write_separation(&dir, &other)
        .expect("separation again");
    assert_eq!(cache.bytes(), on_disk());
}

#[test]
fn a_whole_file_that_replaces_one_without_a_description_counts_only_once() {
    let root = root();
    let dir = root.join("model").join("type").join("track");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(crate::whole::file(&dir, "vocals", "s16le"), [0; 1000]).unwrap();
    let cache = Cache::open(root.to_path_buf(), 1 << 30).expect("open");
    let on_disk = || -> u64 { files(cache.root()).iter().map(|(_, size, _)| size).sum() };
    assert_eq!(cache.bytes(), on_disk());
    let task = Consolidation::at_44100(dir, StemFormat::Pcm16, 10);
    let audio = stemd_core::Audio::new(vec![vec![0.1; 10]; 2], 44_100);
    let bytes = stemd_core::encode_stem(&audio, StemFormat::Pcm16, 1.0).unwrap();
    cache
        .consolidate(&task, "vocals", &bytes, 10, Some(0))
        .expect("stored");
    assert_eq!(cache.bytes(), on_disk());
}

#[test]
fn subtracting_more_bytes_than_counted_stops_at_zero() {
    let cache = open(1 << 20);
    let dir = cache.verified_dir("model", "type", "track");
    cache
        .write_verified(&dir, "vocals", 0, "flac", &[1; 100])
        .expect("write");
    cache.sub_bytes(1000);
    assert_eq!(cache.bytes(), 0);
}

#[test]
fn a_reopened_store_keeps_its_pieces_and_drops_partial_writes() {
    let root = root();
    let cache = Cache::open(root.to_path_buf(), 1 << 20).expect("open");
    let dir = cache.verified_dir("model", "type", "track");
    cache
        .write_verified(&dir, "vocals", 0, "flac", &[1; 100])
        .expect("write");
    std::fs::write(dir.join("vocals").join("0001.flac.part"), [0; 50]).expect("partial");
    drop(cache);

    let reopened = Cache::open(root.to_path_buf(), 1 << 20).expect("reopen");
    assert_eq!(reopened.bytes(), 100);
    assert!(Cache::lookup(&dir, "vocals", 0, "flac").is_some());
    assert!(!dir.join("vocals").join("0001.flac.part").exists());
}

/// Two pieces of 300 bytes for each of `tracks`, oldest first.
fn fill(cache: &Cache, tracks: &[&str]) {
    for track in tracks {
        let dir = cache.verified_dir("model", "type", track);
        for stem in ["vocals", "harmonics"] {
            cache
                .write_verified(&dir, stem, 0, "flac", &[0; 300])
                .expect("write");
        }
        // Give each track a distinct modification time.
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

fn present(cache: &Cache, track: &str) -> Vec<&'static str> {
    ["vocals", "harmonics"]
        .into_iter()
        .filter(|stem| {
            cache
                .verified_dir("model", "type", track)
                .join(stem)
                .join(file(0, "flac"))
                .is_file()
        })
        .collect()
}

#[test]
fn over_the_budget_the_least_recently_used_track_is_deleted() {
    let cache = open(1300);
    fill(&cache, &["a", "b"]);
    // Reading `a` makes `b` the least recently used.
    assert!(
        Cache::lookup(
            &cache.verified_dir("model", "type", "a"),
            "vocals",
            0,
            "flac"
        )
        .is_some()
    );
    std::thread::sleep(std::time::Duration::from_millis(20));
    fill(&cache, &["c"]);
    assert!(cache.bytes() <= 1300, "{} bytes kept", cache.bytes());
    assert_eq!(present(&cache, "b"), Vec::<&str>::new());
    assert_eq!(present(&cache, "a"), vec!["vocals", "harmonics"]);
    assert_eq!(present(&cache, "c"), vec!["vocals", "harmonics"]);
}

#[test]
fn eviction_skips_the_tracks_open_sessions_play() {
    let cache = open(1300);
    cache.set_in_use(Box::new(|| HashSet::from(["a".to_owned()])));
    fill(&cache, &["a", "b", "c"]);
    assert_eq!(present(&cache, "a"), vec!["vocals", "harmonics"]);
    assert_eq!(present(&cache, "b"), Vec::<&str>::new());
}

/// A track whose two stems are stored whole, 300 bytes each.
fn store_whole(cache: &Cache, track: &str) {
    let task = Consolidation::at_44100(
        cache.verified_dir("model", "type", track),
        StemFormat::Pcm16,
        150,
    );
    for stem in ["vocals", "harmonics"] {
        cache
            .consolidate(&task, stem, &[0; 300], 150, Some(0))
            .expect("store");
    }
    std::thread::sleep(std::time::Duration::from_millis(20));
}

fn exists(cache: &Cache, track: &str) -> bool {
    cache.verified_dir("model", "type", track).is_dir()
}

fn rules(max_bytes: u64, partial: u64, whole: u64) -> Rules {
    Rules {
        max_bytes,
        partial_max_age: std::time::Duration::from_millis(partial),
        whole_max_age: std::time::Duration::from_millis(whole),
        unfetched_ttl: std::time::Duration::from_secs(60),
    }
}

#[test]
fn a_track_unused_past_its_age_is_deleted_partial_and_whole_alike() {
    let cache = open_with(rules(1 << 20, 30, 3_600_000));
    fill(&cache, &["partial"]);
    store_whole(&cache, "whole");
    std::thread::sleep(std::time::Duration::from_millis(40));
    cache.clean();
    assert!(!exists(&cache, "partial"), "a partial track past a day");
    assert!(exists(&cache, "whole"), "a whole track inside 30 days");

    let cache = open_with(rules(1 << 20, 3_600_000, 30));
    store_whole(&cache, "whole");
    std::thread::sleep(std::time::Duration::from_millis(40));
    cache.clean();
    assert!(!exists(&cache, "whole"));
    assert_eq!(cache.bytes(), 0);
}

#[test]
fn over_the_budget_partial_tracks_go_before_whole_ones() {
    let cache = open(1 << 20);
    store_whole(&cache, "old-whole");
    fill(&cache, &["new-partial"]);
    store_whole(&cache, "newest-whole");
    // The cache is just over the budget, so deleting the partial track is enough.
    let budget = cache.bytes() - 100;
    let cache = Cache::open(cache.root().to_owned(), budget).expect("reopen");
    assert!(!exists(&cache, "new-partial"), "partial goes first");
    assert!(exists(&cache, "old-whole"));
    assert!(exists(&cache, "newest-whole"));
}

#[test]
fn a_job_result_is_kept_until_every_stem_is_fetched() {
    let cache = open(1300);
    store_whole(&cache, "held");
    let dir = cache.verified_dir("model", "type", "held");
    let output = Output {
        format: StemFormat::Pcm16,
        rate: OutputRate::default(),
        derived: false,
        dsp: stemd_core::DspMode::General.into(),
        parts: Parts::parse("vocals,harmonics").unwrap(),
    };
    let entry = Arc::new(crate::jobs::Entry::stored(&dir, &output, 1.0, 2, None).expect("whole"));
    cache.hold(&entry);
    store_whole(&cache, "next");
    assert!(exists(&cache, "held"), "kept while unfetched");
    entry.mark_fetched("harmonics");
    entry.mark_fetched("vocals");
    store_whole(&cache, "last");
    assert!(!exists(&cache, "held"), "fetched, so it goes first");
}

#[test]
fn a_reopened_cache_drops_the_old_job_entries_and_keeps_tracks() {
    let root = root();
    let cache = Cache::open(root.to_path_buf(), 1 << 20).expect("open");
    store_whole(&cache, "kept");
    let old = root.join("ab".repeat(32));
    std::fs::create_dir_all(&old).expect("an old entry");
    std::fs::write(old.join("vocals.flac"), [0; 10]).expect("an old stem");
    std::fs::write(root.join(LOCK_FILE), []).expect("a lock file");
    drop(cache);
    let cache = Cache::open(root.to_path_buf(), 1 << 20).expect("reopen");
    assert!(!old.exists());
    assert!(exists(&cache, "kept"));
    assert_eq!(cache.stats().tracks, 1);

    cache.wipe();
    assert!(!exists(&cache, "kept"));
    assert!(root.join(LOCK_FILE).exists(), "the lock stays");
    assert_eq!(cache.bytes(), 0);
}

#[test]
fn emptying_the_cache_keeps_the_tracks_sessions_play() {
    let cache = open(1 << 20);
    cache.set_in_use(Box::new(|| HashSet::from(["playing".to_owned()])));
    fill(&cache, &["playing", "idle"]);
    assert_eq!(cache.clear().0, 1);
    assert!(exists(&cache, "playing"));
    assert!(!exists(&cache, "idle"));
}

#[test]
fn emptying_the_cache_frees_its_bytes_and_file_keys_and_keeps_held_results() {
    let cache = open(1 << 20);
    store_whole(&cache, "held");
    store_whole(&cache, "idle");
    cache.record_file("key", "idle").expect("recorded");
    let output = Output {
        format: StemFormat::Pcm16,
        rate: OutputRate::default(),
        derived: false,
        dsp: stemd_core::DspMode::General.into(),
        parts: Parts::parse("vocals,harmonics").unwrap(),
    };
    let dir = cache.verified_dir("model", "type", "held");
    let entry = Arc::new(crate::jobs::Entry::stored(&dir, &output, 1.0, 2, None).expect("whole"));
    cache.hold(&entry);
    let before = cache.bytes();
    let (tracks, freed) = cache.clear();
    assert_eq!(tracks, 1);
    assert!(exists(&cache, "held"));
    assert!(!exists(&cache, "idle"));
    assert_eq!(cache.bytes(), before - freed);
    assert_eq!(cache.file_track("key"), None);
}

#[test]
fn the_track_count_drops_as_soon_as_the_cache_is_emptied() {
    let cache = open(1 << 20);
    fill(&cache, &["a", "b"]);
    assert_eq!(cache.stats().tracks, 2);
    cache.clear();
    assert_eq!(cache.stats().tracks, 0);
    fill(&cache, &["c"]);
    cache.wipe();
    assert_eq!(cache.stats().tracks, 0);
}

#[test]
fn a_track_without_a_used_file_counts_from_its_newest_piece() {
    let root = root();
    let cache = Cache::open(root.to_path_buf(), 1 << 20).expect("open");
    fill(&cache, &["a", "b"]);
    for track in ["a", "b"] {
        std::fs::remove_file(cache.verified_dir("model", "type", track).join(USED))
            .expect("a used file");
    }
    drop(cache);
    let cache = Cache::open(root.to_path_buf(), 1300).expect("reopen");
    fill(&cache, &["c"]);
    assert_eq!(present(&cache, "a"), Vec::<&str>::new());
    assert_eq!(present(&cache, "b"), vec!["vocals", "harmonics"]);
}

#[test]
fn a_track_is_not_stored_whole_while_the_disk_is_low() {
    let cache = crate::testkit::cache(1 << 30, std::time::Duration::from_secs(60));
    cache.set_free_probe(Box::new(|| Some(crate::disk::FLOOR - 1)));
    let task = Consolidation {
        stems: vec!["vocals"],
        ..Consolidation::at_44100(
            cache.verified_dir("model", "type", "track"),
            StemFormat::Flac,
            10,
        )
    };
    assert!(cache.consolidate_later(task.clone()));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while cache.is_consolidating(&task.dir) {
        assert!(std::time::Instant::now() < deadline, "still consolidating");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(cache.consolidation_failed(&task.dir));
    assert!(!cache.consolidate_later(task), "queued again at once");
}

#[test]
fn a_panic_while_storing_whole_fails_that_track_and_the_next_one_still_runs() {
    let cache = crate::testkit::cache(1 << 30, std::time::Duration::from_secs(60));
    cache.set_gate(Box::new(|| panic!("gate")));
    let task = |track: &str| Consolidation {
        stems: vec!["vocals"],
        ..Consolidation::at_44100(
            cache.verified_dir("model", "type", track),
            StemFormat::Flac,
            10,
        )
    };
    let first = task("first");
    assert!(cache.consolidate_later(first.clone()));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while cache.is_consolidating(&first.dir) {
        assert!(std::time::Instant::now() < deadline, "still consolidating");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(cache.consolidation_failed(&first.dir));

    cache.set_gate(Box::new(|| true));
    let second = task("second");
    assert!(cache.consolidate_later(second.clone()));
    while cache.is_consolidating(&second.dir) {
        assert!(std::time::Instant::now() < deadline, "the second never ran");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
