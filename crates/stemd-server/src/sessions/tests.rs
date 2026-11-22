use super::plan::{BEFORE_JOBS, MAX_PRIORITY, MAX_REGIONS, Region};
use super::*;
use crate::ident::track_id;
use crate::output::output_index;
use stemd_core::{DspMode, OutputRate, StemFormat};
use stemd_core::{Grid, Stems};

/// Sessions on a cache no other test touches.
fn sessions(idle: Duration) -> Arc<Sessions> {
    Sessions::new(
        idle,
        4,
        crate::testkit::cache(1 << 30, Duration::from_secs(60)),
    )
}

#[test]
fn spans_merge_what_touches_and_cover_only_within_one_span() {
    let mut spans = Spans::default();
    spans.insert(10..20);
    spans.insert(30..40);
    assert!(spans.covers(&(12..18)));
    assert!(!spans.covers(&(15..35)), "a gap is not covered");
    assert_eq!(spans.overlaps(&(15..35)), vec![15..20, 30..35]);
    spans.insert(20..30);
    assert_eq!(spans.as_pairs(), vec![[10, 40]]);
    assert!(spans.covers(&(15..35)));
    spans.insert(0..5);
    assert_eq!(spans.as_pairs(), vec![[0, 5], [10, 40]]);
    assert_eq!(spans.span_of(4), Some(&(0..5)));
    assert_eq!(spans.span_of(5), None);
}

fn grid() -> Grid {
    // Segments of 1000 every 750 samples.
    Grid::new(100_000, 1000, 0.25).expect("a grid")
}

fn region(at: usize, ahead: usize, behind: usize, priority: u8) -> Region {
    Region {
        at,
        ahead,
        behind,
        priority,
    }
}

fn window(regions: &[Region]) -> WindowRequest {
    WindowRequest {
        regions: regions.to_vec(),
    }
}

#[test]
fn segments_run_nearest_at_first_with_behind_counted_twice() {
    let planned = plan(&grid(), &window(&[region(10_000, 3000, 1500, 0)]));
    let order: Vec<(usize, usize)> = planned.iter().map(|(k, r)| (*k, r.distance)).collect();
    assert_eq!(
        order,
        vec![
            (13, 0),
            (12, 1),
            (14, 500),
            (15, 1250),
            (11, 1501),
            (16, 2000),
            (17, 2750)
        ]
    );
    assert!(planned.iter().all(|(_, r)| r.priority == 0));
}

#[test]
fn a_segment_takes_its_best_rank_across_regions() {
    let request = window(&[
        region(50_000, 1000, 0, 1),
        region(10_000, 1000, 0, 0),
        region(10_000, 0, 0, 7),
    ]);
    let planned = plan(&grid(), &request);
    let order: Vec<(usize, u8)> = planned.iter().map(|(k, r)| (*k, r.priority)).collect();
    assert_eq!(order, vec![(13, 0), (14, 0), (66, 1), (67, 1)]);
    assert_eq!(planned[0].1.class(), Class::BeforeJobs);
    let idle = plan(&grid(), &window(&[region(0, 0, 0, BEFORE_JOBS)]));
    assert_eq!(idle[0].1.class(), Class::WhenIdle);
}

#[test]
fn a_plan_stays_inside_the_track() {
    let planned = plan(&grid(), &window(&[region(99_999, 50_000, 2000, 0)]));
    assert!(planned.iter().all(|(k, _)| *k < grid().len()));
    let mut ks: Vec<usize> = planned.iter().map(|(k, _)| *k).collect();
    ks.sort_unstable();
    ks.dedup();
    assert_eq!(ks.len(), planned.len(), "a segment is planned twice");
}

#[test]
fn a_window_past_its_limits_is_refused() {
    assert!(window(&[region(0, 0, 0, MAX_PRIORITY)]).check().is_ok());
    assert!(
        window(&[region(0, 0, 0, MAX_PRIORITY + 1)])
            .check()
            .is_err()
    );
    assert!(
        window(&vec![region(0, 0, 0, 0); MAX_REGIONS + 1])
            .check()
            .is_err()
    );
}

fn output(format: StemFormat) -> Output {
    Output {
        format,
        rate: OutputRate::default(),
        derived: false,
        dsp: DspMode::default().into(),
        parts: Parts::DEFAULT,
    }
}

#[test]
fn the_gain_leaves_three_decibels_over_the_mix_peak() {
    assert_eq!(output(StemFormat::Pcm32).gain(Some(4.0)), 1.0);
    assert_eq!(output(StemFormat::Pcm16).gain(Some(0.5)), 1.0);
    let full_scale = output(StemFormat::Flac).gain(None);
    assert!((full_scale - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    let loud = output(StemFormat::Pcm16).gain(Some(0.9));
    assert!((loud - 1.0 / (0.9 * std::f32::consts::SQRT_2)).abs() < 1e-6);
}

fn spec() -> TrackSpec {
    TrackSpec {
        frames: 100_000,
        channels: 2,
        sample_rate: 44_100,
        grid: grid(),
        normalises: true,
        parts: Parts::DEFAULT,
    }
}

fn audio(frames: usize, seed: f32) -> Audio {
    Audio::new(
        (0..2)
            .map(|c| (0..frames).map(|i| ((i + c) as f32 * seed).sin()).collect())
            .collect(),
        44_100,
    )
}

fn request() -> Request {
    Request {
        output: output(StemFormat::Pcm32),
        in_format: PcmFormat::F32le,
        track: spec(),
        norm: None,
        peak: None,
        model_id: "model".into(),
    }
}

fn settled(sessions: &Sessions, session: &Session, mix: &Audio) -> Arc<Track> {
    let track = session.track();
    match track.upload(0, mix).expect("upload") {
        Some(Verdict::Verified(id)) => sessions.settle(&track, &id),
        other => panic!("the upload came to {other:?}"),
    }
}

#[test]
fn an_open_past_the_memory_budget_is_refused_until_a_track_ends() {
    let sessions = sessions(Duration::from_secs(30));
    let one = spec().memory();
    sessions.set_memory_budget(2 * one);
    let a = sessions.open(request()).expect("open");
    let b = sessions.open(request()).expect("open");
    assert!(matches!(
        sessions.open(request()),
        Err(OpenError::Full { .. })
    ));
    sessions
        .join(output(StemFormat::Pcm32), PcmFormat::F32le, None, b.track())
        .expect("a join reserves nothing");
    sessions.remove(&a.id, "ended by the client");
    sessions
        .open(request())
        .expect("the ended track's memory is free");

    sessions.set_memory_budget(one - 1);
    assert!(matches!(
        sessions.open(request()),
        Err(OpenError::TooLarge { .. })
    ));
}

#[test]
fn two_sessions_at_one_rank_take_turns_and_a_better_rank_goes_first() {
    let sessions = sessions(Duration::from_secs(30));
    let a = sessions.open(request()).expect("open");
    let b = sessions.open(request()).expect("open");
    for (i, session) in [&a, &b].into_iter().enumerate() {
        session
            .track()
            .upload(0, &audio(100_000, 0.01 + i as f32 / 100.0))
            .expect("upload");
        session
            .set_window(&window(&[region(10_000, 3000, 0, 0)]))
            .expect("window");
    }
    let mut order = Vec::new();
    for _ in 0..6 {
        let (track, k) = sessions.next(Class::BeforeJobs).expect("a segment");
        order.push(if track.uid == a.track().uid { 'a' } else { 'b' });
        track.done(k);
    }
    assert!(
        order.windows(2).all(|w| w[0] != w[1]),
        "the sessions did not take turns: {order:?}"
    );

    b.set_window(&window(&[region(50_000, 0, 0, 0)]))
        .expect("window");
    a.set_window(&window(&[region(60_000, 0, 0, 1)]))
        .expect("window");
    let (track, _) = sessions.next(Class::BeforeJobs).expect("a segment");
    assert_eq!(track.uid, b.track().uid, "priority 0 waited for priority 1");
    assert!(sessions.next(Class::WhenIdle).is_none());
}

#[test]
fn a_track_id_covers_every_sample_and_the_geometry() {
    let a = audio(1000, 0.01);
    let mut b = a.clone();
    b.data[1][999] = f32::from_bits(b.data[1][999].to_bits() ^ 1);
    assert_ne!(track_id(&a), track_id(&b));
    let c = Audio::new(a.data.clone(), 48_000);
    assert_ne!(track_id(&a), track_id(&c));
}

#[test]
fn two_sessions_on_the_same_audio_end_up_on_one_track() {
    let sessions = sessions(Duration::from_secs(30));
    let mix = audio(100_000, 0.01);
    let a = sessions.open(request()).expect("open");
    let b = sessions.open(request()).expect("open");
    assert!(
        !Arc::ptr_eq(&a.track(), &b.track()),
        "each starts on its own"
    );

    let first = settled(&sessions, &a, &mix);
    let merged_from = b.track();
    let second = settled(&sessions, &b, &mix);
    assert!(Arc::ptr_eq(&first, &second));
    assert!(Arc::ptr_eq(&a.track(), &b.track()));
    assert!(
        merged_from.closed().is_some(),
        "the track b left is still open"
    );
    assert!(b.closed().is_none(), "b went down with its old track");

    let c = sessions.open(request()).expect("open");
    let other = settled(&sessions, &c, &audio(100_000, 0.02));
    assert!(!Arc::ptr_eq(&other, &first), "different audio was merged");

    sessions.remove(&a.id, "ended by the client");
    assert!(sessions.has_track(first.uid), "b still plays it");
    sessions.remove(&b.id, "ended by the client");
    assert!(!sessions.has_track(first.uid));
}

#[test]
fn an_upload_that_differs_from_what_the_track_holds_is_refused() {
    let sessions = sessions(Duration::from_secs(30));
    let a = sessions.open(request()).expect("open");
    let mix = audio(100_000, 0.01);
    let track = a.track();
    track.upload(0, &mix.excerpt(0, 60_000)).expect("first");
    track
        .upload(50_000, &mix.excerpt(50_000, 70_000))
        .expect("the same samples again");
    let err = track
        .upload(40_000, &audio(20_000, 0.02))
        .expect_err("different");
    assert!(matches!(err, UploadError::Differs(40_000, 60_000)), "{err}");
}

#[test]
fn client_statistics_that_differ_from_the_mix_keep_the_track_unshared() {
    let sessions = sessions(Duration::from_secs(30));
    let mix = audio(100_000, 0.01);
    let mut wrong = request();
    wrong.norm = Some(Normalisation::new(0.5, 2.0));
    let session = sessions.open(wrong).expect("open");
    let verdict = session.track().upload(0, &mix).expect("upload");
    assert_eq!(verdict, Some(Verdict::Unpublishable(track_id(&mix))));
    assert_eq!(session.track().verified(), None);
}

#[test]
fn a_range_not_ready_is_refused() {
    let sessions = sessions(Duration::from_secs(30));
    let session = sessions.open(request()).expect("open");
    let err = session
        .read("vocals", 0, 10)
        .err()
        .expect("nothing is ready");
    assert!(format!("{err}").contains("not ready"), "{err}");
}

#[test]
fn pieces_at_another_rate_tile_the_output_without_gaps() {
    for out in [24_000u64, 48_000, 96_000] {
        for stride in [257_985usize, 441_600] {
            let ends: Vec<usize> = (0..10)
                .map(|k| output_index(k * stride, 44_100, out))
                .collect();
            for w in ends.windows(2) {
                let expected = (stride as u64 * out) as f64 / 44_100.0;
                let got = (w[1] - w[0]) as f64;
                assert!(
                    (got - expected).abs() < 1.0,
                    "{out} Hz, stride {stride}: {got}"
                );
            }
        }
    }
    assert_eq!(output_index(441_000, 44_100, 48_000), 480_000);
}

#[test]
fn a_polyphase_read_to_the_end_reaches_the_last_output_sample() {
    let frames = 100_000;
    assert_ne!(frames % 147, 0);
    let mut req = request();
    req.output.rate = OutputRate::Hz96000;
    req.output.dsp = DspMode::Mode1.into();
    let sessions = sessions(Duration::from_secs(30));
    let session = sessions.open(req).expect("open");
    let track = session.track();
    track.upload(0, &audio(frames, 0.01)).expect("upload");
    track
        .store(0, frames, &crate::testkit::stems(frames).1)
        .expect("store");
    for from in [0, frames - 10_000] {
        let got = session.read("vocals", from, frames).expect("read");
        assert_eq!(
            got.audio.frames(),
            output_index(frames, 44_100, 96_000) - output_index(from, 44_100, 96_000),
            "from {from}"
        );
    }
}

#[test]
fn a_read_at_another_rate_takes_nothing_from_before_the_ready_span() {
    let (frames, ready) = (100_000, 50_000);
    assert_ne!(ready % 147, 0);
    let mut req = request();
    req.output.rate = OutputRate::Hz48000;
    req.output.derived = true;
    let sessions = sessions(Duration::from_secs(30));
    let session = sessions.open(req).expect("open");
    let track = session.track();
    let mix = audio(frames, 0.01);
    track.upload(0, &mix).expect("upload");
    // From `ready` on the parts sum to the mix, so the derived part is silent there.
    let half = Audio::new(
        mix.data
            .iter()
            .map(|c| c[ready..].iter().map(|v| v * 0.5).collect())
            .collect(),
        44_100,
    );
    let stems = Stems {
        shipped: stemd_core::SHIPPED
            .iter()
            .map(|name| (*name, half.clone()))
            .collect(),
        model_residual_db: 0.0,
        more: Vec::new(),
    };
    track.store(ready, frames, &stems).expect("store");
    let got = session.read("drums", ready, ready + 10_000).expect("read");
    let peak = got
        .audio
        .data
        .iter()
        .flatten()
        .fold(0f32, |m, v| m.max(v.abs()));
    assert!(peak < 1e-6, "the unseparated mix leaked in at {peak}");
}

#[test]
fn the_activity_counts_uploads_plans_and_stored_pieces() {
    let sessions = sessions(Duration::from_secs(30));
    let session = sessions.open(request()).expect("open");
    let track = session.track();
    track
        .upload(0, &audio(40_000, 0.01))
        .expect("upload the first part");
    session
        .set_window(&window(&[region(0, 100_000, 0, 0)]))
        .expect("window");
    track.done(0);
    session.set_cached([1]);

    let [activity] = sessions.activity().try_into().expect("one session");
    assert_eq!(activity.number, 1);
    assert_eq!(activity.pieces, track.spec.grid.len());
    assert_eq!(activity.ready, 1, "the stored piece");
    assert_eq!(activity.arrived, 40_000);
    assert_eq!(activity.covered, track.piece(1).len(), "the stored piece");
    assert_eq!(activity.normal, 0);
    assert!(activity.high > 0, "{activity:?}");
    assert!(!activity.runnable, "runnable without the mix statistics");

    let mix = audio(100_000, 0.01);
    track
        .upload(40_000, &mix.excerpt(40_000, 100_000))
        .expect("upload the rest");
    let [activity] = sessions.activity().try_into().expect("one session");
    assert_eq!(activity.arrived, 100_000);
    assert!(activity.runnable, "{activity:?}");
}

#[test]
fn an_ended_session_is_kept_with_its_reason() {
    let sessions = sessions(Duration::from_millis(20));
    let first = sessions.open(request()).expect("open");
    let second = sessions.open(request()).expect("open");
    let third = sessions.open(request()).expect("open");
    third.track().close("segment 3: out of memory");
    sessions.remove(&first.id, "ended by the client");
    std::thread::sleep(Duration::from_millis(40));
    sessions.reap();
    let fourth = sessions.open(request()).expect("open");
    sessions.close_all("the server switched models");
    let ended: Vec<(u64, String)> = sessions
        .ended()
        .into_iter()
        .map(|e| (e.number, e.reason))
        .collect();
    assert_eq!(
        ended,
        [
            (first.number, "ended by the client".to_owned()),
            (second.number, "idle".to_owned()),
            (third.number, "segment 3: out of memory".to_owned()),
            (fourth.number, "the server switched models".to_owned()),
        ]
    );
    assert!(sessions.activity().is_empty());
}

#[test]
fn a_quiet_session_with_nothing_to_run_is_reaped() {
    let sessions = sessions(Duration::from_millis(20));
    let session = sessions.open(request()).expect("open");
    assert_eq!(sessions.reap(), 0, "reaped at once");
    std::thread::sleep(Duration::from_millis(40));
    assert_eq!(sessions.reap(), 1);
    assert!(session.closed().is_some());
    assert!(!sessions.has_track(session.track().uid));
}
