//! Tests that send requests through the whole router.

mod separations {
    use std::sync::Arc;

    use axum::Router;
    use axum::http::{Method, StatusCode};
    use serde_json::json;

    use crate::api::params::CreateParams;
    use crate::jobs::{MixShape, job_key};
    use crate::output::Output;
    use crate::separations::output_of;
    use crate::state::AppState;
    use crate::testkit::{Answer, call, state};

    const TRACK: &str = "f9c29cbfe3466b674f9903d9";

    const FILE: &str = "3b0c55e19a7d42f08c6e1d2a";

    /// The output a request with `query` names.
    fn output_for(state: &AppState, query: serde_json::Value) -> Output {
        let params: CreateParams = serde_json::from_value(query).expect("params");
        output_of(state, &params).expect("an output")
    }

    /// A queued job for `TRACK` at `output`, with `FILE` naming the track.
    fn queued_job(state: &AppState, output: &Output) -> Arc<crate::jobs::Job> {
        let key = job_key(TRACK, output, &state.switcher.cache_id());
        let mix = MixShape {
            frames: 44_100,
            sample_rate: 44_100,
            channels: 2,
        };
        let Ok(job) = state.jobs.claim(&key, mix) else {
            panic!("a job already holds {key}");
        };
        state.cache.record_file(FILE, TRACK).expect("recorded");
        job
    }

    /// `GET /v2/separations?file=FILE` for 16-bit stems.
    async fn by_file(app: &Router) -> Answer {
        let uri = format!("/v2/separations?file={FILE}&output_format=s16le");
        call(app, Method::GET, &uri, Vec::new()).await
    }

    #[tokio::test]
    async fn a_lookup_by_file_key_while_a_job_runs_answers_its_job() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let job = queued_job(
            &state,
            &output_for(&state, json!({"output_format": "s16le"})),
        );
        let answer = by_file(&app).await;
        assert_eq!(answer.status, StatusCode::OK, "{}", answer.json);
        assert_eq!(answer.json["state"], "separating");
        assert_eq!(answer.json["job"], job.id.as_str());
        assert_eq!(answer.json["track"], TRACK);
        assert_eq!(answer.json["frames"], 44_100);
        assert!(
            answer.json["output"].get("gain").is_none(),
            "{}",
            answer.json
        );
    }

    #[tokio::test]
    async fn a_lookup_after_the_job_ends_answers_stored_without_a_job() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let body: Vec<u8> = (0..44_100 * 2)
            .flat_map(|i: i32| (((i * 37) % 2000 - 1000) as i16 * 8).to_le_bytes())
            .collect();
        let uri = format!("/v2/jobs?file={FILE}&format=s16le&output_format=s16le");
        let posted = call(&app, Method::POST, &uri, body).await;
        assert!(posted.status.is_success(), "{}", posted.json);
        let id = posted.json["id"].as_str().expect("a job id").to_owned();
        let early = by_file(&app).await;
        assert!(
            early.json["state"] == "stored" || early.json["job"] == id.as_str(),
            "{}",
            early.json
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let job = call(&app, Method::GET, &format!("/v2/jobs/{id}"), Vec::new()).await;
            let stage = &job.json["progress"]["stage"];
            if stage == "done" {
                break;
            }
            assert!(stage != "failed", "{}", job.json);
            assert!(std::time::Instant::now() < deadline, "the job never ended");
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let answer = by_file(&app).await;
        assert_eq!(answer.json["state"], "stored", "{}", answer.json);
        assert!(answer.json.get("job").is_none(), "{}", answer.json);
    }

    #[tokio::test]
    async fn a_running_job_for_another_output_is_not_reported() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        queued_job(
            &state,
            &output_for(&state, json!({"output_format": "flac"})),
        );
        let answer = by_file(&app).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND, "{}", answer.json);
    }

    #[tokio::test]
    async fn a_failed_job_is_no_longer_reported_as_separating() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let job = queued_job(
            &state,
            &output_for(&state, json!({"output_format": "s16le"})),
        );
        assert_eq!(by_file(&app).await.json["state"], "separating");
        job.fail("the model failed");
        let answer = by_file(&app).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND, "{}", answer.json);
    }

    #[tokio::test]
    async fn after_a_join_only_the_last_delete_cancels_the_job() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let job = queued_job(
            &state,
            &output_for(&state, json!({"output_format": "s16le"})),
        );
        let uri = format!("/v2/jobs/{}", job.id);
        let joined = call(&app, Method::POST, &format!("{uri}/join"), Vec::new()).await;
        assert_eq!(joined.status, StatusCode::OK, "{}", joined.json);

        let submitter = call(&app, Method::DELETE, &uri, Vec::new()).await;
        assert_eq!(submitter.status, StatusCode::NO_CONTENT);
        let running = call(&app, Method::GET, &uri, Vec::new()).await;
        assert_eq!(running.status, StatusCode::OK, "{}", running.json);
        assert_eq!(by_file(&app).await.json["state"], "separating");

        let joiner = call(&app, Method::DELETE, &uri, Vec::new()).await;
        assert_eq!(joiner.status, StatusCode::NO_CONTENT);
        let gone = call(&app, Method::GET, &uri, Vec::new()).await;
        assert_eq!(gone.status, StatusCode::NOT_FOUND, "{}", gone.json);
        assert_eq!(by_file(&app).await.status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_join_on_an_unknown_job_is_not_found() {
        let app = crate::api::router(state());
        let answer = call(&app, Method::POST, "/v2/jobs/nope-1/join", Vec::new()).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND, "{}", answer.json);
    }

    #[tokio::test]
    async fn a_join_answers_the_job_view() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let job = queued_job(
            &state,
            &output_for(&state, json!({"output_format": "s16le"})),
        );
        let uri = format!("/v2/jobs/{}/join", job.id);
        let answer = call(&app, Method::POST, &uri, Vec::new()).await;
        assert_eq!(answer.status, StatusCode::OK, "{}", answer.json);
        assert_eq!(answer.json["id"], job.id.as_str());
        assert_eq!(
            answer.json["progress"]["stage"], "queued",
            "{}",
            answer.json
        );
    }

    /// Send `GET uri` and return the status, headers and body bytes.
    async fn fetch(app: &Router, uri: &str) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        let request = axum::http::Request::builder()
            .uri(uri)
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app.clone(), request)
            .await
            .unwrap();
        let (status, headers) = (response.status(), response.headers().clone());
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, headers, body.to_vec())
    }

    /// Separate one second under `FILE` as 16-bit stems and return its track id.
    async fn stored_track(app: &Router) -> String {
        let body: Vec<u8> = (0..44_100 * 2)
            .flat_map(|i: i32| (((i * 37) % 2000 - 1000) as i16 * 8).to_le_bytes())
            .collect();
        let uri = format!("/v2/jobs?file={FILE}&format=s16le&output_format=s16le");
        let posted = call(app, Method::POST, &uri, body).await;
        assert!(posted.status.is_success(), "{}", posted.json);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let answer = by_file(app).await;
            if answer.json["state"] == "stored" {
                return answer.json["track"].as_str().expect("a track").to_owned();
            }
            assert!(std::time::Instant::now() < deadline, "{}", answer.json);
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn a_stored_separation_serves_each_stem_and_the_bundle() {
        let app = crate::api::router(state());
        let track = stored_track(&app).await;
        let base = format!("/v2/separations/{track}");
        let (status, headers, vocals) =
            fetch(&app, &format!("{base}/stems/vocals?output_format=s16le")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            headers[axum::http::header::CONTENT_LENGTH],
            vocals.len().to_string()
        );
        let (status, _, _) = fetch(&app, &format!("{base}/stems/bass?output_format=s16le")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let (status, headers, bundle) =
            fetch(&app, &format!("{base}/bundle?output_format=s16le")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            headers[axum::http::header::CONTENT_LENGTH],
            bundle.len().to_string()
        );
        assert_eq!(&bundle[..4], b"STMB");
        let stems = u32::from_le_bytes(bundle[8..12].try_into().unwrap()) as usize;
        assert!(stems >= 2, "{stems} stems");
        assert!(
            bundle.windows(vocals.len()).any(|w| w == vocals.as_slice()),
            "the bundle holds the vocals as served alone"
        );
    }

    #[tokio::test]
    async fn a_stem_of_an_unknown_running_or_mp3_separation_is_refused() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let unknown = call(
            &app,
            Method::GET,
            &format!("/v2/separations/{TRACK}/stems/vocals?output_format=s16le"),
            Vec::new(),
        )
        .await;
        assert_eq!(unknown.status, StatusCode::NOT_FOUND, "{}", unknown.json);
        assert_eq!(unknown.json["code"], "unknown_separation");

        queued_job(
            &state,
            &output_for(&state, json!({"output_format": "s16le"})),
        );
        let running = call(
            &app,
            Method::GET,
            &format!("/v2/separations/{TRACK}/bundle?output_format=s16le"),
            Vec::new(),
        )
        .await;
        assert_eq!(running.status, StatusCode::CONFLICT, "{}", running.json);
        let mp3 = call(
            &app,
            Method::GET,
            &format!("/v2/separations/{TRACK}/stems/vocals?output_format=mp3"),
            Vec::new(),
        )
        .await;
        assert_eq!(mp3.status, StatusCode::BAD_REQUEST, "{}", mp3.json);
    }
}

mod session_api {
    use axum::http::{Method, StatusCode};

    use crate::testkit::{call, state};

    #[tokio::test]
    async fn an_open_past_the_memory_budget_is_413_alone_and_429_beside_others() {
        let state = state();
        let app = crate::api::router(std::sync::Arc::clone(&state));
        let uri = "/v2/sessions?frames=30000";
        let sessions = state.queue.sessions();
        let first = call(&app, Method::POST, uri, Vec::new()).await;
        assert_eq!(first.status, StatusCode::CREATED, "{}", first.json);
        let id = first.json["id"].as_str().unwrap();
        let one = sessions.get(id).unwrap().track().spec.memory();

        sessions.set_memory_budget(2 * one - 1);
        let beside = call(&app, Method::POST, uri, Vec::new()).await;
        assert_eq!(
            beside.status,
            StatusCode::TOO_MANY_REQUESTS,
            "{}",
            beside.json
        );
        sessions.set_memory_budget(one - 1);
        let alone = call(&app, Method::POST, uri, Vec::new()).await;
        assert_eq!(
            alone.status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "{}",
            alone.json
        );
    }

    #[tokio::test]
    async fn a_malformed_window_is_a_400_with_a_json_error() {
        let app = crate::api::router(state());
        let opened = call(&app, Method::POST, "/v2/sessions?frames=30000", Vec::new()).await;
        assert_eq!(opened.status, StatusCode::CREATED, "{}", opened.json);
        let uri = format!(
            "/v2/sessions/{}/window",
            opened.json["id"].as_str().unwrap()
        );
        for body in [
            &b"not json"[..],
            br#"{"regions": [{"at": "soon"}]}"#,
            br#"{}"#,
        ] {
            let answer = call(&app, Method::PUT, &uri, body.to_vec()).await;
            assert_eq!(
                answer.status,
                StatusCode::BAD_REQUEST,
                "{:?}",
                String::from_utf8_lossy(body)
            );
            assert!(answer.json["error"].is_string(), "{}", answer.json);
        }
        let fine = call(&app, Method::PUT, &uri, br#"{"regions": []}"#.to_vec()).await;
        assert_eq!(fine.status, StatusCode::OK);
    }

    /// 16-bit stereo noise, `frames` long.
    fn pcm(frames: usize) -> Vec<u8> {
        (0..frames * 2)
            .flat_map(|i| ((((i * 37) % 2000) as i32 - 1000) as i16 * 8).to_le_bytes())
            .collect()
    }

    /// Wait up to 20 s for `done`.
    async fn until(mut done: impl FnMut() -> bool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while !done() {
            assert!(std::time::Instant::now() < deadline, "timed out");
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn an_open_that_does_not_fit_the_model_or_the_api_is_a_400() {
        let app = crate::api::router(state());
        for query in [
            "frames=30000&output_format=mp3",
            "frames=30000&sample_rate=48000",
            "frames=30000&channels=1",
            "frames=0",
            "frames=30000&mean=0.1",
            "frames=30000&mean=0.1&std=-1",
            "frames=30000&peak=-1",
            "frames=30000&peak=NaN",
            "frames=30000&peak=17",
        ] {
            let answer = call(
                &app,
                Method::POST,
                &format!("/v2/sessions?{query}"),
                Vec::new(),
            )
            .await;
            assert_eq!(
                answer.status,
                StatusCode::BAD_REQUEST,
                "{query}: {}",
                answer.json
            );
            assert!(answer.json["error"].is_string(), "{}", answer.json);
        }
    }

    #[tokio::test]
    async fn a_view_answers_304_while_its_etag_still_holds() {
        let app = crate::api::router(state());
        let opened = call(&app, Method::POST, "/v2/sessions?frames=30000", Vec::new()).await;
        let uri = format!("/v2/sessions/{}", opened.json["id"].as_str().unwrap());
        let first = call(&app, Method::GET, &uri, Vec::new()).await;
        assert_eq!(first.status, StatusCode::OK);
        let tag = first.headers[axum::http::header::ETAG].clone();
        let request = axum::http::Request::builder()
            .uri(format!("{uri}?wait=0"))
            .header(axum::http::header::IF_NONE_MATCH, tag.clone())
            .body(axum::body::Body::empty())
            .unwrap();
        let response = tower::ServiceExt::oneshot(app.clone(), request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(response.headers()[axum::http::header::ETAG], tag);
    }

    #[tokio::test]
    async fn a_piece_is_404_off_the_grid_409_until_ready_and_200_after() {
        let state = state();
        let app = crate::api::router(std::sync::Arc::clone(&state));
        let opened = call(&app, Method::POST, "/v2/sessions?frames=30000", Vec::new()).await;
        let id = opened.json["id"].as_str().unwrap().to_owned();
        let pieces = format!("/v2/sessions/{id}/pieces");
        let unknown = call(&app, Method::GET, &format!("{pieces}/bass/0"), Vec::new()).await;
        assert_eq!(unknown.status, StatusCode::NOT_FOUND);
        let past = call(
            &app,
            Method::GET,
            &format!("{pieces}/vocals/999"),
            Vec::new(),
        )
        .await;
        assert_eq!(past.status, StatusCode::NOT_FOUND);
        let early = call(&app, Method::GET, &format!("{pieces}/vocals/0"), Vec::new()).await;
        assert_eq!(early.status, StatusCode::CONFLICT, "{}", early.json);

        let uploaded = call(
            &app,
            Method::PUT,
            &format!("/v2/sessions/{id}/pcm?offset=0"),
            pcm(30_000),
        )
        .await;
        assert_eq!(uploaded.status, StatusCode::OK, "{}", uploaded.json);
        let window = br#"{"regions": [{"at": 0, "ahead": 30000, "priority": 0}]}"#.to_vec();
        let set = call(
            &app,
            Method::PUT,
            &format!("/v2/sessions/{id}/window"),
            window,
        )
        .await;
        assert_eq!(set.status, StatusCode::OK, "{}", set.json);
        let session = state.queue.sessions().get(&id).expect("the session");
        until(|| session.track().ready_covers(&session.track().piece(0))).await;
        let ready = call(&app, Method::GET, &format!("{pieces}/vocals/0"), Vec::new()).await;
        assert_eq!(ready.status, StatusCode::OK, "{}", ready.json);
        assert_eq!(ready.headers["x-stemd-from"], "0");
    }

    #[tokio::test]
    async fn a_join_needs_an_open_track_and_no_upload() {
        let state = state();
        let app = crate::api::router(std::sync::Arc::clone(&state));
        let unknown = "0".repeat(crate::ident::TRACK_ID_LEN);
        let missing = call(
            &app,
            Method::POST,
            &format!("/v2/sessions?track={unknown}"),
            Vec::new(),
        )
        .await;
        assert_eq!(missing.status, StatusCode::NOT_FOUND, "{}", missing.json);
        assert_eq!(missing.json["code"], "unknown_separation");

        let opened = call(&app, Method::POST, "/v2/sessions?frames=30000", pcm(30_000)).await;
        assert_eq!(opened.status, StatusCode::CREATED, "{}", opened.json);
        let session = state
            .queue
            .sessions()
            .get(opened.json["id"].as_str().unwrap())
            .expect("the session");
        until(|| session.track().verified().is_some()).await;
        let track = session.track().verified().unwrap();
        let uri = format!("/v2/sessions?track={track}");
        let with_body = call(&app, Method::POST, &uri, pcm(10)).await;
        assert_eq!(
            with_body.status,
            StatusCode::BAD_REQUEST,
            "{}",
            with_body.json
        );
        for peak in ["NaN", "1e30"] {
            let refused = call(
                &app,
                Method::POST,
                &format!("{uri}&peak={peak}"),
                Vec::new(),
            )
            .await;
            assert_eq!(
                refused.status,
                StatusCode::BAD_REQUEST,
                "{peak}: {}",
                refused.json
            );
        }
        let joined = call(
            &app,
            Method::POST,
            &format!("{uri}&file={FILE}"),
            Vec::new(),
        )
        .await;
        assert_eq!(joined.status, StatusCode::CREATED, "{}", joined.json);
        assert_ne!(joined.json["id"], opened.json["id"]);
        assert_eq!(
            state.cache.file_track(FILE),
            None,
            "a join records no file key"
        );
    }

    const FILE: &str = "3b0c55e19a7d42f08c6e1d2a";

    /// 16-bit stereo noise, `frames` long, that differs from [`pcm`].
    fn other_pcm(frames: usize) -> Vec<u8> {
        (0..frames * 2)
            .flat_map(|i| ((((i * 53) % 2000) as i32 - 1000) as i16 * 8).to_le_bytes())
            .collect()
    }

    /// Every file under `dir` except the `.used` marker.
    fn files_under(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_owned()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.file_name().is_some_and(|n| n != ".used") {
                    out.push(path);
                }
            }
        }
        out.sort();
        out
    }

    /// Store the first pieces of [`pcm`] under `FILE`, end the session, and return the track id and
    /// its directory.
    async fn partial_track(
        state: &crate::state::AppState,
        app: &axum::Router,
    ) -> (String, std::path::PathBuf) {
        let uri = format!("/v2/sessions?frames=30000&file={FILE}");
        let opened = call(app, Method::POST, &uri, pcm(30_000)).await;
        assert_eq!(opened.status, StatusCode::CREATED, "{}", opened.json);
        let sid = opened.json["id"].as_str().unwrap().to_owned();
        let window = br#"{"regions": [{"at": 0, "ahead": 15000, "priority": 0}]}"#.to_vec();
        let set = call(
            app,
            Method::PUT,
            &format!("/v2/sessions/{sid}/window"),
            window,
        )
        .await;
        assert_eq!(set.status, StatusCode::OK, "{}", set.json);
        state.queue.wake();
        let sessions = state.queue.sessions();
        let session = sessions.get(&sid).expect("the session");
        until(|| sessions.cached_for(&session).len() >= 4).await;
        let id = session.track().verified().expect("hashed");
        let type_name = crate::cache::type_dir(&session.output, session.gain);
        let dir = state
            .cache
            .verified_dir(&state.switcher.cache_id(), &type_name, &id);
        let ended = call(
            app,
            Method::DELETE,
            &format!("/v2/sessions/{sid}"),
            Vec::new(),
        )
        .await;
        assert_eq!(ended.status, StatusCode::NO_CONTENT);
        // A segment that is running when the session ends may still write its pieces.
        let uid = session.track().uid;
        until(|| !sessions.has_track(uid) && state.queue.is_idle()).await;
        let stored = crate::cache::Cache::cached(
            &dir,
            &session.output.served(),
            "flac",
            session.track().spec.grid.len(),
        );
        assert!(stored.len() < session.track().spec.grid.len(), "{stored:?}");
        (id, dir)
    }

    #[tokio::test]
    async fn a_session_on_a_partial_track_serves_its_pieces_then_stores_the_rest() {
        let state = state();
        let app = crate::api::router(std::sync::Arc::clone(&state));
        let (id, dir) = partial_track(&state, &app).await;
        let lookup = format!("/v2/separations?file={FILE}");
        let found = call(&app, Method::GET, &lookup, Vec::new()).await;
        assert_eq!(found.json["state"], "partial", "{}", found.json);

        let uri = format!("/v2/sessions?track={id}&file={FILE}");
        let opened = call(&app, Method::POST, &uri, Vec::new()).await;
        assert_eq!(opened.status, StatusCode::CREATED, "{}", opened.json);
        for stem in opened.json["stems"].as_array().unwrap() {
            assert_eq!(stem["stored"], "pieces", "{stem}");
            assert_eq!(stem["pieces"][0][0], 0, "{stem}");
        }
        let sid = opened.json["id"].as_str().unwrap().to_owned();
        let base = format!("/v2/sessions/{sid}");
        let read = call(
            &app,
            Method::GET,
            &format!("{base}/stems/vocals?from=0&to=1000"),
            Vec::new(),
        )
        .await;
        assert_eq!(read.status, StatusCode::OK, "{}", read.json);

        let uploaded = call(
            &app,
            Method::PUT,
            &format!("{base}/pcm?offset=0"),
            pcm(30_000),
        )
        .await;
        assert_eq!(uploaded.status, StatusCode::OK, "{}", uploaded.json);
        assert_eq!(uploaded.json["track"]["id"], id.as_str());
        let window = br#"{"regions": [{"at": 0, "ahead": 30000, "priority": 0}]}"#.to_vec();
        let set = call(&app, Method::PUT, &format!("{base}/window"), window).await;
        assert_eq!(set.status, StatusCode::OK, "{}", set.json);
        let segments = set.json["grid"]["segments"].as_u64().unwrap();
        assert!(
            set.json["planned"].as_u64().unwrap() < segments,
            "{}",
            set.json
        );
        state.queue.wake();
        let session = state.queue.sessions().get(&sid).expect("the session");
        let served = session.output.served();
        let count = session.track().spec.grid.len();
        until(|| crate::cache::Cache::cached(&dir, &served, "flac", count).len() == count).await;
    }

    #[tokio::test]
    async fn an_upload_that_is_not_the_claimed_track_ends_the_session_and_stores_nothing() {
        let state = state();
        let app = crate::api::router(std::sync::Arc::clone(&state));
        let (id, dir) = partial_track(&state, &app).await;
        let before = files_under(dir.parent().unwrap());

        let uri = format!("/v2/sessions?track={id}&file={FILE}");
        let opened = call(&app, Method::POST, &uri, Vec::new()).await;
        assert_eq!(opened.status, StatusCode::CREATED, "{}", opened.json);
        let base = format!("/v2/sessions/{}", opened.json["id"].as_str().unwrap());
        let window = br#"{"regions": [{"at": 0, "ahead": 30000, "priority": 0}]}"#.to_vec();
        let set = call(&app, Method::PUT, &format!("{base}/window"), window).await;
        assert_eq!(set.status, StatusCode::OK, "{}", set.json);
        let other = other_pcm(30_000);
        let first = call(
            &app,
            Method::PUT,
            &format!("{base}/pcm?offset=0"),
            other[..60_000].to_vec(),
        )
        .await;
        assert_eq!(first.status, StatusCode::OK, "{}", first.json);
        state.queue.wake();
        let rest = call(
            &app,
            Method::PUT,
            &format!("{base}/pcm?offset=15000"),
            other[60_000..].to_vec(),
        )
        .await;
        assert_eq!(rest.json["state"], "closed", "{}", rest.json);
        let reason = format!("upload is not track {id}");
        assert_eq!(rest.json["error"], reason.as_str());
        let gone = call(&app, Method::GET, &base, Vec::new()).await;
        assert_eq!(gone.status, StatusCode::NOT_FOUND);
        assert!(
            state
                .queue
                .sessions()
                .ended()
                .iter()
                .any(|e| e.reason == reason),
            "the window shows why it ended"
        );

        until(|| state.queue.is_idle()).await;
        assert_eq!(
            files_under(dir.parent().unwrap()),
            before,
            "something was written"
        );
        let lookup = format!("/v2/separations?file={FILE}");
        let found = call(&app, Method::GET, &lookup, Vec::new()).await;
        assert_eq!(found.status, StatusCode::NOT_FOUND, "{}", found.json);
    }

    #[tokio::test]
    async fn a_session_on_stored_pieces_takes_their_length_and_an_unknown_id_is_404() {
        let state = state();
        let app = crate::api::router(std::sync::Arc::clone(&state));
        let (id, _) = partial_track(&state, &app).await;
        let longer = call(
            &app,
            Method::POST,
            &format!("/v2/sessions?track={id}&frames=30001"),
            Vec::new(),
        )
        .await;
        assert_eq!(longer.status, StatusCode::BAD_REQUEST, "{}", longer.json);
        let same = call(
            &app,
            Method::POST,
            &format!("/v2/sessions?track={id}&frames=30000"),
            Vec::new(),
        )
        .await;
        assert_eq!(same.status, StatusCode::CREATED, "{}", same.json);

        let unknown = "1".repeat(crate::ident::TRACK_ID_LEN);
        let missing = call(
            &app,
            Method::POST,
            &format!("/v2/sessions?track={unknown}"),
            Vec::new(),
        )
        .await;
        assert_eq!(missing.status, StatusCode::NOT_FOUND, "{}", missing.json);
        assert_eq!(missing.json["code"], "unknown_separation");
        assert!(
            missing.json["error"]
                .as_str()
                .is_some_and(|e| e.contains("no open or stored track")),
            "{}",
            missing.json
        );
    }
}

mod v1 {
    use std::sync::Arc;

    use axum::http::{self, Method, StatusCode};
    use serde_json::{Value, json};

    use crate::v1::{FLOOR_DB, LINK};

    use crate::testkit::{call, state};

    fn keys(value: &Value) -> Vec<&str> {
        let mut keys: Vec<&str> = value
            .as_object()
            .expect("an object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        keys
    }

    fn sorted(mut keys: Vec<&str>) -> Vec<&str> {
        keys.sort_unstable();
        keys
    }

    #[tokio::test]
    async fn health_has_the_0_2_0_fields() {
        let app = crate::api::router(state());
        let answer = call(&app, Method::GET, "/v1/health", Vec::new()).await;
        assert_eq!(answer.status, StatusCode::OK);
        assert_eq!(answer.headers["deprecation"], "true");
        assert_eq!(answer.headers[http::header::LINK], LINK);
        assert_eq!(
            keys(&answer.json),
            sorted(vec![
                "version",
                "backend",
                "model",
                "device",
                "sample_rate",
                "channels",
                "stems",
                "model_id",
                "completed_jobs",
                "queue_depth",
                "cached_tracks",
                "cached_bytes",
                "cache_max_bytes",
                "max_track_seconds",
                "output_sample_rates",
                "dsp_modes",
                "dsp_mode_pairs",
                "default_output_format",
                "default_output_sample_rate",
                "derived_stem",
                "parts",
                "default_parts",
            ])
        );
        assert_eq!(answer.json["stems"], json!(["harmonics", "vocals"]));
        assert_eq!(answer.json["derived_stem"], "drums");
        assert_eq!(answer.json["default_parts"], json!(["harmonics", "vocals"]));
        assert_eq!(answer.json["default_output_format"], "flac");
        assert_eq!(answer.json["default_output_sample_rate"], 44_100);

        let v2 = call(&app, Method::GET, "/v2/health", Vec::new()).await;
        assert!(v2.headers.get("deprecation").is_none());
    }

    #[tokio::test]
    async fn a_job_result_has_the_0_2_0_shape() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let job = state.jobs.create("track".into());
        job.complete(
            crate::testkit::publish(&state.cache, "track", 100, 1.0),
            false,
        );

        let answer = call(
            &app,
            Method::GET,
            &format!("/v1/jobs/{}", job.id),
            Vec::new(),
        )
        .await;
        assert_eq!(answer.status, StatusCode::OK);
        assert_eq!(keys(&answer.json), ["id", "progress", "result"]);
        let result = &answer.json["result"];
        assert_eq!(
            keys(result),
            sorted(vec![
                "sample_rate",
                "channels",
                "frames",
                "format",
                "stems",
                "model_residual_db",
                "derived_stem",
                "separation_secs",
                "realtime_factor",
                "cached",
            ])
        );
        assert_eq!(result["format"], "pcm16");
        assert_eq!(result["sample_rate"], 44_100);
        assert_eq!(result["derived_stem"], "drums");
        let stem = &result["stems"][0];
        assert_eq!(keys(stem), ["bytes", "gain", "name", "path", "url"]);
        assert_eq!(stem["url"], format!("/v1/jobs/{}/stems/harmonics", job.id));
        assert!(
            std::path::Path::new(stem["path"].as_str().unwrap()).is_file(),
            "{stem}"
        );

        let url = stem["url"].as_str().unwrap();
        let download = call(&app, Method::GET, url, Vec::new()).await;
        assert_eq!(download.status, StatusCode::OK);
        assert_eq!(download.headers["deprecation"], "true");

        let empty = call(&app, Method::POST, "/v1/jobs?format=s16le", Vec::new()).await;
        assert_eq!(empty.status, StatusCode::BAD_REQUEST);
        assert_eq!(empty.headers["deprecation"], "true");
    }

    #[tokio::test]
    async fn a_job_on_stems_a_session_stored_reads_the_floor_residual() {
        use crate::cache::{Consolidation, type_dir};
        use stemd_core::Audio;
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let frames = 4410;
        let wave = |scale: f32| -> Vec<f32> {
            (0..frames)
                .map(|i| scale * (i as f32 * 0.01).sin())
                .collect()
        };
        let mix = Audio::new(vec![wave(0.5), wave(0.4)], 44_100);
        let id = crate::ident::track_id(&mix);
        let output = crate::testkit::output();
        let dir =
            state
                .cache
                .verified_dir(&state.switcher.cache_id(), &type_dir(&output, 1.0), &id);
        let task = Consolidation::at_44100(dir, output.format, frames);
        state.cache.write_info(&task.dir, &task.info).unwrap();
        for (stem, scale) in [("harmonics", 0.2), ("vocals", 0.15), ("drums", 0.1)] {
            let part = Audio::new(vec![wave(scale), wave(scale)], 44_100);
            let bytes = stemd_core::encode_stem(&part, output.format, 1.0).unwrap();
            state
                .cache
                .consolidate(&task, stem, &bytes, frames, Some(0))
                .unwrap();
        }
        let body: Vec<u8> = (0..frames)
            .flat_map(|i| [mix.data[0][i], mix.data[1][i]])
            .flat_map(f32::to_le_bytes)
            .collect();

        let answer = call(
            &app,
            Method::POST,
            "/v1/jobs?format=f32le&output_format=pcm16",
            body.clone(),
        )
        .await;
        assert_eq!(answer.status, StatusCode::OK, "{}", answer.json);
        let result = &answer.json["result"];
        assert_eq!(result["cached"], true);
        assert_eq!(result["model_residual_db"].as_f64(), Some(FLOOR_DB));
        assert_eq!(result["realtime_factor"].as_f64(), Some(0.0));

        let id = answer.json["id"].as_str().unwrap();
        let again = call(&app, Method::GET, &format!("/v1/jobs/{id}"), Vec::new()).await;
        let result = &again.json["result"];
        assert_eq!(result["model_residual_db"].as_f64(), Some(FLOOR_DB));
        assert_eq!(result["realtime_factor"].as_f64(), Some(0.0));

        let v2 = call(
            &app,
            Method::POST,
            "/v2/jobs?format=f32le&output_format=pcm16",
            body,
        )
        .await;
        assert_eq!(v2.status, StatusCode::OK, "{}", v2.json);
        let result = &v2.json["result"];
        assert_eq!(result["realtime_factor"].as_f64(), Some(0.0));
        assert_eq!(result["separation_secs"].as_f64(), Some(0.0));
        assert!(result["model_residual_db"].is_null(), "{result}");
    }

    #[tokio::test]
    async fn sessions_and_filters_are_protocol_2_only() {
        let app = crate::api::router(state());
        for (method, path) in [
            (Method::POST, "/v1/sessions?frames=30000"),
            (Method::GET, "/v1/sessions/abc"),
            (Method::PUT, "/v1/sessions/abc/window"),
        ] {
            let answer = call(&app, method, path, Vec::new()).await;
            assert_eq!(answer.status, StatusCode::GONE, "{path}");
            assert_eq!(answer.json["code"], "protocol_1", "{path}");
            assert!(
                answer.json["error"]
                    .as_str()
                    .unwrap()
                    .contains("/v2/sessions"),
                "{path}"
            );
            assert_eq!(answer.headers["deprecation"], "true", "{path}");
        }
        let filter = format!("/v1/filters/{}", "0".repeat(64));
        let answer = call(&app, Method::GET, &filter, Vec::new()).await;
        assert_eq!(answer.status, StatusCode::NOT_FOUND);
    }
}

mod jobs {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use axum::Router;
    use axum::http::{Method, StatusCode};

    use crate::testkit::{call, state, state_with_depth};

    /// One second of 16-bit stereo at 44.1 kHz.
    fn second() -> Vec<u8> {
        (0..44_100 * 2)
            .flat_map(|i: i32| (((i * 37) % 2000 - 1000) as i16 * 8).to_le_bytes())
            .collect()
    }

    /// Post [`second`] as a job and wait for it to finish. Returns the job id.
    async fn finished_job(app: &Router) -> String {
        let posted = call(
            app,
            Method::POST,
            "/v2/jobs?format=s16le&output_format=s16le",
            second(),
        )
        .await;
        assert!(posted.status.is_success(), "{}", posted.json);
        let id = posted.json["id"].as_str().expect("a job id").to_owned();
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let job = call(app, Method::GET, &format!("/v2/jobs/{id}"), Vec::new()).await;
            let stage = &job.json["progress"]["stage"];
            if stage == "done" {
                return id;
            }
            assert!(stage != "failed", "{}", job.json);
            assert!(Instant::now() < deadline, "the job never ended");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn every_request_is_a_503_while_the_server_is_off() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        state.serving.set(false, "toy");
        let answer = call(&app, Method::GET, "/v2/health", Vec::new()).await;
        assert_eq!(
            answer.status,
            StatusCode::SERVICE_UNAVAILABLE,
            "{}",
            answer.json
        );
        assert!(answer.json["error"].as_str().is_some(), "{}", answer.json);
    }

    #[tokio::test]
    async fn a_track_over_the_length_limit_is_a_413() {
        let state = state();
        state.settings.set_max_track_minutes(0.01);
        let app = crate::api::router(Arc::clone(&state));
        let answer = call(&app, Method::POST, "/v2/jobs?format=s16le", second()).await;
        assert_eq!(
            answer.status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "{}",
            answer.json
        );
    }

    #[tokio::test]
    async fn a_job_past_a_full_queue_is_a_429_and_fails() {
        let state = state_with_depth(0);
        let app = crate::api::router(Arc::clone(&state));
        let answer = call(&app, Method::POST, "/v2/jobs?format=s16le", second()).await;
        assert_eq!(
            answer.status,
            StatusCode::TOO_MANY_REQUESTS,
            "{}",
            answer.json
        );
        // A failed job frees its key, so a new request for the same track is refused again instead
        // of joining the failed job.
        let again = call(&app, Method::POST, "/v2/jobs?format=s16le", second()).await;
        assert_eq!(
            again.status,
            StatusCode::TOO_MANY_REQUESTS,
            "{}",
            again.json
        );
    }

    #[tokio::test]
    async fn a_filter_is_201_when_new_200_when_stored_and_readable_after() {
        let app = crate::api::router(state());
        let body = crate::filters::encode(44_100, 96_000, &[1.0; 320]);
        let id = crate::filters::id_of(&body);
        let uri = format!("/v2/filters/{id}");
        let created = call(&app, Method::PUT, &uri, body.clone()).await;
        assert_eq!(created.status, StatusCode::CREATED, "{}", created.json);
        let again = call(&app, Method::PUT, &uri, body).await;
        assert_eq!(again.status, StatusCode::OK, "{}", again.json);
        let read = call(&app, Method::GET, &uri, Vec::new()).await;
        assert_eq!(read.status, StatusCode::OK, "{}", read.json);
        assert_eq!(read.json, created.json);
    }

    #[tokio::test]
    async fn a_filter_past_the_file_limit_is_a_413() {
        let app = crate::api::router(state());
        let body = vec![0; crate::filters::MAX_FILE_BYTES + 1];
        let uri = format!("/v2/filters/{}", "0".repeat(64));
        let put = call(&app, Method::PUT, &uri, body).await;
        assert_eq!(put.status, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn a_filter_under_the_wrong_id_is_a_400_and_an_unknown_one_a_404() {
        let app = crate::api::router(state());
        let body = crate::filters::encode(44_100, 96_000, &[1.0; 320]);
        let wrong = "0".repeat(64);
        let put = call(&app, Method::PUT, &format!("/v2/filters/{wrong}"), body).await;
        assert_eq!(put.status, StatusCode::BAD_REQUEST, "{}", put.json);
        let get = call(
            &app,
            Method::GET,
            &format!("/v2/filters/{wrong}"),
            Vec::new(),
        )
        .await;
        assert_eq!(get.status, StatusCode::NOT_FOUND, "{}", get.json);
        assert_eq!(get.json["code"], "unknown_filter");
    }

    #[tokio::test]
    async fn a_stem_of_a_running_job_is_a_409() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let mix = crate::jobs::MixShape {
            frames: 44_100,
            sample_rate: 44_100,
            channels: 2,
        };
        let Ok(job) = state.jobs.claim("a-key", mix) else {
            panic!("a fresh table");
        };
        let uri = format!("/v2/jobs/{}/stems/vocals", job.id);
        let answer = call(&app, Method::GET, &uri, Vec::new()).await;
        assert_eq!(answer.status, StatusCode::CONFLICT, "{}", answer.json);
    }

    #[tokio::test]
    async fn a_finished_job_serves_its_stems_until_they_leave_the_cache() {
        let state = state();
        let app = crate::api::router(Arc::clone(&state));
        let id = finished_job(&app).await;
        let uri = format!("/v2/jobs/{id}/stems/vocals");
        let served = call(&app, Method::GET, &uri, Vec::new()).await;
        assert_eq!(served.status, StatusCode::OK);
        let unknown = call(
            &app,
            Method::GET,
            &format!("/v2/jobs/{id}/stems/bass"),
            Vec::new(),
        )
        .await;
        assert_eq!(unknown.status, StatusCode::NOT_FOUND);

        let job = state.jobs.get(&id).expect("the job");
        let entry = job.entry.lock().clone().expect("stems");
        std::fs::remove_file(&entry.stem("vocals").expect("vocals").path).expect("removed");
        let gone = call(&app, Method::GET, &uri, Vec::new()).await;
        assert_eq!(gone.status, StatusCode::GONE, "{}", gone.json);
    }
}
