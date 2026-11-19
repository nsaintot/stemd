//! Fixtures shared by the cache, job-store, queue and HTTP tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Method, StatusCode, header};
use serde_json::Value;
use stemd_core::{Audio, DspMode, OutputRate, StemFormat, Stems};
use tower::ServiceExt as _;

use crate::api::AppState;
use crate::cache::{Cache, Entry, Output};

/// Makes each test root unique. A timestamp alone is not enough: tests run in
/// parallel within one process, and two starting inside the same clock tick
/// would share a root and reap each other's entries.
static ROOT_SEQ: AtomicU64 = AtomicU64::new(0);

/// A cache rooted somewhere no other test will touch.
pub fn cache(max_bytes: u64, ttl: Duration) -> Arc<Cache> {
    let root = std::env::temp_dir().join(format!(
        "stemd-test-{}-{}",
        std::process::id(),
        ROOT_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let cache = Cache::new(root, max_bytes, ttl).expect("cache root");
    cache.ignore_disk();
    cache
}

/// A mix and its three parts, each `frames` long.
///
/// Two stems ship, so an entry costs `frames * channels * 2 * 2` bytes at
/// s16le: the cap tests depend on that arithmetic.
pub fn stems(frames: usize) -> (Audio, Stems) {
    let mix = Audio::new(vec![vec![0.25; frames], vec![0.25; frames]], 44100);
    let shipped = stemd_core::SHIPPED
        .iter()
        .map(|name| {
            (
                *name,
                Audio::new(vec![vec![0.1; frames], vec![0.1; frames]], 44100),
            )
        })
        .collect();
    (
        mix,
        Stems {
            shipped,
            model_residual_db: -30.0,
            more: Vec::new(),
        },
    )
}

pub fn publish(cache: &Cache, key: &str, frames: usize, separation_secs: f64) -> Arc<Entry> {
    let (mix, parts) = stems(frames);
    cache
        .publish(
            key,
            &mix,
            &parts,
            Output {
                format: StemFormat::Pcm16,
                rate: OutputRate::default(),
                derived: false,
                dsp: DspMode::default().into(),
                parts: stemd_core::Parts::DEFAULT,
            },
            separation_secs,
        )
        .expect("publish")
}

/// A server on the toy model, with nothing stored.
pub fn state() -> Arc<AppState> {
    let dir = std::env::temp_dir().join(format!(
        "stemd-state-test-{}-{}",
        std::process::id(),
        ROOT_SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let cache = cache(1 << 30, Duration::from_secs(60));
    let build: crate::queue::BuildSeparator = Box::new(|| {
        Ok(Box::new(stemd_core::toy::Toy::new(4410, 0.25)) as Box<dyn stemd_core::Separate>)
    });
    let queue = Arc::new(
        crate::queue::Queue::start(build, 8, Arc::clone(&cache))
            .expect("the toy builds")
            .0,
    );
    let settings = Arc::new(crate::settings::SettingsStore::open(
        dir.join("settings.json"),
    ));
    let switcher = Arc::new(crate::switch::Switcher::new(
        None,
        Arc::clone(&queue),
        Arc::clone(&settings),
        crate::switch::SwitchConfig {
            dirs: Vec::new(),
            cache: dir.join("models"),
            overlap: 0.25,
            precisions: crate::precision::Precisions::stated(None, stemd_core::Accelerator::Cpu),
            offline: true,
            artefact: "toy".into(),
        },
    ));
    Arc::new(AppState {
        store: Arc::new(crate::jobs::JobStore::new(Duration::from_secs(60))),
        cache,
        queue,
        switcher,
        logs: crate::logbuf::LogBuffer::new(16),
        drops: Arc::default(),
        settings,
        max_track_secs: 600.0,
        serving: crate::serving::Serving::new(true, None, "toy"),
        filters: crate::filters::Filters::open(dir.join("filters"), 4).expect("filters"),
    })
}

/// An answer from [`call`], its body parsed as JSON or `null`.
pub struct Answer {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub json: Value,
}

/// Send one request to `app` with a JSON content type.
pub async fn call(app: &Router, method: Method, uri: &str, body: Vec<u8>) -> Answer {
    let request = axum::http::Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    Answer {
        status,
        headers,
        json: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    }
}
