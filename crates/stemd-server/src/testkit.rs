//! Fixtures shared by the cache, job-store and queue tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use stemd_core::{Audio, DspMode, OutputRate, StemFormat, Stems};

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
    Cache::new(root, max_bytes, ttl).expect("cache root")
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
                dsp: DspMode::default(),
            },
            separation_secs,
        )
        .expect("publish")
}
