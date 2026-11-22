//! Switches the model at runtime.
//!
//! A switch can mean a 320 MB download, so it never runs on the UI thread: a request spawns a
//! thread, which publishes progress through [`SwitchState`] and hands the queue a closure that
//! builds the model. The queue's worker runs it between jobs, so a switch during a separation takes
//! effect once the current track is done.
//! The switch thread downloads but never builds: MLX weights must be allocated on the thread that
//! evaluates them, so the separation worker does the last step. See
//! [`crate::queue::BuildSeparator`].
//!
//! Nothing here cancels. A second request while a switch runs is refused.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use parking_lot::Mutex;

use crate::ident;
use crate::models::{self, Preset};
use crate::precision::Precisions;
use crate::queue::Queue;
use crate::settings::SettingsFile;

/// The live state of a model switch, shared between the window and the switch thread.
///
/// The window reads it to draw a progress dialog and to decide whether the model menu accepts
/// input.
#[derive(Debug, Clone, PartialEq)]
pub enum SwitchState {
    Idle,
    /// The bytes of the artefact downloaded so far.
    Downloading {
        preset: Preset,
        file: String,
        done: u64,
        total: u64,
    },
    /// The artefact is on disk and being hashed before use, which takes about a second.
    Verifying(Preset),
    /// Handed to the separation worker, the only thread that may build the model. This takes
    /// seconds on an idle server; on a busy one it also waits for the current track, which finishes
    /// on the old model.
    Loading(Preset),
    Failed {
        preset: Preset,
        message: String,
    },
}

impl SwitchState {
    pub const fn busy(&self) -> bool {
        matches!(
            self,
            Self::Downloading { .. } | Self::Verifying(_) | Self::Loading(_)
        )
    }
}

/// The switch inputs that stay fixed once the server is up.
pub struct SwitchConfig {
    /// The directories searched in order, as at startup: `--models`, then next to the executable,
    /// then the download cache.
    pub dirs: Vec<PathBuf>,
    pub cache: PathBuf,
    pub overlap: f32,
    /// How each model's precision is decided; see [`crate::precision::Precisions`].
    pub precisions: Precisions,
    pub offline: bool,
    /// `--demucs-model` as given. It identifies weights that are not a preset's; presets have a
    /// pinned digest instead.
    pub artefact: String,
}

pub struct Switcher {
    state: Mutex<SwitchState>,
    /// `None` when `--demucs-model` names an artefact that is not a preset, such as a hand-traced
    /// variant. Switching still works, and the window shows no preset selected.
    current: Mutex<Option<Preset>>,
    queue: Arc<Queue>,
    settings: Arc<SettingsFile>,
    config: SwitchConfig,
}

impl Switcher {
    pub fn new(
        current: Option<Preset>,
        queue: Arc<Queue>,
        settings: Arc<SettingsFile>,
        config: SwitchConfig,
    ) -> Self {
        Self {
            state: Mutex::new(SwitchState::Idle),
            current: Mutex::new(current),
            queue,
            settings,
            config,
        }
    }

    pub fn state(&self) -> SwitchState {
        self.state.lock().clone()
    }

    pub fn current(&self) -> Option<Preset> {
        *self.current.lock()
    }

    /// Every input that decides what a separation returns, for the server's own cache key.
    // A hand-traced artefact is identified by its file name, since several artefacts share one
    // manifest model name. The digest stays full length, since it feeds a hash. Clients receive
    // [`Self::published_identity`].
    pub fn identity(&self) -> String {
        identity_of(
            self.current(),
            &self.config.artefact,
            self.config.precisions,
            self.config.overlap,
        )
    }

    /// The cache's directory for the loaded model: [`Self::published_identity`] plus a
    /// digest of [`Self::identity`].
    pub fn cache_id(&self) -> String {
        format!(
            "{}-{}",
            self.published_identity(),
            ident::short_digest(&self.identity())
        )
    }

    /// The model identity as clients see and store it. It leaves out precision and overlap, so it
    /// stays the same when the server runs on another accelerator.
    // A client keys its own stem cache on it, so it becomes a directory name: the artefact name
    // plus `ident::DIGEST_CHARS` of digest, enough to tell artefacts apart. It always stays within
    // the portable charset and length of `ident`. It is narrower than `Self::identity`, whose
    // precision and overlap depend on the machine and the command line.
    pub fn published_identity(&self) -> String {
        let id = match self.current() {
            // The id uses a digest of the model identity, so it changes when the preset's
            // arrangement changes as well as when the weights file does. The form is `<artefact>-<8
            // hex>`.
            Some(preset) => format!(
                "{}-{}",
                preset.artefact(),
                ident::short_digest(&model_identity(preset))
            ),
            // `--demucs-model` is arbitrary input, so it is sanitised and shortened before use.
            None => ident::tagged("custom", &self.config.artefact),
        };
        debug_assert!(ident::is_portable_id(&id), "unportable model_id: {id}");
        id
    }

    /// True when the artefact is on disk, so switching is instant and needs no network. The window
    /// uses this to decide whether to warn first.
    pub fn is_local(&self, preset: Preset) -> bool {
        models::locate(&self.config.dirs, preset).is_some()
    }

    /// Clear a failure so the menu accepts input again.
    pub fn dismiss_error(&self) {
        let mut state = self.state.lock();
        if matches!(*state, SwitchState::Failed { .. }) {
            *state = SwitchState::Idle;
        }
    }

    /// Start a switch. Return false if a switch is already running or the preset is already loaded.
    pub fn request(self: &Arc<Self>, preset: Preset) -> bool {
        {
            let state = self.state.lock();
            if state.busy() {
                return false;
            }
        }
        if self.current() == Some(preset) {
            return false;
        }

        let this = Arc::clone(self);
        std::thread::Builder::new()
            .name("stemd-switch".into())
            .spawn(move || {
                if let Err(err) = this.run(preset) {
                    tracing::error!("switch to {:?} failed: {err:#}", preset);
                    *this.state.lock() = SwitchState::Failed {
                        preset,
                        message: format!("{err:#}"),
                    };
                } else {
                    *this.state.lock() = SwitchState::Idle;
                }
            })
            .expect("spawning the model switch thread");
        true
    }

    /// Make the artefact available and have the separation worker load it.
    fn run(&self, preset: Preset) -> anyhow::Result<()> {
        let dir = self.stage_artefact(preset)?;

        *self.state.lock() = SwitchState::Loading(preset);
        // This blocks until the worker has swapped models, so the steps below happen only after a
        // real swap: the queue publishes what the model reports about itself, and a failed build
        // returns here and becomes `Failed`.
        self.queue.install(crate::models::builder(
            &dir,
            preset.artefact(),
            Some(preset),
            self.config.overlap,
            self.config.precisions,
        ))?;

        *self.current.lock() = Some(preset);
        // Save the preset only after it loads; otherwise a preset that cannot load on this machine
        // would be the first one every later launch tries.
        self.settings.set_preset(preset);
        tracing::info!(
            "model switch complete: {} ({:?})",
            preset.artefact(),
            preset
        );
        Ok(())
    }

    /// The directory holding a verified copy of `preset`, downloaded first if this machine has
    /// none.
    fn stage_artefact(&self, preset: Preset) -> anyhow::Result<PathBuf> {
        if let Some(dir) = models::locate(&self.config.dirs, preset) {
            *self.state.lock() = SwitchState::Verifying(preset);
            models::verify(&dir, &self.config.cache, preset, self.config.offline)?;
            return Ok(dir);
        }

        if self.config.offline {
            anyhow::bail!("{} not installed; --offline set", preset.artefact());
        }

        *self.state.lock() = SwitchState::Downloading {
            preset,
            file: String::new(),
            done: 0,
            total: preset.total_bytes(),
        };
        models::fetch(&self.config.cache, preset.source(), |file, done, total| {
            *self.state.lock() = SwitchState::Downloading {
                preset,
                file: file.to_owned(),
                done,
                total,
            };
        })
        .context("fetching the model")?;
        Ok(self.config.cache.clone())
    }
}

/// The published model identity: the weights digest and the preset's recipe, with nothing about the
/// run. See [`Switcher::published_identity`].
fn model_identity(preset: Preset) -> String {
    format!("m={};r={}", preset.digest(), preset.recipe())
}

/// The cache identity: `m=` (the model), `p=` (precisions) and `o=` (overlap), `;`-separated.
///
/// Every input that changes the samples belongs here, since a cache hit is served without
/// separating again. Append new fields at the end: only `m=` holds free-form text, so parsing from
/// the right stays unambiguous.
fn identity_of(
    model: Option<Preset>,
    artefact: &str,
    precisions: Precisions,
    overlap: f32,
) -> String {
    let m = match model {
        Some(preset) => model_identity(preset),
        None => format!("m=custom:{artefact}"),
    };
    // `{}` on an f32 prints the shortest decimal that round-trips, so two overlaps one ulp apart
    // give different keys; a fixed number of decimal places would round neighbouring values to one
    // key.
    //
    // The precision field is one value when every model in the preset agrees; only a backend that
    // wants different precisions for different models produces the joined form.
    format!("{m};p={};o={overlap}", precisions.key(model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use stemd_core::{Accelerator, Precision};

    /// A separation, a real preset switch, and a second separation, on this machine's GPU. It
    /// catches a model built on one thread and evaluated on another, which Metal allows and CUDA
    /// does not.
    // Ignored by default: it needs several hundred megabytes of weights and a working backend. The
    // unit tests in `crate::queue` check the threading arrangement; this test checks that the
    // arrangement is enough, through both paths that load a model.
    //
    //     STEMD_AB_MLX=<dir with every artefact> \
    //     cargo test --release -p stemd-server -- --ignored across_a_preset_switch --nocapture
    #[test]
    #[ignore]
    fn separating_works_across_a_preset_switch() {
        use crate::jobs::Jobs;
        use crate::output::Output;
        use crate::queue::{Queue, QueuedWork};
        use stemd_core::{Audio, DspMode, OutputRate, StemFormat};

        let Ok(dir) = std::env::var("STEMD_AB_MLX") else {
            eprintln!("SKIPPED: set STEMD_AB_MLX to a directory holding the artefacts.");
            return;
        };
        let dir = PathBuf::from(dir);
        let cache = crate::testkit::cache(u64::MAX, std::time::Duration::from_secs(600));
        let jobs = Jobs::new(std::time::Duration::from_secs(600));

        // Eight seconds of tone and noise: long enough to cross a segment boundary, short enough
        // that Quality finishes quickly. A bass note, a chord, a vibrato lead and a noise burst
        // twice a second give each branch something to find. The two channels differ: a mix with
        // identical channels has no side content, and the BS-RoFormer standardises a band of it to
        // NaN.
        let hz = 44100;
        let tau = 2.0 * std::f32::consts::PI;
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut noise = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // 24 bits over 2^23 gives a value in [-1, 1). A wrong divisor gives samples in the
            // hundreds, which htdemucs standardises away and the BS-RoFormer turns into NaN.
            (seed >> 40) as f32 / 8_388_608.0 - 1.0
        };
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for i in 0..hz * 8 {
            let t = i as f32 / hz as f32;
            let bass = 0.28 * (tau * 82.41 * t).sin();
            let chord = 0.16 * (tau * 329.63 * t).sin() + 0.13 * (tau * 493.88 * t).sin();
            let hit = 0.45 * (-28.0 * ((t * 2.0) % 1.0)).exp() * noise();
            let lead = 0.12 * (tau * (660.0 + 8.0 * (tau * 5.0 * t).sin()) * t).sin();
            left.push(0.7 * (bass + chord + hit + lead));
            right.push(0.7 * (bass + 0.9 * chord + 0.8 * hit + 1.1 * lead));
        }
        let mix = Audio::new(vec![left, right], hz as u32);

        let separate = |queue: &Queue, key: &str| {
            let job = jobs.create(key.to_owned());
            queue
                .submit(QueuedWork {
                    job: Arc::clone(&job),
                    mix: mix.clone(),
                    output: Output {
                        format: StemFormat::Pcm16,
                        rate: OutputRate::default(),
                        derived: false,
                        dsp: DspMode::default().into(),
                        parts: stemd_core::Parts::DEFAULT,
                    },
                    cache_id: "test".into(),
                    from_api: true,
                })
                .expect("queue has room");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(900);
            while !job.progress.lock().stage.is_terminal() {
                assert!(std::time::Instant::now() < deadline, "{key} never finished");
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            assert!(
                job.error.lock().is_none(),
                "{key} failed: {:?}",
                job.error.lock()
            );
            let entry = job.entry.lock().clone().expect("a completed job has stems");
            let names: Vec<String> = entry.stems.iter().map(|s| s.name.clone()).collect();
            assert_eq!(names, stemd_core::SHIPPED, "{key} shipped {names:?}");
            println!(
                "  {key}: {} stems, residual {:.1} dB, {:.1}x realtime",
                names.len(),
                entry.model_residual_db.unwrap_or(f64::NAN),
                entry.realtime_factor()
            );
        };

        let from = Preset::Fast;
        let to = Preset::Quality;
        let (queue, info) = Queue::start(
            crate::models::builder(
                &dir,
                from.artefact(),
                Some(from),
                0.25,
                Precisions::stated(None, Accelerator::Metal),
            ),
            8,
            cache,
        )
        .unwrap_or_else(|e| panic!("loading {}: {e:#}", from.artefact()));
        println!("started on {} via {}", info.model, info.backend);
        let queue = Arc::new(queue);
        separate(&queue, "before-the-switch");

        let scratch = crate::scratch::Dir::new("switch-test");
        let settings = Arc::new(crate::settings::SettingsFile::open(
            scratch.join("settings.json"),
        ));
        let switcher = Arc::new(Switcher::new(
            Some(from),
            Arc::clone(&queue),
            settings,
            SwitchConfig {
                dirs: vec![dir.clone()],
                cache: dir,
                overlap: 0.25,
                precisions: Precisions::stated(None, Accelerator::Metal),
                // Every artefact is already on disk and the download is not under test, so a
                // missing file fails the test.
                offline: true,
                artefact: from.artefact().to_owned(),
            },
        ));
        assert!(switcher.request(to), "the switch was refused");

        // Wait on `current`: `request` returns before its thread sets a state, so `busy` would read
        // `Idle` as finished.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(900);
        while switcher.current() != Some(to) {
            if let SwitchState::Failed { message, .. } = switcher.state() {
                panic!("switching to {to:?} failed: {message}");
            }
            assert!(std::time::Instant::now() < deadline, "the switch hung");
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert_eq!(
            switcher.state(),
            SwitchState::Idle,
            "the switch did not complete"
        );
        println!("switched to {}", queue.info().model);
        assert_ne!(
            queue.info().model,
            info.model,
            "the queue still reports the model it started on"
        );

        separate(&queue, "after-the-switch");
    }

    /// Every id a client can receive passes its sanitiser unchanged, so the stored value matches
    /// what the server reported.
    #[test]
    fn every_preset_publishes_a_portable_identity() {
        for preset in Preset::ALL {
            let digest = preset.digest();
            let id = format!(
                "{}-{}",
                preset.artefact(),
                &digest[..ident::DIGEST_CHARS.min(digest.len())]
            );
            assert!(
                ident::is_portable_id(&id),
                "{:?} publishes {id:?} ({} chars), which a client would rewrite",
                preset,
                id.len()
            );
        }
    }

    /// A cache hit is served without separating again, so precision and overlap each change the
    /// key.
    #[test]
    fn precision_and_overlap_each_change_the_identity() {
        let at = |forced, overlap| {
            identity_of(
                Some(Preset::Fast),
                "unused",
                Precisions::stated(forced, Accelerator::Metal),
                overlap,
            )
        };
        assert_ne!(
            at(None, 0.25),
            at(Some(Precision::F32), 0.25),
            "precision is not in the key"
        );
        assert_ne!(at(None, 0.25), at(None, 0.30), "overlap is not in the key");
        // The same configuration still gives the same string, or the cache would never hit.
        assert_eq!(at(None, 0.25), at(None, 0.25));
    }

    /// The id a client stores stays the same on another accelerator, or on a machine where a
    /// library failed to resolve.
    // It changes with the weights and the arrangement only; the server's own key changes with every
    // input.
    #[test]
    fn what_is_published_survives_a_change_of_machine() {
        for preset in Preset::ALL {
            let published = model_identity(preset);
            for accelerator in [Accelerator::Metal, Accelerator::Cuda, Accelerator::Cpu] {
                for forced in [None, Some(Precision::F32)] {
                    for overlap in [0.25f32, 0.30, 0.5] {
                        let precisions = Precisions::stated(forced, accelerator);
                        // The server's own key changes with every input, since the audio differs.
                        let internal = identity_of(Some(preset), "unused", precisions, overlap);
                        assert!(
                            internal.starts_with(&published),
                            "the internal key stopped covering the model: {internal}"
                        );
                        // The id the client sees does not change.
                        assert_eq!(
                            model_identity(preset),
                            published,
                            "{preset:?} publishes a different id on {accelerator:?}"
                        );
                    }
                }
            }
        }

        // It still tells different presets apart.
        assert_ne!(
            model_identity(Preset::Fast),
            model_identity(Preset::Balanced),
            "two presets share one published id"
        );
    }

    /// docs/api.md shows a `model_id` for a default server, which is a digest of [`identity_of`],
    /// so a new field there changes the documented example. This test fixes the value;
    /// `every_preset_publishes_a_portable_identity` checks the form.
    #[test]
    fn the_documented_model_id_is_what_a_default_server_reports() {
        // Derive the id as `published_identity` does, without the server's own cache key.
        let id = format!(
            "{}-{}",
            models::DEFAULT_PRESET.artefact(),
            ident::short_digest(&model_identity(models::DEFAULT_PRESET))
        );
        assert_eq!(
            id, "htdemucs-22c7f2c9",
            "docs/api.md says htdemucs-22c7f2c9 and this server reports {id}"
        );
    }

    /// No two configurations share a key. The test asserts that distinct inputs give distinct
    /// strings, so a new field is covered by adding inputs, with no expected outputs to rewrite.
    #[test]
    fn distinct_configurations_never_share_an_identity() {
        // 0.25 and its neighbour one ulp up. A key that rounded the overlap to a fixed number of
        // places would serve one separation's stems for the other.
        let overlaps = [0.25f32, 0.5, f32::from_bits(0.25f32.to_bits() + 1)];
        // `--demucs-model` reaches this unfiltered, so the crafted names matter most: a name that
        // contains `;p=` could otherwise collide with a real field.
        let artefacts = ["htdemucs_seg10_mps", "a;p=f16;o=0.25", "a", ""];

        // Sweep every backend and both settings of the flag. The assertion is that an identity
        // determines the configuration; the backend is left out of the key on purpose, so not every
        // input is distinct.
        let mut seen: std::collections::HashMap<String, (String, String, String)> =
            std::collections::HashMap::new();
        let mut record = |id: String, what: (String, String, String)| {
            if let Some(previous) = seen.insert(id.clone(), what.clone()) {
                assert_eq!(previous, what, "two configurations share the identity {id}");
            }
        };

        for on in [Accelerator::Metal, Accelerator::Cuda, Accelerator::Cpu] {
            for forced in [None, Some(Precision::F32)] {
                let precisions = Precisions::stated(forced, on);
                for overlap in overlaps {
                    for preset in Preset::ALL {
                        record(
                            identity_of(Some(preset), "unused", precisions, overlap),
                            (
                                format!("{preset:?}"),
                                precisions.key(Some(preset)),
                                overlap.to_string(),
                            ),
                        );
                    }
                    for artefact in artefacts {
                        record(
                            identity_of(None, artefact, precisions, overlap),
                            (
                                format!("custom:{artefact}"),
                                precisions.key(None),
                                overlap.to_string(),
                            ),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_hand_traced_artefact_publishes_a_portable_identity() {
        // `--demucs-model` is arbitrary input and reaches this unfiltered.
        for artefact in [
            "htdemucs_seg10_mps",
            "../../etc/passwd",
            "a name with spaces and a / slash",
            "",
        ] {
            let id = ident::tagged("custom", artefact);
            assert!(ident::is_portable_id(&id), "{artefact:?} -> {id:?}");
        }
    }
}
