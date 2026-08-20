//! Changing model at runtime.
//!
//! Switching can mean a 320 MB download, so it never happens on the UI thread: a
//! request spawns a worker, publishes progress through [`SwitchState`], and hands
//! the queue a recipe. The queue's worker builds it between jobs, so a switch
//! during a separation takes effect once the track in flight is done.
//!
//! This thread fetches but never builds: MLX weights have to be allocated on the
//! thread that will evaluate them, so the last step belongs to the separation
//! worker. See [`crate::queue::BuildSeparator`].
//!
//! Nothing here cancels. A second request while one is running is refused rather
//! than queued.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use parking_lot::Mutex;

use crate::ident;
use crate::models::{self, Preset};
use crate::precision::Precisions;
use crate::queue::Queue;
use crate::settings::SettingsStore;

/// Live state of a model switch, shared between the window and the thread doing
/// the work.
///
/// The window reads this to draw a progress dialog and to decide whether the
/// model menu is interactive.
#[derive(Debug, Clone, PartialEq)]
pub enum SwitchState {
    Idle,
    /// Bytes of the artefact fetched so far.
    Downloading {
        preset: Preset,
        file: String,
        done: u64,
        total: u64,
    },
    /// Present already; hashing it before trusting it. About a second.
    Verifying(Preset),
    /// Handed to the separation worker, which is the only thread that may build
    /// it. Seconds on an idle server; on a busy one it also covers the track in
    /// flight, which finishes on the old model first.
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

/// Everything a switch needs that does not change once the server is up.
pub struct SwitchConfig {
    /// Searched in order, mirroring startup: `--models`, then next to the
    /// executable, then the download cache.
    pub dirs: Vec<PathBuf>,
    pub cache: PathBuf,
    pub overlap: f32,
    /// How each model's precision is decided; see [`crate::precision::Precisions`].
    pub precisions: Precisions,
    pub offline: bool,
    /// `--demucs-model` as given. Identifies the weights when they are not one
    /// of the presets, which have a pinned digest instead.
    pub artefact: String,
}

pub struct Switcher {
    state: Mutex<SwitchState>,
    /// `None` when `--demucs-model` names an artefact that is not one of the
    /// two presets: a hand-traced variant, say. Switching still works; the
    /// window simply shows nothing selected rather than claiming otherwise.
    current: Mutex<Option<Preset>>,
    queue: Arc<Queue>,
    settings: Arc<SettingsStore>,
    config: SwitchConfig,
}

impl Switcher {
    pub fn new(
        current: Option<Preset>,
        queue: Arc<Queue>,
        settings: Arc<SettingsStore>,
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

    /// Everything that decides what a separation returns, for the server's own cache
    /// key.
    ///
    /// A hand-traced artefact is identified by its filename, not by the manifest's
    /// model name: several artefacts share one. Stays the full digest, since it is an
    /// ingredient of a hash. What goes over the wire is [`Self::published_identity`].
    pub fn identity(&self) -> String {
        identity_of(
            self.current(),
            &self.config.artefact,
            self.config.precisions,
            self.config.overlap,
        )
    }

    /// The same identity as clients see and store it.
    ///
    /// A client keys its own stem cache on this, so it becomes a directory name: the
    /// artefact name plus [`ident::DIGEST_CHARS`] of digest, which keeps the property
    /// the digest is here for. Always within [`ident`]'s portable charset and length.
    ///
    /// Deliberately narrower than [`Self::identity`], and this is the whole point
    /// of there being two. Precision and overlap change the audio, so the
    /// server's own key must carry them; but they are properties of the machine
    /// and the command line, not of the model. Folded in here they made the
    /// published id move whenever the same server came up on a different
    /// accelerator, which on a client is a whole on-media cache thrown away
    /// because a PATH changed. What a client is asking is "are these the same
    /// weights, arranged the same way", and that is what this answers.
    pub fn published_identity(&self) -> String {
        let id = match self.current() {
            // A digest *of* the model identity rather than a prefix of the
            // weights', so the id moves when the arrangement moves and not only
            // when the file does. Same shape either way: `<artefact>-<8 hex>`.
            Some(preset) => format!(
                "{}-{}",
                preset.artefact(),
                ident::short_digest(&model_identity(preset))
            ),
            // `--demucs-model` is arbitrary input, so it is sanitised and
            // shortened rather than interpolated.
            None => ident::tagged("custom", &self.config.artefact),
        };
        debug_assert!(ident::is_portable_id(&id), "unportable model_id: {id}");
        id
    }

    /// True when the artefact is already on disk, i.e. switching is instant and
    /// needs no network. The window uses this to decide whether to warn first.
    pub fn is_local(&self, preset: Preset) -> bool {
        models::locate(&self.config.dirs, preset).is_some()
    }

    /// Clear a failure so the menu becomes interactive again.
    pub fn dismiss_error(&self) {
        let mut state = self.state.lock();
        if matches!(*state, SwitchState::Failed { .. }) {
            *state = SwitchState::Idle;
        }
    }

    /// Begin switching. Returns false if a switch is already in flight or the
    /// preset is already loaded.
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
        // Blocks until the worker has actually swapped, which is what makes
        // everything below true rather than merely intended: the queue is the
        // one publishing what the model says about itself, and a build that
        // failed comes back here to become `Failed`.
        self.queue.install(crate::startup::builder(
            &dir,
            preset.artefact(),
            Some(preset),
            self.config.overlap,
            self.config.precisions,
        ))?;

        *self.current.lock() = Some(preset);
        // Only once it has loaded. Saving the request instead would make a
        // preset that cannot load on this machine the one every later launch
        // tries first.
        self.settings.set_preset(preset);
        tracing::info!(
            "model switch complete: {} ({:?})",
            preset.artefact(),
            preset
        );
        Ok(())
    }

    /// The directory holding a verified copy of `preset`, downloading it first
    /// if this machine does not have one.
    fn stage_artefact(&self, preset: Preset) -> anyhow::Result<PathBuf> {
        if let Some(dir) = models::locate(&self.config.dirs, preset) {
            *self.state.lock() = SwitchState::Verifying(preset);
            models::verify(&dir, &self.config.cache, preset, self.config.offline)?;
            return Ok(dir);
        }

        if self.config.offline {
            anyhow::bail!(
                "{} is not installed and --offline forbids fetching it",
                preset.artefact()
            );
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

/// The cache identity, built from its parts.
///
/// Free-standing because a [`Switcher`] owns a queue and a loaded separator, and
/// the thing worth testing here is the string.
///
/// **Every input that changes the samples belongs in here.** [`crate::cache`]
/// serves a hit without re-separating, which is only sound while this is
/// complete. Precision and overlap both change the audio.
///
/// A `;`-separated `k=v` list, appended to rather than reshaped. Appending is safe
/// because `m=` carries the only value not drawn from a closed set:
/// `--demucs-model` is arbitrary operator input, and every field after it is
/// `;`-free, so reading from the right is unambiguous. A second free-form field
/// would have to be length-prefixed the way [`crate::cache::key`] prefixes its
/// ingredients.
/// Which weights, and what is done with them, and nothing else.
///
/// The second half matters because a preset can change how it uses a file
/// without the file changing, and cached stems from before that are no longer
/// what a fresh separation would produce. Nothing about the run belongs here:
/// see [`Switcher::published_identity`] for why the two identities differ.
fn model_identity(preset: Preset) -> String {
    format!("m={};r={}", preset.digest(), preset.recipe())
}

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
    //  `{}` on an f32 prints the shortest decimal that round-trips, so two overlaps
    //  differing by one ulp differ here. A fixed number of places would round
    //  neighbouring values onto one key.
    //  One value when every model in the preset agrees; only a backend wanting
    //  different precisions from different models produces the joined form.
    format!("{m};p={};o={overlap}", precisions.key(model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use stemd_core::{Accelerator, Precision};

    /// A separation, a real preset switch, and a second separation, on the GPU this
    /// machine actually has.
    ///
    /// Ignored because it wants several hundred megabytes of weights and a working
    /// backend. It covers a bug nothing single-threaded can catch: a model built on
    /// one thread and evaluated on another, which Metal permits and CUDA does not.
    /// The unit tests in [`crate::queue`] pin the arrangement; this pins that the
    /// arrangement is enough. Both paths that load a model are covered.
    ///
    /// ```text
    /// STEMD_AB_MLX=<dir with every artefact> \
    /// cargo test --release -p stemd-server -- --ignored across_a_preset_switch --nocapture
    /// ```
    #[test]
    #[ignore]
    fn separating_works_across_a_preset_switch() {
        use crate::cache::Output;
        use crate::jobs::JobStore;
        use crate::queue::{Queue, QueuedWork};
        use stemd_core::{Audio, DspMode, OutputRate, StemFormat};

        let Ok(dir) = std::env::var("STEMD_AB_MLX") else {
            eprintln!("SKIPPED: set STEMD_AB_MLX to a directory holding the artefacts.");
            return;
        };
        let dir = PathBuf::from(dir);
        let cache = crate::testkit::cache(u64::MAX, std::time::Duration::from_secs(600));
        let store = JobStore::new(std::time::Duration::from_secs(600));

        //  Eight seconds of tone and noise: long enough to cross a segment boundary,
        //  short enough that Quality is not a coffee break. A bass note, a chord, a
        //  vibrato lead and a twice-a-second noise burst give each branch something to
        //  find. Not the same samples twice: a mix whose channels are identical has no
        //  side content, and the BS-RoFormer standardises a band of it to NaN.
        let hz = 44100;
        let tau = 2.0 * std::f32::consts::PI;
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut noise = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // 24 bits over 2^23, so this lands in [-1, 1). Getting the divisor
            // wrong here is not a quieter test signal, it is samples in the
            // hundreds, which htdemucs standardises away and the BS-RoFormer
            // turns into NaN.
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
            let job = store.create(key.to_owned());
            queue
                .submit(QueuedWork {
                    job: Arc::clone(&job),
                    mix: mix.clone(),
                    output: Output {
                        format: StemFormat::Pcm16,
                        rate: OutputRate::default(),
                        derived: false,
                        dsp: DspMode::default(),
                    },
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
                entry.model_residual_db,
                entry.realtime_factor()
            );
        };

        let from = Preset::Fast;
        let to = Preset::Quality;
        let (queue, info) = Queue::start(
            crate::startup::builder(
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

        let settings = Arc::new(crate::settings::SettingsStore::open(
            std::env::temp_dir().join(format!("stemd-switch-test-{}.json", std::process::id())),
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
                // Everything is on disk already; a download is not what is
                // under test and its absence must fail rather than fetch.
                offline: true,
                artefact: from.artefact().to_owned(),
            },
        ));
        assert!(switcher.request(to), "the switch was refused");

        // Waiting on `current` rather than on `busy`, because `request` returns
        // before its thread has set a state and `Idle` would read as finished.
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

    /// Every id a client can be handed has to survive its sanitiser unchanged,
    /// or the value it stores differs from the value the server reported.
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

    /// The property the README's word "provably" rests on. Both of these were
    /// absent from the key, and a hit is served without re-separating: stems
    /// made at f16 came back unchanged from a server restarted with
    /// `--full-precision`, and likewise for a changed `--overlap`.
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
        // And the same configuration still lands on the same string, or the
        // cache would never hit at all.
        assert_eq!(at(None, 0.25), at(None, 0.25));
    }

    /// The other half of the pair above, and the reason there are two
    /// identities rather than one.
    ///
    /// What a client stores has to survive the server coming up on a different
    /// accelerator, or on a machine where a library happened not to resolve. It
    /// changes for the weights and for the arrangement, and for nothing else.
    #[test]
    fn what_is_published_survives_a_change_of_machine() {
        for preset in Preset::ALL {
            let published = model_identity(preset);
            for accelerator in [Accelerator::Metal, Accelerator::Cuda, Accelerator::Cpu] {
                for forced in [None, Some(Precision::F32)] {
                    for overlap in [0.25f32, 0.30, 0.5] {
                        let precisions = Precisions::stated(forced, accelerator);
                        // The server's own key moves with all of it, as it must:
                        // these are genuinely different audio.
                        let internal = identity_of(Some(preset), "unused", precisions, overlap);
                        assert!(
                            internal.starts_with(&published),
                            "the internal key stopped covering the model: {internal}"
                        );
                        // What the client sees does not.
                        assert_eq!(
                            model_identity(preset),
                            published,
                            "{preset:?} publishes a different id on {accelerator:?}"
                        );
                    }
                }
            }
        }

        // And it still separates the things it is for.
        assert_ne!(
            model_identity(Preset::Fast),
            model_identity(Preset::Balanced),
            "two presets share one published id"
        );
    }

    /// docs/api.md prints a `model_id` for a default server, which is a digest of
    /// [`identity_of`], so every field added there falsifies the documented example.
    /// This pins the value; the shape is `every_preset_publishes_a_portable_identity`.
    #[test]
    fn the_documented_model_id_is_what_a_default_server_reports() {
        //  Derived the way `published_identity` derives it, not the way the
        //  server's own cache key is built: this test asserted the latter for a
        //  while after the two were separated, so it went on passing while
        //  agreeing with nothing the server said.
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

    /// No two configurations may share a key. Stated as "distinct inputs,
    /// distinct strings" rather than as a table of expected outputs, so the
    /// next field is covered by widening the inputs rather than by rewriting
    /// the expectations.
    #[test]
    fn distinct_configurations_never_share_an_identity() {
        // 0.25 and its neighbour one ulp up. A key that rounded the overlap to
        // a fixed number of places would serve one separation's stems for the
        // other.
        let overlaps = [0.25f32, 0.5, f32::from_bits(0.25f32.to_bits() + 1)];
        // `--demucs-model` reaches this unfiltered, so the crafted names are
        // the interesting half: one spelled as the fields that follow it is the
        // shape a collision would take.
        let artefacts = ["htdemucs_seg10_mps", "a;p=f16;o=0.25", "a", ""];

        //  Swept over every backend and both settings of the flag. What is asserted is
        //  that an identity determines the configuration, not that every input is
        //  distinct: the backend is deliberately absent from the key.
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
