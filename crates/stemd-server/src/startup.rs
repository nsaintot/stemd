//! Turning parsed arguments into a running set of services.
//!
//! Everything between "the command line is understood" and "the router can be
//! mounted": locating and loading the model, opening the stem cache, starting
//! the worker and the reaper, and advertising over mDNS.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use stemd_core::{
    HybridConfig, HybridSeparator, MlxConfig, MlxSeparator, OutputRate, PcmFormat,
    RoformerSeparator, Separate, StemFormat,
};

use crate::api::AppState;
use crate::cache::{self, Cache};
use crate::cli::Args;
use crate::jobs::JobStore;
use crate::launch::{Launch, Stage};
use crate::logbuf::LogBuffer;
use crate::models::{self, DEFAULT_PRESET, Preset};
use crate::precision::Precisions;
use crate::queue::{BuildSeparator, Queue};
use crate::settings::{self, SettingsStore};
use crate::switch::{self, Switcher};

/// A server wired up and ready to bind.
pub struct Server {
    pub state: Arc<AppState>,
    /// Largest request body axum will buffer, derived from `--max-track-minutes`.
    pub max_upload: usize,
    /// One line describing where stems go, for the startup log.
    pub cache_summary: String,
}

/// Build every service the router depends on.
///
/// The stem cache is opened before the model because the worker that loads the
/// model needs it. See [`crate::queue::BuildSeparator`].
///
/// `addr` is where the listener is already bound, not where it was asked to be, so
/// it is the real port under `--bind :0`. Progress is reported through `launch`.
pub fn prepare(args: &Args, logs: LogBuffer, addr: SocketAddr, launch: &Launch) -> Result<Server> {
    // Asked once. The accelerator cannot change under a running server, and
    // every model loaded from here on is chosen against the same answer.
    let precisions = Precisions::detect(args.forced_precision());
    // Said here rather than in `log_model`, which runs after a model has been
    // loaded at CPU speed. A card whose runtime is missing looks exactly like
    // no card in every later line, and only one of the two has a fix.
    if stemd_core::Accelerator::gpu_refused() {
        if cfg!(windows) {
            tracing::warn!(
                "a GPU is installed but its runtime is not usable from here, so this \
                 runs on the CPU. Run with --install-cuda to fetch it (about 1.2 GB, \
                 once), or put the libraries beside the executable yourself."
            );
        } else {
            tracing::warn!(
                "a GPU is installed but failed a test computation, so this runs on the CPU."
            );
        }
    }
    let settings = Arc::new(open_settings(args)?);
    let (stems, cache_summary) = open_stem_cache(args)?;
    // Failures hold no stems, so the cache cannot decide when to forget them.
    // The same clock that reaps an uncollected separation is the right one.
    let store = Arc::new(JobStore::new(args.unfetched_ttl()));

    let (model, queue, info) = load_model(args, &settings, &stems, precisions, launch)?;
    queue.sessions().set_limits(
        std::time::Duration::from_secs(args.session_idle_secs),
        args.max_sessions,
    );
    queue.sessions().set_pieces(open_pieces(args)?);
    spawn_session_reaper(Arc::clone(&queue));
    let max_upload = max_body_bytes(args.max_track_secs(), highest_rate(), info.channels);
    log_model(&info, precisions, args.max_track_secs(), max_upload);

    // Built before the app state so every exit path can reach it. Headless
    // there is no switch to turn it back on, so it always serves.
    let serving = crate::serving::Serving::new(
        args.headless || settings.get().serving,
        announce(args, addr.port()),
        &info.model,
    );

    // Both identities, once, at debug. The next time the published one moves
    // when it should not, the question is which field did it, and that is not
    // answerable from an eight-character digest.
    let switcher = Arc::new(Switcher::new(
        model.preset,
        Arc::clone(&queue),
        Arc::clone(&settings),
        switch::SwitchConfig {
            dirs: model.search_dirs,
            cache: model.download_cache,
            overlap: args.overlap,
            precisions,
            offline: args.offline,
            artefact: model.artefact,
        },
    ));
    tracing::debug!(
        "cache key: {} | published as {}",
        switcher.identity(),
        switcher.published_identity()
    );

    spawn_reaper(Arc::clone(&stems), Arc::clone(&store), Arc::clone(&queue));

    Ok(Server {
        state: Arc::new(AppState {
            store,
            cache: stems,
            queue,
            switcher,
            logs,
            drops: Arc::default(),
            settings,
            max_track_secs: args.max_track_secs(),
            serving,
        }),
        max_upload,
        cache_summary,
    })
}

/// Open the settings file and apply the flags that override it for this run.
///
/// The flags are parsed here rather than where they are used so a typo fails at
/// launch with the flag quoted, instead of on the first request that would have
/// needed it.
fn open_settings(args: &Args) -> Result<SettingsStore> {
    let path = match args.settings.clone() {
        Some(path) => path,
        None => settings::default_path()?,
    };
    let format = args
        .output_format
        .as_deref()
        .map(|given| {
            given
                .parse::<StemFormat>()
                .with_context(|| format!("bad --output-format {given:?}"))
        })
        .transpose()?;
    let rate = args
        .output_sample_rate
        .as_deref()
        .map(|given| {
            given
                .parse::<OutputRate>()
                .with_context(|| format!("bad --output-sample-rate {given:?}"))
        })
        .transpose()?;

    Ok(SettingsStore::open(path).pin(format, rate))
}

/// Where the model was found, and what the switcher needs to replace it later.
///
/// It holds no separator: the model itself belongs to the queue's worker, which
/// is the only thread allowed to have built it.
struct LoadedModel {
    /// `None` when the artefact is not one of the presets.
    preset: Option<Preset>,
    /// What was actually loaded, which is not always what was asked for.
    artefact: String,
    search_dirs: Vec<PathBuf>,
    download_cache: PathBuf,
}

/// Load the model this run should use: `--demucs-model` if given, otherwise
/// whatever the window last selected.
///
/// A saved preset that will not load falls back to the default rather than failing
/// the launch, since the window is the only way to choose differently. A model
/// named on the command line is not covered.
///
/// Behind a window, a saved preset that is not on disk is not fetched: the window
/// asks which model to download, and the answer is saved once it has loaded.
///
/// [`Queue::start`] does not return until the worker has a model or has failed to
/// get one.
fn load_model(
    args: &Args,
    settings: &SettingsStore,
    stems: &Arc<Cache>,
    precisions: Precisions,
    launch: &Launch,
) -> Result<(LoadedModel, Arc<Queue>, stemd_core::BackendInfo)> {
    let saved = settings.get().preset;
    if args.demucs_model.is_none() && launch.asks() && !args.offline {
        let dirs = search_dirs(&args.models, &models::cache_dir()?);
        if models::locate(&dirs, saved).is_none() {
            let installed = Preset::ALL
                .into_iter()
                .filter(|p| models::locate(&dirs, *p).is_some())
                .collect();
            tracing::info!("no {} model on disk; choose one", saved.label());
            let chosen = launch.choose(installed);
            let loaded = load_artefact(args, chosen.artefact(), stems, precisions, launch)?;
            settings.set_preset(chosen);
            return Ok(loaded);
        }
    }
    let Some(requested) = args.demucs_model.clone() else {
        return load_artefact(args, saved.artefact(), stems, precisions, launch).or_else(|err| {
            if saved == DEFAULT_PRESET {
                return Err(err);
            }
            tracing::warn!(
                "the saved model {} will not load ({err:#}); starting on {} instead",
                saved.artefact(),
                DEFAULT_PRESET.artefact()
            );
            load_artefact(args, DEFAULT_PRESET.artefact(), stems, precisions, launch)
        });
    };
    load_artefact(args, &requested, stems, precisions, launch)
}

fn load_artefact(
    args: &Args,
    artefact: &str,
    stems: &Arc<Cache>,
    precisions: Precisions,
    launch: &Launch,
) -> Result<(LoadedModel, Arc<Queue>, stemd_core::BackendInfo)> {
    let preset = Preset::from_artefact(artefact);
    let shown = preset.map_or_else(|| artefact.to_owned(), |p| p.label().to_owned());
    let download_cache = models::cache_dir()?;
    let search_dirs = search_dirs(&args.models, &download_cache);
    let dir = resolve_models(
        &search_dirs,
        &download_cache,
        preset,
        artefact,
        args.offline,
        launch,
    )?;

    // Starting the worker *is* loading the model, it builds what it will run.
    launch.set(Stage::Loading(shown));
    let (queue, info) = Queue::start(
        builder(&dir, artefact, preset, args.overlap, precisions),
        args.queue_depth,
        Arc::clone(stems),
    )?;

    Ok((
        LoadedModel {
            preset,
            artefact: artefact.to_owned(),
            search_dirs,
            download_cache,
        },
        Arc::new(queue),
        info,
    ))
}

/// [`build`] as a recipe, for whichever thread is going to run the model.
///
/// The arguments are copied rather than borrowed because the closure crosses a
/// thread boundary. Both callers that install a model, the launch and the switch,
/// go through here. See [`crate::queue::BuildSeparator`].
pub fn builder(
    dir: &Path,
    artefact: &str,
    preset: Option<Preset>,
    overlap: f32,
    precisions: Precisions,
) -> BuildSeparator {
    let (dir, artefact) = (dir.to_path_buf(), artefact.to_owned());
    Box::new(move || {
        build(&dir, &artefact, preset, overlap, precisions)
            .with_context(|| format!("loading {artefact} from {}", dir.display()))
    })
}

/// Turn a directory and an artefact name into a loaded model.
///
/// Private, and reached only through [`builder`].
///
/// Which architecture is decided by the preset rather than by sniffing the file,
/// because only a preset can say that two artefacts belong together.
fn build(
    dir: &Path,
    artefact: &str,
    preset: Option<Preset>,
    overlap: f32,
    precisions: Precisions,
) -> Result<Box<dyn Separate>> {
    // Precision is asked per model family; see `crate::precision`.
    let families = Preset::families(preset);
    match preset {
        Some(Preset::Quality) => Ok(Box::new(RoformerSeparator::load(
            dir,
            Preset::Quality.artefact(),
            precisions.of(families[0]),
            overlap,
        )?)),
        // Two of the artefact's four models, not all four: it only ever needed
        // three, and two is both cheaper and better. Not chained: this tier's
        // vocals half is a demucs specialist, and handing on what it leaves
        // behind measured *worse* on every track tried.
        Some(Preset::Balanced) => Ok(Box::new(HybridSeparator::demucs_specialists(
            dir,
            artefact,
            HybridConfig {
                overlap,
                vocals_precision: precisions.of(families[0]),
                drums_precision: precisions.of(families[0]),
                cascade: false,
            },
        )?)),
        _ => Ok(Box::new(MlxSeparator::load(
            dir,
            artefact,
            MlxConfig {
                overlap,
                precision: precisions.of(families[0]),
            },
        )?)),
    }
}

/// Directories searched for a model artefact, in order.
///
/// An `.app` is launched with the working directory set to `/`, so a relative
/// `--models` cannot resolve against the cwd. The download cache comes last, so a
/// local copy wins during development.
fn search_dirs(arg: &Path, cache: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![arg.to_path_buf()];
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        dirs.push(dir.join(arg));
        dirs.push(dir.join("../Resources").join(arg));
    }
    dirs.push(cache.to_path_buf());
    // `join` with an absolute argument returns it unchanged, so an absolute
    // --models lands in the list three times and every error message repeats
    // it. Order matters, so dedupe in place rather than sorting.
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|d| seen.insert(d.clone()));
    dirs
}

/// Find the artefact, downloading it if this is a first run.
///
/// A preset is looked for by every file it needs, not just the one that names
/// it: `Quality` is two artefacts, and a directory holding one of them is not
/// a directory it can be loaded from.
fn resolve_models(
    dirs: &[PathBuf],
    cache: &Path,
    preset: Option<Preset>,
    artefact: &str,
    offline: bool,
    launch: &Launch,
) -> Result<PathBuf> {
    let marker = format!("{artefact}{}", models::WEIGHTS_EXTENSION);
    let present = |dir: &Path| match preset {
        Some(preset) => models::is_complete(dir, preset.source()),
        None => dir.join(&marker).is_file(),
    };
    if let Some(dir) = dirs.iter().find(|d| present(d)) {
        // Presence is not enough: a cached artefact can rot after the download
        // that verified it, and loading damaged weights is not a failure mode
        // worth saving a second on.
        if let Some(preset) = preset {
            launch.set(Stage::Verifying(preset));
            models::verify(dir, cache, preset, offline)?;
        }
        return Ok(dir.canonicalize().unwrap_or_else(|_| dir.clone()));
    }

    // Only a known preset can be fetched; anything else has no pinned URL.
    let Some(preset) = preset else {
        anyhow::bail!(
            "no {marker} in any of {} — it is not one of the built-in presets, \
             so it cannot be downloaded; put the weights in one of those \
             directories yourself",
            joined(dirs)
        );
    };
    if offline {
        anyhow::bail!("{}", offline_refusal(dirs, cache, preset));
    }
    fetch_preset(cache, preset, launch)
}

/// Why an absent model cannot be obtained while `--offline` is in force.
fn offline_refusal(dirs: &[PathBuf], cache: &Path, preset: Preset) -> String {
    let missing: Vec<&str> = preset
        .source()
        .files
        .iter()
        .filter(|f| !cache.join(f.name).is_file())
        .map(|f| f.name)
        .collect();
    format!(
        "no model in {} and {} is missing {} — --offline forbids fetching it",
        joined(dirs),
        cache.display(),
        missing.join(", ")
    )
}

/// Download a preset into the cache and confirm it landed.
///
/// Anything already there is skipped, so a machine with Balanced installed
/// pays only for Quality's other half.
fn fetch_preset(cache: &Path, preset: Preset, launch: &Launch) -> Result<PathBuf> {
    let wanted: u64 = preset
        .source()
        .files
        .iter()
        .filter(|f| !cache.join(f.name).is_file())
        .map(|f| f.bytes)
        .sum();
    tracing::info!(
        "first run: fetching {} ({:.0} MB) into {}",
        preset.artefact(),
        wanted as f64 / 1e6,
        cache.display()
    );
    launch.set(Stage::Downloading {
        preset,
        file: String::new(),
        done: 0,
        total: wanted,
    });
    models::fetch(cache, preset.source(), |name, done, total| {
        let pct = if total > 0 { done * 100 / total } else { 0 };
        tracing::info!(
            "  {name}: {pct}% ({:.0}/{:.0} MB)",
            done as f64 / 1e6,
            total as f64 / 1e6
        );
        launch.set(Stage::Downloading {
            preset,
            file: name.to_owned(),
            done,
            total,
        });
    })
    .context("fetching the model")?;

    if !models::is_complete(cache, preset.source()) {
        anyhow::bail!(
            "fetched {} but {} is still incomplete — the release does not \
             match what stemd expects",
            preset.artefact(),
            cache.display()
        );
    }
    Ok(cache.to_path_buf())
}

fn joined(dirs: &[PathBuf]) -> String {
    dirs.iter()
        .map(|d| d.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn open_stem_cache(args: &Args) -> Result<(Arc<Cache>, String)> {
    let root = cache::root(args)?;
    let max_bytes = args.cache_max_bytes();
    let cache = Cache::new(root.clone(), max_bytes, args.unfetched_ttl())
        .with_context(|| format!("preparing the stem cache in {}", root.display()))?;
    let summary = format!(
        "{} capped at {:.1} GB",
        root.display(),
        max_bytes as f64 / 1e9
    );
    Ok((cache, summary))
}

/// Open the piece store, which keeps pieces across restarts.
fn open_pieces(args: &Args) -> Result<Arc<crate::pieces::Pieces>> {
    let root = match args.pieces_dir.clone() {
        Some(dir) => dir,
        None => dirs::cache_dir()
            .context("no cache directory for this platform")?
            .join("stemd-pieces"),
    };
    let max_bytes = (args.pieces_max_gb * 1e9) as u64;
    let pieces = crate::pieces::Pieces::open(root.clone(), max_bytes)
        .with_context(|| format!("preparing the piece store in {}", root.display()))?;
    tracing::info!(
        "pieces in {}: {:.0} MB of {:.1} GB",
        root.display(),
        pieces.bytes() as f64 / 1e6,
        max_bytes as f64 / 1e9
    );
    Ok(pieces)
}

/// The name and port to announce over mDNS, unless asked not to.
fn announce(args: &Args, port: u16) -> Option<(String, u16)> {
    if args.no_mdns {
        tracing::info!("mDNS advertisement disabled");
        return None;
    }
    Some((args.instance.clone(), port))
}

/// Run the cache rules on a timer, drop job handles they invalidate, and hand
/// MLX's allocator cache back once the server has been idle for [`IDLE_RELEASE`].
fn spawn_reaper(cache: Arc<Cache>, store: Arc<JobStore>, queue: Arc<Queue>) {
    let interval = cache.sweep_interval();
    std::thread::Builder::new()
        .name("stemd-reaper".into())
        .spawn(move || {
            let mut swept = Instant::now();
            let mut idle_since: Option<Instant> = None;
            let mut released = false;
            loop {
                std::thread::sleep(IDLE_TICK);
                if swept.elapsed() >= interval {
                    swept = Instant::now();
                    let reaped = cache.reap();
                    let pruned = store.prune();
                    if reaped.any() {
                        tracing::info!(
                            "reaped {} uncollected and rotated {} for space ({:.0} MB), \
                             dropped {pruned} job handles",
                            reaped.unfetched,
                            reaped.rotated,
                            reaped.bytes as f64 / 1e6
                        );
                    }
                }

                // Writes stop when the disk is low, so eviction has to run
                // from here too, or the store never gives space back.
                if let Some(pieces) = queue.sessions().pieces() {
                    pieces.evict();
                }

                let busy =
                    queue.running().is_some() || queue.depth() > 0 || !queue.sessions().is_empty();
                if busy {
                    idle_since = None;
                    released = false;
                    continue;
                }
                let since = *idle_since.get_or_insert_with(Instant::now);
                // After a minute idle: the gap between two tracks costs a
                // fifth of the second one if the cache is empty when it starts;
                // see `stemd_core::memory`. Once per idle spell.
                if !released && since.elapsed() >= IDLE_RELEASE {
                    released = true;
                    let freed = stemd_core::memory::release();
                    if freed > 0 {
                        tracing::info!(
                            "idle, so returned {:.0} MB of MLX allocator cache; \
                             {:.0} MB still live",
                            freed as f64 / 1e6,
                            stemd_core::memory::active() as f64 / 1e6
                        );
                    }
                }
            }
        })
        .expect("spawning the cache reaper");
}

/// End idle sessions once a second, and let the worker free what they held.
fn spawn_session_reaper(queue: Arc<Queue>) {
    std::thread::Builder::new()
        .name("stemd-sessions".into())
        .spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                if queue.sessions().reap() > 0 {
                    queue.wake();
                }
            }
        })
        .expect("spawning the session reaper");
}

/// Idle time before MLX's allocator cache is handed back.
const IDLE_RELEASE: Duration = Duration::from_secs(60);

/// How often the reaper looks at the queue.
const IDLE_TICK: Duration = Duration::from_secs(5);

/// Bytes the longest permitted track can occupy, at the widest sample format the
/// API accepts. `f32le`, the larger of the two. This bounds what axum buffers
/// before the handler runs, so it has to be an upper bound.
///
/// Taken at the highest rate the API names rather than the model's, since an
/// upload arrives at whatever rate the client holds and is converted here. Above
/// that rate the body is refused by length; the duration limit still applies
/// underneath it, so this only decides how far past 44.1 kHz an upload may go.
fn max_body_bytes(max_track_secs: f64, sample_rate: u32, channels: usize) -> usize {
    let per_second =
        f64::from(sample_rate) * channels as f64 * PcmFormat::F32le.bytes_per_sample() as f64;
    (max_track_secs * per_second) as usize
}

/// The highest rate `output_sample_rate` names, and the highest an upload can
/// carry within the body limit.
fn highest_rate() -> u32 {
    stemd_core::OutputRate::ALL
        .iter()
        .map(|rate| rate.hz())
        .max()
        .unwrap_or(96_000)
}

fn log_model(
    info: &stemd_core::BackendInfo,
    precisions: Precisions,
    max_track_secs: f64,
    max_upload: usize,
) {
    // The backend by name rather than `info.device`'s coarse "gpu": on a
    // cross-platform build the difference between metal, cuda and a CPU
    // fallback is the first thing worth knowing, and it decides the precision
    // each model runs at.
    tracing::info!(
        "loaded {} via {} on {} ({} Hz, {} ch)",
        info.model,
        info.backend,
        precisions.accelerator().backend(),
        info.sample_rate,
        info.channels,
    );
    tracing::info!(
        "accepting tracks up to {:.0} minutes at any rate up to {} Hz ({:.0} MB \
         at f32le)",
        max_track_secs / 60.0,
        highest_rate(),
        max_upload as f64 / 1e6
    );
    tracing::info!(
        "output: {} stems [{}]",
        info.stems.len(),
        info.stems.join(", ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every preset ships the same two stems, so a client's rebuild never changes
    /// with the preset.
    ///
    /// The presets build `harmonics` in different ways, `Fast` as the model's
    /// `bass + other` and the other two as `mix - vocals - drums`, and none of that
    /// reaches the wire. What ships is `harmonics` and `vocals` under all three, so
    /// `mix - harmonics - vocals` is the client's job under all three.
    ///
    /// ```text
    /// STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with every artefact> \
    /// cargo test --release -p stemd-server -- --ignored every_preset --nocapture
    /// ```
    #[test]
    #[ignore]
    fn every_preset_ships_the_same_two_stems() {
        use stemd_core::{Audio, PcmFormat, Silent};

        let (Ok(pcm), Ok(dir)) = (std::env::var("STEMD_AB_PCM"), std::env::var("STEMD_AB_MLX"))
        else {
            eprintln!("SKIPPED: set STEMD_AB_PCM and STEMD_AB_MLX.");
            return;
        };
        let bytes = std::fs::read(&pcm).expect("reading the track");
        let mix = Audio::from_interleaved(&bytes, PcmFormat::S16le, 2, 44100).expect("stereo pcm");

        for preset in Preset::ALL {
            let mut sep = build(
                Path::new(&dir),
                preset.artefact(),
                Some(preset),
                0.25,
                Precisions::stated(None, stemd_core::Accelerator::Metal),
            )
            .unwrap_or_else(|e| panic!("building {preset:?}: {e:#}"));

            let advertised: Vec<String> = sep.info().stems;
            let stems = sep
                .separate(&mix, stemd_core::Parts::DEFAULT, &Silent)
                .expect("separating");
            let shipped: Vec<&str> = stems.shipped.iter().map(|(n, _)| *n).collect();

            // What the player would rebuild, and what it gets at unity.
            let h = &stems
                .shipped
                .iter()
                .find(|(n, _)| *n == "harmonics")
                .expect("harmonics")
                .1;
            let v = &stems
                .shipped
                .iter()
                .find(|(n, _)| *n == "vocals")
                .expect("vocals")
                .1;
            let mut worst = 0.0f32;
            for c in 0..mix.channels() {
                for i in 0..mix.frames() {
                    let drums = mix.data[c][i] - h.data[c][i] - v.data[c][i];
                    let rebuilt = drums + h.data[c][i] + v.data[c][i];
                    worst = worst.max((rebuilt - mix.data[c][i]).abs());
                }
            }

            println!(
                "{:<9} advertises [{}]  ships [{}]  unity error {worst:.2e}",
                preset.label(),
                advertised.join(", "),
                shipped.join(", "),
            );

            assert_eq!(
                shipped,
                stemd_core::SHIPPED,
                "{preset:?} ships something other than the two stems the \
                 protocol promises, so a client would have to know which \
                 preset produced its audio"
            );
            assert_eq!(
                advertised,
                stemd_core::SHIPPED,
                "{preset:?} advertises something else"
            );
            assert!(
                worst < 1e-6,
                "{preset:?}: rebuilding drums and mixing back at unity is off \
                 by {worst:.2e}, so the parts do not sum"
            );
        }
    }

    /// How much percussion each preset leaves in the vocals stem.
    ///
    /// Measured as how much of the vocals stem and the rebuilt drums is shared
    /// content, `dot^2 / (|v|^2 |d|^2)`, squared cosine similarity, so lower is
    /// better. It is symmetric: the directional quantity is `alpha = <v,d>/<d,d>`,
    /// which `stemd_core::hybrid` reports beside it. Whatever is in both stems is in
    /// the wrong one.
    ///
    /// The drums are the client's own `mix - harmonics - vocals`, so this is the
    /// artefact as a player meets it.
    ///
    /// ```text
    /// STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with every artefact> \
    /// cargo test --release -p stemd-server -- --ignored leaves_in_the_vocals --nocapture
    /// ```
    #[test]
    #[ignore]
    fn what_each_preset_leaves_in_the_vocals() {
        use stemd_core::{Audio, PcmFormat, Silent};

        let (Ok(pcm), Ok(dir)) = (std::env::var("STEMD_AB_PCM"), std::env::var("STEMD_AB_MLX"))
        else {
            eprintln!("SKIPPED: set STEMD_AB_PCM and STEMD_AB_MLX.");
            return;
        };
        let bytes = std::fs::read(&pcm).expect("reading the track");
        let mix = Audio::from_interleaved(&bytes, PcmFormat::S16le, 2, 44100).expect("stereo pcm");

        println!(
            "  {:<9}{:>12}{:>22}",
            "preset", "vocals rms", "shared with drums"
        );
        for preset in Preset::ALL {
            let mut sep = build(
                Path::new(&dir),
                preset.artefact(),
                Some(preset),
                0.25,
                Precisions::stated(None, stemd_core::Accelerator::Metal),
            )
            .unwrap_or_else(|e| panic!("building {preset:?}: {e:#}"));
            let stems = sep
                .separate(&mix, stemd_core::Parts::DEFAULT, &Silent)
                .expect("separating");
            let pick = |want: &str| {
                stems
                    .shipped
                    .iter()
                    .find(|(n, _)| *n == want)
                    .map(|(_, a)| a.clone())
                    .expect("a shipped stem")
            };
            let (h, v) = (pick("harmonics"), pick("vocals"));

            // Shared content, as squared cosine similarity. Symmetric by
            // construction: `alpha^2 * bb / aa` with `alpha = dot / bb` is
            // `dot^2 / (aa * bb)`, so there is one number here and not two.
            let shared = |a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>| {
                let (mut dot, mut bb, mut aa) = (0.0f64, 0.0f64, 0.0f64);
                for (ca, cb) in a.iter().zip(b) {
                    for (x, y) in ca.iter().zip(cb) {
                        dot += f64::from(*x) * f64::from(*y);
                        bb += f64::from(*y).powi(2);
                        aa += f64::from(*x).powi(2);
                    }
                }
                (10.0 * (dot.powi(2) / (aa * bb)).log10(), aa)
            };

            let drums: Vec<Vec<f32>> = (0..mix.channels())
                .map(|c| {
                    (0..mix.frames())
                        .map(|i| mix.data[c][i] - h.data[c][i] - v.data[c][i])
                        .collect()
                })
                .collect();

            let (ticks, vocal_energy) = shared(&v.data, &drums);
            let rms = (vocal_energy / (mix.frames() * mix.channels()) as f64).sqrt();
            println!("  {:<9}{rms:>12.4}{ticks:>19.1} dB", preset.label());
        }
    }

    #[test]
    fn the_body_limit_admits_the_longest_track_at_every_rate_it_names() {
        // The two limits are one number seen twice; the body limit must never
        // reject audio the duration gate would have allowed, or a legal upload
        // fails before the handler can explain why. An upload arrives at
        // whatever rate the client holds, so it has to hold at each of them.
        let secs = 600.0;
        let limit = max_body_bytes(secs, highest_rate(), 2);

        for rate in stemd_core::OutputRate::ALL {
            let f32_track = (secs * f64::from(rate.hz()) * 2.0 * 4.0) as usize;
            let s16_track = (secs * f64::from(rate.hz()) * 2.0 * 2.0) as usize;
            assert!(limit >= f32_track, "{rate}: {limit} < {f32_track}");
            assert!(limit >= s16_track);
        }

        // And one second more at the highest of them must not fit, or the gate
        // is the only thing holding the line.
        let widest = f64::from(highest_rate()) * 2.0 * 4.0;
        assert!(limit < ((secs + 1.0) * widest) as usize);
    }

    #[test]
    fn the_body_limit_follows_the_rate_and_channel_count() {
        assert_eq!(
            max_body_bytes(600.0, 48000, 2),
            max_body_bytes(600.0, 44100, 2) * 48000 / 44100
        );
        assert_eq!(
            max_body_bytes(600.0, 44100, 4),
            max_body_bytes(600.0, 44100, 2) * 2
        );
    }

    #[test]
    fn an_absolute_models_path_is_not_searched_twice() {
        // `join` with an absolute path returns it unchanged, so without the
        // dedupe every error message repeats the same directory.
        let dirs = search_dirs(Path::new("/opt/models"), Path::new("/var/cache"));
        let mut unique = dirs.clone();
        unique.dedup();
        assert_eq!(dirs.len(), unique.len(), "{dirs:?}");
    }

    #[test]
    fn the_download_cache_is_searched_last() {
        let dirs = search_dirs(Path::new("models"), Path::new("/var/cache"));
        assert_eq!(dirs.first().unwrap(), Path::new("models"));
        assert_eq!(
            dirs.last().unwrap(),
            Path::new("/var/cache"),
            "a local copy of the weights must win over a downloaded one"
        );
    }
}
