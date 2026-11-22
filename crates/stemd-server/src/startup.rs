//! Turns the parsed arguments into running services.
//!
//! It covers every step between parsing the command line and mounting the router: locating and
//! loading the model, opening the stem cache, starting the worker and the reaper, and advertising
//! over mDNS.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::cache::Cache;
use crate::jobs::Jobs;
use crate::launch::{Launch, Stage};
use crate::logbuf::LogBuffer;
use crate::models::{self, DEFAULT_PRESET, Preset};
use crate::precision::Precisions;
use crate::queue::Queue;
use crate::settings::SettingsFile;
use crate::state::AppState;
use crate::switch::{self, Switcher};

/// A server wired up and ready to bind.
pub struct Server {
    pub state: Arc<AppState>,
    /// One line describing where stems go, for the startup log.
    pub cache_summary: String,
}

/// What startup takes from the command line.
pub struct Config {
    /// Directory holding the model artefacts.
    pub models: PathBuf,
    /// The artefact named on the command line, which overrides the saved model.
    pub demucs_model: Option<String>,
    /// A precision forced for every model, or `None` to let each model choose.
    pub precision: Option<stemd_core::Precision>,
    pub overlap: f32,
    pub cache_root: PathBuf,
    /// True when `--cache-dir` named the cache root.
    pub cache_dir_given: bool,
    pub queue_depth: usize,
    pub session_idle: Duration,
    pub max_sessions: usize,
    pub partial_max_age: Duration,
    pub unfetched_ttl: Duration,
    pub ephemeral: bool,
    /// The mDNS instance name.
    pub instance: String,
    /// Advertise over mDNS.
    pub mdns: bool,
    pub offline: bool,
    pub headless: bool,
}

impl Config {
    /// The cache's deletion rules, with the size and whole-track age `settings` holds.
    pub fn cache_rules(&self, settings: &crate::settings::Settings) -> crate::cache::Rules {
        let (max_bytes, whole_max_age) = settings.cache_limits();
        crate::cache::Rules {
            max_bytes,
            partial_max_age: self.partial_max_age,
            whole_max_age,
            unfetched_ttl: self.unfetched_ttl,
        }
    }
}

/// Build every service the router depends on.
///
/// The stem cache is opened before the model because the worker that loads the
/// model needs it. See [`crate::queue::BuildSeparator`].
///
/// `endpoint` holds where the listener is already bound, which is the real port
/// under `--bind :0`. Progress is reported through `launch`.
pub fn prepare(
    config: &Config,
    logs: LogBuffer,
    settings: Arc<SettingsFile>,
    endpoint: crate::endpoint::Endpoint,
    launch: &Launch,
) -> Result<Server> {
    // Detect once: the accelerator cannot change under a running server, so every model loaded
    // later uses the same answer.
    let precisions = Precisions::detect(config.precision);
    // Warn here, before a model loads at CPU speed: in every later log line, a GPU whose runtime is
    // missing looks like no GPU, and only a missing runtime has a fix.
    warn_gpu_refused();
    let (stems, cache_summary) = open_stem_cache(config, &settings.get())?;
    let jobs = Arc::new(Jobs::new(config.unfetched_ttl));

    let (model, queue, info) = load_model(config, &settings, &stems, precisions, launch)?;
    queue
        .sessions()
        .set_limits(config.session_idle, config.max_sessions);
    stems.set_gate(queue.consolidation_gate());
    spawn_session_reaper(Arc::clone(&queue));
    let longest = crate::disk::longest_track_minutes();
    let saved = settings.get().max_track_minutes;
    if saved > f64::from(longest) {
        tracing::warn!(
            "a {saved:.0}-minute track does not fit this machine's memory; \
             accepting up to {longest} minutes"
        );
    }
    log_model(&info, precisions, settings.get().max_track_secs(longest));

    // Build this before the app state so every exit path can reach it. A headless server has no
    // switch to turn it back on, so it always serves.
    let serving = crate::serving::Serving::new(
        config.headless || settings.get().serving,
        announce(config, endpoint.port()),
        &info.model,
    );

    // Log both identities once at debug, so when the published id changes unexpectedly, the log
    // shows which field changed; an eight-character digest cannot.
    let switcher = Arc::new(Switcher::new(
        model.preset,
        Arc::clone(&queue),
        Arc::clone(&settings),
        switch::SwitchConfig {
            dirs: model.search_dirs,
            cache: model.download_cache,
            overlap: config.overlap,
            precisions,
            offline: config.offline,
            artefact: model.artefact,
        },
    ));
    tracing::debug!(
        "cache key: {} | published as {}",
        switcher.identity(),
        switcher.published_identity()
    );

    spawn_reaper(Arc::clone(&stems), Arc::clone(&jobs), Arc::clone(&queue));
    let filters = open_filters()?;

    Ok(Server {
        state: Arc::new(AppState {
            jobs,
            cache: stems,
            queue,
            switcher,
            logs,
            drops: Arc::default(),
            settings,
            serving,
            filters,
            ephemeral: config.ephemeral,
            endpoint,
        }),
        cache_summary,
    })
}

/// Warn that a GPU is present but unusable, with the fix where there is one.
fn warn_gpu_refused() {
    if !stemd_core::Accelerator::gpu_refused() {
        return;
    }
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

/// Where the model was found, and what the switcher needs to replace it later.
///
/// It holds no separator: the model belongs to the queue's worker, the only thread allowed to build
/// it.
struct LoadedModel {
    /// `None` when the artefact is not one of the presets.
    preset: Option<Preset>,
    /// The artefact that was loaded, which can differ from the one asked for.
    artefact: String,
    search_dirs: Vec<PathBuf>,
    download_cache: PathBuf,
}

/// Load the model for this run: `--demucs-model` if given, otherwise the one the window last
/// selected.
// A saved preset that does not load falls back to the default and the launch continues, since the
// window is the only way to choose another. A model named on the command line gets no fallback.
//
// Behind a window, a saved preset that is not on disk is not downloaded: the window asks which
// model to download, and saves the answer once it has loaded.
//
// [`Queue::start`] returns only once the worker has a model or has failed to get one.
fn load_model(
    config: &Config,
    settings: &SettingsFile,
    stems: &Arc<Cache>,
    precisions: Precisions,
    launch: &Launch,
) -> Result<(LoadedModel, Arc<Queue>, stemd_core::BackendInfo)> {
    let saved = settings.get().preset;
    if config.demucs_model.is_none() && launch.asks() && !config.offline {
        let dirs = search_dirs(&config.models, &models::cache_dir()?);
        if models::locate(&dirs, saved).is_none() {
            let installed = Preset::ALL
                .into_iter()
                .filter(|p| models::locate(&dirs, *p).is_some())
                .collect();
            tracing::info!("no {} model on disk; choose one", saved.label());
            let chosen = launch.choose(installed);
            let loaded = load_artefact(config, chosen.artefact(), stems, precisions, launch)?;
            settings.set_preset(chosen);
            return Ok(loaded);
        }
    }
    let Some(requested) = config.demucs_model.clone() else {
        return load_artefact(config, saved.artefact(), stems, precisions, launch).or_else(|err| {
            if saved == DEFAULT_PRESET {
                return Err(err);
            }
            tracing::warn!(
                "the saved model {} will not load ({err:#}); starting on {} instead",
                saved.artefact(),
                DEFAULT_PRESET.artefact()
            );
            load_artefact(config, DEFAULT_PRESET.artefact(), stems, precisions, launch)
        });
    };
    load_artefact(config, &requested, stems, precisions, launch)
}

fn load_artefact(
    config: &Config,
    artefact: &str,
    stems: &Arc<Cache>,
    precisions: Precisions,
    launch: &Launch,
) -> Result<(LoadedModel, Arc<Queue>, stemd_core::BackendInfo)> {
    let preset = Preset::from_artefact(artefact);
    let shown = preset.map_or_else(|| artefact.to_owned(), |p| p.label().to_owned());
    let download_cache = models::cache_dir()?;
    let search_dirs = search_dirs(&config.models, &download_cache);
    let dir = resolve_models(
        &search_dirs,
        &download_cache,
        preset,
        artefact,
        config.offline,
        launch,
    )?;

    // Starting the worker loads the model: the worker builds what it runs.
    launch.set(Stage::Loading(shown));
    let (queue, info) = Queue::start(
        models::builder(&dir, artefact, preset, config.overlap, precisions),
        config.queue_depth,
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
    // `join` with an absolute argument returns it unchanged, so an absolute --models would appear
    // three times and every error message would repeat it. Order matters, so dedupe in place
    // without sorting.
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|d| seen.insert(d.clone()));
    dirs
}

/// Find the artefact, downloading it if this is a first run.
///
/// A directory counts only if it holds every file the preset needs, since a preset can span several
/// files.
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
        // Presence is not enough: a cached artefact can rot after the download that verified it,
        // and hashing costs only about a second.
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
        "no model in {}; {} missing {}; --offline set",
        joined(dirs),
        cache.display(),
        missing.join(", ")
    )
}

/// Download a preset into the cache and confirm it landed.
///
/// Files already in the cache are skipped.
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

fn open_stem_cache(
    config: &Config,
    settings: &crate::settings::Settings,
) -> Result<(Arc<Cache>, String)> {
    let root = config.cache_root.clone();
    let rules = config.cache_rules(settings);
    let cache = Cache::open_with(root.clone(), rules)
        .with_context(|| format!("preparing the stem cache in {}", root.display()))?;
    if config.ephemeral {
        cache.wipe();
    }
    // Remove the old pieces directory only next to the default root, since `--cache-dir` may name
    // that old directory.
    if !config.cache_dir_given {
        crate::migrate::remove_old_pieces_dir();
    }
    let stats = cache.stats();
    let summary = format!(
        "{}: {} tracks, {:.0} MB of {:.1} GB",
        root.display(),
        stats.tracks,
        stats.bytes as f64 / 1e6,
        rules.max_bytes as f64 / 1e9
    );
    Ok((cache, summary))
}

/// Open the uploaded filters, which are kept across restarts.
fn open_filters() -> Result<Arc<crate::filters::Filters>> {
    let dir = models::support_dir()?.join("filters");
    let filters = crate::filters::Filters::open(dir.clone(), crate::filters::MAX_FILTERS)
        .with_context(|| format!("preparing the filters directory in {}", dir.display()))?;
    tracing::info!(
        "filters in {}: {} of {}",
        dir.display(),
        filters.len(),
        crate::filters::MAX_FILTERS
    );
    Ok(filters)
}

/// The name and port to announce over mDNS, unless asked not to.
fn announce(config: &Config, port: u16) -> Option<(String, u16)> {
    if !config.mdns {
        tracing::info!("mDNS advertisement disabled");
        return None;
    }
    Some((config.instance.clone(), port))
}

/// Apply the cache's age rules every [`crate::cache::CLEAN_EVERY`], its
/// budget every tick, drop stale job handles, and hand MLX's allocator cache
/// back once the server has been idle for [`IDLE_RELEASE`].
fn spawn_reaper(cache: Arc<Cache>, jobs: Arc<Jobs>, queue: Arc<Queue>) {
    let interval = crate::cache::CLEAN_EVERY;
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
                    cache.clean();
                }
                let pruned = jobs.prune();
                if pruned > 0 {
                    tracing::debug!("dropped {pruned} job handles");
                }
                // Writes stop when the disk is low, so apply the budget here too, or the cache
                // never gives space back.
                cache.evict();

                let busy =
                    queue.running().is_some() || queue.depth() > 0 || !queue.sessions().is_empty();
                if busy {
                    idle_since = None;
                    released = false;
                    continue;
                }
                let since = *idle_since.get_or_insert_with(Instant::now);
                // Release after a minute idle, once per idle spell: releasing in the gap between
                // two tracks would slow the second track by a fifth; see `stemd_core::memory`.
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

/// The idle time before MLX's allocator cache is released.
const IDLE_RELEASE: Duration = Duration::from_secs(60);

/// How often the reaper looks at the queue.
const IDLE_TICK: Duration = Duration::from_secs(5);

fn log_model(info: &stemd_core::BackendInfo, precisions: Precisions, max_track_secs: f64) {
    // Log the backend by name (metal, cuda or a CPU fallback), since `info.device` only says "gpu".
    // On a cross-platform build the backend matters most, and it decides each model's precision.
    tracing::info!(
        "loaded {} via {} on {} ({} Hz, {} ch)",
        info.model,
        info.backend,
        precisions.accelerator().backend(),
        info.sample_rate,
        info.channels,
    );
    tracing::info!(
        "accepting tracks up to {:.0} minutes at any rate up to {} Hz",
        max_track_secs / 60.0,
        crate::state::highest_rate(),
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

    /// Every preset ships the same two stems, so a client's rebuild does not change with the
    /// preset.
    ///
    /// The presets build `harmonics` in different ways: `Fast` as the model's `bass + other`, the
    /// other two as `mix - vocals - drums`. None of that reaches the client. All three ship
    /// `harmonics` and `vocals`, and the client computes `mix - harmonics - vocals`.
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
            let mut sep = crate::models::builder(
                Path::new(&dir),
                preset.artefact(),
                Some(preset),
                0.25,
                Precisions::stated(None, stemd_core::Accelerator::Metal),
            )()
            .unwrap_or_else(|e| panic!("building {preset:?}: {e:#}"));

            let advertised: Vec<String> = sep.info().stems;
            let stems = sep
                .separate(&mix, stemd_core::Parts::DEFAULT, &Silent)
                .expect("separating");
            let shipped: Vec<&str> = stems.shipped.iter().map(|(n, _)| *n).collect();

            // The stems the player would rebuild, at unity gain.
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

    /// How much percussion each preset leaves in the vocals stem, as the squared cosine similarity
    /// `dot^2 / (|v|^2 |d|^2)` of the vocals stem and the rebuilt drums; lower is better.
    // The measure is symmetric; `stemd_core::hybrid` reports the directional `alpha = <v,d>/<d,d>`
    // beside it. Content present in both stems is bleed in one of them. The drums are the client's
    // own `mix - harmonics - vocals`, so this measures the stems as a player receives them.
    //
    //     STEMD_AB_PCM=track.pcm STEMD_AB_MLX=<dir with every artefact> \
    //     cargo test --release -p stemd-server -- --ignored leaves_in_the_vocals --nocapture
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
            let mut sep = crate::models::builder(
                Path::new(&dir),
                preset.artefact(),
                Some(preset),
                0.25,
                Precisions::stated(None, stemd_core::Accelerator::Metal),
            )()
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

            // Shared content as squared cosine similarity. It is symmetric: `alpha^2 * bb / aa`
            // with `alpha = dot / bb` equals `dot^2 / (aa * bb)`, so one number covers both
            // directions.
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
    fn an_absolute_models_path_is_not_searched_twice() {
        // `join` with an absolute path returns it unchanged, so without the dedupe every error
        // message would repeat the same directory.
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
