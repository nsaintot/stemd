//! Preferences the window changes and the next launch remembers.
//!
//! Three choices are worth persisting: which model loads, and the output format
//! and sample rate a client gets when it does not ask for one.
//!
//! Precedence is flag, then file, then built-in default. A flag is for one run and
//! is never written back, so a launch script that pins `--output-format f32le`
//! does not redefine what the window shows next time. Every change the window
//! makes is saved.
//!
//! Nothing here is allowed to stop the server. A missing file is a first run, and
//! a damaged one is worth a warning and a default.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use stemd_core::{OutputRate, StemFormat};

use crate::models::{self, DEFAULT_PRESET, Preset, RETIRED};

/// What the window can change and the next launch reads back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settings {
    /// The preset loaded at boot. Only ever a preset: `--demucs-model` can name
    /// a hand-traced artefact, but that is a flag for one run, so it is not
    /// something the window can select or the file can hold.
    pub preset: Preset,
    /// Format for a request that does not name one.
    pub format: StemFormat,
    /// Output rate for a request that does not name one.
    pub rate: OutputRate,
}

impl Settings {
    /// Bring the rate down to one the format can actually carry.
    ///
    /// The pair is what a client gets, so it has to be a pair that exists. MP3
    /// is the only format that constrains the rate (it has no 96 kHz mode)
    /// and LAME does not refuse one it lacks: it resamples to 48 kHz on its
    /// own, slowly, and writes a file at a rate nobody chose. Choosing 48 kHz
    /// deliberately gets the same audio in a fraction of the time and says so.
    ///
    /// Held here rather than at each control because the pair can be made
    /// invalid four ways: two flags, two menus, a hand-edited file, and a file
    /// written by a version that allowed it.
    fn reconcile(&mut self) {
        if self.format.carries(self.rate) {
            return;
        }
        let was = self.rate;
        // The highest it can carry, being the nearest thing to what was asked.
        self.rate = self.format.rates().last().unwrap_or_default();
        tracing::info!(
            "{} has no {} Hz mode; writing {} Hz instead",
            self.format,
            was.hz(),
            self.rate.hz()
        );
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            preset: DEFAULT_PRESET,
            format: StemFormat::Flac,
            rate: OutputRate::default(),
        }
    }
}

/// The on-disk shape, one step removed from [`Settings`].
///
/// Every field is optional and stored as the same string a flag would take, so the
/// file stays hand-editable and one unreadable field costs that field rather than
/// the whole file. Serialising the enums directly would tie the format to their
/// variant names.
///
/// Unknown keys are ignored, so a file written by a later version still loads.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct Stored {
    /// Artefact name, as `--demucs-model` takes it.
    model: Option<String>,
    output_format: Option<String>,
    output_sample_rate: Option<u32>,
}

impl Stored {
    fn from_settings(settings: &Settings) -> Self {
        Self {
            model: Some(settings.preset.artefact().to_owned()),
            output_format: Some(settings.format.to_string()),
            output_sample_rate: Some(settings.rate.hz()),
        }
    }

    /// Defaults for anything absent or unrecognised, with a line about what was
    /// dropped: a preference that silently reverts is worse than one that says
    /// it could not be read.
    fn into_settings(self) -> Settings {
        let mut settings = Settings::default();

        if let Some(model) = self.model {
            match Preset::from_artefact(&model) {
                Some(preset) => settings.preset = preset,
                // A preset that was retired between versions is the one case
                // where the file is not wrong, it is just old. Saying so beats
                // "ignoring model ..." for something the user did pick once.
                None if RETIRED.contains(&model.as_str()) => tracing::info!(
                    "settings: {model} is no longer offered; starting on {}. \
                     --demucs-model {model} still loads it",
                    settings.preset.artefact()
                ),
                None => warn_ignored("model", &model, settings.preset.artefact()),
            }
        }
        if let Some(format) = self.output_format {
            match format.parse() {
                Ok(parsed) => settings.format = parsed,
                Err(_) => warn_ignored("output_format", &format, &settings.format.to_string()),
            }
        }
        if let Some(hz) = self.output_sample_rate {
            match OutputRate::from_hz(hz) {
                Some(rate) => settings.rate = rate,
                None => warn_ignored(
                    "output_sample_rate",
                    &hz.to_string(),
                    &settings.rate.hz().to_string(),
                ),
            }
        }
        settings.reconcile();
        settings
    }
}

fn warn_ignored(field: &str, found: &str, fallback: &str) {
    tracing::warn!("settings: ignoring {field} {found:?}, using {fallback}");
}

/// Output fields a flag fixed for this run. A pinned field is not editable, and
/// the window greys its control out.
///
/// The model is deliberately not here. `--demucs-model` decides what boots, but
/// switching model in the window is an explicit act with a download behind it,
/// and refusing it would be worse than honouring it.
#[derive(Debug, Default, Clone, Copy)]
pub struct Pinned {
    pub format: bool,
    pub rate: bool,
}

/// What the file says, and what this run is actually using.
///
/// The two differ exactly where a flag overrides one. Writes serialise `saved`, so
/// a flag has no path to the file at all. Disabling the pinned control is not
/// enough on its own: a write is one document, so changing the rate in the window
/// would otherwise carry a `--output-format` flag into the file alongside it.
#[derive(Debug, Clone, Copy)]
struct State {
    saved: Settings,
    current: Settings,
}

/// The settings file, and the live copy every reader sees.
pub struct SettingsStore {
    path: PathBuf,
    state: Mutex<State>,
    pinned: Pinned,
}

impl SettingsStore {
    /// Read the file, or start from defaults if there is not a usable one.
    ///
    /// Infallible on purpose: this runs before the server exists, and no state of
    /// a preferences file is worth failing a launch over.
    pub fn open(path: PathBuf) -> Self {
        let saved = load(&path).unwrap_or_else(|err| {
            // Not a warning when the file is simply absent: that is every first
            // run, and it is not a problem.
            if path.exists() {
                tracing::warn!(
                    "settings: {} is unreadable ({err:#}); using defaults",
                    path.display()
                );
            }
            Settings::default()
        });
        Self {
            path,
            state: Mutex::new(State {
                saved,
                current: saved,
            }),
            pinned: Pinned::default(),
        }
    }

    /// Apply the flags that override what the file said, for this run only.
    pub fn pin(mut self, format: Option<StemFormat>, rate: Option<OutputRate>) -> Self {
        let mut state = self.state.lock();
        if let Some(format) = format {
            state.current.format = format;
            self.pinned.format = true;
        }
        if let Some(rate) = rate {
            state.current.rate = rate;
            self.pinned.rate = true;
        }
        // A flag can name a pair that does not exist just as easily as the file
        // can. Only `current`: the file still says what it said.
        state.current.reconcile();
        drop(state);
        self
    }

    pub const fn pinned(&self) -> Pinned {
        self.pinned
    }

    /// What this run is using, flags included.
    pub fn get(&self) -> Settings {
        self.state.lock().current
    }

    /// Record the preset to load on the next launch.
    pub fn set_preset(&self, preset: Preset) {
        self.write(|s| s.preset = preset);
    }

    pub fn set_format(&self, format: StemFormat) {
        if !self.pinned.format {
            self.write(|s| s.format = format);
        }
    }

    pub fn set_rate(&self, rate: OutputRate) {
        if !self.pinned.rate {
            self.write(|s| s.rate = rate);
        }
    }

    /// Change both copies and write the file.
    ///
    /// Only ever reached for a field no flag pinned, so applying the change to `saved`
    /// cannot carry a flag into the file. A failed write is a warning: the change still
    /// applies to the running server, and only the next launch is affected.
    fn write(&self, change: impl Fn(&mut Settings)) {
        let settings = {
            let mut state = self.state.lock();
            change(&mut state.saved);
            change(&mut state.current);
            state.saved.reconcile();
            state.current.reconcile();
            state.saved
        };
        if let Err(err) = save(&self.path, &settings) {
            tracing::warn!(
                "settings: could not write {} ({err:#}); the change applies to this run only",
                self.path.display()
            );
        }
    }
}

/// `settings.json` in the data directory, beside the models. See
/// [`models::support_dir`] for where that is on each platform.
pub fn default_path() -> Result<PathBuf> {
    Ok(models::support_dir()?.join("settings.json"))
}

fn load(path: &Path) -> Result<Settings> {
    let text = std::fs::read_to_string(path)?;
    let stored: Stored = serde_json::from_str(&text)?;
    Ok(stored.into_settings())
}

/// Write through a temporary file in the same directory.
///
/// A settings file is rewritten whenever a menu changes, including while the
/// machine is shutting down. Truncating the real file first would make an
/// interrupted write leave a half-written one behind, which is exactly the state
/// [`SettingsStore::open`] has to throw away.
fn save(path: &Path, settings: &Settings) -> Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

    let text = serde_json::to_string_pretty(&Stored::from_settings(settings))?;
    let staging = path.with_extension("json.new");
    std::fs::write(&staging, &text).with_context(|| format!("writing {}", staging.display()))?;
    std::fs::rename(&staging, path).with_context(|| format!("installing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn scratch() -> PathBuf {
        std::env::temp_dir().join(format!(
            "stemd-settings-{}-{}/settings.json",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn a_first_run_has_no_file_and_gets_the_defaults() {
        let store = SettingsStore::open(scratch());
        assert_eq!(store.get(), Settings::default());
    }

    #[test]
    fn what_the_window_changes_is_what_the_next_launch_reads() {
        let path = scratch();
        let store = SettingsStore::open(path.clone());
        store.set_preset(Preset::Balanced);
        store.set_rate(OutputRate::Hz96000);
        store.set_format(StemFormat::Pcm32);

        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().preset, Preset::Balanced);
        assert_eq!(reopened.get().rate, OutputRate::Hz96000);
        assert_eq!(reopened.get().format, StemFormat::Pcm32);
    }

    /// The window offers only rates the format carries, but the file is
    /// hand-editable and a flag can name either half of the pair. Whatever the
    /// route in, what comes out has to be a pair that exists.
    #[test]
    fn a_format_and_a_rate_that_cannot_go_together_are_reconciled() {
        let path = scratch();
        let store = SettingsStore::open(path.clone());
        store.set_rate(OutputRate::Hz96000);
        store.set_format(StemFormat::Mp3);
        assert_eq!(store.get().rate, OutputRate::Hz48000);

        // And it survives the round trip rather than coming back invalid.
        assert_eq!(SettingsStore::open(path).get().rate, OutputRate::Hz48000);
    }

    /// A flag naming the other half of an impossible pair is reconciled too, and
    /// without writing the correction to the file.
    #[test]
    fn a_flag_cannot_pin_a_pair_that_does_not_exist() {
        let path = scratch();
        SettingsStore::open(path.clone()).set_rate(OutputRate::Hz96000);

        let pinned = SettingsStore::open(path.clone()).pin(Some(StemFormat::Mp3), None);
        assert_eq!(pinned.get().rate, OutputRate::Hz48000);
        drop(pinned);

        // The file still says 96000: a flag is for one run, including this one.
        let reopened = SettingsStore::open(path);
        assert_eq!(reopened.get().format, StemFormat::Flac);
        assert_eq!(reopened.get().rate, OutputRate::Hz96000);
    }

    /// A flag is for one run. It has to change what the server does without
    /// touching what the window will offer next time.
    #[test]
    fn a_flag_overrides_the_file_without_rewriting_it() {
        let path = scratch();
        SettingsStore::open(path.clone()).set_rate(OutputRate::Hz48000);

        let run = SettingsStore::open(path.clone()).pin(Some(StemFormat::Pcm32), None);
        assert_eq!(run.get().format, StemFormat::Pcm32, "the flag is in force");
        assert_eq!(
            run.get().rate,
            OutputRate::Hz48000,
            "the file still applies"
        );

        // A control the flag pinned cannot write, so the flag has no way to
        // reach the file. One that it did not pin still can.
        run.set_format(StemFormat::Pcm16);
        run.set_rate(OutputRate::Hz96000);
        assert_eq!(run.get().format, StemFormat::Pcm32, "pinned for this run");

        let next = SettingsStore::open(path).get();
        assert_eq!(next.format, StemFormat::Flac, "the flag must not persist");
        assert_eq!(next.rate, OutputRate::Hz96000);
    }

    /// The file is meant to be editable by hand, so the names in it have to be
    /// the ones the flags and the API already use.
    #[test]
    fn the_file_spells_settings_the_way_the_flags_do() {
        let path = scratch();
        let store = SettingsStore::open(path.clone());
        store.set_preset(Preset::Balanced);
        store.set_rate(OutputRate::Hz48000);

        let text = std::fs::read_to_string(&path).expect("written");
        // The artefact name, whatever it currently is: the point is that the
        // file holds what `--demucs-model` takes, not which model that is.
        let artefact = Preset::Balanced.artefact();
        assert!(text.contains(&format!("\"{artefact}\"")), "{text}");
        assert!(text.contains("\"flac\""), "{text}");
        assert!(text.contains("48000"), "{text}");
    }

    /// A preferences file is not a reason to fail a launch, whatever state it is
    /// in. Each of these must leave a usable server.
    #[test]
    fn nothing_in_the_file_can_stop_the_server() {
        for content in [
            "",
            "{",
            "null",
            "[]",
            r#"{"model": "a model that does not exist"}"#,
            r#"{"output_format": "opus"}"#,
            r#"{"output_sample_rate": 22050}"#,
            r#"{"output_sample_rate": "44100"}"#,
            r#"{"unknown_key": 1}"#,
        ] {
            let path = scratch();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, content).unwrap();
            assert_eq!(
                SettingsStore::open(path).get(),
                Settings::default(),
                "{content:?} should have fallen back to the defaults"
            );
        }
    }

    /// One unreadable field must not discard the two beside it.
    #[test]
    fn a_bad_field_costs_that_field_and_no_others() {
        let path = scratch();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            format!(
                r#"{{"model": "{}", "output_format": "opus", "output_sample_rate": 96000}}"#,
                Preset::Balanced.artefact()
            ),
        )
        .unwrap();

        let settings = SettingsStore::open(path).get();
        assert_eq!(settings.preset, Preset::Balanced);
        assert_eq!(settings.rate, OutputRate::Hz96000);
        assert_eq!(settings.format, StemFormat::Flac, "the default stands in");
    }

    /// Every preset has to survive the round trip, or switching to one in the
    /// window would quietly load a different one next launch.
    #[test]
    fn every_preset_the_window_offers_round_trips_through_the_file() {
        for preset in Preset::ALL {
            let path = scratch();
            SettingsStore::open(path.clone()).set_preset(preset);
            assert_eq!(SettingsStore::open(path).get().preset, preset);
        }
    }

    #[test]
    fn every_rate_and_format_round_trips_through_the_file() {
        for rate in OutputRate::ALL {
            for format in [StemFormat::Flac, StemFormat::Pcm16, StemFormat::Pcm32] {
                let path = scratch();
                let store = SettingsStore::open(path.clone());
                store.set_rate(rate);
                store.set_format(format);
                let read = SettingsStore::open(path).get();
                assert_eq!((read.rate, read.format), (rate, format));
            }
        }
    }

    /// An interrupted write must not be able to leave the real file truncated.
    #[test]
    fn a_write_never_exposes_a_half_written_file() {
        let path = scratch();
        SettingsStore::open(path.clone()).set_rate(OutputRate::Hz48000);

        let staging = path.with_extension("json.new");
        assert!(!staging.exists(), "the temporary file outlived the write");
        assert!(serde_json::from_str::<Stored>(&std::fs::read_to_string(&path).unwrap()).is_ok());
    }
}
