//! Preferences the window changes and the next launch remembers.
//!
//! The file keeps which model loads, the output format and sample rate a client gets when it does
//! not ask for one, the parts a drop writes, whether the server answers and on which port, the
//! longest track accepted, and the cache's size and how long it keeps whole tracks.
//!
//! Precedence is flag, then file, then built-in default. A flag is for one run and
//! is never written back, so a launch script that pins `--output-format f32le`
//! does not redefine what the window shows next time. Every change the window
//! makes is saved.
//!
//! Nothing here may stop the server: a missing file means a first run, and a damaged one gets a
//! warning and the defaults.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use stemd_core::{OutputRate, PartSet, Parts, StemFormat};

use crate::models::{self, DEFAULT_PRESET, Preset, RETIRED};

/// What the window can change and the next launch reads back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// The preset loaded at launch. It is always a preset: `--demucs-model` can name a hand-traced
    /// artefact, but only for one run, so neither the window nor the file can hold one.
    pub preset: Preset,
    /// The format for a request that does not name one.
    pub format: StemFormat,
    /// The output rate for a request that does not name one.
    pub rate: OutputRate,
    /// The parts a dropped track writes next to it. Jobs over HTTP name their own.
    pub drop_parts: DropParts,
    /// Whether the HTTP API answers. The window has a switch; headless always
    /// serves.
    pub serving: bool,
    /// The disk space the cache may occupy, in gigabytes.
    pub cache_max_gb: f64,
    /// Days a whole track is kept unused; 0 keeps it until the cache is full.
    pub whole_max_days: f64,
    /// The port the API listens on, on every interface.
    pub port: u16,
    /// The longest track accepted, in minutes; 0 for the longest that this machine's memory holds.
    pub max_track_minutes: f64,
}

/// The port the API listens on when nothing else is set.
pub const DEFAULT_PORT: u16 = 8420;

/// Which parts a drop writes.
///
/// `harmonics` and `bass` or `other` cannot be on together: ticking one side
/// clears the other. At least one part stays ticked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DropParts(PartSet);

impl DropParts {
    /// `harmonics`, `vocals` and `drums`.
    pub const DEFAULT: Self = Self(Parts::DEFAULT_WITH_DERIVED);

    pub fn has(self, part: &str) -> bool {
        self.0.has(part)
    }

    /// Tick or untick `part`, keeping the set valid.
    #[must_use]
    pub fn toggle(self, part: &str) -> Self {
        let set = if self.0.has(part) {
            self.0.without(part)
        } else if part == "harmonics" {
            self.0.without("bass").without("other").with(part)
        } else if part == "bass" || part == "other" {
            self.0.without("harmonics").with(part)
        } else {
            self.0.with(part)
        };
        if set.is_empty() { self } else { Self(set) }
    }

    pub fn names(self) -> impl Iterator<Item = &'static str> {
        self.0.names()
    }

    /// What to ask the separator for so every ticked part exists: `bass` and
    /// `other` split out when either is ticked, `drums` always as the derived part.
    pub fn request(self) -> Parts {
        if self.has("bass") || self.has("other") {
            Parts::parse("bass,other,vocals").expect("a valid list")
        } else {
            Parts::DEFAULT
        }
    }

    fn parse(list: &str) -> Option<Self> {
        let set = PartSet::parse(list).ok()?;
        let clash = set.has("harmonics") && (set.has("bass") || set.has("other"));
        (!set.is_empty() && !clash).then_some(Self(set))
    }
}

impl std::fmt::Display for DropParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// `secs` as a Duration, at least zero and at most `Duration::MAX`.
fn saturating_secs(secs: f64) -> Duration {
    Duration::try_from_secs_f64(secs.max(0.0)).unwrap_or(Duration::MAX)
}

/// A maximum age of `secs`, where 0 or less means no limit.
pub fn age(secs: f64) -> Duration {
    if secs > 0.0 {
        saturating_secs(secs)
    } else {
        Duration::MAX
    }
}

impl Settings {
    /// The cache's byte cap, and how long it keeps a whole track unused.
    pub fn cache_limits(&self) -> (u64, Duration) {
        let max_bytes = (self.cache_max_gb.max(0.0) * 1e9) as u64;
        let whole_max_age = age(self.whole_max_days * 86400.0);
        (max_bytes, whole_max_age)
    }

    /// The longest track accepted, in seconds, never past `longest` minutes.
    pub fn max_track_secs(&self, longest: u32) -> f64 {
        let longest = f64::from(longest);
        let minutes = if self.max_track_minutes > 0.0 {
            self.max_track_minutes.min(longest)
        } else {
            longest
        };
        minutes * 60.0
    }

    /// Lower the rate to one the format can carry.
    // MP3 is the only format that limits the rate (it has no 96 kHz mode), and LAME does not
    // refuse a missing rate: it resamples to 48 kHz on its own, slowly. Choosing 48 kHz here gives
    // the same audio in a fraction of the time, and the setting shows it.
    //
    // The check lives here because the pair can become invalid in several ways: two flags, two
    // menus, a hand-edited file, and a file written by an older version.
    fn reconcile(&mut self) {
        if self.format.carries(self.rate) {
            return;
        }
        let was = self.rate;
        // The highest rate the format carries is the closest to what was asked.
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
            drop_parts: DropParts::DEFAULT,
            serving: true,
            cache_max_gb: 8.0,
            whole_max_days: 30.0,
            port: DEFAULT_PORT,
            max_track_minutes: 10.0,
        }
    }
}

/// The on-disk form of [`Settings`].
// Every field is optional and stored as the string a flag would take, so the file stays editable
// by hand and an unreadable field loses only that field. Serialising the enums directly would tie
// the file format to their variant names.
//
// Unknown keys are ignored, so a file written by a later version still loads.
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct Stored {
    /// The artefact name, as `--demucs-model` takes it.
    model: Option<String>,
    output_format: Option<String>,
    output_sample_rate: Option<u32>,
    /// Comma-separated, as [`DropParts`] prints it.
    drop_parts: Option<String>,
    serving: Option<bool>,
    cache_max_gb: Option<f64>,
    whole_max_days: Option<f64>,
    port: Option<u16>,
    max_track_minutes: Option<f64>,
}

impl Stored {
    fn from_settings(settings: &Settings) -> Self {
        Self {
            model: Some(settings.preset.artefact().to_owned()),
            output_format: Some(settings.format.to_string()),
            output_sample_rate: Some(settings.rate.hz()),
            drop_parts: Some(settings.drop_parts.to_string()),
            serving: Some(settings.serving),
            cache_max_gb: Some(settings.cache_max_gb),
            whole_max_days: Some(settings.whole_max_days),
            port: Some(settings.port),
            max_track_minutes: Some(settings.max_track_minutes),
        }
    }

    /// Convert to [`Settings`], with defaults for anything absent or unrecognised, and log each
    /// field that was dropped.
    fn into_settings(self) -> Settings {
        let mut settings = Settings::default();

        if let Some(model) = self.model {
            match Preset::from_artefact(&model) {
                Some(preset) => settings.preset = preset,
                // A preset retired between versions means the file is from an older version, and
                // the log says so.
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
        if let Some(list) = self.drop_parts {
            match DropParts::parse(&list) {
                Some(parts) => settings.drop_parts = parts,
                None => warn_ignored("drop_parts", &list, &settings.drop_parts.to_string()),
            }
        }
        if let Some(serving) = self.serving {
            settings.serving = serving;
        }
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let at_least_zero = |v: f64| v.is_finite() && v >= 0.0;
        take(
            "cache_max_gb",
            self.cache_max_gb,
            positive,
            &mut settings.cache_max_gb,
        );
        take(
            "whole_max_days",
            self.whole_max_days,
            at_least_zero,
            &mut settings.whole_max_days,
        );
        take("port", self.port, |port| port > 0, &mut settings.port);
        take(
            "max_track_minutes",
            self.max_track_minutes,
            at_least_zero,
            &mut settings.max_track_minutes,
        );
        settings.reconcile();
        settings
    }
}

/// Store `found` in `slot` when `valid`, else keep the default and say so.
fn take<T: Copy + std::fmt::Display>(
    field: &str,
    found: Option<T>,
    valid: impl Fn(T) -> bool,
    slot: &mut T,
) {
    match found {
        Some(v) if valid(v) => *slot = v,
        Some(v) => warn_ignored(field, &v.to_string(), &slot.to_string()),
        None => {}
    }
}

fn warn_ignored(field: &str, found: &str, fallback: &str) {
    tracing::warn!("settings: ignoring {field} {found:?}, using {fallback}");
}

/// The fields a flag fixed for this run. A pinned field cannot be edited, and the window greys out
/// its control.
///
/// The model is not here: `--demucs-model` decides what loads at launch, but the window still lets
/// the user switch models, since a switch is an explicit choice with a download behind it.
#[derive(Debug, Default, Clone, Copy)]
pub struct Pinned {
    pub format: bool,
    pub rate: bool,
    pub cache_max_gb: bool,
    pub whole_max_days: bool,
    pub port: bool,
    pub max_track_minutes: bool,
}

/// What the file says, and what this run uses. Writes serialise `saved`, so a flag never reaches
/// the file.
// The two differ where a flag overrides a field. A write saves the whole document, so with a single
// copy, changing the rate in the window would also save the value of an `--output-format` flag.
#[derive(Debug, Clone, Copy)]
struct State {
    saved: Settings,
    current: Settings,
}

/// The settings file, and the live copy every reader sees.
pub struct SettingsFile {
    path: PathBuf,
    state: Mutex<State>,
    pinned: Pinned,
}

impl SettingsFile {
    /// Read the file, or start from defaults if it is not usable.
    ///
    /// This never fails: it runs before the server exists, and no state of a preferences file
    /// should fail a launch.
    pub fn open(path: PathBuf) -> Self {
        let saved = load(&path).unwrap_or_else(|err| {
            // An absent file means a first run, so it gets no warning.
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
        // A flag can name an impossible pair just as the file can. Reconcile only `current`; the
        // file keeps what it says.
        state.current.reconcile();
        drop(state);
        self
    }

    /// Apply `--cache-max-gb` and `--whole-max-days`, for this run only.
    pub fn pin_cache(mut self, max_gb: Option<f64>, whole_days: Option<f64>) -> Self {
        let mut state = self.state.lock();
        if let Some(gb) = max_gb {
            state.current.cache_max_gb = gb;
            self.pinned.cache_max_gb = true;
        }
        if let Some(days) = whole_days {
            state.current.whole_max_days = days;
            self.pinned.whole_max_days = true;
        }
        drop(state);
        self
    }

    /// Apply `--bind` and `--max-track-minutes`, for this run only.
    pub fn pin_server(mut self, port: Option<u16>, max_track_minutes: Option<f64>) -> Self {
        let mut state = self.state.lock();
        if let Some(port) = port {
            state.current.port = port;
            self.pinned.port = true;
        }
        if let Some(minutes) = max_track_minutes {
            state.current.max_track_minutes = minutes;
            self.pinned.max_track_minutes = true;
        }
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

    pub fn set_drop_parts(&self, parts: DropParts) {
        self.write(|s| s.drop_parts = parts);
    }

    pub fn set_serving(&self, serving: bool) {
        self.write(|s| s.serving = serving);
    }

    pub fn set_rate(&self, rate: OutputRate) {
        if !self.pinned.rate {
            self.write(|s| s.rate = rate);
        }
    }

    pub fn set_cache_max_gb(&self, gb: f64) {
        if !self.pinned.cache_max_gb {
            self.write(|s| s.cache_max_gb = gb);
        }
    }

    pub fn set_whole_max_days(&self, days: f64) {
        if !self.pinned.whole_max_days {
            self.write(|s| s.whole_max_days = days);
        }
    }

    pub fn set_port(&self, port: u16) {
        if !self.pinned.port {
            self.write(|s| s.port = port);
        }
    }

    pub fn set_max_track_minutes(&self, minutes: f64) {
        if !self.pinned.max_track_minutes {
            self.write(|s| s.max_track_minutes = minutes);
        }
    }

    /// Change both copies and write the file.
    ///
    /// This is reached only for a field that no flag pinned, so applying the change to `saved`
    /// cannot carry a flag into the file. A failed write is logged as a warning; the change still
    /// applies to the running server, and only the next launch misses it.
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
/// The file is rewritten whenever a menu changes, even while the machine shuts down. Writing the
/// real file in place would let an interrupted write leave it half-written, which
/// [`SettingsFile::open`] would then discard.
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

    /// A settings path in a fresh directory, which lives as long as the guard.
    fn scratch() -> (crate::scratch::Dir, PathBuf) {
        let dir = crate::scratch::Dir::new("settings");
        let path = dir.join("settings.json");
        (dir, path)
    }

    #[test]
    fn harmonics_and_its_halves_clear_each_other() {
        let split = DropParts::DEFAULT.toggle("bass");
        assert_eq!(split.to_string(), "vocals,drums,bass");
        assert_eq!(split.request().to_string(), "vocals,bass,other");
        assert_eq!(
            split.toggle("harmonics").to_string(),
            "harmonics,vocals,drums"
        );
        assert_eq!(DropParts::DEFAULT.request(), Parts::DEFAULT);
    }

    #[test]
    fn the_last_ticked_part_cannot_be_unticked() {
        let one = DropParts::parse("vocals").unwrap();
        assert_eq!(one.toggle("vocals"), one);
    }

    #[test]
    fn drop_parts_survive_a_relaunch_and_a_bad_list_is_ignored() {
        let (_dir, path) = scratch();
        let file = SettingsFile::open(path.clone());
        file.set_drop_parts(DropParts::parse("bass,other,vocals,drums").unwrap());
        assert_eq!(
            SettingsFile::open(path).get().drop_parts.to_string(),
            "vocals,drums,bass,other"
        );
        assert!(DropParts::parse("harmonics,bass").is_none());
        assert!(DropParts::parse("piano").is_none());
    }

    #[test]
    fn a_first_run_has_no_file_and_gets_the_defaults() {
        let (_dir, path) = scratch();
        let file = SettingsFile::open(path);
        assert_eq!(file.get(), Settings::default());
    }

    #[test]
    fn what_the_window_changes_is_what_the_next_launch_reads() {
        let (_dir, path) = scratch();
        let file = SettingsFile::open(path.clone());
        file.set_preset(Preset::Balanced);
        file.set_rate(OutputRate::Hz96000);
        file.set_format(StemFormat::Pcm32);

        let reopened = SettingsFile::open(path);
        assert_eq!(reopened.get().preset, Preset::Balanced);
        assert_eq!(reopened.get().rate, OutputRate::Hz96000);
        assert_eq!(reopened.get().format, StemFormat::Pcm32);
    }

    /// The window offers only rates the format carries, but the file can be edited by hand and a
    /// flag can name either half of the pair. Every route ends with a valid pair.
    #[test]
    fn a_format_and_a_rate_that_cannot_go_together_are_reconciled() {
        let (_dir, path) = scratch();
        let file = SettingsFile::open(path.clone());
        file.set_rate(OutputRate::Hz96000);
        file.set_format(StemFormat::Mp3);
        assert_eq!(file.get().rate, OutputRate::Hz48000);

        // The corrected pair survives the round trip.
        assert_eq!(SettingsFile::open(path).get().rate, OutputRate::Hz48000);
    }

    /// A flag naming the other half of an impossible pair is reconciled too, and
    /// without writing the correction to the file.
    #[test]
    fn a_flag_cannot_pin_a_pair_that_does_not_exist() {
        let (_dir, path) = scratch();
        SettingsFile::open(path.clone()).set_rate(OutputRate::Hz96000);

        let pinned = SettingsFile::open(path.clone()).pin(Some(StemFormat::Mp3), None);
        assert_eq!(pinned.get().rate, OutputRate::Hz48000);
        drop(pinned);

        // The file still says 96000: a flag is for one run, including this one.
        let reopened = SettingsFile::open(path);
        assert_eq!(reopened.get().format, StemFormat::Flac);
        assert_eq!(reopened.get().rate, OutputRate::Hz96000);
    }

    /// A flag applies to one run: it changes what the server does without changing what the window
    /// offers next time.
    #[test]
    fn a_flag_overrides_the_file_without_rewriting_it() {
        let (_dir, path) = scratch();
        SettingsFile::open(path.clone()).set_rate(OutputRate::Hz48000);

        let run = SettingsFile::open(path.clone()).pin(Some(StemFormat::Pcm32), None);
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

        let next = SettingsFile::open(path).get();
        assert_eq!(next.format, StemFormat::Flac, "the flag must not persist");
        assert_eq!(next.rate, OutputRate::Hz96000);
    }

    #[test]
    fn the_cache_settings_persist_and_a_flag_pins_them_for_one_run() {
        let (_dir, path) = scratch();
        let file = SettingsFile::open(path.clone());
        file.set_cache_max_gb(16.0);
        file.set_whole_max_days(0.0);
        let next = SettingsFile::open(path.clone()).get();
        assert_eq!((next.cache_max_gb, next.whole_max_days), (16.0, 0.0));
        let (bytes, age) = next.cache_limits();
        assert_eq!(bytes, 16_000_000_000);
        assert_eq!(age, Duration::MAX, "0 days keeps whole tracks until full");

        let run = SettingsFile::open(path.clone()).pin_cache(Some(2.0), None);
        assert_eq!(run.get().cache_max_gb, 2.0, "the flag is in force");
        assert!(run.pinned().cache_max_gb && !run.pinned().whole_max_days);
        run.set_cache_max_gb(32.0);
        run.set_whole_max_days(7.0);
        assert_eq!(run.get().cache_max_gb, 2.0, "pinned for this run");
        let next = SettingsFile::open(path.clone()).get();
        assert_eq!((next.cache_max_gb, next.whole_max_days), (16.0, 7.0));

        std::fs::write(
            &path,
            r#"{"cache_max_gb": -1, "whole_max_days": -3, "serving": false}"#,
        )
        .unwrap();
        let next = SettingsFile::open(path).get();
        assert_eq!((next.cache_max_gb, next.whole_max_days), (8.0, 30.0));
        assert!(!next.serving, "the other fields still load");
    }

    #[test]
    fn the_port_and_the_track_length_persist_and_a_flag_pins_them_for_one_run() {
        let (_dir, path) = scratch();
        let file = SettingsFile::open(path.clone());
        file.set_port(8421);
        file.set_max_track_minutes(0.0);
        let next = SettingsFile::open(path.clone()).get();
        assert_eq!(next.port, 8421);
        assert_eq!(next.max_track_secs(71), 71.0 * 60.0, "0 takes the longest");

        let run = SettingsFile::open(path.clone()).pin_server(Some(9000), Some(20.0));
        assert_eq!(
            (run.get().port, run.get().max_track_secs(71)),
            (9000, 1200.0)
        );
        assert_eq!(
            run.get().max_track_secs(15),
            900.0,
            "memory caps a flag too"
        );
        run.set_port(9100);
        run.set_max_track_minutes(5.0);
        assert_eq!(run.get().port, 9000, "pinned for this run");
        let next = SettingsFile::open(path.clone()).get();
        assert_eq!((next.port, next.max_track_minutes), (8421, 0.0));

        std::fs::write(&path, r#"{"port": 0, "max_track_minutes": -1}"#).unwrap();
        let next = SettingsFile::open(path).get();
        assert_eq!((next.port, next.max_track_minutes), (8420, 10.0));
    }

    /// The file is editable by hand, so it uses the same names as the flags and the API.
    #[test]
    fn the_file_spells_settings_the_way_the_flags_do() {
        let (_dir, path) = scratch();
        let file = SettingsFile::open(path.clone());
        file.set_preset(Preset::Balanced);
        file.set_rate(OutputRate::Hz48000);

        let text = std::fs::read_to_string(&path).expect("written");
        // Use the artefact name, whatever it is now: the file holds the value `--demucs-model`
        // takes.
        let artefact = Preset::Balanced.artefact();
        assert!(text.contains(&format!("\"{artefact}\"")), "{text}");
        assert!(text.contains("\"flac\""), "{text}");
        assert!(text.contains("48000"), "{text}");
    }

    /// No state of the preferences file fails a launch; each case below leaves a usable server.
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
            let (_dir, path) = scratch();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, content).unwrap();
            assert_eq!(
                SettingsFile::open(path).get(),
                Settings::default(),
                "{content:?} should have fallen back to the defaults"
            );
        }
    }

    /// One unreadable field does not discard the fields beside it.
    #[test]
    fn a_bad_field_costs_that_field_and_no_others() {
        let (_dir, path) = scratch();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            format!(
                r#"{{"model": "{}", "output_format": "opus", "output_sample_rate": 96000}}"#,
                Preset::Balanced.artefact()
            ),
        )
        .unwrap();

        let settings = SettingsFile::open(path).get();
        assert_eq!(settings.preset, Preset::Balanced);
        assert_eq!(settings.rate, OutputRate::Hz96000);
        assert_eq!(settings.format, StemFormat::Flac, "the default stands in");
    }

    /// Every preset survives the round trip, so switching to it in the window loads the same preset
    /// at the next launch.
    #[test]
    fn every_preset_the_window_offers_round_trips_through_the_file() {
        for preset in Preset::ALL {
            let (_dir, path) = scratch();
            SettingsFile::open(path.clone()).set_preset(preset);
            assert_eq!(SettingsFile::open(path).get().preset, preset);
        }
    }

    #[test]
    fn every_rate_and_format_round_trips_through_the_file() {
        for rate in OutputRate::ALL {
            for format in [StemFormat::Flac, StemFormat::Pcm16, StemFormat::Pcm32] {
                let (_dir, path) = scratch();
                let file = SettingsFile::open(path.clone());
                file.set_rate(rate);
                file.set_format(format);
                let read = SettingsFile::open(path).get();
                assert_eq!((read.rate, read.format), (rate, format));
            }
        }
    }

    /// An interrupted write cannot leave the real file truncated.
    #[test]
    fn a_write_never_exposes_a_half_written_file() {
        let (_dir, path) = scratch();
        SettingsFile::open(path.clone()).set_rate(OutputRate::Hz48000);

        let staging = path.with_extension("json.new");
        assert!(!staging.exists(), "the temporary file outlived the write");
        assert!(serde_json::from_str::<Stored>(&std::fs::read_to_string(&path).unwrap()).is_ok());
    }

    #[test]
    fn a_whole_track_age_past_what_a_duration_holds_keeps_tracks_forever() {
        for days in [1e300, f64::INFINITY] {
            let settings = Settings {
                whole_max_days: days,
                ..Settings::default()
            };
            assert_eq!(settings.cache_limits().1, Duration::MAX);
        }
    }
}
