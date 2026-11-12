//! Finding and installing a newer release.
//!
//! | host    | check                                     | install                              |
//! | ------- | ----------------------------------------- | ------------------------------------ |
//! | macOS   | Sparkle, from the release's `appcast.xml` | Sparkle                              |
//! | Windows | GitHub's newest `v*` release              | download Setup, run it once idle     |
//! | Linux   | GitHub's newest `v*` release              | open the release page in the browser |
//!
//! The state is process-wide: the title bar and the status bar read it with
//! [`status`] every frame, and the checks write it from their own threads.

#[cfg(any(not(target_os = "macos"), test))]
mod github;
#[cfg(windows)]
mod installer;
#[cfg(not(target_os = "macos"))]
mod release;
#[cfg(target_os = "macos")]
mod sparkle;

#[cfg(not(target_os = "macos"))]
use release as imp;
#[cfg(target_os = "macos")]
use sparkle as imp;

use std::sync::OnceLock;

use eframe::egui;

/// The running version, such as `0.1.4`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// What the update slot in the title bar shows. The slot is empty while
/// `Idle`.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Idle,
    /// A check the user asked for is running.
    Checking,
    /// A check the user asked for found nothing newer.
    UpToDate,
    /// A check the user asked for, or the download, failed. `title` is what
    /// the slot says, `error` the reason.
    Failed {
        title: &'static str,
        error: String,
    },
    /// A newer release, by its version.
    Available(String),
    /// The installer download, 0 to 1.
    #[cfg_attr(not(windows), allow(dead_code))]
    Downloading(f32),
    /// Setup is downloaded and starts once nothing is running, or when the
    /// app quits.
    #[cfg_attr(not(windows), allow(dead_code))]
    Ready,
}

impl Status {
    /// Whether a click on the slot acts on it.
    pub fn opens(&self) -> bool {
        matches!(self, Self::Available(_))
    }

    /// Whether the slot draws it in the update's gradient.
    pub fn sheen(&self) -> bool {
        matches!(self, Self::Available(_) | Self::Ready)
    }
}

static CTX: OnceLock<egui::Context> = OnceLock::new();

/// Start the updater: Sparkle and its menu item on macOS, a background check
/// elsewhere. Call once, on the main thread.
pub fn start(ctx: &egui::Context) {
    if CTX.set(ctx.clone()).is_ok() {
        imp::start();
    }
}

/// Repaint the window, from any thread.
fn repaint() {
    if let Some(ctx) = CTX.get() {
        ctx.request_repaint();
    }
}

/// What the update slot shows now.
pub fn status() -> Status {
    imp::status()
}

/// Whether this build has an updater: macOS needs Sparkle in the bundle.
pub fn present() -> bool {
    imp::present()
}

/// Whether a click on the version checks for updates.
#[cfg_attr(all(unix, not(target_os = "macos")), allow(dead_code))]
pub fn can_check() -> bool {
    imp::can_check()
}

/// Check for updates and report the result in the slot.
#[cfg_attr(all(unix, not(target_os = "macos")), allow(dead_code))]
pub fn check() {
    imp::check()
}

/// The slot was clicked: act on the update it shows.
pub fn open() {
    imp::open()
}

/// The version the install question is about, while it is on screen.
pub fn asking() -> Option<String> {
    #[cfg(windows)]
    return release::asking();
    #[cfg(not(windows))]
    None
}

/// Answer the install question: download and install, or not now.
pub fn answer(install: bool) {
    #[cfg(windows)]
    release::answer(install);
    #[cfg(not(windows))]
    let _ = install;
}

/// Start Setup once it is ready and `idle`; true when it has started and the window should close.
pub fn tick(idle: bool) -> bool {
    #[cfg(windows)]
    return release::tick(idle);
    #[cfg(not(windows))]
    {
        let _ = idle;
        false
    }
}

/// Start Setup without `/relaunch` if it is ready but not started. Called as the app quits.
pub fn run_installer() {
    #[cfg(windows)]
    release::run_installer();
}

/// Open `url` in the default browser.
#[cfg(not(target_os = "macos"))]
fn open_in_browser(url: &str) {
    #[cfg(windows)]
    let result = installer::shell_open(std::path::Path::new(url), None);
    #[cfg(not(windows))]
    let result = std::process::Command::new("xdg-open")
        .arg(url)
        // The packaged launcher points this at the bundled CUDA libraries.
        .env_remove("LD_LIBRARY_PATH")
        .spawn()
        .map(|mut child| {
            // Wait on the child so it does not stay a zombie.
            std::thread::spawn(move || child.wait());
        })
        .map_err(|e| e.to_string());
    if let Err(err) = result {
        tracing::warn!("could not open {url}: {err}");
    }
}

/// A version. Fields in comparison order: a pre-release sorts before its
/// release, so `0.2.0-rc1` < `0.2.0`, and `rc2` < `rc10`.
// Only the tests use this on macOS, where Sparkle compares versions.
#[cfg_attr(target_os = "macos", allow(dead_code))]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Version {
    numbers: [u64; 3],
    release: bool,
    pre: Vec<(String, u64)>,
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
impl Version {
    /// Parse `1.2.3`, `v1.2.3` or `1.2.3-rc1`.
    fn parse(text: &str) -> Option<Self> {
        let text = text.trim().trim_start_matches('v');
        let (core, pre) = text.split_once('-').unwrap_or((text, ""));
        let mut numbers = [0; 3];
        let mut parts = core.split('.');
        for n in &mut numbers {
            *n = parts.next()?.parse().ok()?;
        }
        if parts.next().is_some() {
            return None;
        }
        Some(Self {
            numbers,
            release: pre.is_empty(),
            pre: pre.split('.').map(split_number).collect(),
        })
    }
}

/// A pre-release part as its text and trailing number: `rc10` is `("rc", 10)`
/// and `2` is `("", 2)`, so numbers compare as numbers.
fn split_number(part: &str) -> (String, u64) {
    let digits = part.len() - part.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    let (text, number) = part.split_at(part.len() - digits);
    (text.to_owned(), number.parse().unwrap_or(0))
}

/// Whether `latest` is newer than `running`. False when either does not parse.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub fn newer(latest: &str, running: &str) -> bool {
    match (Version::parse(latest), Version::parse(running)) {
        (Some(latest), Some(running)) => latest > running,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newer_compares_numbers_not_text() {
        assert!(newer("v0.2.0", "0.1.4"));
        assert!(newer("0.1.10", "0.1.9"));
        assert!(newer("1.0.0", "0.99.99"));
        assert!(!newer("0.1.4", "0.1.4"));
        assert!(!newer("v0.1.3", "0.1.4"));
    }

    #[test]
    fn a_release_is_newer_than_its_pre_release() {
        assert!(newer("0.2.0", "0.2.0-rc1"));
        assert!(!newer("0.2.0-rc1", "0.2.0"));
        assert!(newer("0.2.0-rc2", "0.2.0-rc1"));
        assert!(newer("0.2.0-rc10", "0.2.0-rc2"));
        assert!(newer("0.2.0-beta.11", "0.2.0-beta.2"));
        assert!(newer("0.2.0-rc1", "0.2.0-beta9"));
        assert!(newer("0.2.0-rc1", "0.1.9"));
    }

    #[test]
    fn unparsable_versions_are_never_newer() {
        for bad in ["", "v", "0.2", "0.2.0.1", "latest", "0.x.0"] {
            assert!(!newer(bad, "0.1.4"), "{bad:?}");
            assert!(!newer("0.1.4", bad), "{bad:?}");
        }
    }
}
