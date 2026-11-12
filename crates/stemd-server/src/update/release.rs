//! Windows and Linux: check GitHub's newest release at launch, once a day
//! and on a click on the version.
//!
//! On Linux, and for a Windows copy Setup did not install (the zip), a newer
//! release opens its page in the browser. On an installed Windows copy, it
//! offers to download Setup. Once Setup is ready and nothing is running, the
//! app starts it and closes; Setup installs and starts Stemd again.

use std::time::{Duration, Instant, SystemTime};

use parking_lot::Mutex;

use super::github::{self, Release};
use super::{Status, VERSION, repaint};

/// Time between background checks, in wall time so a computer that sleeps still checks daily.
const RECHECK: Duration = Duration::from_secs(24 * 60 * 60);

/// How often the check thread looks at the clock.
const WAKE: Duration = Duration::from_secs(60 * 60);

/// How long "Up to date" and "Check failed" stay in the slot.
const NOTICE: Duration = Duration::from_millis(2600);

/// A download that has not moved for this long has stalled.
#[cfg(windows)]
const STALL: Duration = Duration::from_secs(60);

#[derive(Clone, Debug)]
enum Phase {
    Idle,
    Checking {
        user: bool,
    },
    UpToDate {
        until: Instant,
    },
    /// `after` is what the slot returns to once the notice expires.
    Failed {
        title: &'static str,
        error: String,
        until: Instant,
        after: Option<Release>,
    },
    Available(Release),
    /// The install question is on screen.
    #[cfg(windows)]
    Asking(Release),
    /// `attempt` numbers the download, so the result of an abandoned one is ignored.
    #[cfg(windows)]
    Downloading {
        release: Release,
        fraction: f32,
        moved: Instant,
        attempt: u64,
    },
    /// Setup is downloaded and verified.
    #[cfg(windows)]
    Ready {
        release: Release,
        setup: super::installer::Setup,
        mode: super::installer::Mode,
        launch: Launch,
    },
}

/// Progress of starting Setup.
#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Launch {
    /// Until nothing is running, queued or downloading.
    Waiting,
    /// Hashing the file and waiting on the consent prompt, off the UI thread.
    Starting,
    /// Setup runs; the window closes on the next frame.
    Started,
    /// The window was asked to close.
    Closing,
}

static PHASE: Mutex<Phase> = Mutex::new(Phase::Idle);

fn set(phase: Phase) {
    *PHASE.lock() = phase;
    repaint();
}

pub fn start() {
    #[cfg(windows)]
    super::installer::remove_downloads();
    std::thread::Builder::new()
        .name("update-check".into())
        .spawn(|| {
            let mut last: Option<SystemTime> = None;
            loop {
                let due = last.is_none_or(|t| t.elapsed().is_ok_and(|e| e >= RECHECK));
                if due {
                    check_now(false);
                    last = Some(SystemTime::now());
                }
                std::thread::sleep(WAKE);
            }
        })
        .expect("spawning the update check");
}

/// Ask GitHub for the newest release. `user` is a click, which reports what
/// it found; a background check stays silent unless there is an update.
fn check_now(user: bool) {
    let before = {
        let mut phase = PHASE.lock();
        match &mut *phase {
            // A click during a background check reports its result.
            Phase::Checking { user: reports } => {
                *reports |= user;
                return;
            }
            #[cfg(windows)]
            Phase::Asking(_) | Phase::Downloading { .. } | Phase::Ready { .. } => return,
            _ => {}
        }
        std::mem::replace(&mut *phase, Phase::Checking { user })
    };
    repaint();
    // An update already known survives a failed check.
    let known = match before {
        Phase::Available(release) => Some(release),
        Phase::Failed { after, .. } => after,
        _ => None,
    };
    let latest = github::latest();
    let user = matches!(*PHASE.lock(), Phase::Checking { user: true });
    let until = Instant::now() + NOTICE;
    set(match latest {
        Ok(release) if super::newer(release.version(), VERSION) => {
            tracing::info!("update available: {}", release.version());
            Phase::Available(release)
        }
        Ok(_) if user => Phase::UpToDate { until },
        Ok(_) => Phase::Idle,
        Err(err) => {
            tracing::warn!("update check failed: {err:#}");
            match (user, known) {
                (true, after) => Phase::Failed {
                    title: "Check failed",
                    error: format!("{err:#}"),
                    until,
                    after,
                },
                (false, Some(release)) => Phase::Available(release),
                (false, None) => Phase::Idle,
            }
        }
    });
}

pub fn status() -> Status {
    let mut phase = PHASE.lock();
    let now = Instant::now();
    match &mut *phase {
        Phase::Idle | Phase::Checking { user: false } => Status::Idle,
        Phase::Checking { user: true } => Status::Checking,
        Phase::UpToDate { until } if now >= *until => {
            *phase = Phase::Idle;
            Status::Idle
        }
        Phase::Failed { until, after, .. } if now >= *until => {
            let after = after.take();
            let status = after.as_ref().map_or(Status::Idle, |release| {
                Status::Available(release.version().to_owned())
            });
            *phase = after.map_or(Phase::Idle, Phase::Available);
            status
        }
        Phase::UpToDate { .. } => Status::UpToDate,
        Phase::Failed { title, error, .. } => Status::Failed {
            title,
            error: error.clone(),
        },
        Phase::Available(release) => Status::Available(release.version().to_owned()),
        #[cfg(windows)]
        Phase::Asking(release) => Status::Available(release.version().to_owned()),
        // A blocked read cannot be interrupted, so a stalled download is
        // abandoned, and its thread's result is ignored by `attempt`.
        #[cfg(windows)]
        Phase::Downloading { release, moved, .. } if now.duration_since(*moved) >= STALL => {
            tracing::warn!("update download stalled");
            let error = format!("no data for {} s", STALL.as_secs());
            *phase = Phase::Failed {
                title: "Download failed",
                error: error.clone(),
                until: now + NOTICE,
                after: Some(release.clone()),
            };
            Status::Failed {
                title: "Download failed",
                error,
            }
        }
        #[cfg(windows)]
        Phase::Downloading { fraction, .. } => Status::Downloading(*fraction),
        #[cfg(windows)]
        Phase::Ready { .. } => Status::Ready,
    }
}

pub fn present() -> bool {
    true
}

pub fn can_check() -> bool {
    match &*PHASE.lock() {
        Phase::Checking { user } => !user,
        #[cfg(windows)]
        Phase::Asking(_) | Phase::Downloading { .. } | Phase::Ready { .. } => false,
        _ => true,
    }
}

pub fn check() {
    std::thread::spawn(|| check_now(true));
}

pub fn open() {
    let phase = PHASE.lock().clone();
    if let Phase::Available(release) = phase {
        #[cfg(windows)]
        if installable(&release) {
            set(Phase::Asking(release));
            return;
        }
        super::open_in_browser(&release.html_url);
    }
}

/// Whether the app can install `release`: Setup has a SHA-256 and this account may update it.
#[cfg(windows)]
fn installable(release: &Release) -> bool {
    use super::installer;
    release.installer().and_then(|a| a.sha256()).is_some()
        && installer::install_mode().is_some_and(installer::can_install)
}

#[cfg(windows)]
pub fn asking() -> Option<String> {
    match &*PHASE.lock() {
        Phase::Asking(release) => Some(release.version().to_owned()),
        _ => None,
    }
}

#[cfg(windows)]
pub fn answer(install: bool) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ATTEMPTS: AtomicU64 = AtomicU64::new(0);

    let Phase::Asking(release) = PHASE.lock().clone() else {
        return;
    };
    let mode = match super::installer::install_mode() {
        Some(mode) if install => mode,
        _ => {
            set(Phase::Available(release));
            return;
        }
    };
    let attempt = ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    set(Phase::Downloading {
        release: release.clone(),
        fraction: 0.0,
        moved: Instant::now(),
        attempt,
    });
    std::thread::spawn(move || {
        let progress = |fraction: f32| {
            if let Phase::Downloading {
                fraction: f,
                moved,
                attempt: a,
                ..
            } = &mut *PHASE.lock()
                && *a == attempt
            {
                *f = fraction;
                *moved = Instant::now();
            }
            repaint();
        };
        let result = super::installer::download(&release, progress);
        let mut phase = PHASE.lock();
        if !matches!(&*phase, Phase::Downloading { attempt: a, .. } if *a == attempt) {
            return;
        }
        *phase = match result {
            Ok(setup) => Phase::Ready {
                release,
                setup,
                mode,
                launch: Launch::Waiting,
            },
            Err(err) => {
                tracing::warn!("update download failed: {err:#}");
                Phase::Failed {
                    title: "Download failed",
                    error: format!("{err:#}"),
                    until: Instant::now() + NOTICE,
                    after: Some(release),
                }
            }
        };
        drop(phase);
        repaint();
    });
}

#[cfg(windows)]
pub fn tick(idle: bool) -> bool {
    let mut phase = PHASE.lock();
    let Phase::Ready {
        release,
        setup,
        mode,
        launch,
    } = &mut *phase
    else {
        return false;
    };
    match launch {
        Launch::Waiting if idle => {
            *launch = Launch::Starting;
            let (release, setup, mode) = (release.clone(), setup.clone(), *mode);
            std::thread::spawn(move || {
                let started = setup.run(mode, true);
                let mut phase = PHASE.lock();
                if let Phase::Ready { launch, .. } = &mut *phase {
                    match started {
                        Ok(()) => *launch = Launch::Started,
                        Err(err) => {
                            tracing::warn!("could not start Setup: {err:#}");
                            *phase = Phase::Failed {
                                title: "Install failed",
                                error: format!("{err:#}"),
                                until: Instant::now() + NOTICE,
                                after: Some(release),
                            };
                        }
                    }
                }
                drop(phase);
                repaint();
            });
            false
        }
        Launch::Started => {
            *launch = Launch::Closing;
            true
        }
        _ => false,
    }
}

#[cfg(windows)]
pub fn run_installer() {
    let ready = match &*PHASE.lock() {
        Phase::Ready {
            setup,
            mode,
            launch: Launch::Waiting,
            ..
        } => Some((setup.clone(), *mode)),
        _ => None,
    };
    if let Some((setup, mode)) = ready
        && let Err(err) = setup.run(mode, false)
    {
        tracing::warn!("could not start Setup: {err:#}");
    }
}
