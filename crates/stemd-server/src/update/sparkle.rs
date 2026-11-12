//! macOS: Sparkle 2, loaded at run time from `Contents/Frameworks`.
//!
//! Sparkle reads the feed named by `SUFeedURL` in `Info.plist`, checks once a
//! day, shows its own window, verifies the `.dmg` against `SUPublicEDKey`,
//! replaces the bundle and relaunches. A click on the version runs a probe,
//! which reports to the slot without a window; a click on "New update
//! available!" opens Sparkle's window. The app menu also gets "Check for
//! Updates…". A build without the framework, such as `cargo run`, has no
//! updater. Every call here runs on the main thread.

use std::cell::RefCell;
use std::time::{Duration, Instant};

use objc2::rc::{Allocated, Retained};
use objc2::runtime::{AnyClass, AnyObject, NSObject};
use objc2::{AllocAnyThread, define_class, msg_send, sel};
use objc2_foundation::{NSBundle, NSString};
use parking_lot::Mutex;

use super::{Status, repaint};

/// How long "Up to date" and "Check failed" stay in the slot.
const NOTICE: Duration = Duration::from_millis(2600);

/// Sparkle's error code for a check that found nothing newer.
const NO_UPDATE: isize = 1001;

/// How long a probe may go unanswered before it counts as failed.
const NO_ANSWER: Duration = Duration::from_secs(60);

#[derive(Clone, Debug)]
enum Phase {
    Idle,
    /// A probe the user asked for.
    Checking {
        since: Instant,
    },
    UpToDate {
        until: Instant,
    },
    Failed {
        error: String,
        until: Instant,
    },
    Available(String),
}

static PHASE: Mutex<Phase> = Mutex::new(Phase::Idle);

fn set(phase: Phase) {
    *PHASE.lock() = phase;
    repaint();
}

thread_local! {
    /// The controller, and its delegate, which it holds weakly.
    static UPDATER: RefCell<Option<(Retained<AnyObject>, Retained<Delegate>)>> =
        const { RefCell::new(None) };
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements and Delegate has no
    // Drop.
    #[unsafe(super(NSObject))]
    #[name = "StemdUpdaterDelegate"]
    struct Delegate;

    // The SPUUpdaterDelegate methods the app answers. Sparkle asks
    // `respondsToSelector:` for each.
    impl Delegate {
        #[unsafe(method(updater:didFindValidUpdate:))]
        fn did_find(&self, _updater: &AnyObject, item: &AnyObject) {
            // SAFETY: SUAppcastItem's `displayVersionString`, an NSString.
            let version: Option<Retained<NSString>> =
                unsafe { msg_send![item, displayVersionString] };
            set(Phase::Available(
                version.map(|v| v.to_string()).unwrap_or_default(),
            ));
        }

        #[unsafe(method(updaterDidNotFindUpdate:))]
        fn did_not_find(&self, _updater: &AnyObject) {
            let checking = matches!(*PHASE.lock(), Phase::Checking { .. });
            set(if checking {
                Phase::UpToDate {
                    until: Instant::now() + NOTICE,
                }
            } else {
                Phase::Idle
            });
        }

        // Called after the two above, so a probe still checking here failed.
        #[unsafe(method(updater:didFinishUpdateCycleForUpdateCheck:error:))]
        fn did_finish(&self, _updater: &AnyObject, _check: isize, error: Option<&AnyObject>) {
            if !matches!(*PHASE.lock(), Phase::Checking { .. }) {
                return;
            }
            let error = error.and_then(|error| {
                // SAFETY: NSError's `code` and `localizedDescription`.
                let code: isize = unsafe { msg_send![error, code] };
                let text: Option<Retained<NSString>> =
                    unsafe { msg_send![error, localizedDescription] };
                (code != NO_UPDATE).then(|| text.map(|t| t.to_string()).unwrap_or_default())
            });
            set(match error {
                Some(error) => {
                    tracing::warn!("update check failed: {error}");
                    Phase::Failed {
                        error,
                        until: Instant::now() + NOTICE,
                    }
                }
                None => Phase::Idle,
            });
        }
    }
);

impl Delegate {
    fn new() -> Retained<Self> {
        let this = Self::alloc().set_ivars(());
        // SAFETY: NSObject's init.
        unsafe { msg_send![super(this), init] }
    }
}

pub fn start() {
    let Some(framework) = std::env::current_exe()
        .ok()
        .and_then(|exe| Some(exe.parent()?.parent()?.join("Frameworks/Sparkle.framework")))
        .filter(|path| path.exists())
    else {
        tracing::debug!("no Sparkle.framework in the bundle, so no updater");
        return;
    };
    let path = NSString::from_str(&framework.to_string_lossy());
    // SAFETY: loading a framework on the main thread.
    let loaded = NSBundle::bundleWithPath(&path).is_some_and(|bundle| unsafe { bundle.load() });
    let Some(class) = loaded
        .then(|| AnyClass::get(c"SPUStandardUpdaterController"))
        .flatten()
    else {
        tracing::warn!("Sparkle.framework would not load, so no updater");
        return;
    };
    let delegate = Delegate::new();
    // SAFETY: SPUStandardUpdaterController's designated initialiser. The
    // delegate is kept beside the controller, which holds it weakly.
    let controller: Option<Retained<AnyObject>> = unsafe {
        let obj: Allocated<AnyObject> = msg_send![class, alloc];
        msg_send![
            obj,
            initWithStartingUpdater: false,
            updaterDelegate: &*delegate,
            userDriverDelegate: Option::<&AnyObject>::None
        ]
    };
    // Started here so a failure is seen: the initialiser would only show an
    // alert, and an updater that did not start ignores every check.
    // SAFETY: SPUStandardUpdaterController's `updater`, and SPUUpdater's
    // `startUpdater:`, which takes a null error pointer.
    let started = controller.as_ref().is_some_and(|controller| unsafe {
        let updater: Retained<AnyObject> = msg_send![&**controller, updater];
        msg_send![&*updater, startUpdater: std::ptr::null_mut::<*mut AnyObject>()]
    });
    let Some(controller) = controller.filter(|_| started) else {
        tracing::warn!("Sparkle's updater would not start: check SUFeedURL and SUPublicEDKey");
        return;
    };
    add_menu_item(&controller);
    UPDATER.with(|u| *u.borrow_mut() = Some((controller, delegate)));
}

/// Put "Check for Updates…" under "About" in the app menu, which winit builds.
fn add_menu_item(controller: &AnyObject) {
    // SAFETY: AppKit calls on the main thread; every object is checked for nil.
    unsafe {
        let Some(app_class) = AnyClass::get(c"NSApplication") else {
            return;
        };
        let app: Option<Retained<AnyObject>> = msg_send![app_class, sharedApplication];
        let Some(app) = app else { return };
        let menu: Option<Retained<AnyObject>> = msg_send![&*app, mainMenu];
        let Some(menu) = menu else { return };
        let count: isize = msg_send![&*menu, numberOfItems];
        if count == 0 {
            return;
        }
        let first: Option<Retained<AnyObject>> = msg_send![&*menu, itemAtIndex: 0isize];
        let submenu: Option<Retained<AnyObject>> =
            first.and_then(|item| msg_send![&*item, submenu]);
        let Some(submenu) = submenu else { return };
        let Some(item_class) = AnyClass::get(c"NSMenuItem") else {
            return;
        };
        let item: Allocated<AnyObject> = msg_send![item_class, alloc];
        let item: Retained<AnyObject> = msg_send![
            item,
            initWithTitle: &*NSString::from_str("Check for Updates…"),
            action: sel!(checkForUpdates:),
            keyEquivalent: &*NSString::from_str("")
        ];
        let _: () = msg_send![&*item, setTarget: controller];
        let _: () = msg_send![&*submenu, insertItem: &*item, atIndex: 1isize];
    }
}

pub fn status() -> Status {
    let mut phase = PHASE.lock();
    match &*phase {
        Phase::Idle => Status::Idle,
        Phase::Checking { since } if since.elapsed() >= NO_ANSWER => {
            let error = format!("no answer from Sparkle in {} s", NO_ANSWER.as_secs());
            *phase = Phase::Failed {
                error: error.clone(),
                until: Instant::now() + NOTICE,
            };
            Status::Failed {
                title: "Check failed",
                error,
            }
        }
        Phase::Checking { .. } => Status::Checking,
        Phase::UpToDate { until } | Phase::Failed { until, .. } if Instant::now() >= *until => {
            *phase = Phase::Idle;
            Status::Idle
        }
        Phase::UpToDate { .. } => Status::UpToDate,
        Phase::Failed { error, .. } => Status::Failed {
            title: "Check failed",
            error: error.clone(),
        },
        Phase::Available(version) => Status::Available(version.clone()),
    }
}

pub fn present() -> bool {
    UPDATER.with(|u| u.borrow().is_some())
}

pub fn can_check() -> bool {
    present() && !matches!(*PHASE.lock(), Phase::Checking { .. })
}

/// The controller, cloned out of the borrow: Sparkle may call the delegate
/// before a call into it returns.
fn controller() -> Option<Retained<AnyObject>> {
    UPDATER.with(|u| u.borrow().as_ref().map(|(c, _)| c.clone()))
}

/// Probe the feed without a window. While Sparkle's window is up, bring it
/// forward instead: Sparkle ignores a probe during a session.
pub fn check() {
    let Some(controller) = controller() else {
        return;
    };
    // SAFETY: SPUStandardUpdaterController's `updater`, and SPUUpdater's
    // `sessionInProgress` and `checkForUpdateInformation`, on the main thread.
    unsafe {
        let updater: Retained<AnyObject> = msg_send![&*controller, updater];
        let busy: bool = msg_send![&*updater, sessionInProgress];
        if busy {
            let _: () = msg_send![&*controller, checkForUpdates: Option::<&AnyObject>::None];
            return;
        }
        set(Phase::Checking {
            since: Instant::now(),
        });
        let _: () = msg_send![&*updater, checkForUpdateInformation];
    }
}

/// Open Sparkle's window on the update the slot shows.
pub fn open() {
    if let Some(controller) = controller() {
        // SAFETY: an IBAction that takes a nil sender.
        unsafe {
            let _: () = msg_send![&*controller, checkForUpdates: Option::<&AnyObject>::None];
        }
    }
}
