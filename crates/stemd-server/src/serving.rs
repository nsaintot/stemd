//! Whether the HTTP API answers, and the mDNS advertisement that goes with it.
//!
//! Turning the server off keeps the port bound, so a second instance still fails
//! at launch; requests get 503 until it is turned back on. Drops in the window
//! work either way.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::discovery::Advertiser;

pub struct Serving {
    shared: Arc<Shared>,
    /// Instance name and port to advertise under; `None` with `--no-mdns`.
    announce: Option<(String, u16)>,
}

#[derive(Default)]
struct Shared {
    on: AtomicBool,
    /// Bumped on every change, so an advertisement that finishes starting after
    /// the switch moved again withdraws itself.
    generation: AtomicU64,
    advertiser: Mutex<Option<Arc<Advertiser>>>,
    /// Held while registering, so a quick off and on cannot register twice.
    registering: Mutex<()>,
}

impl Serving {
    /// Start in `on`, advertising `model` if on and announcing is allowed.
    pub fn new(on: bool, announce: Option<(String, u16)>, model: &str) -> Self {
        let serving = Self {
            shared: Arc::new(Shared {
                on: AtomicBool::new(on),
                ..Shared::default()
            }),
            announce,
        };
        if on {
            serving.advertise(model);
        }
        serving
    }

    pub fn is_on(&self) -> bool {
        self.shared.on.load(Ordering::SeqCst)
    }

    /// Turn the API on or off, advertising `model` when it comes on.
    pub fn set(&self, on: bool, model: &str) {
        if self.shared.on.swap(on, Ordering::SeqCst) == on {
            return;
        }
        if on {
            self.shared.generation.fetch_add(1, Ordering::SeqCst);
            self.advertise(model);
            tracing::info!("server on");
        } else {
            self.withdraw();
            tracing::info!("server off: requests get 503 until it is turned back on");
        }
    }

    /// Take the advertisement down, for turning off and for quitting.
    pub fn withdraw(&self) {
        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        if let Some(advertiser) = self.shared.advertiser.lock().take() {
            advertiser.withdraw();
        }
    }

    /// Announce over mDNS on a background thread: registering can take long
    /// enough to stall the window. A failure is a warning: a client with the
    /// address configured out of band still works.
    fn advertise(&self, model: &str) {
        let Some((instance, port)) = self.announce.clone() else {
            return;
        };
        let shared = Arc::clone(&self.shared);
        let model = model.to_owned();
        let generation = shared.generation.load(Ordering::SeqCst);
        let spawned = std::thread::Builder::new()
            .name("stemd-mdns".into())
            .spawn(move || {
                let _one = shared.registering.lock();
                if shared.generation.load(Ordering::SeqCst) != generation {
                    return;
                }
                match Advertiser::start(&instance, port, &model) {
                    Ok(advertiser) => {
                        let mut slot = shared.advertiser.lock();
                        if shared.generation.load(Ordering::SeqCst) == generation {
                            *slot = Some(Arc::new(advertiser));
                        } else {
                            advertiser.withdraw();
                        }
                    }
                    Err(err) => tracing::warn!("could not advertise over mDNS: {err:#}"),
                }
            });
        if let Err(err) = spawned {
            tracing::warn!("could not advertise over mDNS: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_reports_what_it_was_set_to() {
        let serving = Serving::new(false, None, "model");
        assert!(!serving.is_on());
        serving.set(true, "model");
        assert!(serving.is_on());
        serving.set(false, "model");
        assert!(!serving.is_on());
    }
}
