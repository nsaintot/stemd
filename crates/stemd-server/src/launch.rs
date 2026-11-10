//! Startup progress, shared between the thread preparing the server and the
//! window that shows it.
//!
//! The window opens before the model is ready, so a first run that downloads
//! hundreds of megabytes shows a progress bar instead of nothing.

use std::sync::{Arc, OnceLock};

use parking_lot::{Condvar, Mutex};

use crate::api::AppState;
use crate::models::Preset;

/// What startup is doing, for the window to show.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Stage {
    /// Detecting the GPU and opening the caches.
    #[default]
    Starting,
    /// No model on disk yet: waiting for the user to pick one to download.
    /// Holds the presets already installed, if any.
    Choosing(Vec<Preset>),
    /// Fetching the model's weights.
    Downloading {
        preset: Preset,
        file: String,
        done: u64,
        total: u64,
    },
    /// Hashing weights already on disk.
    Verifying(Preset),
    /// Building the model on the separation worker. Holds the name to show: a
    /// preset's label, or the artefact given to `--demucs-model`.
    Loading(String),
    /// Startup stopped. The message is the error chain.
    Failed(String),
}

#[derive(Default)]
pub struct Launch {
    stage: Mutex<Stage>,
    ready: OnceLock<Arc<AppState>>,
    /// True when a window can ask which model to download.
    asks: bool,
    choice: Mutex<Option<Preset>>,
    chosen: Condvar,
}

impl Launch {
    /// A launch behind a window, which asks for a model instead of fetching one.
    pub fn windowed() -> Self {
        Self {
            asks: true,
            ..Self::default()
        }
    }

    pub fn asks(&self) -> bool {
        self.asks
    }

    /// Show the model choice and block until the window answers.
    pub fn choose(&self, installed: Vec<Preset>) -> Preset {
        self.set(Stage::Choosing(installed));
        let mut choice = self.choice.lock();
        loop {
            if let Some(preset) = choice.take() {
                return preset;
            }
            self.chosen.wait(&mut choice);
        }
    }

    /// Answer [`Self::choose`].
    pub fn pick(&self, preset: Preset) {
        *self.choice.lock() = Some(preset);
        self.chosen.notify_all();
    }

    pub fn stage(&self) -> Stage {
        self.stage.lock().clone()
    }

    pub fn set(&self, stage: Stage) {
        *self.stage.lock() = stage;
    }

    /// The running server, once startup has finished.
    pub fn ready(&self) -> Option<&Arc<AppState>> {
        self.ready.get()
    }

    pub fn finish(&self, state: Arc<AppState>) {
        let _ = self.ready.set(state);
    }
}
