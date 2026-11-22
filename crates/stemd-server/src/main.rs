//! stemd, a stem separation server with a desktop window.
//!
//! ```text
//! cli       parse the command line
//! serve     take the port, then later run the window or wait for a signal
//! startup   load the model, open the caches, start the worker
//! shutdown  the one way the process ends, however it was asked to
//! ```
//!
//! The cache root is locked and the port taken first, so a second launch stops
//! before it loads a model or clears the cache of the server already running.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod api;
mod cache;
mod cli;
#[cfg(windows)]
mod cuda;
mod discovery;
mod disk;
mod drops;
mod endpoint;
mod filters;
mod held;
#[cfg(test)]
mod http_tests;
mod ident;
mod jobs;
mod launch;
mod logbuf;
mod migrate;
mod models;
mod output;
mod panics;
mod precision;
mod queue;
#[cfg(test)]
mod scratch;
mod separations;
mod serve;
mod serving;
mod session_api;
#[cfg(test)]
mod session_tests;
mod sessions;
mod settings;
mod shutdown;
mod startup;
mod state;
mod stem_http;
mod switch;
#[cfg(test)]
mod testkit;
mod ui;
mod update;
mod v1;
mod whole;

use anyhow::Result;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::logbuf::{LogBuffer, LogBufferLayer};

fn main() -> Result<()> {
    adopt_parent_console();
    #[cfg(windows)]
    cuda::point_at_bundled_headers();
    let args = cli::Args::from_process();
    let logs = init_logging();
    // Install after the subscriber exists, so the first panic is reported.
    panics::install();
    if args.install_cuda {
        return install_cuda();
    }
    // Lock before `prepare`, which clears the cache root, and hold the lock until the process ends.
    let _instance = cache::lock(&args.cache_root()?)?;
    let settings = std::sync::Arc::new(startup::open_settings(&args)?);
    let (listener, taken) = match args.bind {
        Some(addr) => (endpoint::bind(addr)?, None),
        None => endpoint::bind_saved(settings.get().port)?,
    };
    if args.headless {
        serve::headless(&args, logs, settings, listener, taken)
    } else {
        serve::windowed(args, logs, settings, listener, taken)
    }
}

/// Fetch the CUDA runtime and stop, without loading a model or binding a port.
///
/// This runs before `startup::prepare`: on a machine where the GPU was refused, preparing first
/// would spend a minute loading a model at CPU speed for a process about to exit.
///
/// The probe that picks the backend also decides whether there is anything to install: nothing is
/// needed with a working GPU or with no NVIDIA card.
#[cfg(windows)]
fn install_cuda() -> Result<()> {
    use stemd_core::Accelerator;

    let backend = Accelerator::detect();
    if backend != Accelerator::Cpu {
        println!("CUDA is already usable here ({backend}). Nothing to install.");
        return Ok(());
    }
    if !Accelerator::gpu_refused() {
        anyhow::bail!(
            "no NVIDIA driver here, so there is no GPU for the CUDA runtime to \
             reach. This build runs on the CPU on this machine whatever is \
             installed beside it."
        );
    }

    let into = cuda::beside_the_executable()?;
    let placed = cuda::install(&into, |name, done, total| {
        if total > 0 {
            tracing::info!(
                "{name}: {:.0} MB of {:.0} MB",
                done as f64 / 1e6,
                total as f64 / 1e6
            );
        }
    })?;
    println!(
        "{placed} CUDA libraries installed in {}. Start stemd again to use the GPU.",
        into.display()
    );
    Ok(())
}

#[cfg(not(windows))]
fn install_cuda() -> Result<()> {
    anyhow::bail!(
        "--install-cuda is for Windows, where CUDA is delay-loaded and can be \
         added after the fact. Here the libraries are ordinary link-time \
         dependencies, so install them through the package manager and rebuild."
    )
}

/// Write to the terminal that launched the process, if there is one. Without this, `--headless`
/// would send its log nowhere, because the crate is built for the windows subsystem.
// `AttachConsole` adopts the console of the terminal that launched the process. When the app is
// started by a double click, there is no parent console and the call fails; the window's log view
// then shows the log, and the failure needs no handling.
//
// The shell does not wait for a windows-subsystem process, so its prompt returns at once and the
// log interleaves with it.
#[cfg(windows)]
fn adopt_parent_console() {
    const ATTACH_PARENT_PROCESS: u32 = u32::MAX;
    unsafe extern "system" {
        fn AttachConsole(process_id: u32) -> i32;
    }
    unsafe { AttachConsole(ATTACH_PARENT_PROCESS) };
}

#[cfg(not(windows))]
fn adopt_parent_console() {}

/// Install the tracing subscriber and return the buffer that the window and `GET /v2/logs` read.
///
/// Levels:
///
/// * `error`: work that was asked for did not happen.
/// * `warn`: the work happened differently than asked.
/// * `info`: the server's lifecycle, and what it did to a client's audio.
/// * `debug`: mechanics, such as timings, bookkeeping, and code paths that did no work.
///
/// The two sinks filter separately: the console shows `info` and above, and the buffer also keeps
/// `debug` for the window's level dropdown. `RUST_LOG` overrides both.
fn init_logging() -> LogBuffer {
    let logs = LogBuffer::new(logbuf::CAPACITY);
    let filter = |default: &str| {
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| default.into())
    };
    // Use colour only on a terminal: the installer shows --install-cuda's progress in its own
    // window, and people read redirected logs as plain text.
    let ansi = std::io::IsTerminal::is_terminal(&std::io::stdout());
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(ansi)
                .with_filter(filter("stemd_server=info,stemd_core=info,tower_http=warn")),
        )
        .with(LogBufferLayer::new(logs.clone()).with_filter(filter(
            "stemd_server=debug,stemd_core=debug,tower_http=warn",
        )))
        .init();
    logs
}
