//! stemd: host-side stem separation service.
//!
//! ```text
//! cli       parse the command line
//! serve     take the port, then later run the window or wait for a signal
//! startup   load the model, open the caches, start the worker
//! shutdown  the one way the process ends, however it was asked to
//! ```
//!
//! The port is taken first so that a launch which cannot have it stops before it
//! loads a model or clears a cache another server is serving from. See
//! [`serve::bind`].
#![cfg_attr(windows, windows_subsystem = "windows")]

mod api;
mod cache;
mod cli;
#[cfg(windows)]
mod cuda;
mod discovery;
mod drops;
mod ident;
mod jobs;
mod logbuf;
mod models;
mod panics;
mod precision;
mod queue;
mod serve;
mod settings;
mod shutdown;
mod startup;
mod switch;
#[cfg(test)]
mod testkit;
mod ui;

use anyhow::{Context, Result};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::logbuf::{LogBuffer, LogBufferLayer};

fn main() -> Result<()> {
    adopt_parent_console();
    let args = cli::Args::from_process();
    let logs = init_logging();
    // After the subscriber exists, or the first panic is reported to nothing.
    panics::install();
    if args.install_cuda {
        return install_cuda();
    }
    // Before `prepare`, and the reason is in `serve::bind`: a launch that cannot
    // have the port should find that out before it loads a model or touches the
    // cache that another server is using.
    let listener = serve::bind(args.bind)?;
    let addr = listener.local_addr().context("reading the bound address")?;
    let server = startup::prepare(&args, logs, addr)?;
    serve::run(server, listener, args.headless)
}

/// Fetch the CUDA runtime and stop, without loading a model or binding a port.
///
/// Before `startup::prepare`: this runs on a machine where the GPU was refused, so
/// preparing first would spend a minute loading a model at CPU speed for a process
/// about to exit.
///
/// The probe that decides the backend also decides whether there is anything to
/// do: a working GPU wants nothing, and no card wants nothing either.
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

/// Write to the terminal that launched us, if there was one.
///
/// The crate is built for the windows subsystem, because a windowed app that opens
/// a console beside itself looks broken. That choice is made at link time, so
/// `--headless` would otherwise be a server whose log goes nowhere.
///
/// `AttachConsole` gives both: launched from a terminal it adopts that one;
/// double-clicked there is no parent console, the call fails, and the window's own
/// log view is the answer. The failure needs no handling because it is the
/// ordinary case.
///
/// The shell does not wait on a windows-subsystem process, so its prompt returns
/// immediately and the log interleaves with it.
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

/// Install the tracing subscriber, returning the buffer the window and
/// `GET /v1/logs` both read from.
///
/// Four levels:
///
/// * `error`: work that was asked for did not happen.
/// * `warn`: it happened, but not the way it was asked for.
/// * `info`: the server's own life, and the record of what it did to somebody's
///   audio.
/// * `debug`: mechanics. Timings, bookkeeping, and paths that did no work.
///
/// The two sinks are filtered apart. The console takes `info` and up; the buffer
/// keeps `debug` as well, so the window's level dropdown has something to reveal.
/// `RUST_LOG` overrides both at once.
fn init_logging() -> LogBuffer {
    let logs = LogBuffer::new(logbuf::CAPACITY);
    let filter = |default: &str| {
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| default.into())
    };
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_filter(filter("stemd_server=info,stemd_core=info,tower_http=warn")),
        )
        .with(LogBufferLayer::new(logs.clone()).with_filter(filter(
            "stemd_server=debug,stemd_core=debug,tower_http=warn",
        )))
        .init();
    logs
}
