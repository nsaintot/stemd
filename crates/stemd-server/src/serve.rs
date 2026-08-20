//! Binding the listener and the two ways the process ends.
//!
//! The GUI must own the main thread on macOS, so the HTTP server always runs on
//! a runtime we drive ourselves rather than via `#[tokio::main]`.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::extract::DefaultBodyLimit;

use crate::api::{self, AppState};
use crate::startup::Server;

/// Take the port, before anything else is done.
///
/// Called ahead of [`crate::startup::prepare`], which is the point of it being
/// separate from [`run`]. Two things follow.
///
/// A second launch fails here, in milliseconds, instead of after the model is
/// loaded, which was four seconds on a warm CUDA machine and over a minute on a
/// CPU-only one.
///
/// And it fails before [`crate::cache::Cache::new`] clears the cache root. Two
/// instances share a root by default, so the second used to empty the first's
/// cache on its way to discovering it was the second.
///
/// Non-blocking because tokio requires it of a listener adopted from std, and
/// doing it here means [`listen`] cannot forget to.
pub fn bind(addr: SocketAddr) -> Result<std::net::TcpListener> {
    let listener = std::net::TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
    listener
        .set_nonblocking(true)
        .context("putting the listener in non-blocking mode")?;
    Ok(listener)
}

/// Serve until the window closes or a signal arrives.
pub fn run(server: Server, listener: std::net::TcpListener, headless: bool) -> Result<()> {
    // The address the socket actually got, which is the one to show and to
    // advertise. It answers `--bind :0`, where the requested address names no
    // port anyone could connect to.
    let bind = listener.local_addr().context("reading the bound address")?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building the tokio runtime")?;

    let serving = runtime.spawn(listen(
        Arc::clone(&server.state),
        listener,
        server.max_upload,
        server.cache_summary,
    ));

    // Both ways out, not just the headless one: a windowed server started from a
    // terminal is still Ctrl-C'd from that terminal, and without this it would
    // die on the default action with the worker still in the model and the mDNS
    // advertisement still standing.
    watch_for_signals(&runtime, &server.state);

    if headless {
        runtime.block_on(serving).context("server task panicked")?
    } else {
        window(&server.state, bind)
    }
}

/// Serve on a listener that is already bound.
///
/// The binding is [`bind`]'s, done before the model was loaded. All this does is
/// hand the socket to tokio, which has to happen on a runtime thread and so
/// cannot happen there.
async fn listen(
    state: Arc<AppState>,
    listener: std::net::TcpListener,
    max_upload: usize,
    cache_summary: String,
) -> Result<()> {
    let defaults = state.settings.get();
    let app = api::router(state)
        .layer(DefaultBodyLimit::max(max_upload))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener =
        tokio::net::TcpListener::from_std(listener).context("handing the listener to tokio")?;
    tracing::info!(
        "listening on http://{} (stems in {cache_summary}, output {} at {} Hz by default)",
        listener.local_addr()?,
        defaults.format,
        defaults.rate.hz()
    );
    axum::serve(listener, app).await.context("server failed")
}

/// Shut down cleanly when the system asks the process to stop.
///
/// A signal does not unwind the stack, so no `Drop` runs: without this the
/// worker would still be in the model when the process tore itself down, and
/// clients would chase a dead server until its TTL expired.
fn watch_for_signals(runtime: &tokio::runtime::Runtime, state: &Arc<AppState>) {
    let state = Arc::clone(state);
    runtime.spawn(async move {
        if asked_to_stop().await {
            crate::shutdown::now(&state, "signal received");
        }
    });
}

/// Resolve once the system asks this process to stop.
///
/// `false` means no handler could be installed, so nothing will ever arrive to
/// wait for and the default action stands. The caller must not shut down on
/// that: it is the absence of a request, not one.
#[cfg(unix)]
async fn asked_to_stop() -> bool {
    let mut term = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(sig) => sig,
        Err(err) => {
            tracing::warn!("no SIGTERM handler: {err}");
            return false;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
    true
}

/// Windows has no signals; the console control handlers are the equivalent.
///
/// Three of them, because SIGINT and SIGTERM do not divide the same way here.
/// `ctrl_c` is Ctrl-C and Ctrl-Break, `ctrl_close` is the console's close box, and
/// `ctrl_shutdown` is the machine going down. All three run on a thread the system
/// spawns and give it a few seconds before killing the process.
#[cfg(windows)]
async fn asked_to_stop() -> bool {
    let (mut close, mut shutdown) = match (
        tokio::signal::windows::ctrl_close(),
        tokio::signal::windows::ctrl_shutdown(),
    ) {
        (Ok(close), Ok(shutdown)) => (close, shutdown),
        (Err(err), _) | (_, Err(err)) => {
            tracing::warn!("no console control handler: {err}");
            return false;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = close.recv() => {}
        _ = shutdown.recv() => {}
    }
    true
}

/// Run the window. Closing it ends the process.
///
/// On macOS `ui::Window::on_exit` is what shuts down, because AppKit calls
/// `exit()` the moment its delegate returns, so the only path reaching the line
/// below is `run_native` failing before the window is up. Windows and Linux return
/// normally when the event loop ends.
fn window(state: &Arc<AppState>, bind: SocketAddr) -> Result<()> {
    crate::ui::run(Arc::clone(state), bind).map_err(|e| anyhow::anyhow!("{e}"))?;
    crate::shutdown::now(state, "window closed")
}
