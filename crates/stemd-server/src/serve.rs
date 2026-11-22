//! Binding the listener and the two ways the process ends.
//!
//! The GUI must own the main thread on macOS, so the HTTP server always runs on
//! a runtime we drive ourselves rather than via `#[tokio::main]`.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Request, State};

use crate::api::{self};
use crate::cli::Args;
use crate::endpoint::{Endpoint, Moves};
use crate::launch::{Launch, Stage};
use crate::logbuf::LogBuffer;
use crate::settings::SettingsFile;
use crate::state::AppState;

/// Prepare the server, then serve until a signal arrives.
pub fn headless(
    args: &Args,
    logs: LogBuffer,
    settings: Arc<SettingsFile>,
    listener: std::net::TcpListener,
    taken: Option<u16>,
) -> Result<()> {
    let addr = listener.local_addr().context("reading the bound address")?;
    let (endpoint, moves) = Endpoint::new(addr, taken);
    let launch = Arc::new(Launch::default());
    let server = crate::startup::prepare(args, logs, settings, endpoint, &launch)?;
    let runtime = runtime()?;
    launch.finish(Arc::clone(&server.state));
    watch_for_signals(&runtime, &launch);
    runtime
        .block_on(listen(server.state, listener, moves, server.cache_summary))
        .context("serving")
}

/// Open the window at once and prepare the server behind it.
///
/// With no model on disk the window asks which one to download, then shows the
/// download and the load. Serving starts when the model is loaded. A failed
/// startup is shown in the window until it is closed.
pub fn windowed(
    args: Args,
    logs: LogBuffer,
    settings: Arc<SettingsFile>,
    listener: std::net::TcpListener,
    taken: Option<u16>,
) -> Result<()> {
    // The address the socket actually got, which is the one to show and to
    // advertise. It answers `--bind :0`, where the requested address names no
    // port anyone could connect to.
    let bind = listener.local_addr().context("reading the bound address")?;
    let (endpoint, moves) = Endpoint::new(bind, taken);
    let runtime = runtime()?;
    let launch = Arc::new(Launch::windowed());

    // Both ways out, not just the headless one: a windowed server started from a
    // terminal is still Ctrl-C'd from that terminal, and without this it would
    // die on the default action with the worker still in the model and the mDNS
    // advertisement still standing.
    watch_for_signals(&runtime, &launch);

    let handle = runtime.handle().clone();
    let behind = Arc::clone(&launch);
    std::thread::Builder::new()
        .name("stemd-startup".into())
        .spawn(move || {
            let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::startup::prepare(&args, logs, settings, endpoint, &behind)
            }));
            match prepared {
                Ok(Ok(server)) => {
                    let state = Arc::clone(&server.state);
                    handle.spawn(async move {
                        if let Err(err) =
                            listen(server.state, listener, moves, server.cache_summary).await
                        {
                            tracing::error!("{err:#}");
                        }
                    });
                    behind.finish(state);
                }
                Ok(Err(err)) => {
                    tracing::error!("startup failed: {err:#}");
                    behind.set(Stage::Failed(format!("{err:#}")));
                }
                // The panic hook has already logged where.
                Err(_) => behind.set(Stage::Failed(
                    "startup panicked; the log has the details".into(),
                )),
            }
        })
        .context("spawning the startup thread")?;

    window(&launch)
}

fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building the tokio runtime")
}

/// Serve on a listener that is already bound, then on each one `moves` hands
/// over. The first binding is [`crate::endpoint::bind`]'s, done before the model was loaded.
async fn listen(
    state: Arc<AppState>,
    listener: std::net::TcpListener,
    moves: Moves,
    cache_summary: String,
) -> Result<()> {
    let defaults = state.settings.get();
    let app = api::router(Arc::clone(&state))
        .layer(axum::middleware::from_fn_with_state(state, limit_body))
        .layer(DefaultBodyLimit::disable())
        .layer(tower_http::trace::TraceLayer::new_for_http());

    tracing::info!(
        "listening on http://{} (stems in {cache_summary}, output {} at {} Hz by default)",
        listener.local_addr()?,
        defaults.format,
        defaults.rate.hz()
    );
    serve_moving(app, listener, moves).await
}

/// Serve `app` on `listener`, then on each listener `moves` hands over. The old
/// listener stops accepting; its open requests finish on their own.
async fn serve_moving(
    app: axum::Router,
    mut listener: std::net::TcpListener,
    mut moves: Moves,
) -> Result<()> {
    loop {
        let tokio_listener =
            tokio::net::TcpListener::from_std(listener).context("handing the listener to tokio")?;
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let served = axum::serve(
            tokio_listener,
            app.clone()
                .into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async {
            let _ = stopped.await;
        });
        let mut task = tokio::spawn(served.into_future());
        tokio::select! {
            next = moves.recv() => match next {
                Some(next) => {
                    let _ = stop.send(());
                    listener = next;
                }
                None => return task.await.context("server task")?.context("server failed"),
            },
            done = &mut task => return done.context("server task")?.context("server failed"),
        }
    }
}

/// Refuse a body past the longest track the settings allow, read per request so
/// a change applies at once.
async fn limit_body(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let limit = state.max_body_bytes();
    next.run(request.map(|body| axum::body::Body::new(http_body_util::Limited::new(body, limit))))
        .await
}

/// Shut down cleanly when the system asks the process to stop.
///
/// A signal does not unwind the stack, so no `Drop` runs: without this the
/// worker would still be in the model when the process tore itself down, and
/// clients would chase a dead server until its TTL expired.
fn watch_for_signals(runtime: &tokio::runtime::Runtime, launch: &Arc<Launch>) {
    let launch = Arc::clone(launch);
    runtime.spawn(async move {
        if asked_to_stop().await {
            crate::shutdown::quit(&launch, "signal received");
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
fn window(launch: &Arc<Launch>) -> Result<()> {
    crate::ui::run(Arc::clone(launch)).map_err(|e| anyhow::anyhow!("{e}"))?;
    crate::shutdown::quit(launch, "window closed")
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get(addr: std::net::SocketAddr) -> std::io::Result<String> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(addr).await?;
        stream
            .write_all(b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .await?;
        let mut reply = String::new();
        stream.read_to_string(&mut reply).await?;
        Ok(reply)
    }

    #[tokio::test]
    async fn a_move_serves_on_the_new_port_and_closes_the_old_one() {
        let app = axum::Router::new().route("/", axum::routing::get(|| async { "here" }));
        let first = crate::endpoint::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let old = first.local_addr().unwrap();
        let (endpoint, moves) = Endpoint::new(old, None);
        tokio::spawn(serve_moving(app, first, moves));
        assert!(get(old).await.unwrap().ends_with("here"));

        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = free.local_addr().unwrap().port();
        drop(free);
        endpoint.move_to(port).unwrap();
        let new = std::net::SocketAddr::new(old.ip(), port);
        let mut reply = Err(std::io::ErrorKind::NotConnected.into());
        for _ in 0..50 {
            reply = get(new).await;
            if reply.is_ok() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert!(reply.unwrap().ends_with("here"));
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(get(old).await.is_err(), "the old port still answers");
    }
}
