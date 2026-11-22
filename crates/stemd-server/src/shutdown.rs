//! Ends the process.
//!
//! Every exit goes through [`now`]: Cmd-Q, closing the window, SIGINT and SIGTERM. It does two
//! things in order.
//!
//! First, it stops the separation worker. MLX is a statically linked C++ runtime that holds a GPU
//! device and an allocator in globals; their destructors run at exit, and running them while a
//! forward pass is still allocating aborts the process from a destructor, where nothing can catch
//! it.
//!
//! Then it exits without running those destructors. Stems and the settings file are installed by
//! rename, so the files on disk are already complete; the network is the only party to notify, and
//! [`now`] withdraws the advertisement itself. The destructors only free memory that the kernel
//! reclaims anyway. With `--ephemeral`, [`now`] also empties the cache.

use std::io::Write;

use crate::launch::Launch;
use crate::queue::STOP_GRACE;
use crate::state::AppState;

unsafe extern "C" {
    /// `_exit(2)`: end the process at once, without atexit handlers or C++ static destructors.
    /// [`std::process::exit`] runs both, which this module must avoid.
    safe fn _exit(status: i32) -> !;
}

/// End the process, whether or not startup has finished.
///
/// During startup there is no worker or advertisement to stop. A download in progress leaves only
/// its `.part` file, which the next run replaces.
pub fn quit(launch: &Launch, reason: &str) -> ! {
    match launch.ready() {
        Some(state) => now(state, reason),
        None => {
            tracing::info!("{reason} during startup, shutting down");
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            _exit(0)
        }
    }
}

/// Stop the worker, withdraw the advertisement, and end the process.
///
/// It never returns and is safe to call from any thread.
pub fn now(state: &AppState, reason: &str) -> ! {
    tracing::info!("{reason}, shutting down");
    let began = std::time::Instant::now();

    // Stop the worker first: nothing below is safe while the model runs.
    if !state.queue.stop(STOP_GRACE) {
        tracing::warn!("quitting with the worker still in the model");
    }
    let worker_stopped = began.elapsed();

    // Withdraw the advertisement so clients learn at once that the server is gone; this costs a few
    // milliseconds and spares every client a TTL of retries.
    state.serving.withdraw();
    let withdrawn = began.elapsed();

    if state.ephemeral {
        state.cache.wipe();
    }

    // Quitting should look instant; a window that vanishes while the process lingers looks like a
    // crash. Both steps have timeouts measured in seconds, so when one of them is slow, the log
    // says which one.
    report(worker_stopped, withdrawn - worker_stopped);

    // `_exit` flushes nothing and a piped stdout is block-buffered, so flush here, or the last log
    // lines, including the two above, are lost.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    _exit(0)
}

/// Quits faster than this are not logged, since nobody notices the delay.
const NOTICEABLE: std::time::Duration = std::time::Duration::from_millis(400);

/// Log a quit slow enough for a person to notice, and which of the two steps was slow.
fn report(worker: std::time::Duration, advertisement: std::time::Duration) {
    let total = worker + advertisement;
    if total < NOTICEABLE {
        tracing::debug!("shut down in {total:.0?}");
        return;
    }
    tracing::warn!(
        "shutting down took {total:.1?}: {worker:.1?} stopping the worker, \
         {advertisement:.1?} withdrawing the advertisement"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The threshold stays far below the timeouts it is meant to catch: a three-second withdrawal
    /// must be well above it.
    #[test]
    fn the_threshold_stays_smaller_than_what_it_is_watching_for() {
        assert!(NOTICEABLE < Duration::from_secs(1));
        assert!(NOTICEABLE * 4 < crate::queue::STOP_GRACE);
    }

    /// Both branches of `report` are reachable without a real shutdown.
    #[test]
    fn reporting_either_way_does_not_panic() {
        report(Duration::from_millis(1), Duration::from_millis(2));
        report(Duration::from_millis(80), Duration::from_secs(3));
        report(Duration::from_secs(10), Duration::ZERO);
    }
}
