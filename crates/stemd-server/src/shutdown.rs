//! The one way this process ends.
//!
//! Every exit, Cmd-Q, closing the window, SIGINT, SIGTERM, goes through [`now`].
//! Two things have to happen in order, and neither is the default.
//!
//! First the separation worker has to be out of the model. A statically linked C++
//! runtime holding a GPU device and an allocator in globals runs their destructors
//! on the way out, and doing that while a forward pass is still allocating aborts
//! the process from a destructor, where nothing catches it.
//!
//! Then the process leaves without running that teardown at all. Nothing here
//! needs it: stems and the settings file are both installed by rename, so what is
//! on disk is already complete, and the only thing that has to be told is the
//! network, which [`now`] does itself. Those destructors free memory the kernel is
//! about to reclaim anyway.

use std::io::Write;

use crate::api::AppState;
use crate::queue::STOP_GRACE;

unsafe extern "C" {
    /// `_exit(2)`: end the process immediately, without atexit handlers or C++
    /// static destructors. [`std::process::exit`] runs both, which is the whole
    /// problem this module exists to avoid.
    safe fn _exit(status: i32) -> !;
}

/// Stop the worker, withdraw the advertisement, and end the process.
///
/// Never returns, and is safe to reach from any thread.
pub fn now(state: &AppState, reason: &str) -> ! {
    tracing::info!("{reason}, shutting down");
    let began = std::time::Instant::now();

    // Before anything else: nothing below is safe while the model is running.
    if !state.queue.stop(STOP_GRACE) {
        tracing::warn!("quitting with the worker still in the model");
    }
    let worker_stopped = began.elapsed();

    // A withdrawal a client can act on, rather than one it waits out. Costs a
    // few milliseconds and saves every client on the network a TTL of chasing a
    // server that is gone.
    if let Some(advertiser) = &state.advertiser {
        advertiser.withdraw();
    }
    let withdrawn = began.elapsed();

    // Quitting is meant to look instant. It is the last thing anyone sees the
    // program do, and a window that vanishes while the process lingers reads as
    // a crash. Both steps have timeouts measured in seconds, so when one of them
    // is actually used the difference is visible and the log should say which,
    // rather than leaving someone to guess between the model and the network.
    report(worker_stopped, withdrawn - worker_stopped);

    // `_exit` flushes nothing, and a piped stdout is block-buffered, so the last
    // lines of the log would otherwise be lost: including the two above.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    _exit(0)
}

/// Below this nobody perceives a delay, so nothing is said about it.
const NOTICEABLE: std::time::Duration = std::time::Duration::from_millis(400);

/// Anything a person would notice, and nothing they would not.
///
/// A quit that takes a few milliseconds is not worth a line; one that takes
/// seconds is the difference between a program that closes and a program that
/// hangs, and the only useful thing to say about it is which half was slow.
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

    /// The threshold measures human patience, not machinery. If it ever drifts
    /// up to the size of the timeouts it is meant to catch, it stops catching
    /// them: a three second withdrawal has to be above it, by a lot.
    #[test]
    fn the_threshold_stays_smaller_than_what_it_is_watching_for() {
        assert!(NOTICEABLE < Duration::from_secs(1));
        assert!(NOTICEABLE * 4 < crate::queue::STOP_GRACE);
    }

    /// `report` is about what gets said, and both branches have to be reachable
    /// without a real shutdown behind them.
    #[test]
    fn reporting_either_way_does_not_panic() {
        report(Duration::from_millis(1), Duration::from_millis(2));
        report(Duration::from_millis(80), Duration::from_secs(3));
        report(Duration::from_secs(10), Duration::ZERO);
    }
}
