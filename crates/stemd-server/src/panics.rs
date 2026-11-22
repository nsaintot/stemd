//! A panic hook that logs the location and thread through `tracing`, then runs the previous hook.
//!
//! [`std::panic::catch_unwind`] returns only the payload, and an `.app` has no stderr, so the
//! location reaches the window's log view and `GET /v2/logs` only through this hook. It also
//! covers threads nothing catches, such as the one that decodes a drop.

use std::any::Any;
use std::backtrace::{Backtrace, BacktraceStatus};
use std::panic::{Location, PanicHookInfo};

/// Install the hook.
///
/// Call it once, after logging is initialised; a panic before that is reported nowhere.
pub fn install() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        report(info);
        previous(info);
    }));
}

/// Log one panic.
fn report(info: &PanicHookInfo<'_>) {
    let thread = std::thread::current();
    // The worker and drop threads are named, and the thread name says most of what went wrong.
    let named = thread.name().unwrap_or("unnamed").to_owned();
    tracing::error!(
        "panicked at {} on thread {named}: {}",
        at(info.location()),
        message(info.payload())
    );

    // `capture` returns a backtrace only when `RUST_BACKTRACE` is set; an unrequested backtrace
    // would push the explaining lines out of a buffer that holds `logbuf::CAPACITY` lines.
    let trace = Backtrace::capture();
    if matches!(trace.status(), BacktraceStatus::Captured) {
        tracing::error!("backtrace:\n{trace}");
    }
}

/// `file:line:column`, or a phrase when the panic has no location, such as after
/// [`std::panic::resume_unwind`].
fn at(location: Option<&Location<'_>>) -> String {
    location.map_or_else(
        || "an unknown location".to_owned(),
        |l| format!("{}:{}:{}", l.file(), l.line(), l.column()),
    )
}

/// The panic's message from a `&str` or `String` payload.
pub fn message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        return (*text).to_owned();
    }
    if let Some(text) = payload.downcast_ref::<String>() {
        return text.clone();
    }
    "no message".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two payload shapes that `panic!` produces are read.
    #[test]
    fn both_payload_shapes_are_recovered() {
        assert_eq!(message(&"a literal"), "a literal");
        assert_eq!(message(&String::from("a formatted 1")), "a formatted 1");
    }

    /// A payload of another type is reported as an unknown payload.
    #[test]
    fn an_unreadable_payload_says_so() {
        assert_eq!(message(&42u8), "no message");
    }

    /// The location is rendered in the shape a person pastes into an editor, and
    /// the file is the one the panic came from.
    #[test]
    fn a_location_is_rendered_as_file_line_and_column() {
        let rendered = at(Some(Location::caller()));
        assert!(rendered.contains("panics.rs"), "{rendered}");
        assert_eq!(rendered.matches(':').count(), 2, "{rendered}");
    }

    /// A re-raised panic has no location, and the log line still reads as a full sentence after
    /// "panicked at".
    #[test]
    fn a_missing_location_is_still_a_phrase() {
        assert_eq!(at(None), "an unknown location");
    }
}
