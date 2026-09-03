//! Where a panic happened, put somewhere the person who hit it can reach.
//!
//! A caught panic gives up its location. [`std::panic::catch_unwind`] hands back
//! the payload and nothing else, so [`crate::queue`]'s worker can say *what* a
//! separation panicked with and never *where*. The default hook does print the
//! location, but it prints it to stderr, and this ships as an `.app`, where
//! stderr goes nowhere. The first bug report against a release accordingly read
//! `the separation panicked: index out of bounds: the len is 0 but the index is
//! 0` with no file, no line, and no way to narrow it from here.
//!
//! So the hook is replaced by one that logs the location through `tracing`, which
//! puts it in the ring buffer behind the window's log view and `GET /v1/logs`.
//! The hook that was there runs afterwards, so a launch from a terminal keeps the
//! output it always had, `RUST_BACKTRACE` included.
//!
//! This is not only for the caught ones. A drop is decoded on its own thread and
//! nothing catches that, so a panic before the job is submitted used to leave a
//! row sitting at `Reading` for ever with nothing written anywhere. It is logged
//! now, which is the difference between a puzzling hang and a report somebody can
//! act on.

use std::any::Any;
use std::backtrace::{Backtrace, BacktraceStatus};
use std::panic::{Location, PanicHookInfo};

/// Install the hook.
///
/// Call once, and after logging is initialised: a panic reported before there is
/// a subscriber to receive it is a panic reported nowhere.
pub fn install() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        report(info);
        previous(info);
    }));
}

/// Log one panic.
///
/// The job it belonged to is not known here. `queue` logs that separately
/// against its own id, so a caught panic leaves two lines: this one saying where,
/// and that one saying which job died of it.
fn report(info: &PanicHookInfo<'_>) {
    let thread = std::thread::current();
    // The worker and the drop threads are both named, and which of them it was
    // is most of the diagnosis.
    let named = thread.name().unwrap_or("unnamed").to_owned();
    tracing::error!(
        "panicked at {} on thread {named}: {}",
        at(info.location()),
        message(info.payload())
    );

    // Only when it was asked for. `capture` is disabled unless `RUST_BACKTRACE`
    // is set, and an unasked-for backtrace would push the lines explaining it out
    // of a buffer that holds `logbuf::CAPACITY` of them.
    let trace = Backtrace::capture();
    if matches!(trace.status(), BacktraceStatus::Captured) {
        tracing::error!("backtrace:\n{trace}");
    }
}

/// `file:line:column`, or a phrase saying there was none.
///
/// A location is an `Option` because a panic raised through
/// [`std::panic::resume_unwind`] carries none, which is exactly what
/// `cache::write_stems` does to re-raise an encoder panic across a thread join.
fn at(location: Option<&Location<'_>>) -> String {
    location.map_or_else(
        || "an unknown location".to_owned(),
        |l| format!("{}:{}:{}", l.file(), l.line(), l.column()),
    )
}

/// What a panic said, as far as it can be recovered.
///
/// A payload is `Any`, and the two shapes `panic!` produces are the only ones
/// worth naming. Anything else is a custom payload whose message, if it has one,
/// is not reachable from here.
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

    /// The two payload shapes `panic!` produces, which are the two a report has
    /// any chance of being useful from.
    #[test]
    fn both_payload_shapes_are_recovered() {
        assert_eq!(message(&"a literal"), "a literal");
        assert_eq!(message(&String::from("a formatted 1")), "a formatted 1");
    }

    /// A payload that is neither says so rather than pretending to be empty.
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

    /// A re-raised panic carries no location, and the line still has to read as
    /// a sentence rather than trailing off after "panicked at".
    #[test]
    fn a_missing_location_is_still_a_phrase() {
        assert_eq!(at(None), "an unknown location");
    }
}
