//! In-memory ring buffer of recent log lines.
//!
//! Exposed at `GET /v1/logs` so a headless server is still debuggable, and so
//! the optional desktop window is a thin client over the same buffer rather
//! than a second logging path.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::Serialize;
use tracing::field::{Field, Visit};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

#[derive(Debug, Clone, Serialize)]
pub struct LogLine {
    /// Milliseconds since the Unix epoch.
    pub at_ms: u64,
    pub level: String,
    pub target: String,
    pub message: String,
}

#[derive(Clone, Default)]
pub struct LogBuffer {
    lines: Arc<Mutex<VecDeque<LogLine>>>,
    capacity: usize,
}

/// Lines held in memory, and the most the window asks for.
///
/// One constant because the two must agree: a window showing fewer than the
/// buffer holds would hide lines that `GET /v1/logs` still serves.
pub const CAPACITY: usize = 2000;

impl LogBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            lines: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
        }
    }

    pub fn push(&self, line: LogLine) {
        let mut lines = self.lines.lock();
        if lines.len() == self.capacity {
            lines.pop_front();
        }
        lines.push_back(line);
    }

    /// Most recent `limit` lines, oldest first.
    pub fn recent(&self, limit: usize) -> Vec<LogLine> {
        let lines = self.lines.lock();
        let skip = lines.len().saturating_sub(limit);
        lines.iter().skip(skip).cloned().collect()
    }

    /// Drop everything buffered. Returns how many lines went.
    ///
    /// This is the buffer `GET /v1/logs` reads, so clearing it in the window
    /// clears it for a client too. That is the right behaviour for what it is:
    /// a shared tail of what just happened, not a record, but it does mean the
    /// button is not local to the window.
    pub fn clear(&self) -> usize {
        let mut lines = self.lines.lock();
        let dropped = lines.len();
        lines.clear();
        dropped
    }
}

/// Tracing layer that feeds a [`LogBuffer`].
pub struct LogBufferLayer {
    buffer: LogBuffer,
}

impl LogBufferLayer {
    pub const fn new(buffer: LogBuffer) -> Self {
        Self { buffer }
    }
}

impl<S: tracing::Subscriber> Layer<S> for LogBufferLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let meta = event.metadata();
        self.buffer.push(LogLine {
            at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or_default(),
            level: meta.level().to_string(),
            target: meta.target().to_owned(),
            message: visitor.message,
        });
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else {
            if !self.message.is_empty() {
                self.message.push(' ');
            }
            let _ = write!(self.message, "{}={value:?}", field.name());
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            if !self.message.is_empty() {
                self.message.push(' ');
            }
            let _ = write!(self.message, "{}={value}", field.name());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_drops_oldest() {
        let buf = LogBuffer::new(2);
        for i in 0..5 {
            buf.push(LogLine {
                at_ms: i,
                level: "INFO".into(),
                target: "t".into(),
                message: format!("line {i}"),
            });
        }
        let recent = buf.recent(10);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].message, "line 3");
        assert_eq!(recent[1].message, "line 4");
    }
}
