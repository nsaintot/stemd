//! The Clients list, which shows API clients' whole-track jobs and sessions, and the number of
//! session segments waiting.
//!
//! Rows name jobs and sessions by number, such as "Job 2" and "Session 3"; the list never reads or
//! shows a track id or title. Client jobs are listed in queue order with the running job first, as
//! queued drops are listed under Separated. The list shows only while it has rows.

use std::time::Duration;

use eframe::egui::{self, Rect, Sense, Stroke, pos2, vec2};
use stemd_core::Progress;

use crate::sessions::{Activity, Ended};
use crate::state::AppState;

use super::eta::{Eta, left_text};
use super::icons;
use super::recents::ROW_HEIGHT;
use super::status::percent;
use super::theme::{self, Brushes, Palette, Weight, white};

const RADIUS: f32 = 16.0;

/// The number of rows shown before the list scrolls; the partly visible row fades out.
const MAX_VISIBLE: f32 = 2.75;

/// How long an ended session stays in the list.
const ENDED_FOR: Duration = Duration::from_secs(15);

/// A client job in the list.
#[derive(Clone, Debug)]
pub struct JobRow {
    /// The API job's number, counting from 1 since the server started.
    pub number: u64,
    pub progress: Progress,
    /// Seconds left, once there is an estimate. Only a job that is separating has one.
    pub left: Option<f64>,
    /// True when a protocol 1 client waits on the job.
    pub protocol_1: bool,
}

/// What the list shows, read from the server state at most once per refresh.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    /// Client jobs, the running one first, then the waiting ones in queue order.
    pub jobs: Vec<JobRow>,
    pub sessions: Vec<Activity>,
    /// Sessions that ended within [`ENDED_FOR`], oldest first.
    pub ended: Vec<Ended>,
}

impl Snapshot {
    /// Read the server state. `eta` follows the running client job, and `now` is the window's
    /// clock.
    pub fn read(state: &AppState, eta: &mut Eta, now: f64) -> Self {
        let queue = &state.queue;
        let segment_secs = queue.segment_run_seconds();
        let jobs = queue
            .client_jobs()
            .into_iter()
            .map(|client| JobRow {
                number: client.number,
                protocol_1: client.protocol_1,
                left: client
                    .running
                    .then(|| eta.update(client.number, now, &client.progress, segment_secs))
                    .flatten(),
                progress: client.progress,
            })
            .collect();
        let sessions = queue.sessions();
        Self {
            jobs,
            sessions: sessions.activity(),
            ended: sessions
                .ended()
                .into_iter()
                .filter(|e| e.at.elapsed() < ENDED_FOR)
                .collect(),
        }
    }

    /// The number of rows in the list: client jobs, open sessions and ended sessions.
    pub fn rows(&self) -> usize {
        self.jobs.len() + self.sessions.len() + self.ended.len()
    }

    /// Whether the list shows at all.
    pub fn any(&self) -> bool {
        self.rows() > 0
    }
}

/// The height the list takes for `snapshot`, heading included; 0 when the list is hidden.
pub fn height(snapshot: &Snapshot) -> f32 {
    if !snapshot.any() {
        return 0.0;
    }
    let rows = (snapshot.rows() as f32).min(MAX_VISIBLE);
    super::section::HEIGHT + 6.0 + ROW_HEIGHT * rows + 6.0
}

/// `m:ss` for a length in samples.
pub fn length(frames: usize, sample_rate: u32) -> String {
    let secs = frames / sample_rate.max(1) as usize;
    format!("{}:{:02}", secs / 60, secs % 60)
}

/// The number of session segments waiting, shown at the right of the heading; empty when none are
/// waiting.
pub fn queue_text(snapshot: &Snapshot) -> String {
    let high: usize = snapshot.sessions.iter().map(|s| s.high).sum();
    let normal: usize = snapshot.sessions.iter().map(|s| s.normal).sum();
    if high + normal == 0 {
        return String::new();
    }
    format!("{high} high, {normal} normal")
}

/// What a session is doing, which picks its glyph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionState {
    Separating,
    /// Planned segments wait for audio the client has not uploaded.
    Uploading,
    /// Planned segments wait for something else, such as the statistics.
    Waiting,
    Complete,
    Idle,
}

pub fn session_state(session: &Activity) -> SessionState {
    let planned = session.high + session.normal > 0;
    if session.runnable {
        SessionState::Separating
    } else if planned && session.arrived < session.frames {
        SessionState::Uploading
    } else if planned {
        SessionState::Waiting
    } else if session.covered >= session.frames {
        SessionState::Complete
    } else {
        SessionState::Idle
    }
}

/// A session's row detail, such as `uploading 25% · 20/40 pieces`.
pub fn session_detail(session: &Activity) -> String {
    let dot = theme::DOT;
    let pieces = format!("{}/{} pieces", session.ready, session.pieces);
    match session_state(session) {
        SessionState::Uploading => format!(
            "uploading {}{dot}{pieces}",
            percent(session.arrived as f32 / session.frames.max(1) as f32)
        ),
        SessionState::Waiting => format!("waiting{dot}{pieces}"),
        SessionState::Separating | SessionState::Complete | SessionState::Idle => pieces,
    }
}

/// The tooltip on a session row, with every count in full.
fn session_tooltip(session: &Activity) -> String {
    format!(
        "Uploaded: {}\n\
         Ready: {} of {} pieces, {}\n\
         Waiting: {} high, {} normal",
        percent(session.arrived as f32 / session.frames.max(1) as f32),
        session.ready,
        session.pieces,
        percent(session.covered as f32 / session.frames.max(1) as f32),
        session.high,
        session.normal,
    )
}

/// An ended session's detail, and whether it ended on a failure.
pub fn ended_detail(ended: &Ended) -> (String, bool) {
    match ended.reason.as_str() {
        "idle" => ("ended: idle".to_owned(), false),
        reason if reason.starts_with("segment ") => (format!("failed: {reason}"), true),
        reason if reason.starts_with("ended") => (reason.to_owned(), false),
        reason => (format!("ended: {reason}"), false),
    }
}

/// A client job's detail, such as `separating · 39% · about 40 s left`. A
/// waiting job reads `waiting for the worker`, as a waiting drop does.
pub fn job_detail(progress: &Progress, left: Option<f64>) -> String {
    let dot = theme::DOT;
    if progress.stage == stemd_core::Stage::Queued {
        return super::dropzone::stage_label(progress.stage).to_owned();
    }
    let detail = format!(
        "{}{dot}{}",
        super::dropzone::stage_label(progress.stage),
        percent(progress.fraction)
    );
    match left {
        Some(secs) => format!("{detail}{dot}{}", left_text(secs)),
        None => detail,
    }
}

fn queued(job: &JobRow) -> bool {
    job.progress.stage == stemd_core::Stage::Queued
}

enum Row<'a> {
    Job(&'a JobRow),
    Session(&'a Activity),
    Ended(&'a Ended),
}

/// Draw the heading and the list.
pub fn show(ui: &mut egui::Ui, palette: &Palette, brushes: &Brushes, snapshot: &Snapshot) {
    let open = snapshot.jobs.len() + snapshot.sessions.len();
    let summary = queue_text(snapshot);
    super::section::heading_with(
        ui,
        palette,
        "Clients",
        (open > 0).then(|| open.to_string()),
        (!summary.is_empty()).then_some(summary),
    );

    let rows: Vec<Row> = snapshot
        .jobs
        .iter()
        .map(Row::Job)
        .chain(snapshot.sessions.iter().map(Row::Session))
        .chain(snapshot.ended.iter().map(Row::Ended))
        .collect();

    let height = ROW_HEIGHT * (rows.len() as f32).min(MAX_VISIBLE);
    let now = ui.input(|i| i.time) as f32;
    if snapshot.jobs.iter().any(|j| !queued(j)) || snapshot.sessions.iter().any(|s| s.runnable) {
        // Spinners turn every frame.
        theme::request_frame(ui.ctx());
    }
    super::widgets::glass_list(ui, brushes, (height, RADIUS, ROW_HEIGHT), "clients", |ui| {
        for (n, item) in rows.iter().enumerate() {
            row(ui, palette, brushes, item, n == 0, now);
        }
    });
}

fn row(ui: &mut egui::Ui, palette: &Palette, brushes: &Brushes, item: &Row, first: bool, now: f32) {
    let (rect, response) =
        ui.allocate_exact_size(vec2(ui.available_width(), ROW_HEIGHT), Sense::hover());
    let painter = ui.painter();
    if !first {
        painter.line_segment(
            [
                rect.left_top() + vec2(0.0, 0.5),
                rect.right_top() + vec2(0.0, 0.5),
            ],
            Stroke::new(1.0, palette.hairline),
        );
    }

    let (title, length, detail, fill, tooltip) = match item {
        Row::Job(job) => (
            format!("Job {}", job.number),
            job.protocol_1.then(|| "protocol 1".to_owned()),
            job_detail(&job.progress, job.left),
            (!queued(job)).then(|| {
                let id = egui::Id::new(("client-fill", "job", job.number));
                (job.progress.fraction, id)
            }),
            None,
        ),
        Row::Session(session) => (
            format!("Session {}", session.number),
            Some(length(session.frames, session.sample_rate)),
            session_detail(session),
            Some((
                session.covered as f32 / session.frames.max(1) as f32,
                egui::Id::new(("client-fill", "session", session.number)),
            )),
            Some(session_tooltip(session)),
        ),
        Row::Ended(ended) => (
            format!("Session {}", ended.number),
            None,
            ended_detail(ended).0,
            None,
            None,
        ),
    };
    if let Some(tooltip) = tooltip {
        response.on_hover_text(tooltip);
    }

    let icon = icons::square(pos2(rect.left() + 14.0 + 6.0, rect.center().y), 12.0);
    let failed = matches!(item, Row::Ended(e) if ended_detail(e).1);
    match item {
        Row::Job(job) if queued(job) => icons::clock(painter, icon, palette.muted, 1.4),
        Row::Job(_) => icons::spinner(painter, icon, palette.text, 1.6, now.rem_euclid(1.0)),
        Row::Session(s) => match session_state(s) {
            SessionState::Separating => {
                icons::spinner(painter, icon, palette.text, 1.6, now.rem_euclid(1.0));
            }
            SessionState::Uploading | SessionState::Waiting => {
                icons::clock(painter, icon, palette.muted, 1.4);
            }
            SessionState::Complete => icons::check(painter, icon, palette.good, 1.6),
            SessionState::Idle => icons::pause(painter, icon, palette.muted, 1.4),
        },
        Row::Ended(_) if failed => icons::cross(painter, icon.shrink(1.0), palette.danger, 1.5),
        Row::Ended(_) => icons::cross(painter, icon.shrink(1.0), palette.muted, 1.5),
    }

    let meta_font = theme::font(12.0, Weight::Regular);
    let colour = if failed {
        palette.danger
    } else {
        palette.muted
    };
    let mut job = egui::text::LayoutJob::single_section(
        detail,
        egui::TextFormat::simple(meta_font.clone(), colour),
    );
    job.wrap = egui::text::TextWrapping::truncate_at_width(rect.width() * 0.6);
    let meta = painter.layout_job(job);
    let meta_left = rect.right() - 14.0 - meta.size().x;
    painter.galley(
        pos2(meta_left, rect.center().y - meta.size().y / 2.0),
        meta,
        colour,
    );

    let title_left = icon.right() + 10.0;
    let ended = matches!(item, Row::Ended(_));
    let title_colour = if ended { palette.muted } else { palette.text };
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &title,
        0.0,
        egui::TextFormat::simple(theme::font(13.0, Weight::Medium), title_colour),
    );
    if let Some(length) = length {
        job.append(
            &length,
            6.0,
            egui::TextFormat::simple(meta_font, palette.muted),
        );
    }
    job.wrap =
        egui::text::TextWrapping::truncate_at_width((meta_left - 10.0 - title_left).max(20.0));
    let galley = painter.layout_job(job);
    painter.galley(
        pos2(title_left, rect.center().y - galley.size().y / 2.0),
        galley,
        title_colour,
    );

    // The share of the track ready, along the foot of the row.
    if let Some((fraction, id)) = fill {
        let fraction =
            super::dropzone::eased_fill(ui.ctx(), id, fraction, super::dropzone::TRACK_EASE);
        let track = Rect::from_min_max(
            pos2(title_left, rect.bottom() - 6.0),
            pos2(rect.right() - 14.0, rect.bottom() - 4.0),
        );
        painter.rect_filled(track, 1.0, white(0.06));
        let fill = Rect::from_min_max(
            track.min,
            pos2(track.left() + track.width() * fraction, track.bottom()),
        );
        brushes.spectrum(painter, track, fill, 1.0, white(0.8));
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn session() -> Activity {
        Activity {
            number: 2,
            frames: 441_000,
            sample_rate: 44_100,
            pieces: 40,
            ready: 20,
            arrived: 441_000,
            covered: 220_500,
            high: 3,
            normal: 5,
            runnable: true,
        }
    }

    #[test]
    fn a_session_row_reads_state_then_progress() {
        assert_eq!(session_detail(&session()), "20/40 pieces");
        assert_eq!(session_state(&session()), SessionState::Separating);
        let uploading = Activity {
            runnable: false,
            arrived: 110_250,
            ..session()
        };
        assert_eq!(session_detail(&uploading), "uploading 25% · 20/40 pieces");
        let idle = Activity {
            runnable: false,
            high: 0,
            normal: 0,
            ..session()
        };
        assert_eq!(session_state(&idle), SessionState::Idle);
        assert_eq!(session_detail(&idle), "20/40 pieces");
        let complete = Activity {
            covered: 441_000,
            ready: 40,
            ..idle
        };
        assert_eq!(session_state(&complete), SessionState::Complete);
        assert_eq!(
            job_detail(
                &Progress::counted(stemd_core::Stage::Separating, 3, 8),
                None
            ),
            "separating · 39%"
        );
        assert_eq!(
            job_detail(
                &Progress::counted(stemd_core::Stage::Separating, 3, 8),
                Some(37.0)
            ),
            "separating · 39% · about 40 s left"
        );
        assert_eq!(
            job_detail(&Progress::new(stemd_core::Stage::Queued), None),
            "waiting for the worker",
            "a waiting job must read like a waiting drop"
        );
    }

    #[test]
    fn the_heading_sums_session_segments() {
        assert_eq!(queue_text(&Snapshot::default()), "");
        let busy = Snapshot {
            sessions: vec![session(), session()],
            ..Snapshot::default()
        };
        assert_eq!(queue_text(&busy), "6 high, 10 normal");
    }

    #[test]
    fn an_ended_session_says_why_and_flags_a_failure() {
        let ended = |reason: &str| Ended {
            number: 1,
            reason: reason.to_owned(),
            at: Instant::now(),
        };
        assert_eq!(ended_detail(&ended("idle")), ("ended: idle".into(), false));
        assert_eq!(
            ended_detail(&ended("ended by the client")),
            ("ended by the client".into(), false)
        );
        assert_eq!(
            ended_detail(&ended("the server switched models")),
            ("ended: the server switched models".into(), false)
        );
        assert!(ended_detail(&ended("segment 3: out of memory")).1);
    }

    #[test]
    fn the_list_hides_when_there_is_nothing_in_it() {
        assert_eq!(height(&Snapshot::default()), 0.0);
        let many = Snapshot {
            sessions: vec![session(); 5],
            ..Snapshot::default()
        };
        assert_eq!(
            height(&many),
            super::super::section::HEIGHT + 12.0 + 2.75 * ROW_HEIGHT
        );
        assert_eq!(length(441_000 * 22, 44_100), "3:40");
    }
}
