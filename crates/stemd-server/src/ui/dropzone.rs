//! The main region: the globe, what is happening, and the way to add a track.
//!
//! The zone takes whatever height the window has left. When idle, it invites a drop; while a file
//! is held over it, it lights up with a dashed edge; while working, it shows the track's progress.
//! The container stays the same in every state, so the layout does not jump under the pointer.

use eframe::egui::{self, Color32, Rect, Sense, Stroke, pos2, vec2};

use crate::drops::{Dropped, State};
use crate::models::Preset;
use crate::settings::DropParts;

use super::globe::{self, Motion};
use super::theme::{self, Brushes, Palette, Tween, Weight, white};
use super::widgets::{self, Column, Pill};

/// The height below which the zone stops shrinking.
pub const MIN_HEIGHT: f32 = 300.0;

const RADIUS: f32 = 22.0;

/// The width of the progress bar.
const BAR_WIDTH: f32 = 260.0;
const BAR_HEIGHT: f32 = 8.0;

/// The width reserved for the percentage beside the bar.
const PERCENT_WIDTH: f32 = 38.0;
const PERCENT_GAP: f32 = 12.0;

/// Seconds the fill takes to catch up with a track's progress.
pub(super) const TRACK_EASE: f32 = 0.16;
/// Seconds the fill takes to catch up with a download, whose byte count arrives in uneven steps.
const DOWNLOAD_EASE: f32 = 0.6;

/// The height of the row with the bar and the percentage.
const BAR_ROW_HEIGHT: f32 = 16.0;

const BUTTON_HEIGHT: f32 = 30.0;

/// The widest title before it is cut short with an ellipsis.
const TITLE_WIDTH: f32 = 420.0;

/// The hint under the title, listing the ticked parts.
fn hint(parts: DropParts) -> String {
    let names: Vec<&str> = parts.names().collect();
    let list = match names.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => names.join(""),
    };
    format!("{list}, beside the file")
}

/// A model switch, shown in place of the invitation to drop.
#[derive(Clone, Copy)]
pub enum Switch<'a> {
    Downloading {
        preset: Preset,
        file: &'a str,
        done: u64,
        total: u64,
    },
    Verifying(Preset),
    /// `waiting` is true while the worker finishes the track in flight.
    Loading {
        preset: Preset,
        waiting: bool,
    },
    Failed {
        preset: Preset,
        message: &'a str,
    },
}

/// The hand-over from the startup screen: the globe glides from where it was,
/// and everything else in the zone fades in.
#[derive(Clone, Copy)]
pub struct Intro {
    /// The globe's stage on the startup screen.
    pub from: Rect,
    /// How far its slices were spread there.
    pub spread: f32,
    /// The eased progress, from 0 to 1.
    pub t: f32,
}

/// What a click in the zone asked for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    ChooseFile,
    KeepModel,
}

/// Tracks dropped since the window was last idle.
#[derive(Clone, Default)]
pub struct Batch {
    /// Whether each track, in drop order, has finished or failed.
    pub finished: Vec<bool>,
    /// The track the zone shows.
    pub shown: Option<usize>,
}

impl Batch {
    fn total(&self) -> usize {
        self.finished.len()
    }

    fn done(&self) -> usize {
        self.finished.iter().filter(|f| **f).count()
    }
}

pub struct Zone<'a> {
    pub palette: &'a Palette,
    pub brushes: &'a Brushes,
    pub motion: &'a Motion,
    /// A file is being dragged over the window right now.
    pub armed: bool,
    /// The drop being shown, while it is worked on.
    pub current: Option<&'a Dropped>,
    pub parts: DropParts,
    pub batch: Batch,
    /// Shown in place of the subtitle while set.
    pub note: Option<&'a str>,
    /// Time left on the current drop, added to its caption.
    pub left: Option<&'a str>,
    /// True when Quality is selected, which shows the globe's halo.
    pub maxed: bool,
    pub switch: Option<Switch<'a>>,
    pub intro: Option<Intro>,
}

/// `text` on one line centred on `mid` at `y`, cut to `width`.
fn centred_line(
    painter: &egui::Painter,
    text: String,
    font: egui::FontId,
    colour: egui::Color32,
    mid: f32,
    y: f32,
    width: f32,
) {
    let mut job =
        egui::text::LayoutJob::single_section(text, egui::TextFormat::simple(font, colour));
    job.wrap = egui::text::TextWrapping::truncate_at_width(width);
    let galley = painter.layout_job(job);
    painter.galley(pos2(mid - galley.size().x / 2.0, y), galley, colour);
}

impl Zone<'_> {
    /// Draws the zone and returns what its button asked for.
    pub fn show(&self, ui: &mut egui::Ui) -> Option<Action> {
        let height = ui.available_height().max(MIN_HEIGHT);
        let (rect, response) =
            ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
        let ctx = ui.ctx().clone();
        let now = ctx.input(|i| i.time);
        let id = ui.id().with("dropzone");
        // The globe stays opaque while the rest fades in.
        let globe_painter = ui.painter().clone();
        if let Some(intro) = self.intro {
            ui.multiply_opacity(intro.t);
        }

        let working = self.current.is_some_and(Dropped::is_running);
        let fx = self.motion.fx(now);
        let cheering = matches!(fx, Some((globe::Fx::Cheer, _)));

        self.surface(ui, rect, id);

        let switch = self.switch.filter(|_| !self.armed && !working && !cheering);
        let title = self.title(switch, working, cheering);
        let title_colour = if matches!(switch, Some(Switch::Failed { .. })) {
            self.palette.danger
        } else {
            self.palette.text
        };
        let progress = match switch {
            Some(Switch::Downloading { done, total, .. }) => Some((
                (total > 0).then(|| done as f32 / total as f32),
                String::new(),
            )),
            Some(Switch::Verifying(_)) => Some((None, String::new())),
            _ => self.current.filter(|_| working).map(progress_of),
        };
        let ease = if switch.is_some() {
            DOWNLOAD_EASE
        } else {
            TRACK_EASE
        };
        let sub = self.caption(switch, progress.as_ref());
        let title_font = theme::font(20.0, Weight::Semibold);
        let sub_font = theme::font(12.5, Weight::Regular);
        let (title_h, sub_h) =
            ctx.fonts_mut(|f| (f.row_height(&title_font), f.row_height(&sub_font)));
        let stacked = progress.is_some() && switch.is_none() && self.batch.total() > 1;
        let button = match switch {
            Some(Switch::Failed { .. }) => Some(("Keep current model", Action::KeepModel)),
            Some(_) => None,
            None if working => Some(("Queue another track", Action::ChooseFile)),
            None => Some(("Choose a file", Action::ChooseFile)),
        };

        // Measure each row's height and the gap under it once, so the block can be centred.
        let mut column = Column::default();
        let stage_at = column.add(globe::STAGE.y, 8.0);
        let title_at = column.add(title_h, 6.0);
        let sub_at = column.add(sub_h, 18.0);
        let bar_at = progress
            .is_some()
            .then(|| column.add(BAR_ROW_HEIGHT, if stacked { 6.0 } else { 14.0 }));
        let segments_at = stacked.then(|| column.add(4.0, 14.0));
        let button_at = button.is_some().then(|| column.add(BUTTON_HEIGHT, 0.0));
        let top = rect.center().y - column.height() / 2.0;
        let mid = rect.center().x;

        let stage = Rect::from_min_size(
            pos2(mid - globe::STAGE.x / 2.0, top + stage_at),
            globe::STAGE,
        );
        self.paint_globe(
            &globe_painter,
            rect,
            stage,
            id,
            working,
            response.contains_pointer(),
        );

        let painter = ui.painter();
        centred_line(
            painter,
            title,
            title_font,
            title_colour,
            mid,
            top + title_at,
            TITLE_WIDTH.min(rect.width() - 40.0),
        );
        centred_line(
            painter,
            sub,
            sub_font,
            self.palette.muted,
            mid,
            top + sub_at,
            rect.width() - 40.0,
        );

        if let (Some((fraction, _)), Some(bar_at)) = (progress, bar_at) {
            // The bar is centred, and the percentage sits past its right end.
            let row = Rect::from_min_size(
                pos2(mid - BAR_WIDTH / 2.0, top + bar_at),
                vec2(BAR_WIDTH + PERCENT_GAP + PERCENT_WIDTH, BAR_ROW_HEIGHT),
            );
            progress_row(
                ui,
                row,
                fraction,
                ease,
                self.palette,
                self.brushes,
                self.motion.clock,
            );
            if let Some(segments_at) = segments_at {
                self.segments(
                    ui,
                    Rect::from_min_size(pos2(row.left(), top + segments_at), vec2(BAR_WIDTH, 4.0)),
                    fraction,
                );
            }
        }

        let ((label, action), button_at) = button.zip(button_at)?;
        let width = widgets::pill_width(ui, label, BUTTON_HEIGHT);
        let rect = Rect::from_min_size(
            pos2(mid - width / 2.0, top + button_at),
            vec2(width, BUTTON_HEIGHT),
        );
        widgets::pill(
            ui,
            rect,
            id.with("choose"),
            label,
            Pill::Ghost,
            true,
            self.palette,
            self.brushes,
        )
        .clicked()
        .then_some(action)
    }

    /// The zone's title line.
    fn title(&self, switch: Option<Switch<'_>>, working: bool, cheering: bool) -> String {
        if self.armed {
            "Release to separate".to_owned()
        } else if cheering {
            "All done".to_owned()
        } else if let Some(item) = self.current.filter(|_| working) {
            item.title.clone()
        } else if let Some(switch) = switch {
            switch_title(switch)
        } else {
            "Drop a track".to_owned()
        }
    }

    /// The line under the title: the hint, the switch, a note, or the progress caption.
    fn caption(
        &self,
        switch: Option<Switch<'_>>,
        progress: Option<&(Option<f32>, String)>,
    ) -> String {
        if self.armed {
            hint(self.parts)
        } else if let Some(switch) = switch {
            switch_caption(switch)
        } else if let Some(note) = self.note {
            note.to_owned()
        } else if let Some((_, caption)) = progress {
            let caption = match self.left.filter(|_| switch.is_none()) {
                Some(left) => format!("{caption}{}{left}", theme::DOT),
                None => caption.clone(),
            };
            if let Some(shown) = self.batch.shown
                && self.batch.total() > 1
            {
                format!(
                    "{caption}{}track {} of {}",
                    theme::DOT,
                    shown + 1,
                    self.batch.total()
                )
            } else {
                caption.clone()
            }
        } else {
            hint(self.parts)
        }
    }

    /// The globe at `stage`, or between the startup screen's pose and it during the intro.
    fn paint_globe(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        stage: Rect,
        id: egui::Id,
        working: bool,
        hovered: bool,
    ) {
        let ctx = painter.ctx();
        let now = ctx.input(|i| i.time);
        let spread = ctx.animate_bool_with_time_and_easing(
            id.with("spread"),
            self.armed || working,
            0.5,
            ease_spread,
        );
        let lift = Tween::follow(
            ctx,
            id.with("lift"),
            if hovered { 3.0 } else { 0.0 },
            0.3,
            theme::ease_spring,
        );
        let halo = ctx.animate_bool_with_time(id.with("halo"), self.maxed, 0.7);
        // From the startup screen's pose to this one.
        let (stage, spread, halo) = match self.intro {
            Some(intro) => (
                Rect::from_min_size(intro.from.min.lerp(stage.min, intro.t), stage.size()),
                intro.spread + (spread - intro.spread) * intro.t,
                halo * intro.t,
            ),
            None => (stage, spread, halo),
        };
        // Clip the halo and rings to the card. During the intro the globe can still be outside the
        // card, so the clip also covers the globe's box.
        let globe_painter = painter.with_clip_rect(rect.union(stage));
        globe::paint(
            &globe_painter,
            self.brushes,
            self.palette,
            stage,
            self.motion,
            globe::Look { spread, lift, halo },
            now,
        );
    }

    /// The zone's own surface: faint at rest, brighter with a dashed edge and an
    /// inner blue glow while a file is held over it.
    fn surface(&self, ui: &egui::Ui, rect: Rect, id: egui::Id) {
        let painter = ui.painter();
        let armed = ui
            .ctx()
            .animate_bool_with_time(id.with("armed"), self.armed, 0.2);
        painter.rect_filled(rect, RADIUS, white(0.025 + 0.045 * armed));
        theme::inner_glow(
            painter,
            rect,
            RADIUS,
            48.0,
            Color32::from_rgba_unmultiplied(91, 132, 216, (0.22 * 255.0 * armed) as u8),
        );
        if armed < 1.0 {
            painter.rect_stroke(
                rect,
                RADIUS,
                Stroke::new(1.5, white(0.06 * (1.0 - armed))),
                egui::StrokeKind::Inside,
            );
        }
        if armed > 0.0 {
            theme::dashed_outline(
                painter,
                rect,
                RADIUS,
                Stroke::new(1.5, self.palette.band.gamma_multiply(armed)),
            );
        }
    }

    /// One short bar per track in the batch, under the main bar. A batch too
    /// large for 4-point bars gets one bar for the whole batch.
    fn segments(&self, ui: &egui::Ui, row: Rect, fraction: Option<f32>) {
        let total = self.batch.total();
        let gap = 3.0;
        let mut n = total;
        let mut width = (row.width() - gap * (n as f32 - 1.0)) / n as f32;
        if width < 4.0 {
            n = 1;
            width = row.width();
        }
        for i in 0..n {
            let seg = Rect::from_min_size(
                pos2(row.left() + i as f32 * (width + gap), row.top()),
                vec2(width, row.height()),
            );
            ui.painter().rect_filled(seg, 2.0, white(0.12));
            let target = if n < total {
                (self.batch.done() as f32 + fraction.unwrap_or(0.0)) / total as f32
            } else if self.batch.finished[i] {
                1.0
            } else if self.batch.shown == Some(i) {
                fraction.unwrap_or(0.0)
            } else {
                0.0
            };
            let shown = eased_fill(ui.ctx(), ui.id().with(("segment", i)), target, TRACK_EASE);
            if shown > 0.0 {
                ui.painter().rect_filled(
                    Rect::from_min_size(seg.min, vec2(width * shown, seg.height())),
                    2.0,
                    self.palette.soft,
                );
            }
        }
    }
}

fn switch_title(switch: Switch) -> String {
    match switch {
        Switch::Downloading { preset, .. } => format!("Downloading the {} model", preset.label()),
        Switch::Verifying(preset) => format!("Checking the {} model", preset.label()),
        Switch::Loading { preset, .. } => format!("Loading {}", preset.label()),
        Switch::Failed { preset, .. } => format!("Could not switch to {}", preset.label()),
    }
}

fn switch_caption(switch: Switch) -> String {
    match switch {
        Switch::Downloading {
            file, done, total, ..
        } => format!(
            "{file}{}{:.0} / {:.0} MB",
            theme::DOT,
            done as f64 / 1e6,
            total as f64 / 1e6
        ),
        Switch::Verifying(_) => "verifying local data".to_owned(),
        Switch::Loading { waiting: true, .. } => "after the track in flight finishes".to_owned(),
        Switch::Loading { waiting: false, .. } => "onto the GPU, a few seconds".to_owned(),
        Switch::Failed { message, .. } => message.to_owned(),
    }
}

/// The `cubic-bezier(.2, .9, .2, 1)` easing of the slices settling apart.
fn ease_spread(t: f32) -> f32 {
    theme::cubic_bezier(0.2, 0.9, 0.2, 1.0, t)
}

/// The fraction done, if it can be counted, and the stage in words.
fn progress_of(item: &Dropped) -> (Option<f32>, String) {
    match item.state() {
        State::Reading => (None, "reading the file".to_owned()),
        State::Working(progress) => {
            let caption = match progress.total {
                0 => stage_label(progress.stage).to_owned(),
                total => format!(
                    "{}{}{}/{}",
                    stage_label(progress.stage),
                    theme::DOT,
                    progress.completed,
                    total
                ),
            };
            (Some(progress.fraction), caption)
        }
        // Not reachable while running, but a state read across two locks can
        // land between them.
        State::Done(_) | State::Failed(_) => (None, "finishing".to_owned()),
    }
}

/// Draw the bar and the percentage beside it in `row`.
///
/// The fill is a red, green and blue gradient anchored to the full length of the bar, with a gloss
/// and a moving shimmer. A `None` fraction, for a stage that cannot be counted, draws an empty bar
/// with the shimmer running across it.
pub(super) fn progress_row(
    ui: &egui::Ui,
    row: Rect,
    fraction: Option<f32>,
    ease: f32,
    palette: &Palette,
    brushes: &Brushes,
    clock: f32,
) {
    let painter = ui.painter();
    let track = Rect::from_min_size(
        pos2(row.left(), row.center().y - BAR_HEIGHT / 2.0),
        vec2(
            BAR_WIDTH.min(row.width() - PERCENT_GAP - PERCENT_WIDTH),
            BAR_HEIGHT,
        ),
    );
    let r = BAR_HEIGHT / 2.0;
    painter.rect_filled(track, r, theme::black(0.3));
    // A shade along the top (`inset 0 1px 2px`) and a light line under the bottom (`0 1px 0`).
    painter.line_segment(
        [
            pos2(track.left() + r, track.top() + 0.75),
            pos2(track.right() - r, track.top() + 0.75),
        ],
        Stroke::new(1.5, theme::black(0.3)),
    );
    painter.line_segment(
        [
            pos2(track.left() + r, track.bottom() + 0.5),
            pos2(track.right() - r, track.bottom() + 0.5),
        ],
        Stroke::new(1.0, white(0.10)),
    );

    let shown = eased_fill(
        ui.ctx(),
        ui.id().with("progress-fill"),
        fraction.unwrap_or(0.0),
        ease,
    );
    let fill = Rect::from_min_size(track.min, vec2(track.width() * shown, BAR_HEIGHT));
    if fill.width() > 0.5 {
        brushes.spectrum(painter, track, fill, r, Color32::WHITE);
        brushes.gloss(painter, fill, r);
    }
    let lane = if fraction.is_some() { fill } else { track };
    let alpha = if fraction.is_some() { 0.55 } else { 0.25 };
    shimmer(painter, lane, r, clock, alpha);

    let figure = fraction.map_or_else(String::new, super::status::percent);
    painter.text(
        pos2(track.right() + PERCENT_GAP, row.center().y),
        egui::Align2::LEFT_CENTER,
        figure,
        theme::font(13.0, Weight::Medium),
        palette.text,
    );
}

/// A progress fill under `id` easing out to `fraction` over `secs`.
pub(super) fn eased_fill(ctx: &egui::Context, id: egui::Id, fraction: f32, secs: f32) -> f32 {
    Tween::follow(ctx, id, fraction.clamp(0.0, 1.0), secs, theme::ease_out)
}

/// A 60-point band of light sweeping across `lane` every 1.6 seconds, clipped to
/// its rounded ends.
fn shimmer(painter: &egui::Painter, lane: Rect, radius: f32, clock: f32, alpha: f32) {
    const BAND: f32 = 60.0;
    if lane.width() < 1.0 {
        return;
    }
    let t = theme::ease_in_out((clock / 1.6).rem_euclid(1.0));
    let start = lane.left() - BAND + t * (lane.width() + BAND);
    let (from, to) = (start.max(lane.left()), (start + BAND).min(lane.right()));
    if to <= from {
        return;
    }
    // Half the height of the rounded lane at `x`.
    let half = |x: f32| {
        let h = lane.height() / 2.0;
        let edge = (x - lane.left()).min(lane.right() - x);
        if edge >= radius {
            h
        } else {
            let dx = radius - edge;
            (radius * radius - dx * dx).max(0.0).sqrt().min(h)
        }
    };
    let mut mesh = egui::Mesh::default();
    let steps = ((to - from) / 2.0).ceil().max(1.0) as u32;
    for n in 0..=steps {
        let x = from + (to - from) * n as f32 / steps as f32;
        let u = (x - start) / BAND;
        let colour = white(alpha * (1.0 - (2.0 * u - 1.0).abs()));
        let h = half(x);
        mesh.colored_vertex(pos2(x, lane.center().y - h), colour);
        mesh.colored_vertex(pos2(x, lane.center().y + h), colour);
    }
    for n in 0..steps {
        let a = 2 * n;
        mesh.add_triangle(a, a + 1, a + 2);
        mesh.add_triangle(a + 1, a + 3, a + 2);
    }
    painter.add(mesh);
}

/// The stage in words.
pub fn stage_label(stage: stemd_core::Stage) -> &'static str {
    use stemd_core::Stage;
    match stage {
        Stage::Queued => "waiting for the worker",
        Stage::Analysing => "analysing",
        Stage::Separating => "separating",
        Stage::Reconstructing => "reconstructing",
        Stage::Writing => "writing stems",
        Stage::Done => "done",
        Stage::Failed => "failed",
        Stage::Cancelled => "stopped",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fill this frame for `fraction`, at window time `time`.
    fn fill_at(ctx: &egui::Context, time: f64, fraction: f32) -> f32 {
        let input = egui::RawInput {
            time: Some(time),
            ..Default::default()
        };
        let mut shown = 0.0;
        let _ = ctx.run_ui(input, |ui| {
            shown = eased_fill(ui.ctx(), egui::Id::new("fill"), fraction, TRACK_EASE);
        });
        shown
    }

    #[test]
    fn a_fill_eases_out_to_each_new_fraction() {
        let ctx = egui::Context::default();
        assert_eq!(
            fill_at(&ctx, 0.0, 0.2),
            0.2,
            "the first fraction shows as is"
        );
        assert_eq!(
            fill_at(&ctx, 1.0, 0.6),
            0.2,
            "a new fraction starts from the old"
        );
        let half = fill_at(&ctx, 1.0 + f64::from(TRACK_EASE) / 2.0, 0.6);
        assert!(half > 0.4 && half < 0.6, "ease-out is past halfway: {half}");
        assert_eq!(fill_at(&ctx, 1.0 + f64::from(TRACK_EASE), 0.6), 0.6);
        assert_eq!(
            fill_at(&ctx, 2.0, 1.4),
            0.6,
            "clamped to the end, from the old"
        );
        assert_eq!(fill_at(&ctx, 3.0, 1.4), 1.0);
    }
}
