//! The main region: where a track is dropped, and what is happening to it.
//!
//! It takes whatever height the window has left rather than a fixed box, so the
//! target is the window rather than a rectangle drawn inside one.
//!
//! Three states, one rectangle: idle it invites a drop, armed it lights up,
//! working it becomes the progress of the track. Swapping the contents rather than
//! the container stops the layout jumping under the pointer mid-drop.

use eframe::egui::{self, Color32, Stroke};

use crate::drops::{Dropped, State};

use super::theme::{Palette, Sheen, ZONE_RADIUS};

/// Below this the zone stops shrinking and the window scrolls instead. Enough
/// for the icon, the title and its line of help.
pub const MIN_HEIGHT: f32 = 150.0;

/// The drop icon, square.
const ARROW: f32 = 44.0;

/// Height of the working layout (title, bar and caption with their spacing)
/// used to centre it. The inviting layout measures itself instead; this one is
/// still laid out with widgets, because the progress bar is one.
const WORKING_BLOCK: f32 = 124.0;

/// Room for the figure beside the bar. Fixed rather than measured, so the bar
/// does not twitch as the number goes from one digit to three.
pub(super) const PERCENT_WIDTH: f32 = 34.0;

/// Height of the bar-and-figure row. Tall enough for the figure, which is what
/// decides it: the bar itself is five points.
const BAR_ROW_HEIGHT: f32 = 16.0;

pub struct Zone<'a> {
    pub palette: &'a Palette,
    pub sheen: &'a mut Sheen,
    /// A file is being dragged over the window right now.
    pub armed: bool,
    /// The most recent drop, if it is still being worked on.
    pub current: Option<&'a Dropped>,
}

impl Zone<'_> {
    /// Draws the zone and returns whether it was clicked, which opens a picker.
    pub fn show(&mut self, ui: &mut egui::Ui) -> bool {
        let height = ui.available_height().max(MIN_HEIGHT);
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        // A fixed id rather than an allocation counter, for the same reason the
        // separated rows use one: the counter shifts as the layout above changes
        // and an id that moves between frames cannot pair a press with its
        // release.
        let response = ui.interact(rect, ui.id().with("dropzone"), egui::Sense::click());

        let busy = self.current.is_some_and(Dropped::is_running);
        let lit = self.armed || (response.hovered() && !busy);
        // Idle, the surface carries no outline at all: the gradient is what
        // separates it from the window behind. A border here would draw the eye
        // to the edge of the target rather than to the middle of it, and the
        // edge is not the part anyone needs to find.
        let (fill, stroke) = if lit {
            (
                Palette::wash(self.palette.accent, 0x24),
                Stroke::new(1.5, self.palette.accent),
            )
        } else {
            (self.palette.card, Stroke::NONE)
        };
        self.sheen.card(ui, rect, ZONE_RADIUS, fill, stroke);

        match self.current {
            Some(item) if item.is_running() => {
                let mut content = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(rect.shrink(20.0))
                        .layout(egui::Layout::top_down(egui::Align::Center)),
                );
                content.add_space(((rect.height() - WORKING_BLOCK) / 2.0).max(0.0));
                self.working(&mut content, item);
            }
            _ => self.inviting(ui, rect),
        }

        !busy
            && response
                .on_hover_cursor(egui::CursorIcon::PointingHand)
                .clicked()
    }

    /// Painted rather than laid out as widgets.
    ///
    /// A `ui.label` registers its own interaction rect on top of the zone's, so egui
    /// hands the pointer to the label and the zone beneath stops seeing the hover. The
    /// painter takes no part in interaction.
    fn inviting(&self, ui: &egui::Ui, rect: egui::Rect) {
        let colour = if self.armed {
            self.palette.accent
        } else {
            self.palette.muted
        };
        let title = if self.armed {
            "Release to separate"
        } else {
            "Drop a track"
        };

        let title_font = egui::FontId::proportional(19.0);
        let hint_font = egui::FontId::proportional(11.5);
        let (title_h, hint_h) = ui
            .ctx()
            .fonts_mut(|fonts| (fonts.row_height(&title_font), fonts.row_height(&hint_font)));

        // Centred as one block, so the zone reads the same whether it is 150
        // points tall or 400.
        let block = ARROW + 16.0 + title_h + 5.0 + hint_h;
        let mut y = rect.center().y - block / 2.0;
        let mid = rect.center().x;

        arrow(ui, egui::pos2(mid, y), colour, self.armed);
        y += ARROW + 16.0;

        let painter = ui.painter();
        painter.text(
            egui::pos2(mid, y),
            egui::Align2::CENTER_TOP,
            title,
            title_font,
            self.palette.text,
        );
        y += title_h + 5.0;
        painter.text(
            egui::pos2(mid, y),
            egui::Align2::CENTER_TOP,
            "harmonics, vocals and drums, beside the file",
            hint_font,
            self.palette.muted,
        );
    }

    fn working(&self, ui: &mut egui::Ui, item: &Dropped) {
        ui.label(
            egui::RichText::new(&item.title)
                .size(16.0)
                .color(self.palette.text),
        );
        ui.add_space(16.0);

        let (fraction, caption) = match item.state() {
            State::Reading => (None, "reading the file".to_owned()),
            State::Working(progress) => {
                let caption = match progress.total {
                    0 => stage_label(progress.stage).to_owned(),
                    total => format!(
                        "{} · {}/{}",
                        stage_label(progress.stage),
                        progress.completed,
                        total
                    ),
                };
                (Some(progress.fraction), caption)
            }
            // Not reachable while `is_running`, but a state machine read across
            // two locks can always land between them.
            State::Done(_) | State::Failed(_) => (None, "finishing".to_owned()),
        };

        progress_row(ui, fraction, self.palette);
        ui.add_space(14.0);
        ui.label(
            egui::RichText::new(caption)
                .size(11.5)
                .color(self.palette.muted),
        );
    }
}

/// The bar and the figure beside it, as one centred row.
///
/// The figure sits beside the bar rather than inside it: the bar is five points
/// tall and egui would clip the text. Allocating exactly the row's width lets the
/// enclosing `Align::Center` do its job, which a plain `ui.horizontal` prevents by
/// taking the whole width.
pub(super) fn progress_row(ui: &mut egui::Ui, fraction: Option<f32>, palette: &Palette) {
    let bar = match fraction {
        Some(fraction) => egui::ProgressBar::new(fraction),
        // An indeterminate stage is honest about it rather than showing a bar
        // parked at zero, which reads as stuck.
        None => egui::ProgressBar::new(0.0).animate(true),
    };
    let gap = ui.spacing().item_spacing.x;
    // Clamped, not just capped: a narrow enough window would otherwise ask for
    // a negative width once the figure's share is taken off.
    let width = (ui.available_width() - PERCENT_WIDTH - gap).clamp(60.0, 260.0);

    ui.allocate_ui_with_layout(
        egui::vec2(width + gap + PERCENT_WIDTH, BAR_ROW_HEIGHT),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.add(
                bar.desired_width(width)
                    .desired_height(5.0)
                    .corner_radius(3)
                    .fill(palette.accent),
            );
            // Blank while indeterminate, so the row does not change width when
            // a stage that cannot be counted starts or ends.
            let figure = fraction.map_or_else(String::new, crate::ui::status::percent);
            ui.add_sized(
                [PERCENT_WIDTH, BAR_ROW_HEIGHT],
                egui::Label::new(egui::RichText::new(figure).size(11.5).color(palette.muted)),
            );
        },
    );
}

/// The stage in words rather than `{:?}`, which is what the old window showed.
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

/// A downward arrow into a tray, painted rather than typed.
///
/// egui bundles three fonts and most arrow glyphs are in none of them; a missing
/// glyph draws nothing at all. `top_centre` is the middle of its top edge; it
/// occupies [`ARROW`] square.
fn arrow(ui: &egui::Ui, top_centre: egui::Pos2, colour: Color32, armed: bool) {
    const SIZE: f32 = ARROW;
    let rect = egui::Rect::from_min_size(
        egui::pos2(top_centre.x - SIZE / 2.0, top_centre.y),
        egui::vec2(SIZE, SIZE),
    );
    let painter = ui.painter();
    let stroke = Stroke::new(if armed { 2.4 } else { 1.9 }, colour);
    let mid = rect.center().x;
    // Armed, the arrow sits lower in its box: the smallest hint that the file
    // is on its way in.
    let travel = if armed { 4.0 } else { 0.0 };
    let top = rect.top() + 2.0 + travel;
    let stem_bottom = rect.top() + SIZE * 0.54 + travel;

    painter.line_segment([egui::pos2(mid, top), egui::pos2(mid, stem_bottom)], stroke);
    let head = SIZE * 0.16;
    for side in [-1.0, 1.0] {
        painter.line_segment(
            [
                egui::pos2(mid + side * head, stem_bottom - head),
                egui::pos2(mid, stem_bottom),
            ],
            stroke,
        );
    }

    // The tray: two shoulders and a floor, open at the top.
    let tray_top = rect.top() + SIZE * 0.70;
    let bottom = rect.bottom() - 2.0;
    let (left, right) = (rect.left() + 6.0, rect.right() - 6.0);
    for x in [left, right] {
        painter.line_segment([egui::pos2(x, tray_top), egui::pos2(x, bottom)], stroke);
    }
    painter.line_segment(
        [egui::pos2(left, bottom), egui::pos2(right, bottom)],
        stroke,
    );
}
