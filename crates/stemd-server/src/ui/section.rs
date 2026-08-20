//! A collapsible section header.
//!
//! egui's `CollapsingHeader` draws a triangle, a body indent and a vertical rule
//! down the left of its contents, which reads as an inspector panel. This is a
//! quiet rule with a label on it: a chevron, a name, an optional count, and no
//! indent.

use eframe::egui::{self, Color32, Stroke};

use super::theme::Palette;

const HEADER_HEIGHT: f32 = 26.0;
const CHEVRON_BOX: f32 = 14.0;

/// Indent that lines a body up with the label above it rather than the chevron.
/// The right default for a list of rows; the wrong one for anything that wants
/// the width more than the alignment, which is what [`FLUSH`] is for.
pub const UNDER_LABEL: f32 = CHEVRON_BOX + 4.0;

/// No indent. Log lines are long and monospaced, so eighteen points of
/// alignment costs about three characters off the end of every line.
pub const FLUSH: f32 = 0.0;

/// Draw the header and, if it is open, its body. Returns nothing: `open` is the
/// state, and the caller owns it.
pub fn show(
    ui: &mut egui::Ui,
    palette: &Palette,
    label: &str,
    trailing: Option<String>,
    open: &mut bool,
    indent: f32,
    body: impl FnOnce(&mut egui::Ui),
) {
    let width = ui.available_width();
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, HEADER_HEIGHT), egui::Sense::click());
    if response.clicked() {
        *open = !*open;
    }

    let lit = response.hovered();
    let colour = if lit { palette.text } else { palette.muted };

    let chevron_rect = egui::Rect::from_center_size(
        egui::pos2(rect.left() + CHEVRON_BOX / 2.0, rect.center().y),
        egui::vec2(CHEVRON_BOX, CHEVRON_BOX),
    );
    chevron(ui, chevron_rect, *open, colour);

    ui.painter().text(
        egui::pos2(rect.left() + CHEVRON_BOX + 4.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(12.0),
        colour,
    );

    if let Some(trailing) = trailing {
        ui.painter().text(
            egui::pos2(rect.right(), rect.center().y),
            egui::Align2::RIGHT_CENTER,
            trailing,
            egui::FontId::proportional(11.0),
            palette.muted,
        );
    }

    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    if *open {
        ui.add_space(2.0);
        let mut body_ui = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(egui::Rect::from_min_max(
                    egui::pos2(ui.min_rect().left() + indent, ui.cursor().top()),
                    ui.max_rect().max,
                ))
                .layout(egui::Layout::top_down(egui::Align::Min)),
        );
        body(&mut body_ui);
        ui.advance_cursor_after_rect(body_ui.min_rect());
        ui.add_space(6.0);
    }
}

/// A chevron pointing right when closed and down when open.
///
/// Painted rather than typed for the same reason every other mark here is: egui
/// bundles three fonts, most arrow glyphs are in none of them, and a missing one
/// draws nothing at all instead of a placeholder.
fn chevron(ui: &egui::Ui, rect: egui::Rect, open: bool, colour: Color32) {
    let painter = ui.painter();
    let stroke = Stroke::new(1.4, colour);
    let c = rect.center();
    let arm = 3.2;
    let (a, b, tip) = if open {
        (
            egui::pos2(c.x - arm, c.y - arm * 0.6),
            egui::pos2(c.x + arm, c.y - arm * 0.6),
            egui::pos2(c.x, c.y + arm * 0.7),
        )
    } else {
        (
            egui::pos2(c.x - arm * 0.6, c.y - arm),
            egui::pos2(c.x - arm * 0.6, c.y + arm),
            egui::pos2(c.x + arm * 0.7, c.y),
        )
    };
    painter.line_segment([a, tip], stroke);
    painter.line_segment([b, tip], stroke);
}
