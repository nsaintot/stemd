//! The "Separated" heading above the list: a muted label and a count.

use eframe::egui::{self, Sense, pos2, vec2};

use super::theme::{self, Palette, Weight};

pub const HEIGHT: f32 = 18.0;

/// Left of the label, level with the icons in the list's rows.
const INSET: f32 = 14.0;

pub fn heading(ui: &mut egui::Ui, palette: &Palette, label: &str, count: Option<String>) {
    let font = theme::font(12.0, Weight::Regular);
    let painter = ui.painter();
    let label = painter.layout_no_wrap(label.to_owned(), font.clone(), palette.muted);
    let count = count.map(|c| painter.layout_no_wrap(c, font, palette.text));
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), HEIGHT), Sense::hover());
    let painter = ui.painter();
    let y = rect.center().y;
    let x = rect.left() + INSET;
    let count_x = x + label.size().x + 6.0;
    painter.galley(pos2(x, y - label.size().y / 2.0), label, palette.muted);
    if let Some(count) = count {
        painter.galley(pos2(count_x, y - count.size().y / 2.0), count, palette.text);
    }
}
