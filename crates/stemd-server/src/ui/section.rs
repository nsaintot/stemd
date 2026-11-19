//! The heading above a list: a muted label, a count, and an optional detail at the right.

use eframe::egui::{self, Sense, pos2, vec2};

use super::theme::{self, Palette, Weight};

pub const HEIGHT: f32 = 18.0;

/// Left of the label, level with the icons in the list's rows.
const INSET: f32 = 14.0;

pub fn heading(ui: &mut egui::Ui, palette: &Palette, label: &str, count: Option<String>) {
    heading_with(ui, palette, label, count, None);
}

/// The heading with `detail` in muted text at the right.
pub fn heading_with(
    ui: &mut egui::Ui,
    palette: &Palette,
    label: &str,
    count: Option<String>,
    detail: Option<String>,
) {
    let font = theme::font(12.0, Weight::Regular);
    let painter = ui.painter();
    let label = painter.layout_no_wrap(label.to_owned(), font.clone(), palette.muted);
    let count = count.map(|c| painter.layout_no_wrap(c, font.clone(), palette.text));
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), HEIGHT), Sense::hover());
    let painter = ui.painter();
    let y = rect.center().y;
    let x = rect.left() + INSET;
    let count_x = x + label.size().x + 6.0;
    let mut used = count_x;
    painter.galley(pos2(x, y - label.size().y / 2.0), label, palette.muted);
    if let Some(count) = count {
        used += count.size().x;
        painter.galley(pos2(count_x, y - count.size().y / 2.0), count, palette.text);
    }
    if let Some(detail) = detail {
        let mut job = egui::text::LayoutJob::single_section(
            detail,
            egui::TextFormat::simple(font, palette.muted),
        );
        job.wrap = egui::text::TextWrapping::truncate_at_width(
            (rect.right() - INSET - used - 16.0).max(0.0),
        );
        let detail = painter.layout_job(job);
        let at = pos2(
            rect.right() - INSET - detail.size().x,
            y - detail.size().y / 2.0,
        );
        painter.galley(at, detail, palette.muted);
    }
}
