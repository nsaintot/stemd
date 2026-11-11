//! The list of tracks this window has separated.
//!
//! One glass panel, a hairline between rows. A finished row opens its folder;
//! hovering it swaps the detail for "Open folder" and a cross that removes the row
//! and deletes its stems. On a running row the cross stops the drop. A row whose
//! folder has been deleted removes itself when asked to open it.

use std::sync::Arc;

use eframe::egui::{self, Rect, Sense, Stroke, pos2, vec2};

use crate::drops::{Dropped, Drops, State};

use super::icons;
use super::theme::{self, Brushes, Glass, Palette, Weight, white};

pub const ROW_HEIGHT: f32 = 40.0;

const RADIUS: f32 = 16.0;

/// Hit target and drawn box of the remove control.
const CROSS_BOX: f32 = 26.0;

/// The list's height for `rows` rows in `max_height`, never under one row.
pub fn height(rows: usize, max_height: f32) -> f32 {
    (ROW_HEIGHT * rows as f32).min(max_height.max(ROW_HEIGHT))
}

/// Draw the list. Returns a running drop whose cross was clicked, for the
/// caller to stop.
pub fn list(
    ui: &mut egui::Ui,
    palette: &Palette,
    brushes: &Brushes,
    drops: &Drops,
    items: &[Arc<Dropped>],
    max_height: f32,
) -> Option<Arc<Dropped>> {
    if items.is_empty() {
        placeholder(ui, palette, brushes, max_height);
        return None;
    }

    let height = height(items.len(), max_height);
    let (panel, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    brushes.glass(ui.painter(), panel, RADIUS, Glass::List);

    let mut inside = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(panel)
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    inside.set_clip_rect(panel.intersect(ui.clip_rect()));
    inside.spacing_mut().item_spacing = vec2(0.0, 0.0);
    inside.spacing_mut().scroll = egui::style::ScrollStyle {
        bar_width: 6.0,
        ..egui::style::ScrollStyle::floating()
    };
    let now = ui.input(|i| i.time) as f32;
    let mut stop = None;
    egui::ScrollArea::vertical()
        .id_salt("recents")
        .max_height(height)
        .auto_shrink([false, true])
        .show(&mut inside, |ui| {
            for (n, item) in items.iter().enumerate() {
                if row(ui, palette, drops, item, n == 0, n + 1 == items.len(), now) {
                    stop = Some(Arc::clone(item));
                }
            }
        });
    stop
}

/// The empty list: the panel at three rows, or what fits, faded, with faint
/// sample rows under one line of text.
fn placeholder(ui: &mut egui::Ui, palette: &Palette, brushes: &Brushes, max_height: f32) {
    let height = height(3, max_height);
    let (panel, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    let mut painter = ui.painter().with_clip_rect(ui.clip_rect());
    painter.multiply_opacity(0.5);
    brushes.glass(&painter, panel, RADIUS, Glass::List);
    painter.set_clip_rect(panel.intersect(ui.clip_rect()));

    // Sample rows: a dot where the icon goes and bars for the title and detail,
    // each row fainter than the one above. The bars stop short of the centred text.
    let rows = (height / ROW_HEIGHT).floor().max(1.0) as usize;
    let titles = [84.0, 64.0, 76.0];
    for n in 0..rows {
        let row = Rect::from_min_size(
            panel.min + vec2(0.0, ROW_HEIGHT * n as f32),
            vec2(panel.width(), ROW_HEIGHT),
        );
        let alpha = 1.0 - 0.3 * n as f32;
        if n > 0 {
            painter.line_segment(
                [
                    row.left_top() + vec2(0.0, 0.5),
                    row.right_top() + vec2(0.0, 0.5),
                ],
                Stroke::new(1.0, palette.hairline.gamma_multiply(alpha)),
            );
        }
        let ghost = white(0.05 * alpha);
        let y = row.center().y;
        painter.circle_filled(pos2(row.left() + 20.0, y), 6.0, ghost);
        let title = Rect::from_min_size(
            pos2(row.left() + 36.0, y - 4.0),
            vec2(titles[n % titles.len()], 8.0),
        );
        painter.rect_filled(title, 4.0, ghost);
        let detail = Rect::from_min_size(pos2(row.right() - 14.0 - 64.0, y - 3.0), vec2(64.0, 6.0));
        painter.rect_filled(detail, 3.0, ghost);
    }

    painter.set_opacity(1.0);
    let text = painter.layout_no_wrap(
        "Separated tracks will appear here.".to_owned(),
        theme::font(12.0, Weight::Regular),
        palette.muted,
    );
    painter.galley(panel.center() - text.size() / 2.0, text, palette.muted);
}

fn row(
    ui: &mut egui::Ui,
    palette: &Palette,
    drops: &Drops,
    item: &Arc<Dropped>,
    first: bool,
    last: bool,
    now: f32,
) -> bool {
    let state = item.state();
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), ROW_HEIGHT), Sense::hover());

    // Keyed on the source path: an automatic id is a counter that shifts as the
    // list changes, and an id that moves between frames cannot pair a press with
    // its release. Two files can share a title, so not the title either.
    let id = ui.id().with(("recent", &item.source));
    let response = ui.interact(rect, id.with("row"), Sense::click());

    let openable = matches!(state, State::Done(_));
    let running = item.is_running();
    let cross_rect = Rect::from_center_size(
        pos2(rect.right() - 8.0 - CROSS_BOX / 2.0, rect.center().y),
        vec2(CROSS_BOX, CROSS_BOX),
    );
    // Registered after the row and on top of it, so it gets the pointer first.
    let cross = ui.interact(cross_rect, id.with("remove"), Sense::click());
    let touched = response.hovered() || cross.hovered();

    let painter = ui.painter();
    let hover = ui
        .ctx()
        .animate_bool_with_time(id.with("hover"), touched, 0.12);
    if hover > 0.0 {
        let corners = egui::CornerRadius {
            nw: if first { RADIUS as u8 - 1 } else { 0 },
            ne: if first { RADIUS as u8 - 1 } else { 0 },
            sw: if last { RADIUS as u8 - 1 } else { 0 },
            se: if last { RADIUS as u8 - 1 } else { 0 },
        };
        painter.rect_filled(rect.shrink(1.0), corners, white(0.05 * hover));
    }
    if !first {
        painter.line_segment(
            [
                rect.left_top() + vec2(0.0, 0.5),
                rect.right_top() + vec2(0.0, 0.5),
            ],
            Stroke::new(1.0, palette.hairline),
        );
    }

    //  Painted: a `ui.label` registers its own interaction rect on top of the
    //  row's and takes the pointer from it.
    let icon = icons::square(pos2(rect.left() + 14.0 + 6.0, rect.center().y), 12.0);
    let waiting = matches!(&state, State::Working(p) if p.stage == stemd_core::Stage::Queued);
    match &state {
        State::Done(_) => icons::check(painter, icon, palette.good, 1.6),
        State::Failed(_) => icons::cross(painter, icon.shrink(1.0), palette.danger, 1.5),
        _ if waiting => icons::clock(painter, icon, palette.muted, 1.4),
        State::Reading | State::Working(_) => {
            icons::spinner(painter, icon, palette.text, 1.6, now.rem_euclid(1.0));
        }
    }

    let swapped = touched && openable;
    let meta_font = theme::font(12.0, Weight::Regular);
    let meta_right = if touched {
        cross_rect.left() - 4.0 - 6.0
    } else {
        rect.right() - 8.0 - 6.0
    };
    let meta_left = if swapped {
        let open =
            painter.layout_no_wrap("Open folder".to_owned(), meta_font.clone(), palette.text);
        let x = cross_rect.left() - 4.0 - 6.0 - open.size().x;
        painter.galley(
            pos2(x, rect.center().y - open.size().y / 2.0),
            open,
            palette.text,
        );
        x
    } else {
        let colour = if matches!(state, State::Failed(_)) {
            palette.danger
        } else {
            palette.muted
        };
        let mut job = egui::text::LayoutJob::single_section(
            detail(&state),
            egui::TextFormat::simple(meta_font, colour),
        );
        job.wrap = egui::text::TextWrapping::truncate_at_width(rect.width() * 0.45);
        let meta = painter.layout_job(job);
        let x = meta_right - meta.size().x;
        painter.galley(pos2(x, rect.center().y - meta.size().y / 2.0), meta, colour);
        x
    };

    let title_left = icon.right() + 10.0;
    let mut job = egui::text::LayoutJob::single_section(
        item.title.clone(),
        egui::TextFormat::simple(theme::font(13.0, Weight::Medium), palette.text),
    );
    job.wrap =
        egui::text::TextWrapping::truncate_at_width((meta_left - 10.0 - title_left).max(20.0));
    let title = painter.layout_job(job);
    painter.galley(
        pos2(title_left, rect.center().y - title.size().y / 2.0),
        title,
        palette.text,
    );

    if touched {
        let lit = ui
            .ctx()
            .animate_bool_with_time(id.with("cross"), cross.hovered(), 0.12);
        if lit > 0.0 {
            painter.rect_filled(cross_rect, 8.0, white(0.08 * lit));
        }
        icons::cross(
            painter,
            icons::square(cross_rect.center(), 9.0),
            theme::mix(palette.muted, palette.danger, lit),
            1.5,
        );
    }

    if cross
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(match &state {
            State::Done(_) => "Remove from the list and delete the stems",
            _ if running => "Cancel and remove from the list",
            _ => "Remove from the list",
        })
        .clicked()
    {
        if running {
            return true;
        }
        drops.discard(item);
        return false;
    }

    if openable
        && response
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .clicked()
        && let State::Done(finished) = &state
    {
        // Deleted behind the window's back. Opening nothing and saying nothing
        // reads as broken, so the row goes instead.
        if finished.dir.is_dir() {
            reveal(&finished.dir);
        } else {
            tracing::info!("{} is gone; dropping the row", finished.dir.display());
            drops.forget(item);
        }
    }
    false
}

fn detail(state: &State) -> String {
    let dot = theme::DOT;
    match state {
        State::Reading => "reading".to_owned(),
        State::Working(progress) => super::dropzone::stage_label(progress.stage).to_owned(),
        State::Done(finished) if finished.cached => {
            format!("{} parts{dot}already separated", finished.parts.len())
        }
        State::Done(finished) => {
            format!("{} parts{dot}{:.0} s", finished.parts.len(), finished.secs)
        }
        State::Failed(why) => why.clone(),
    }
}

/// Open the stems folder in the system's file manager.
///
/// A failure is worth a log line and nothing more: the stems are written either
/// way.
fn reveal(dir: &std::path::Path) {
    // The absolute path on macOS: a bundle launched from Finder inherits a minimal
    // PATH.
    #[cfg(target_os = "macos")]
    let mut cmd = std::process::Command::new("/usr/bin/open");
    #[cfg(windows)]
    let mut cmd = std::process::Command::new("explorer.exe");
    #[cfg(target_os = "linux")]
    let mut cmd = std::process::Command::new("xdg-open");

    if let Err(err) = cmd.arg(dir).spawn() {
        tracing::warn!("could not reveal {}: {err}", dir.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_takes_its_rows_up_to_the_room_it_has() {
        assert_eq!(
            height(2, 1000.0),
            2.0 * ROW_HEIGHT,
            "two rows grew past themselves"
        );
        assert_eq!(height(20, 300.0), 300.0);
        assert_eq!(height(20, 0.0), ROW_HEIGHT, "no room hid the list entirely");
    }
}
