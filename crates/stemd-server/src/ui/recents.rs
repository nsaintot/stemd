//! The list of tracks this window has separated.
//!
//! A finished row's only job is to get you to the stems, so the whole row opens the
//! folder. Failures stay in the list with their reason rather than disappearing.
//!
//! Rows leave two ways: the cross removes one and deletes its stems, and a row
//! whose folder has been deleted behind the window's back removes itself when
//! anyone asks it to open that folder.

use std::sync::Arc;

use eframe::egui::{self, Color32, Stroke};

use crate::drops::{Dropped, Drops, State};

use super::theme::{CARD_RADIUS, Palette, Sheen};

const ROW_HEIGHT: f32 = 40.0;
const DOT_RADIUS: f32 = 3.5;

/// How many rows the list shows before it scrolls, in rows rather than points
/// so it stays four of them if [`ROW_HEIGHT`] ever moves. The half is the point
/// of it: a whole number of rows ends the list on a clean edge and looks like
/// the whole list, while half a row showing under the fourth says there is more
/// below. Without a cap the window grew by a row per track and twelve of them
/// took it past the bottom of the screen.
const VISIBLE_ROWS: f32 = 4.5;

/// How tall the list gets before it scrolls.
///
/// Read off the ui rather than stated as a constant because the gap between
/// rows comes from the theme, and a cap that ignored it would be a row short of
/// what it claims the moment the spacing moves.
pub fn visible_height(ui: &egui::Ui) -> f32 {
    ROW_HEIGHT * VISIBLE_ROWS + ui.spacing().item_spacing.y * (VISIBLE_ROWS - 1.0)
}

/// Hit target for the delete control. Bigger than the cross it draws: a
/// nine-point glyph is not something to ask anyone to hit.
const CROSS_BOX: f32 = 22.0;

/// Arm length of the painted cross, from its centre.
const CROSS_ARM: f32 = 3.5;

pub fn list(
    ui: &mut egui::Ui,
    palette: &Palette,
    sheen: &mut Sheen,
    drops: &Drops,
    items: &[Arc<Dropped>],
) {
    if items.is_empty() {
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new("Nothing yet. Dropped tracks show up here.")
                .size(11.5)
                .color(palette.muted),
        );
        return;
    }

    // The window sizes itself to this panel, so the cap here is what stops a
    // long list from resizing the window off the screen. `auto_shrink` on the
    // vertical keeps a short list the height of its own rows: a fixed box would
    // leave empty space under one row, which reads as a list that lost
    // something.
    egui::ScrollArea::vertical()
        .id_salt("recents")
        .max_height(visible_height(ui))
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for item in items {
                row(ui, palette, sheen, drops, item);
            }
        });
}

fn row(
    ui: &mut egui::Ui,
    palette: &Palette,
    sheen: &mut Sheen,
    drops: &Drops,
    item: &Arc<Dropped>,
) {
    let state = item.state();
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, ROW_HEIGHT), egui::Sense::hover());
    let inner = rect.shrink2(egui::vec2(12.0, 6.0));

    // Interacted by an id derived from the track rather than by allocation
    // order. An automatic id is a counter over the widgets before it, so it
    // shifts whenever the list above changes shape, and an id that changes
    // between frames cannot correlate a press with its release, which is
    // exactly a row that highlights on hover and does nothing when clicked.
    //
    // Keyed on the source path rather than the title: two different files can
    // share a title (different folders, same name), and `title` alone would
    // hand both rows the same id, colliding in exactly the way this comment
    // warns about.
    let id = ui.id().with(("recent", &item.source));
    let response = ui.interact(rect, id.with("row"), egui::Sense::click());

    // Registered after the row and overlapping it, so egui hands it the pointer
    // first: the last widget to claim a spot is the one on top of it.
    let removable = !item.is_running();
    let cross_rect = egui::Rect::from_center_size(
        egui::pos2(inner.right() - CROSS_BOX / 2.0, inner.center().y),
        egui::Vec2::splat(CROSS_BOX),
    );
    let cross = removable.then(|| ui.interact(cross_rect, id.with("remove"), egui::Sense::click()));

    let openable = matches!(state, State::Done(_));
    // The cross sits inside the row, so pointing at it takes the hover off the
    // row underneath. Without this the control would vanish as it was reached.
    let touched = response.hovered() || cross.as_ref().is_some_and(egui::Response::hovered);
    sheen.card(
        ui,
        rect,
        CARD_RADIUS,
        if touched && openable {
            palette.card_hover
        } else {
            palette.card
        },
        Stroke::new(1.0, palette.outline),
    );

    //  Painted, not laid out as widgets. A `ui.label` inside the row registers its own
    //  interaction rect on top of the row's, so egui hands the pointer to the label and
    //  the row beneath stops seeing the hover: the whole card clickable except over its
    //  own text. The painter takes no part in interaction.
    let painter = ui.painter();
    painter.circle_filled(
        egui::pos2(inner.left() + DOT_RADIUS, inner.center().y),
        DOT_RADIUS,
        colour_for(palette, &state),
    );

    let text_left = inner.left() + DOT_RADIUS * 2.0 + 8.0;
    painter.text(
        egui::pos2(text_left, inner.center().y - 1.0),
        egui::Align2::LEFT_BOTTOM,
        &item.title,
        egui::FontId::proportional(12.5),
        palette.text,
    );
    painter.text(
        egui::pos2(text_left, inner.center().y + 2.0),
        egui::Align2::LEFT_TOP,
        detail(&state),
        egui::FontId::proportional(10.5),
        match &state {
            State::Failed(_) => palette.bad,
            _ => palette.muted,
        },
    );

    if touched && openable {
        painter.text(
            egui::pos2(cross_rect.left() - 6.0, inner.center().y),
            egui::Align2::RIGHT_CENTER,
            "Show in Finder",
            egui::FontId::proportional(10.5),
            palette.accent,
        );
    }

    // Only under the pointer: twelve permanent crosses down the side of the list
    // would be the loudest thing in the window, and the least often wanted.
    if let Some(cross) = &cross
        && touched
    {
        draw_cross(
            ui,
            cross_rect.center(),
            if cross.hovered() {
                palette.bad
            } else {
                palette.muted
            },
        );
    }

    if let Some(cross) = cross
        && cross
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text(match &state {
                State::Done(_) => "Remove this, and delete its stems",
                _ => "Remove this from the list",
            })
            .clicked()
    {
        drops.discard(item);
        return;
    }

    if openable
        && response
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .clicked()
        && let State::Done(finished) = &state
    {
        // Deleted behind the window's back: in Finder, or by the cross on a
        // second copy of the same track. Opening nothing and saying nothing is
        // the one outcome that reads as broken, so the row goes instead.
        if finished.dir.is_dir() {
            reveal(&finished.dir);
        } else {
            tracing::info!("{} is gone; dropping the row", finished.dir.display());
            drops.forget(item);
        }
    }
}

/// A cross, painted rather than typed, for the same reason the drop zone's arrow
/// is: egui bundles three fonts and most symbol glyphs are in none of them.
fn draw_cross(ui: &egui::Ui, centre: egui::Pos2, colour: Color32) {
    let painter = ui.painter();
    let stroke = Stroke::new(1.3, colour);
    for dx in [-CROSS_ARM, CROSS_ARM] {
        painter.line_segment(
            [
                egui::pos2(centre.x - dx, centre.y - CROSS_ARM),
                egui::pos2(centre.x + dx, centre.y + CROSS_ARM),
            ],
            stroke,
        );
    }
}

fn colour_for(palette: &Palette, state: &State) -> Color32 {
    match state {
        State::Reading | State::Working(_) => palette.accent,
        State::Done(_) => palette.good,
        State::Failed(_) => palette.bad,
    }
}

fn detail(state: &State) -> String {
    match state {
        State::Reading => "reading".to_owned(),
        State::Working(progress) => super::dropzone::stage_label(progress.stage).to_owned(),
        State::Done(finished) if finished.cached => {
            format!("{} parts · already separated", finished.parts.len())
        }
        State::Done(finished) => format!(
            "{} parts · separated in {:.0}s",
            finished.parts.len(),
            finished.secs
        ),
        State::Failed(why) => why.clone(),
    }
}

/// Open the stems folder in the system's file manager.
///
/// A failure here is worth a log line and nothing more: the stems are written
/// either way, and the row still shows where they are.
fn reveal(dir: &std::path::Path) {
    // macOS takes the absolute path rather than bare `open`: a bundle launched
    // from Finder inherits a minimal PATH, and "the button does nothing" is a
    // bad way to find that out. The other two are resolved through PATH because
    // that is the only place they are.
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
