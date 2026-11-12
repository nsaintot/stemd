//! The bottom bar: the server switch and port, what is stored, and the way to
//! the log. On Linux, also a newer release.
//!
//! The same state `/v1/health` reports: a second view, never a second source.

use eframe::egui::{self, Sense, pos2, vec2};

use crate::api::AppState;

use super::icons;
use super::theme::{self, Palette, Weight, white};

/// Height of the bar.
pub const HEIGHT: f32 = 20.0;

const TEXT: f32 = 12.0;
const GAP: f32 = 14.0;

/// A fraction as a whole number of per cent.
///
/// Rounded down, so a bar that has not finished never reads 100%.
pub fn percent(fraction: f32) -> String {
    format!("{}%", (fraction.clamp(0.0, 1.0) * 100.0).floor())
}

/// Free bytes on the cache's volume, read at most every two seconds.
fn free_space(ctx: &egui::Context, state: &AppState) -> Option<u64> {
    let id = egui::Id::new("stemd-free-space");
    let now = ctx.input(|i| i.time);
    if let Some((at, free)) = ctx.data(|d| d.get_temp::<(f64, Option<u64>)>(id))
        && now - at < 2.0
    {
        return free;
    }
    let free = crate::disk::free_bytes(state.cache.root());
    ctx.data_mut(|d| d.insert_temp(id, (now, free)));
    free
}

/// What a click in the bar asked for. Emptying the cache is done in place.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    ToggleLogs,
    ToggleServer,
}

const SWITCH: egui::Vec2 = vec2(26.0, 14.0);

/// The text beside the switch: the server's state, with the port while it is on,
/// or the disk warning, which takes precedence.
fn status_text(port: u16, on: bool, separating: Option<usize>, full: Option<u64>) -> String {
    let dot = theme::DOT;
    if let Some(free) = full {
        return format!("disk nearly full{dot}{:.1} GB free", free as f64 / 1e9);
    }
    match (on, separating) {
        (false, _) => "Server off".to_owned(),
        (true, None) => format!("Server{dot}port {port}"),
        (true, Some(0)) => format!("Server{dot}port {port}{dot}separating"),
        (true, Some(queued)) => {
            format!("Server{dot}port {port}{dot}separating{dot}{queued} queued")
        }
    }
}

/// Draw the bar. Returns what was clicked.
#[must_use]
pub fn bar(
    ui: &mut egui::Ui,
    state: &AppState,
    palette: &Palette,
    port: u16,
    logs_open: bool,
) -> Option<Action> {
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), HEIGHT), Sense::hover());
    let rect = rect.shrink2(vec2(6.0, 0.0));
    let font = theme::font(TEXT, Weight::Regular);
    let y = rect.center().y;
    let mut action = None;

    let on = state.serving.is_on();
    let separating = state.queue.running().map(|_| state.queue.depth());
    let free = free_space(ui.ctx(), state);
    let full = free.filter(|free| *free < crate::disk::FLOOR);
    let status = status_text(port, on, separating, full);
    let text = ui
        .painter()
        .layout_no_wrap(status, font.clone(), egui::Color32::PLACEHOLDER);

    // The switch and its label are one control.
    let track = egui::Rect::from_center_size(pos2(rect.left() + SWITCH.x / 2.0, y), SWITCH);
    let hit = egui::Rect::from_min_max(
        track.left_top() - vec2(4.0, 3.0),
        pos2(
            track.right() + 8.0 + text.size().x + 4.0,
            track.bottom() + 3.0,
        ),
    );
    let response = ui
        .interact(hit, ui.id().with("server-switch"), Sense::click())
        .on_hover_text(if on {
            format!("Answering players on port {port}. Click to stop.")
        } else {
            format!(
                "Not answering. Click to serve on port {port}. \
                 Drops in this window work either way."
            )
        });
    let lit = ui
        .ctx()
        .animate_bool_with_time(response.id.with("on"), on, 0.18);
    let painter = ui.painter();
    let r = SWITCH.y / 2.0;
    painter.rect_filled(track, r, theme::mix(palette.danger, palette.good, lit));
    let knob = pos2(
        track.left() + r + (SWITCH.x - SWITCH.y) * lit,
        track.center().y,
    );
    painter.circle_filled(knob, r - 2.0, palette.text);
    let colour = if full.is_some() {
        palette.danger
    } else {
        palette.muted
    };
    painter.galley(
        pos2(track.right() + 8.0, y - text.size().y / 2.0),
        text,
        colour,
    );
    if response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .clicked()
    {
        action = Some(Action::ToggleServer);
    }

    // Right to left: the log button, "Empty cache", what is stored.
    let logs = icons::square(pos2(rect.right() + 5.0 - 12.0, y), 24.0);
    let logs_response = ui
        .interact(logs, ui.id().with("logs"), Sense::click())
        .on_hover_text(if logs_open {
            "Hide the log"
        } else {
            "Show the log"
        });
    let lit = ui.ctx().animate_bool_with_time(
        logs_response.id.with("hover"),
        logs_response.hovered() || logs_open,
        0.15,
    );
    if lit > 0.0 {
        ui.painter().rect_filled(logs, 7.0, white(0.1 * lit));
    }
    icons::lines(
        ui.painter(),
        icons::square(logs.center(), 14.0),
        theme::mix(palette.muted, egui::Color32::WHITE, lit),
        1.4,
    );

    let stats = state.cache.stats();
    let empty = ui.painter().layout_no_wrap(
        "Empty cache".to_owned(),
        font.clone(),
        egui::Color32::PLACEHOLDER,
    );
    let empty_rect = egui::Rect::from_min_size(
        pos2(
            logs.left() + 6.0 - GAP - empty.size().x,
            y - empty.size().y / 2.0,
        ),
        empty.size(),
    );
    let has_any = stats.tracks > 0;
    let empty_response = ui.interact(
        empty_rect,
        ui.id().with("empty-cache"),
        if has_any {
            Sense::click()
        } else {
            Sense::hover()
        },
    );
    let danger = ui.ctx().animate_bool_with_time(
        empty_response.id.with("hover"),
        empty_response.hovered() && has_any,
        0.15,
    );
    let colour = if has_any {
        theme::mix(palette.text, palette.danger, danger)
    } else {
        palette.muted.gamma_multiply(0.6)
    };
    ui.painter().galley(empty_rect.min, empty, colour);
    // No confirmation: every byte is re-computable from audio that still exists,
    // so a misclick costs one separation.
    if empty_response
        .on_hover_text("Delete all separated tracks")
        .on_hover_cursor(if has_any {
            egui::CursorIcon::PointingHand
        } else {
            egui::CursorIcon::Default
        })
        .clicked()
    {
        let (tracks, freed) = state.cache.clear();
        tracing::info!("cache reset: {tracks} tracks, {:.0} MB", freed as f64 / 1e6);
    }

    let _stats = ui.painter().text(
        pos2(empty_rect.left() - GAP, y),
        egui::Align2::RIGHT_CENTER,
        format!(
            "{} cached{}{:.0} MB",
            stats.tracks,
            theme::DOT,
            stats.bytes as f64 / 1e6
        ),
        font,
        palette.muted,
    );

    // Linux has no title bar of its own to carry the version, so a newer
    // release shows here, between the two groups, where it fits.
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let status = crate::update::status();
        if status.opens()
            && let Some(x) = super::update::fits(ui, &status, hit.right(), _stats.left())
        {
            let anchor = pos2(x, y);
            super::update::slot(ui, anchor, egui::Align2::LEFT_CENTER, palette, 1.0);
        }
    }

    if logs_response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .clicked()
    {
        action = Some(Action::ToggleLogs);
    }
    action
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_status_reads_port_then_state() {
        assert_eq!(status_text(8420, true, None, None), "Server · port 8420");
        assert_eq!(status_text(8420, false, Some(2), None), "Server off");
        assert_eq!(
            status_text(8420, true, Some(2), None),
            "Server · port 8420 · separating · 2 queued"
        );
        assert_eq!(
            status_text(8420, true, None, Some(1_200_000_000)),
            "disk nearly full · 1.2 GB free"
        );
    }
}
