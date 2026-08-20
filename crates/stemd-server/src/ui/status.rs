//! The bottom bar: is the server up, what is the worker doing, what is stored.
//!
//! One line, and the same [`LogBuffer`]-backed state `/v1/health` reports: a second
//! view, never a second source.
//!
//! [`LogBuffer`]: crate::logbuf::LogBuffer

use eframe::egui::{self, Color32};

use crate::api::AppState;

use super::theme::Palette;

/// Text size along the bar. Everything here is secondary to the drop zone, but
/// the previous 11.0 was small enough to read as disabled rather than quiet.
const TEXT: f32 = 12.5;

/// A fraction as a whole number of per cent.
///
/// Rounded down rather than to nearest, so a bar that has not finished never
/// reads 100%. Watching it sit on "100%" for the last stretch of a four-minute
/// separation is worse than watching it sit on 99%.
pub fn percent(fraction: f32) -> String {
    format!("{}%", (fraction.clamp(0.0, 1.0) * 100.0).floor())
}

/// Height of the bar. Fixed, and every item is centred in it: laying the row out
/// by its natural height let the tallest widget, the button, set the baseline,
/// so the label beside it sat visibly high.
const BAR_HEIGHT: f32 = 30.0;

/// Draw the bar. Returns nothing: everything here is read-only except the cache
/// reset, which acts on the state it was given.
pub fn bar(ui: &mut egui::Ui, state: &AppState, palette: &Palette, bind: std::net::SocketAddr) {
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), BAR_HEIGHT),
        egui::Sense::hover(),
    );
    let mut ui = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    let ui = &mut ui;

    {
        let running = state.queue.running();
        // Green is up and free, amber is busy. Neither is a fault: the colours
        // read as a traffic light, which is what "can I give it a track right
        // now" wants. Grey would be the one to use if the server were down, and
        // there is no such state here: the window is the server.
        dot(
            ui,
            if running.is_some() {
                palette.warn
            } else {
                palette.good
            },
        );

        match &running {
            Some(id) => {
                ui.label(
                    egui::RichText::new("separating")
                        .size(TEXT)
                        .color(palette.text),
                );
                if let Some(job) = state.store.get(id) {
                    let progress = job.progress.lock().clone();
                    ui.add(
                        egui::ProgressBar::new(progress.fraction)
                            .desired_width(96.0)
                            .desired_height(5.0)
                            .corner_radius(3)
                            .fill(palette.accent),
                    );
                    ui.label(
                        egui::RichText::new(percent(progress.fraction))
                            .size(TEXT)
                            .color(palette.muted),
                    );
                }
            }
            None => {
                ui.label(
                    egui::RichText::new(format!("serving on {bind}"))
                        .size(TEXT)
                        .color(palette.muted),
                );
            }
        }

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            cache_controls(ui, state, palette);
        });
    }
}

/// What is stored, and a button to drop it.
///
/// Disabled when there is nothing to drop, so the button doubles as the answer
/// to whether anything is stored. No confirmation: every byte is re-computable
/// from audio that still exists, so a misclick costs one separation.
fn cache_controls(ui: &mut egui::Ui, state: &AppState, palette: &Palette) {
    let stats = state.cache.stats();
    // Not "Clear": the log section has one of those, and two buttons reading
    // "Clear" in one window is a coin toss about which one you are about to
    // press. This one names what it empties.
    if ui
        .add_enabled(
            stats.tracks > 0,
            egui::Button::new(egui::RichText::new("Empty cache").size(TEXT)),
        )
        .on_hover_text("Delete every separated stem. Tracks separate again on the next request.")
        .clicked()
    {
        let (tracks, freed) = state.cache.clear();
        tracing::info!("cache reset: {tracks} tracks, {:.0} MB", freed as f64 / 1e6);
    }
    ui.label(
        egui::RichText::new(format!(
            "{} cached · {:.0} MB",
            stats.tracks,
            stats.bytes as f64 / 1e6
        ))
        .size(TEXT)
        .color(palette.muted),
    );
}

/// The status dot, painted rather than typed.
///
/// The obvious characters (U+25CF and U+25CB) are not both in egui's bundled fonts:
/// U+25CF resolves to `.notdef` and draws nothing. Always filled, now that colour
/// carries the state.
fn dot(ui: &mut egui::Ui, colour: Color32) {
    let height = ui.text_style_height(&egui::TextStyle::Body);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(height * 0.6, height), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.5, colour);
}
