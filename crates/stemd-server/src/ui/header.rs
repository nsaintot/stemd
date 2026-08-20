//! The top row: which model is loaded, and what the stems come out as.
//!
//! Both are saved as they change, see [`crate::settings`], so this row is also
//! what the next launch will do, not just this one.

use std::sync::Arc;

use eframe::egui;
use stemd_core::{OutputRate, StemFormat};

use crate::api::AppState;
use crate::models::Preset;
use crate::switch::{SwitchState, Switcher};

use super::theme::Palette;

const MODEL_WIDTH: f32 = 112.0;
const FORMAT_WIDTH: f32 = 88.0;
const RATE_WIDTH: f32 = 82.0;
const DIALOG_WIDTH: f32 = 300.0;

/// Text size in the header row.
const TEXT: f32 = 12.0;

/// Cap on a dropdown's list height.
///
/// `ComboBox` puts its contents in a `ScrollArea` capped at
/// `spacing.combo_height`, 200 points by default, and a list that reaches the cap
/// grows a scrollbar. This is deliberately far above what any list here needs
/// rather than computed to fit: a computed height a few points short clips the
/// list and puts the scrollbar back. The popup sizes itself to its contents; this
/// only says where scrolling starts.
///
/// Not enough on its own: see [`borderless_rows`].
pub const MENU_CAP: f32 = 600.0;

/// Take the border off the rows inside a dropdown.
///
/// A `selectable_label` in a menu picks up the window's one-point button outline,
/// which is drawn outside the row's rect, so a list of them measures a fraction of
/// a point taller than the box holding it and the `ScrollArea` grows a scroll bar.
/// Raising [`MENU_CAP`] cannot fix it: the overflow is against the list's own
/// height.
///
/// Pinned by [`no_dropdown_grows_a_scroll_bar`](tests::no_dropdown_grows_a_scroll_bar).
pub fn borderless_rows(ui: &mut egui::Ui) {
    let widgets = &mut ui.style_mut().visuals.widgets;
    for widget in [
        &mut widgets.inactive,
        &mut widgets.hovered,
        &mut widgets.active,
        &mut widgets.open,
        &mut widgets.noninteractive,
    ] {
        widget.bg_stroke = egui::Stroke::NONE;
    }
}

/// Marks a preset the machine does not have yet.
///
/// U+2B07 rather than a typographically nicer arrow: egui bundles only
/// Ubuntu-Light, NotoEmoji and emoji-icon-font, and most arrow glyphs are in
/// none of the three. A missing glyph draws nothing at all, so anything used
/// here has to be checked against those fonts rather than assumed.
const DOWNLOAD_MARK: &str = " ⬇";

/// Marks a preset that cannot meet the latency target the player expects.
const SLOW_MARK: &str = " ⚠";

/// Shown on an output control a command-line flag fixed for this run.
const PINNED_HINT: &str = "Set on the command line for this run. Restart without \
                           the flag to change it here.";

/// No wordmark here: the title bar two points above already says "stemd", and
/// saying it twice in forty pixels is the kind of thing that reads as unfinished.
/// The row is the controls.
pub fn row(ui: &mut egui::Ui, state: &AppState, palette: &Palette) {
    ui.horizontal(|ui| {
        model(ui, state, palette);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            output(ui, state);
        });
    });
}

/// The model dropdown, plus whatever the switch is currently doing.
///
/// Presets are named for the trade rather than the checkpoint. Two things are
/// marked because both cost you something you would otherwise find out the hard
/// way: an artefact that is not on disk yet, and one that cannot meet the
/// latency target.
fn model(ui: &mut egui::Ui, state: &AppState, palette: &Palette) {
    let switcher = Arc::clone(&state.switcher);
    let switch = switcher.state();
    let current = switcher.current();

    // "Speed" on its own says nothing about what it is the speed of. The presets
    // are named for a trade, and the label is what tells you which trade.
    ui.label(egui::RichText::new("Model").size(TEXT).color(palette.muted));

    ui.add_enabled_ui(!switch.busy(), |ui| {
        let selected = current.map_or_else(
            || "custom".to_owned(),
            |preset| entry_for(&switcher, preset),
        );
        egui::ComboBox::from_id_salt("model-preset")
            .selected_text(egui::RichText::new(selected).size(TEXT))
            .width(MODEL_WIDTH)
            .height(MENU_CAP)
            .show_ui(ui, |ui| {
                borderless_rows(ui);
                for preset in Preset::ALL {
                    if ui
                        .selectable_label(current == Some(preset), entry_for(&switcher, preset))
                        .on_hover_text(hint_for(&switcher, preset))
                        .clicked()
                    {
                        switcher.request(preset);
                    }
                }
            })
            .response
            .on_hover_text(match current {
                Some(preset) => preset.detail().to_owned(),
                None => "a hand-traced artefact, not one of the presets".to_owned(),
            });
    });

    // After the menu, not before it: the row reads left to right now, so a
    // marker belongs beside the thing it is marking rather than ahead of it.
    if switch.busy() {
        ui.spinner();
    } else if matches!(switch, SwitchState::Failed { .. }) {
        ui.label(
            egui::RichText::new("switch failed")
                .size(11.0)
                .color(palette.bad),
        );
    }
    if current.is_some_and(Preset::over_budget) {
        ui.label(egui::RichText::new("slow").size(11.0).color(palette.warn));
    }
}

/// What a client gets when it does not ask for a format or a rate, and what a
/// drop is written as.
fn output(ui: &mut egui::Ui, state: &AppState) {
    let settings = Arc::clone(&state.settings);
    let current = settings.get();
    let pinned = settings.pinned();

    ui.add_enabled_ui(!pinned.rate, |ui| {
        egui::ComboBox::from_id_salt("output-rate")
            .selected_text(egui::RichText::new(rate_label(current.rate)).size(TEXT))
            .width(RATE_WIDTH)
            .height(MENU_CAP)
            .show_ui(ui, |ui| {
                borderless_rows(ui);
                // Only the rates the chosen format can carry. Offering 96 kHz
                // beside MP3 would be offering something that does not exist:
                // picking it used to cost eighteen seconds a stem and produce a
                // 48 kHz file anyway.
                for rate in current.format.rates() {
                    if ui
                        .selectable_label(current.rate == rate, rate_label(rate))
                        .on_hover_text(rate_hint(rate))
                        .clicked()
                    {
                        settings.set_rate(rate);
                    }
                }
            });
    })
    .response
    .on_disabled_hover_text(PINNED_HINT);

    ui.add_enabled_ui(!pinned.format, |ui| {
        egui::ComboBox::from_id_salt("output-format")
            .selected_text(egui::RichText::new(current.format.label()).size(TEXT))
            .width(FORMAT_WIDTH)
            .height(MENU_CAP)
            .show_ui(ui, |ui| {
                borderless_rows(ui);
                for format in StemFormat::ALL {
                    if ui
                        .selectable_label(current.format == format, format.label())
                        .on_hover_text(format_hint(format))
                        .clicked()
                    {
                        settings.set_format(format);
                    }
                }
            });
    })
    .response
    .on_disabled_hover_text(PINNED_HINT);
}

/// The download / load dialog. Only drawn while a switch is in flight or has
/// failed, so there is nothing to dismiss in the common case.
pub fn switch_dialog(ctx: &egui::Context, state: &AppState, palette: &Palette) {
    let switcher = Arc::clone(&state.switcher);
    let switch = switcher.state();
    if matches!(switch, SwitchState::Idle) {
        return;
    }

    // `Loading` covers two situations that look nothing alike to whoever is
    // waiting: the worker building the model, which takes seconds, and the
    // worker still separating a track, which takes as long as the track. The
    // state cannot tell them apart, because from the switch's side both are the
    // same blocked call. The queue can.
    let separating = state.queue.running().is_some();

    egui::Window::new("Switching model")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.add_space(4.0);
            dialog_body(ui, &switch, &switcher, palette, separating);
            ui.add_space(4.0);
        });
}

/// The download dialog: a bar, and underneath it what is being fetched and how far
/// along.
///
/// The counts sit below the bar rather than inside it: egui draws a progress bar's
/// text over a six-point bar and clips it, and a bar tall enough to hold text
/// would not be the hairline the rest of this window uses.
pub(super) fn downloading(
    ui: &mut egui::Ui,
    preset: Preset,
    file: &str,
    done: u64,
    total: u64,
    palette: &Palette,
) {
    ui.label(format!("Downloading the {} model", preset.label()));
    ui.label(
        egui::RichText::new(preset.detail())
            .size(11.0)
            .color(palette.muted),
    );
    ui.add_space(8.0);
    let fraction = if total > 0 {
        done as f32 / total as f32
    } else {
        0.0
    };
    ui.add(
        egui::ProgressBar::new(fraction)
            .desired_width(DIALOG_WIDTH)
            .desired_height(6.0)
            .corner_radius(3)
            .fill(palette.accent),
    );
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        let small = |text: String| egui::RichText::new(text).size(10.5).color(palette.muted);
        ui.label(small(file.to_owned()));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(small(format!(
                "{} · {:.0} / {:.0} MB",
                super::status::percent(fraction),
                done as f64 / 1e6,
                total as f64 / 1e6
            )));
        });
    });
    ui.add_space(6.0);
    ui.label(
        egui::RichText::new("Separation keeps running on the current model.")
            .size(10.5)
            .color(palette.muted),
    );
}

fn dialog_body(
    ui: &mut egui::Ui,
    state: &SwitchState,
    switcher: &Switcher,
    palette: &Palette,
    separating: bool,
) {
    match state {
        SwitchState::Downloading {
            preset,
            file,
            done,
            total,
        } => downloading(ui, *preset, file, *done, *total, palette),
        SwitchState::Verifying(preset) => {
            ui.label(format!("Checking the {} model", preset.label()));
            ui.add_space(8.0);
            ui.spinner();
        }
        // Waiting for the track in flight, not loading anything. Saying
        // "Loading onto the GPU" through a six-minute separation is a dialog
        // that looks wedged, next to a progress bar that is still moving, and
        // invites someone to force-quit a working program.
        SwitchState::Loading(preset) if separating => {
            ui.label(format!(
                "{} will load when the track in flight finishes",
                preset.label()
            ));
            ui.label(
                egui::RichText::new("It keeps running on the current model.")
                    .size(10.5)
                    .color(palette.muted),
            );
            ui.add_space(8.0);
            ui.spinner();
        }
        SwitchState::Loading(preset) => {
            ui.label(format!("Loading {} onto the GPU", preset.label()));
            ui.add_space(8.0);
            ui.spinner();
        }
        SwitchState::Failed { preset, message } => {
            ui.label(
                egui::RichText::new(format!("Could not switch to {}", preset.label()))
                    .color(palette.bad),
            );
            ui.add_space(6.0);
            ui.label(egui::RichText::new(message).size(11.0));
            ui.add_space(8.0);
            // The old model is untouched, so this is dismissable rather than
            // fatal.
            if ui.button("Keep current model").clicked() {
                switcher.dismiss_error();
            }
        }
        SwitchState::Idle => {}
    }
}

/// One dropdown row: the preset name plus any marks.
fn entry_for(switcher: &Switcher, preset: Preset) -> String {
    let mut entry = preset.label().to_owned();
    if !switcher.is_local(preset) {
        entry.push_str(DOWNLOAD_MARK);
    }
    if preset.over_budget() {
        entry.push_str(SLOW_MARK);
    }
    entry
}

/// Hover text for a dropdown row, warning about a download before it starts.
fn hint_for(switcher: &Switcher, preset: Preset) -> String {
    let mut hint = preset.detail().to_owned();
    if !switcher.is_local(preset) {
        hint.push_str(&format!(
            "\n\nnot installed — choosing this downloads {:.0} MB",
            preset.total_bytes() as f64 / 1e6
        ));
    }
    hint
}

/// A rate as a person reads it. `Display` gives the number the API takes, which
/// is the right answer in a URL and the wrong one in a menu.
fn rate_label(rate: OutputRate) -> String {
    match rate.hz() {
        44_100 => "44.1 kHz".to_owned(),
        hz => format!("{} kHz", hz / 1000),
    }
}

fn rate_hint(rate: OutputRate) -> &'static str {
    match rate {
        OutputRate::Hz24000 => "Half the bandwidth, half the bytes.",
        OutputRate::Hz44100 => "The model's own rate. No conversion runs.",
        OutputRate::Hz48000 => "Converted from 44.1 kHz.",
        OutputRate::Hz96000 => "Converted from 44.1 kHz. Twice the bytes, no more detail.",
    }
}

fn format_hint(format: StemFormat) -> &'static str {
    match format {
        StemFormat::Flac => "Lossless, about half the bytes of a WAV, and openable everywhere.",
        StemFormat::Wav => "The same 16-bit samples as FLAC, uncompressed. For anything fussy.",
        StemFormat::Mp3 => {
            "320 kbps, and the only lossy option. Stems will not sum back to \
                            the mix, so it is for listening rather than for a deck."
        }
        StemFormat::Pcm16 => "Raw 16-bit samples: no container, no decode step.",
        StemFormat::Pcm32 => {
            "Raw floats. Twice the bytes, no ceiling at full scale, and the \
                              only route to an exact null."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Characters checked by hand against the cmap and glyf tables of egui's
    /// bundled fonts. Extend this only after checking a new one the same way.
    const COVERED_BY_BUNDLED_FONTS: [char; 3] = ['⬇', '⚠', '·'];

    /// Every non-ASCII mark in the window must exist in one of egui's three
    /// bundled fonts. A missing glyph draws nothing at all rather than a
    /// placeholder, so it fails silently: U+25CF and U+2913 both did.
    #[test]
    fn the_menu_marks_are_glyphs_the_bundled_fonts_have() {
        for mark in [DOWNLOAD_MARK, SLOW_MARK] {
            for c in mark.chars().filter(|c| !c.is_whitespace()) {
                assert!(
                    COVERED_BY_BUNDLED_FONTS.contains(&c),
                    "{c:?} (U+{:04X}) is not known to render; check it against \
                     Ubuntu-Light, NotoEmoji and emoji-icon-font before using it",
                    c as u32
                );
            }
        }
    }

    #[test]
    fn a_rate_reads_as_kilohertz_including_the_awkward_one() {
        assert_eq!(rate_label(OutputRate::Hz44100), "44.1 kHz");
        assert_eq!(rate_label(OutputRate::Hz48000), "48 kHz");
        assert_eq!(rate_label(OutputRate::Hz24000), "24 kHz");
        assert_eq!(rate_label(OutputRate::Hz96000), "96 kHz");
    }
}
