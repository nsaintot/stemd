//! The top row: a segmented control for the model, and a capsule with the parts,
//! format and rate menus.
//!
//! Everything here is saved as it changes, see [`crate::settings`], so this row is
//! also what the next launch will do.

use std::sync::Arc;

use eframe::egui::{self, Color32, Rect, Sense, Stroke, pos2, vec2};
use stemd_core::{OutputRate, PART_NAMES, StemFormat};

use crate::api::AppState;
use crate::models::Preset;
use crate::settings::{DropParts, SettingsStore};
use crate::switch::{SwitchState, Switcher};

use super::icons;
use super::theme::{self, Brushes, Glass, Palette, Tween, Weight, white};
use super::widgets::{self, menu_row, pick_menu};

/// Height of both header controls.
pub const HEIGHT: f32 = 32.0;

/// Segment widths in the segmented control, Fast, Balanced and Quality.
const SEGMENTS: [f32; 3] = [54.0, 86.0, 88.0];
const SEGMENT_HEIGHT: f32 = 24.0;
/// Space between the control's border and its segments.
const INSET: f32 = 3.0;

/// Seconds the thumb takes to reach each preset: quick to Fast, slow to Quality.
const THUMB_TIME: [f32; 3] = [0.12, 0.24, 0.5];

const TEXT: f32 = 13.0;
const CAP_HEIGHT: f32 = 28.0;
const CHEVRON: f32 = 8.0;

/// Shown on an output control a command-line flag fixed for this run.
const PINNED_HINT: &str = "Set on the command line for this run. Restart without \
                           the flag to change it here.";

/// Least room between the model control and the capsule.
const ROW_GAP: f32 = 12.0;

/// Draw the row. Returns the width it needs without the two controls touching.
pub fn row(
    ui: &mut egui::Ui,
    state: &AppState,
    palette: &Palette,
    brushes: &Brushes,
    clock: f32,
) -> f32 {
    // The glass's drop shadow reaches past the panel; let it.
    ui.set_clip_rect(ui.ctx().content_rect());
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), HEIGHT), Sense::hover());
    let left = models(ui, rect, state, palette, brushes, clock);
    let right = capsule(ui, rect, state, palette, brushes);
    left + ROW_GAP + right
}

/// The preset the segmented control shows: the one being loaded once its
/// download and check are done, else the loaded one.
pub fn shown_preset(switcher: &Switcher) -> Option<Preset> {
    match switcher.state() {
        SwitchState::Loading(preset) => Some(preset),
        _ => switcher.current(),
    }
}

/// The preset a switch in flight is fetching or checking.
fn pending_preset(switcher: &Switcher) -> Option<Preset> {
    match switcher.state() {
        SwitchState::Downloading { preset, .. } | SwitchState::Verifying(preset) => Some(preset),
        _ => None,
    }
}

fn index_of(preset: Preset) -> usize {
    Preset::ALL.iter().position(|p| *p == preset).unwrap_or(0)
}

/// Fast, Balanced and Quality as one control with a sliding thumb. Returns its
/// width.
fn models(
    ui: &mut egui::Ui,
    row: Rect,
    state: &AppState,
    palette: &Palette,
    brushes: &Brushes,
    clock: f32,
) -> f32 {
    let switcher = Arc::clone(&state.switcher);
    let busy = switcher.state().busy();
    let shown = shown_preset(&switcher);
    let pending = pending_preset(&switcher);
    let width = 2.0 * (1.0 + INSET) + SEGMENTS.iter().sum::<f32>();
    let rect = Rect::from_min_size(row.min, vec2(width, HEIGHT));
    let painter = ui.painter();
    brushes.glass(painter, rect, HEIGHT / 2.0, Glass::Panel);

    let lefts: [f32; 3] = std::array::from_fn(|i| SEGMENTS[..i].iter().sum::<f32>());
    let top = rect.top() + 1.0 + INSET;
    let segment = |left: f32, width: f32| {
        Rect::from_min_size(
            pos2(rect.left() + 1.0 + INSET + left, top),
            vec2(width, SEGMENT_HEIGHT),
        )
    };

    if let Some(preset) = shown {
        let i = index_of(preset);
        let id = ui.id().with("model-thumb");
        let x = Tween::follow(
            ui.ctx(),
            id.with("x"),
            lefts[i],
            THUMB_TIME[i],
            theme::ease_spring,
        );
        let w = Tween::follow(
            ui.ctx(),
            id.with("w"),
            SEGMENTS[i],
            THUMB_TIME[i],
            theme::ease_spring,
        );
        let thumb = segment(x, w);
        if preset == Preset::Quality {
            brushes.sheen(painter, thumb, SEGMENT_HEIGHT / 2.0, 0.4 * clock);
        } else {
            brushes.pill(painter, thumb, SEGMENT_HEIGHT / 2.0, 1.0);
        }
    }

    for (i, preset) in Preset::ALL.into_iter().enumerate() {
        let seg = segment(lefts[i], SEGMENTS[i]);
        let response = ui
            .interact(seg, ui.id().with(("model", i)), Sense::click())
            .on_hover_text(hint_for(&switcher, preset));
        let selected = shown == Some(preset);
        let hover = ui.ctx().animate_bool_with_time(
            response.id.with("hover"),
            response.hovered() && !busy,
            0.15,
        );
        let colour = if selected {
            palette.ink
        } else {
            theme::mix(palette.soft, Color32::WHITE, hover)
        };
        let weight = if selected {
            Weight::Semibold
        } else {
            Weight::Medium
        };
        let remote = !selected && !switcher.is_local(preset);
        let galley = ui.painter().layout_no_wrap(
            preset.label().to_owned(),
            theme::font(TEXT, weight),
            colour,
        );
        let icon = if remote { 10.0 + 5.0 } else { 0.0 };
        let left = seg.center().x - (galley.size().x + icon) / 2.0;
        let text_pos = pos2(left, seg.center().y - galley.size().y / 2.0);
        let text_width = galley.size().x;
        ui.painter().galley(text_pos, galley, colour);
        let icon_box = icons::square(pos2(left + text_width + 5.0 + 5.0, seg.center().y), 10.0);
        if pending == Some(preset) {
            icons::spinner(ui.painter(), icon_box, colour, 1.3, clock.rem_euclid(1.0));
        } else if remote {
            icons::download(ui.painter(), icon_box, colour, 1.3);
        }
        if !busy && response.clicked() && !selected {
            switcher.request(preset);
        }
        if !busy && !selected {
            response.on_hover_cursor(egui::CursorIcon::PointingHand);
        }
    }
    width
}

/// Which capsule menu.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Menu {
    Parts,
    Format,
    Rate,
}

/// Parts, format and rate, in one glass capsule on the right. Returns its width.
fn capsule(
    ui: &mut egui::Ui,
    row: Rect,
    state: &AppState,
    palette: &Palette,
    brushes: &Brushes,
) -> f32 {
    let settings = Arc::clone(&state.settings);
    let current = settings.get();
    let pinned = settings.pinned();
    let count = current.drop_parts.names().count();
    let items = [
        (
            Menu::Parts,
            if count == 1 {
                "1 part".to_owned()
            } else {
                format!("{count} parts")
            },
            (13.0, 10.0),
            false,
        ),
        (
            Menu::Format,
            match current.format {
                // The menu says "float"; the capsule has no room for it.
                StemFormat::Pcm32 => "PCM 32-bit".to_owned(),
                format => format.label().to_owned(),
            },
            (12.0, 10.0),
            pinned.format,
        ),
        (
            Menu::Rate,
            rate_label(current.rate),
            (12.0, 11.0),
            pinned.rate,
        ),
    ];

    let font = theme::font(TEXT, Weight::Regular);
    let widths: Vec<f32> = items
        .iter()
        .map(|(_, label, (left, right), _)| {
            let text = ui
                .painter()
                .layout_no_wrap(label.clone(), font.clone(), palette.text)
                .size()
                .x;
            left + text + 6.0 + CHEVRON + right
        })
        .collect();
    // Two hairline dividers and a point of padding at each end.
    let width = widths.iter().sum::<f32>() + 2.0 + 2.0;
    let rect = Rect::from_min_size(pos2(row.right() - width, row.top()), vec2(width, HEIGHT));
    brushes.glass(ui.painter(), rect, HEIGHT / 2.0, Glass::Panel);

    let mut x = rect.left() + 1.0;
    for (n, ((menu, label, (pad_left, _), disabled), item_width)) in
        items.into_iter().zip(widths).enumerate()
    {
        if n > 0 {
            ui.painter().line_segment(
                [
                    pos2(x + 0.5, rect.center().y - 7.0),
                    pos2(x + 0.5, rect.center().y + 7.0),
                ],
                Stroke::new(1.0, palette.divider),
            );
            x += 1.0;
        }
        let button = Rect::from_min_size(
            pos2(x, rect.center().y - CAP_HEIGHT / 2.0),
            vec2(item_width, CAP_HEIGHT),
        );
        x += item_width;
        let id = ui.id().with(("capsule", n));
        let response = ui.interact(
            button,
            id,
            if disabled {
                Sense::hover()
            } else {
                Sense::click()
            },
        );
        let open = widgets::menu(
            &response,
            if menu == Menu::Rate {
                egui::RectAlign::BOTTOM_END
            } else {
                egui::RectAlign::BOTTOM_START
            },
            if menu == Menu::Parts {
                egui::PopupCloseBehavior::CloseOnClickOutside
            } else {
                egui::PopupCloseBehavior::CloseOnClick
            },
            palette,
            |ui| match menu {
                Menu::Parts => parts_menu(ui, &settings, current.drop_parts, state, palette),
                Menu::Format => format_menu(ui, &settings, current.format, palette),
                Menu::Rate => rate_menu(ui, &settings, current.format, current.rate, palette),
            },
        );
        cap_button(
            ui, button, &response, &label, pad_left, open, disabled, palette, brushes,
        );
        if disabled {
            response.on_hover_text(PINNED_HINT);
        } else if !open {
            response
                .on_hover_cursor(egui::CursorIcon::PointingHand)
                .on_hover_text(match menu {
                    Menu::Parts => format!(
                        "What a dropped track writes: {}.",
                        current.drop_parts.names().collect::<Vec<_>>().join(", ")
                    ),
                    Menu::Format => format_hint(current.format).to_owned(),
                    Menu::Rate => rate_hint(current.rate).to_owned(),
                });
        }
    }
    width
}

/// One capsule button: label and chevron, a faint hover, the light pill while
/// its menu is open.
#[allow(clippy::too_many_arguments)]
fn cap_button(
    ui: &egui::Ui,
    rect: Rect,
    response: &egui::Response,
    label: &str,
    pad_left: f32,
    open: bool,
    disabled: bool,
    palette: &Palette,
    brushes: &Brushes,
) {
    let painter = ui.painter();
    let hover = ui.ctx().animate_bool_with_time(
        response.id.with("hover"),
        response.hovered() && !disabled,
        0.15,
    );
    let lit = ui
        .ctx()
        .animate_bool_with_time(response.id.with("open"), open, 0.15);
    if hover > 0.0 && lit < 1.0 {
        painter.rect_filled(rect, CAP_HEIGHT / 2.0, white(0.09 * hover * (1.0 - lit)));
    }
    if lit > 0.0 {
        brushes.pill(painter, rect, CAP_HEIGHT / 2.0, lit);
    }
    let alpha = if disabled { 0.45 } else { 1.0 };
    let text_colour = theme::mix(palette.text, palette.ink, lit).gamma_multiply(alpha);
    let weight = if open {
        Weight::Semibold
    } else {
        Weight::Regular
    };
    let galley = painter.layout_no_wrap(label.to_owned(), theme::font(TEXT, weight), text_colour);
    let text_width = galley.size().x;
    let y = rect.center().y - galley.size().y / 2.0;
    painter.galley(pos2(rect.left() + pad_left, y), galley, text_colour);
    let chevron_colour = theme::mix(palette.muted, palette.ink, lit).gamma_multiply(alpha);
    icons::chevron(
        painter,
        icons::square(
            pos2(
                rect.left() + pad_left + text_width + 6.0 + CHEVRON / 2.0,
                rect.center().y,
            ),
            CHEVRON,
        ),
        chevron_colour,
        1.6,
        std::f32::consts::PI * lit,
    );
}

/// Which parts a dropped track writes, as ticks. Requests over HTTP name their
/// own parts and are not affected.
fn parts_menu(
    ui: &mut egui::Ui,
    settings: &SettingsStore,
    current: DropParts,
    state: &AppState,
    palette: &Palette,
) {
    ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
    let split_costs = state.switcher.current() == Some(Preset::Balanced);
    let width = widgets::row_width(236.0);
    for part in PART_NAMES {
        let (rect, response) = menu_row(ui, width, 34.0, ui.id().with(("part", part)));
        let on = current.has(part);
        let painter = ui.painter();
        let tick = Rect::from_min_size(
            pos2(rect.left() + 8.0, rect.center().y - 7.0),
            vec2(14.0, 14.0),
        );
        if on {
            painter.rect_filled(tick, 4.0, palette.band);
            icons::check(painter, tick.shrink(2.5), palette.ink, 2.0);
        } else {
            painter.rect_stroke(
                tick,
                4.0,
                Stroke::new(1.0, white(0.3)),
                egui::StrokeKind::Inside,
            );
        }
        painter.text(
            pos2(tick.right() + 10.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            part,
            theme::font(TEXT, Weight::Regular),
            palette.text,
        );
        let slow = split_costs && matches!(part, "bass" | "other");
        let hint = match part {
            "harmonics" => "bass + other",
            _ if slow => "1.5x time",
            _ => "",
        };
        painter.text(
            pos2(rect.right() - 8.0, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            hint,
            theme::font(11.5, Weight::Regular),
            palette.muted,
        );
        if response
            .on_hover_text(if slow { SPLIT_COST } else { part_hint(part) })
            .clicked()
        {
            settings.set_drop_parts(current.toggle(part));
        }
    }
}

fn format_menu(
    ui: &mut egui::Ui,
    settings: &SettingsStore,
    current: StemFormat,
    palette: &Palette,
) {
    let labels: Vec<_> = StemFormat::ALL
        .iter()
        .map(|f| (f.label().to_owned(), *f == current, format_hint(*f)))
        .collect();
    if let Some(n) = pick_menu(ui, 170.0, &labels, palette) {
        settings.set_format(StemFormat::ALL[n]);
    }
}

/// Only the rates the chosen format can carry: 96 kHz beside MP3 would be a
/// choice that produces a 48 kHz file anyway.
fn rate_menu(
    ui: &mut egui::Ui,
    settings: &SettingsStore,
    format: StemFormat,
    current: OutputRate,
    palette: &Palette,
) {
    let rates: Vec<OutputRate> = format.rates().collect();
    let labels: Vec<_> = rates
        .iter()
        .map(|r| (rate_label(*r), *r == current, rate_hint(*r)))
        .collect();
    if let Some(n) = pick_menu(ui, 140.0, &labels, palette) {
        settings.set_rate(rates[n]);
    }
}

/// Shown on `bass` and `other` while Balanced is loaded.
const SPLIT_COST: &str = "On Balanced, bass and other need one more model: about 1.5x the time.";

fn part_hint(part: &str) -> &'static str {
    match part {
        "harmonics" => "bass and other together; clears bass and other",
        "bass" | "other" => "split out of harmonics; clears harmonics",
        "drums" => "rebuilt as the mix minus the other parts",
        _ => "",
    }
}

/// Hover text for a segment, warning about a download before it starts.
fn hint_for(switcher: &Switcher, preset: Preset) -> String {
    let mut hint = preset.detail().to_owned();
    if !switcher.is_local(preset) {
        hint.push_str(&format!(
            "\n\nNot installed. Choosing this downloads {:.0} MB.",
            preset.total_bytes() as f64 / 1e6
        ));
    }
    hint
}

/// A rate as a person reads it. `Display` gives the number the API takes.
fn rate_label(rate: OutputRate) -> String {
    match rate.hz() {
        44_100 => "44.1 kHz".to_owned(),
        hz => format!("{} kHz", hz / 1000),
    }
}

fn rate_hint(rate: OutputRate) -> &'static str {
    match rate {
        OutputRate::Hz24000 => "Half the bandwidth, half the size.",
        OutputRate::Hz44100 => "Model's native rate, no conversion.",
        OutputRate::Hz48000 => "Converted from 44.1 kHz.",
        OutputRate::Hz96000 => "Converted from 44.1 kHz. Twice the size with no quality gain.",
    }
}

fn format_hint(format: StemFormat) -> &'static str {
    match format {
        StemFormat::Flac => "Lossless, about half the size of a WAV, and opens everywhere.",
        StemFormat::Wav => "The same 16-bit samples as FLAC, uncompressed. For anything fussy.",
        StemFormat::Mp3 => "320 kbps lossy. The outputs cannot be summed back.",
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

    #[test]
    fn a_rate_reads_as_kilohertz_including_the_awkward_one() {
        assert_eq!(rate_label(OutputRate::Hz44100), "44.1 kHz");
        assert_eq!(rate_label(OutputRate::Hz48000), "48 kHz");
        assert_eq!(rate_label(OutputRate::Hz24000), "24 kHz");
        assert_eq!(rate_label(OutputRate::Hz96000), "96 kHz");
    }
}
