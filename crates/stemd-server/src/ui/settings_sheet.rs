//! The settings sheet: the longest track accepted, the port, and the cache.
//!
//! The sheet opens from the status bar over the dimmed window. A change applies at once and is
//! saved (see [`crate::settings`]). A value that a flag fixed for this run is greyed out.

use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui::{self, Align2, Color32, Rect, Sense, Stroke, pos2, vec2};
use parking_lot::Mutex;

use crate::state::AppState;

use super::blur::{self, Frosted};
use super::header::PINNED_HINT;
use super::icons;
use super::theme::{self, Brushes, Palette, Weight, white};
use super::widgets::{self, Pill};

/// The sizes the menu offers, in gigabytes.
const SIZES: [f64; 6] = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

/// The ages the menu offers, in days; 0 keeps whole tracks until the cache is full.
const AGES: [f64; 4] = [7.0, 30.0, 90.0, 0.0];

const WIDTH: f32 = 452.0;
/// The gap between the window's top and the sheet.
const TOP: f32 = 40.0;
const MARGIN: i8 = 18;
const ROW: f32 = 56.0;
const CONTROL: egui::Vec2 = vec2(120.0, 28.0);
const MENU_WIDTH: f32 = 150.0;
/// The height of the cache's usage block: two lines, a gap and the meter.
const USAGE: f32 = 76.0;

/// Seconds the port's confirmation stays up.
const MOVED_SECS: f64 = 2.6;
/// How long the note on what a lower limit deleted stays up.
const NOTE_FOR: Duration = Duration::from_secs(4);

/// Colours the sheet uses in addition to the palette.
const FOCUS: Color32 = Color32::from_rgb(187, 206, 247);
const FOCUS_RING: Color32 = Color32::from_rgba_unmultiplied_const(91, 132, 216, 72);
const BAD_RING: Color32 = Color32::from_rgba_unmultiplied_const(212, 104, 95, 64);
const FAINT: Color32 = Color32::from_rgb(0x8A, 0x92, 0xA6);

#[derive(Default)]
pub struct Sheet {
    open: bool,
    /// The longest track as typed, in minutes; empty for any length.
    minutes: String,
    /// The port as typed.
    port: String,
    /// Why the server could not move to the typed port.
    port_failed: Option<String>,
    /// When the server last moved, for the confirmation under the port.
    moved_at: Option<f64>,
    /// What the last lowered limit deleted, and when. The thread that applies the limit sets it.
    note: Arc<Mutex<Option<(String, Instant)>>>,
}

impl Sheet {
    pub const fn is_open(&self) -> bool {
        self.open
    }

    /// Open the sheet with fields showing the values this run uses, or close it.
    pub fn toggle(&mut self, state: &AppState) {
        self.open = !self.open;
        if self.open {
            self.minutes = minutes_text(state.settings.get().max_track_minutes);
            self.port = state.endpoint.port().to_string();
            self.port_failed = None;
            self.moved_at = None;
        }
    }
}

/// How a text field's edit ended this frame.
#[derive(PartialEq, Eq)]
enum Edit {
    None,
    Changed,
    Commit,
    Revert,
}

/// `10`, or empty for 0, which accepts any length.
fn minutes_text(minutes: f64) -> String {
    match minutes {
        m if m <= 0.0 => String::new(),
        m if m.fract() == 0.0 => format!("{m:.0}"),
        m => m.to_string(),
    }
}

/// Why `text` is not a valid length of up to `longest` minutes; `None` when it is. An empty text
/// means `longest`.
fn minutes_error(text: &str, longest: u32) -> Option<String> {
    if text.is_empty() {
        return None;
    }
    if !text.bytes().all(|b| b.is_ascii_digit()) {
        return Some("Whole minutes only.".to_owned());
    }
    match text.parse::<u32>() {
        Ok(n) if (1..=longest).contains(&n) => None,
        _ => Some(format!("Between 1 and {longest} on this machine.")),
    }
}

fn port_error(text: &str) -> Option<&'static str> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Some("Numbers only.");
    }
    match text.parse::<u32>() {
        Ok(n) if n < 1024 => Some("Pick 1024 or higher."),
        Ok(n) if n <= 65535 => None,
        _ => Some("Ports stop at 65535."),
    }
}

/// `8 GB`, or `1.5 GB` for a size set by hand.
fn size_label(gb: f64) -> String {
    if gb.fract() == 0.0 {
        format!("{gb:.0} GB")
    } else {
        format!("{gb} GB")
    }
}

/// `30 days`, `1 day`, or `Until full` for 0.
fn age_label(days: f64) -> String {
    match days {
        0.0 => "Until full".to_owned(),
        1.0 => "1 day".to_owned(),
        d if d.fract() == 0.0 => format!("{d:.0} days"),
        d => format!("{d} days"),
    }
}

/// `410 MB`, `5.2 GB`, `16 GB`.
fn bytes_label(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        let gb = format!("{:.1}", bytes as f64 / 1e9);
        format!("{} GB", gb.strip_suffix(".0").unwrap_or(&gb))
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

/// `12 tracks · 410 MB`, or `Empty`.
fn usage_label(tracks: usize, bytes: u64) -> String {
    match tracks {
        0 => "Empty".to_owned(),
        1 => format!("1 track{}{}", theme::DOT, bytes_label(bytes)),
        n => format!("{n} tracks{}{}", theme::DOT, bytes_label(bytes)),
    }
}

/// The sheet's panel.
fn look() -> Frosted {
    Frosted {
        radius: 20.0,
        margin: MARGIN,
        fill: Color32::from_rgba_unmultiplied_const(26, 33, 62, 209),
        outline: white(0.14),
    }
}

/// Draw the sheet while it is open or fading out.
pub fn show(
    sheet: &mut Sheet,
    state: &AppState,
    ctx: &egui::Context,
    palette: &Palette,
    brushes: &Brushes,
) {
    let id = egui::Id::new("stemd-settings");
    let t = ctx.animate_bool_with_time(id.with("open"), sheet.open, 0.2);
    if t <= 0.0 {
        return;
    }
    let eased = theme::ease_out(t);
    let screen = ctx.content_rect();
    let width = WIDTH.min(screen.width() - 32.0);
    let now = ctx.input(|i| i.time);
    let area = egui::Modal::default_area(id)
        .anchor(Align2::CENTER_TOP, vec2(0.0, TOP - 8.0 * (1.0 - eased)));
    let mut close = false;
    let shown = egui::Modal::new(id)
        .area(area)
        .backdrop_color(Color32::TRANSPARENT)
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            ui.multiply_opacity(eased);
            ui.painter().add(blur::shape(screen, 0.0));
            ui.painter()
                .rect_filled(screen, 0.0, Color32::from_rgba_unmultiplied(6, 9, 20, 140));
            blur::frosted(ui, &look(), |ui| {
                let inner = width - 2.0 * f32::from(MARGIN);
                ui.set_width(inner);
                close = contents(ui, sheet, state, palette, brushes, inner, now);
            });
        });
    if sheet.open && (close || shown.should_close()) {
        sheet.open = false;
    }
}

/// The sheet's contents. Returns true when its close button was clicked.
fn contents(
    ui: &mut egui::Ui,
    sheet: &mut Sheet,
    state: &AppState,
    palette: &Palette,
    brushes: &Brushes,
    width: f32,
    now: f64,
) -> bool {
    ui.spacing_mut().item_spacing = vec2(0.0, 8.0);
    let close = title(ui, palette, width);
    ui.add_space(4.0);
    label(ui, palette, width, "General");
    let general_card = card(ui, width, 2.0 * ROW);
    general(ui, sheet, state, palette, general_card, now);
    ui.add_space(8.0);
    label(ui, palette, width, "Cache");
    let cache_card = card(ui, width, USAGE + 2.0 * ROW);
    cache(ui, sheet, state, palette, brushes, cache_card);
    ui.add_space(6.0);
    let (rect, _) = ui.allocate_exact_size(vec2(width, 16.0), Sense::hover());
    ui.painter().text(
        rect.center(),
        Align2::CENTER_CENTER,
        "Changes are saved as you make them.",
        theme::font(11.5, Weight::Regular),
        FAINT,
    );
    close
}

/// "Settings" and the close button. Returns true when it was clicked.
fn title(ui: &mut egui::Ui, palette: &Palette, width: f32) -> bool {
    let (rect, _) = ui.allocate_exact_size(vec2(width, 28.0), Sense::hover());
    ui.painter().text(
        pos2(rect.left(), rect.center().y),
        Align2::LEFT_CENTER,
        "Settings",
        theme::font(17.0, Weight::Semibold),
        palette.text,
    );
    let button = icons::square(pos2(rect.right() + 6.0 - 14.0, rect.center().y), 28.0);
    let id = ui.id().with("close");
    let response = ui
        .interact(button, id, Sense::click())
        .on_hover_text("Close (Esc)")
        .on_hover_cursor(egui::CursorIcon::PointingHand);
    let hover = widgets::fade(ui, id, response.hovered());
    if hover > 0.0 {
        ui.painter().rect_filled(button, 9.0, white(0.1 * hover));
    }
    icons::cross(
        ui.painter(),
        icons::square(button.center(), 10.0),
        theme::mix(palette.muted, Color32::WHITE, hover),
        1.5,
    );
    response.clicked()
}

/// A section's muted name above its card.
fn label(ui: &mut egui::Ui, palette: &Palette, width: f32, text: &str) {
    let (rect, _) = ui.allocate_exact_size(vec2(width, 16.0), Sense::hover());
    ui.painter().text(
        pos2(rect.left() + 4.0, rect.center().y),
        Align2::LEFT_CENTER,
        text,
        theme::font(12.0, Weight::Regular),
        palette.muted,
    );
}

/// A rounded card `height` tall. Returns its rect.
fn card(ui: &mut egui::Ui, width: f32, height: f32) -> Rect {
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 14.0, white(0.05));
    painter.rect_stroke(
        rect,
        14.0,
        Stroke::new(1.0, white(0.10)),
        egui::StrokeKind::Inside,
    );
    painter.add(theme::top_highlight(rect, 14.0, white(0.10)));
    rect
}

/// Row `n` of a card, `from` below its top, with a hairline above all but the
/// first.
fn row_rect(ui: &egui::Ui, card: Rect, from: f32, n: usize) -> Rect {
    let top = card.top() + from + ROW * n as f32;
    let rect = Rect::from_min_size(pos2(card.left(), top), vec2(card.width(), ROW));
    if from > 0.0 || n > 0 {
        ui.painter()
            .hline(card.x_range(), top, Stroke::new(1.0, white(0.08)));
    }
    rect
}

/// The longest track and the port.
fn general(
    ui: &mut egui::Ui,
    sheet: &mut Sheet,
    state: &AppState,
    palette: &Palette,
    card: Rect,
    now: f64,
) {
    let pinned = state.settings.pinned();
    let current = state.settings.get();

    let longest = crate::disk::longest_track_minutes();
    let error = minutes_error(&sheet.minutes, longest);
    let longest_text = format!("Up to {longest} min on this machine.");
    let longest_label = longest.to_string();
    let hint = if pinned.max_track_minutes {
        ("Set by --max-track-minutes for this run.", palette.muted)
    } else if let Some(error) = &error {
        (error.as_str(), palette.danger)
    } else if sheet.minutes.is_empty() {
        (longest_text.as_str(), palette.muted)
    } else {
        ("Longer tracks are refused.", palette.muted)
    };
    let edit = text_row(
        ui,
        palette,
        row_rect(ui, card, 0.0, 0),
        &Field {
            name: "Max track length",
            hint: hint.0,
            hint_colour: hint.1,
            check: false,
            placeholder: &longest_label,
            suffix: Some("min"),
            pinned: pinned.max_track_minutes,
            bad: error.is_some(),
        },
        &mut sheet.minutes,
    );
    match edit {
        Edit::Commit if error.is_none() => {
            let minutes = sheet.minutes.parse::<f64>().unwrap_or(0.0);
            if minutes != current.max_track_minutes {
                state.settings.set_max_track_minutes(minutes);
                tracing::info!(
                    "accepting tracks up to {:.0} minutes",
                    state.settings.get().max_track_secs(longest) / 60.0
                );
            }
        }
        Edit::Revert => sheet.minutes = minutes_text(current.max_track_minutes),
        _ => {}
    }

    let port = state.endpoint.port();
    if pinned.port {
        sheet.port = port.to_string();
    }
    let typed = port_error(&sheet.port);
    let moved = sheet
        .moved_at
        .filter(|at| now - at < MOVED_SECS && sheet.port == port.to_string());
    if let Some(at) = moved {
        ui.ctx()
            .request_repaint_after(Duration::from_secs_f64(MOVED_SECS - (now - at)));
    }
    let moved_text = format!("Server restarted on {port}.");
    let taken_text;
    let (hint, colour) = if pinned.port {
        ("Set by --bind for this run.", palette.muted)
    } else if let Some(error) = typed {
        (error, palette.danger)
    } else if let Some(failed) = &sheet.port_failed {
        (failed.as_str(), palette.danger)
    } else if moved.is_some() {
        (moved_text.as_str(), palette.good)
    } else if let Some(taken) = state.endpoint.taken()
        && sheet.port == port.to_string()
    {
        taken_text = format!("Port {taken} was busy at launch; serving on {port}.");
        (taken_text.as_str(), palette.danger)
    } else if sheet.port != port.to_string() {
        ("Press Return to restart the server on it.", palette.muted)
    } else {
        ("Players connect here.", palette.muted)
    };
    let field = Field {
        name: "Server port",
        hint,
        hint_colour: colour,
        check: moved.is_some(),
        placeholder: "8420",
        suffix: None,
        pinned: pinned.port,
        bad: !pinned.port && (typed.is_some() || sheet.port_failed.is_some()),
    };
    match text_row(
        ui,
        palette,
        row_rect(ui, card, 0.0, 1),
        &field,
        &mut sheet.port,
    ) {
        Edit::Changed => sheet.port_failed = None,
        Edit::Commit if typed.is_none() => {
            if let Ok(wanted) = sheet.port.parse::<u16>()
                && wanted != port
            {
                match state.move_to_port(wanted) {
                    Ok(()) => sheet.moved_at = Some(now),
                    Err(err) => sheet.port_failed = Some(err.to_string()),
                }
            }
        }
        Edit::Revert => {
            sheet.port = port.to_string();
            sheet.port_failed = None;
        }
        _ => {}
    }
}

/// The cache section: what the cache holds, the button that empties it, and its two limits.
fn cache(
    ui: &mut egui::Ui,
    sheet: &Sheet,
    state: &AppState,
    palette: &Palette,
    brushes: &Brushes,
    card: Rect,
) {
    let current = state.settings.get();
    let pinned = state.settings.pinned();
    let stats = state.cache.stats();
    let painter = ui.painter().clone();
    let left = card.left() + 14.0;
    let top = card.top() + 12.0;

    let mut usage = egui::text::LayoutJob::default();
    usage.append(
        &usage_label(stats.tracks, stats.bytes),
        0.0,
        egui::TextFormat::simple(theme::font(13.0, Weight::Medium), palette.text),
    );
    usage.append(
        &format!("of {}", bytes_label(stats.max_bytes)),
        5.0,
        egui::TextFormat::simple(theme::font(13.0, Weight::Regular), palette.muted),
    );
    let usage = painter.layout_job(usage);
    painter.galley(pos2(left, top), usage, palette.text);

    let used = if stats.max_bytes == 0 {
        0.0
    } else {
        (stats.bytes as f64 / stats.max_bytes as f64).min(1.0) as f32
    };
    let note = sheet
        .note
        .lock()
        .clone()
        .filter(|(_, at)| at.elapsed() < NOTE_FOR);
    if let Some((_, at)) = &note {
        ui.ctx()
            .request_repaint_after(NOTE_FOR.saturating_sub(at.elapsed()));
    }
    let (note, colour) = match note {
        Some((text, _)) => (text, palette.good),
        None if stats.tracks == 0 => (
            "Separated tracks are kept here for players to reload.".to_owned(),
            palette.muted,
        ),
        None => (format!("{:.0}% used", used * 100.0), palette.muted),
    };
    painter.text(
        pos2(left, top + 26.0),
        Align2::LEFT_CENTER,
        note,
        theme::font(11.5, Weight::Regular),
        colour,
    );

    let label = "Empty cache";
    let pill_width = widgets::pill_width(ui, label, CONTROL.y);
    let button = Rect::from_min_size(
        pos2(
            card.right() - 12.0 - pill_width,
            top + 17.0 - CONTROL.y / 2.0,
        ),
        vec2(pill_width, CONTROL.y),
    );
    // No confirmation: every byte can be recomputed from audio that still exists.
    if widgets::pill(
        ui,
        button,
        ui.id().with("empty-cache"),
        label,
        Pill::Danger,
        stats.tracks > 0,
        palette,
        brushes,
    )
    .on_hover_text(
        "Deletes every separated track no session is playing. Each one can be separated again.",
    )
    .clicked()
    {
        state.empty_cache();
    }

    let track = Rect::from_min_size(
        pos2(left, top + 34.0 + 10.0),
        vec2(card.right() - 12.0 - left, 6.0),
    );
    painter.rect_filled(track, 3.0, theme::black(0.3));
    let shown = ui
        .ctx()
        .animate_value_with_time(ui.id().with("used"), used, 0.4);
    let fill = Rect::from_min_size(track.min, vec2(track.width() * shown, track.height()));
    brushes.spectrum(&painter, track, fill, 3.0, Color32::WHITE);
    if fill.width() > 0.0 {
        brushes.gloss(&painter, fill, 3.0);
    }

    let picked = menu_row(
        ui,
        palette,
        row_rect(ui, card, USAGE, 0),
        "Size limit",
        "When full, tracks are deleted, oldest last used first.",
        &size_label(current.cache_max_gb),
        pinned.cache_max_gb,
        &SIZES.map(|gb| (size_label(gb), gb == current.cache_max_gb)),
        egui::RectAlign::BOTTOM_END,
    );
    if let Some(n) = picked {
        state.settings.set_cache_max_gb(SIZES[n]);
        apply(state, &sheet.note);
    }

    let picked = menu_row(
        ui,
        palette,
        row_rect(ui, card, USAGE, 1),
        "Keep unused stems",
        "How long an unused track is kept.",
        &age_label(current.whole_max_days),
        pinned.whole_max_days,
        &AGES.map(|days| (age_label(days), days == current.whole_max_days)),
        egui::RectAlign::TOP_END,
    );
    if let Some(n) = picked {
        state.settings.set_whole_max_days(AGES[n]);
        apply(state, &sheet.note);
    }
}

/// Apply the saved size and age to the cache off the window's thread, since a lower limit deletes
/// tracks. The result goes in `note`.
fn apply(state: &AppState, note: &Arc<Mutex<Option<(String, Instant)>>>) {
    let note = Arc::clone(note);
    state.apply_cache_limits(move |tracks, freed| {
        if tracks > 0 {
            let noun = if tracks == 1 { "track" } else { "tracks" };
            *note.lock() = Some((
                format!(
                    "Freed {}: deleted {tracks} {noun}, oldest last used first.",
                    bytes_label(freed)
                ),
                Instant::now(),
            ));
        }
    });
}

/// A setting's name and the line under it, at the left of `row`.
fn name_and_hint(
    painter: &egui::Painter,
    palette: &Palette,
    row: Rect,
    right: f32,
    name: &str,
    (hint, hint_colour, check): (&str, Color32, bool),
) {
    let left = row.left() + 14.0;
    let y = row.center().y;
    painter.text(
        pos2(left, y - 8.5),
        Align2::LEFT_CENTER,
        name,
        theme::font(13.0, Weight::Medium),
        palette.text,
    );
    let mut x = left;
    if check {
        icons::check(
            painter,
            icons::square(pos2(left + 5.5, y + 9.0), 11.0),
            palette.good,
            1.8,
        );
        x += 16.0;
    }
    let mut job = egui::text::LayoutJob::single_section(
        hint.to_owned(),
        egui::TextFormat::simple(theme::font(11.5, Weight::Regular), hint_colour),
    );
    job.wrap = egui::text::TextWrapping::truncate_at_width((right - 16.0 - x).max(0.0));
    let galley = painter.layout_job(job);
    painter.galley(
        pos2(x, y + 9.0 - galley.size().y / 2.0),
        galley,
        hint_colour,
    );
}

/// The control's rect at the right of `row`.
fn control_rect(row: Rect) -> Rect {
    Rect::from_min_size(
        pos2(
            row.right() - 12.0 - CONTROL.x,
            row.center().y - CONTROL.y / 2.0,
        ),
        CONTROL,
    )
}

/// What a text row shows.
struct Field<'a> {
    name: &'a str,
    hint: &'a str,
    hint_colour: Color32,
    /// A tick before the hint, for a change that went through.
    check: bool,
    placeholder: &'a str,
    /// A unit at the right of the field.
    suffix: Option<&'a str>,
    pinned: bool,
    bad: bool,
}

/// A setting typed into a field at the right of `row`. Return or leaving the
/// field commits; Escape reverts.
fn text_row(
    ui: &mut egui::Ui,
    palette: &Palette,
    row: Rect,
    field: &Field<'_>,
    text: &mut String,
) -> Edit {
    let mut painter = ui.painter().clone();
    if field.pinned {
        painter.multiply_opacity(0.45);
    }
    let control = control_rect(row);
    name_and_hint(
        &painter,
        palette,
        row,
        control.left(),
        field.name,
        (field.hint, field.hint_colour, field.check),
    );
    let behind = painter.add(egui::Shape::Noop);

    let id = ui.id().with(("field", field.name));
    let around = ui.interact(control, id.with("around"), Sense::click());
    let font = theme::font(13.0, Weight::Medium);
    let suffix = field.suffix.map(|suffix| {
        painter.layout_no_wrap(
            suffix.to_owned(),
            theme::font(12.0, Weight::Regular),
            palette.muted,
        )
    });
    let suffix_width = suffix.as_ref().map_or(0.0, |s| s.size().x + 6.0);
    let inside = Rect::from_min_max(
        pos2(control.left() + 12.0, control.top()),
        pos2(control.right() - 12.0 - suffix_width, control.bottom()),
    );
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(inside)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    if field.pinned {
        child.multiply_opacity(0.45);
    }
    let response = child.add(
        egui::TextEdit::singleline(text)
            .id(id)
            .frame(egui::Frame::NONE)
            .margin(egui::Margin::ZERO)
            .font(font)
            .text_color(palette.text)
            .hint_text(egui::RichText::new(field.placeholder).color(FAINT))
            .horizontal_align(egui::Align::RIGHT)
            .desired_width(inside.width())
            .char_limit(5)
            .interactive(!field.pinned),
    );
    if around.clicked() && !field.pinned {
        response.request_focus();
    }
    let focused = response.has_focus();

    let r = CONTROL.y / 2.0;
    let mut shapes = vec![egui::Shape::rect_filled(
        control,
        r,
        white(if focused { 0.09 } else { 0.06 }),
    )];
    let (edge, ring) = if field.bad {
        (palette.danger, Some(BAD_RING))
    } else if focused {
        (FOCUS.gamma_multiply(0.8), Some(FOCUS_RING))
    } else {
        (white(0.12), None)
    };
    if let Some(ring) = ring {
        shapes.push(egui::Shape::Rect(egui::epaint::RectShape::stroke(
            control.expand(2.0),
            r + 2.0,
            Stroke::new(3.0, ring),
            egui::StrokeKind::Middle,
        )));
    }
    shapes.push(egui::Shape::Rect(egui::epaint::RectShape::stroke(
        control,
        r,
        Stroke::new(1.0, edge),
        egui::StrokeKind::Inside,
    )));
    painter.set(behind, egui::Shape::Vec(shapes));
    if let Some(suffix) = suffix {
        let at = pos2(
            control.right() - 12.0 - suffix.size().x,
            control.center().y - suffix.size().y / 2.0,
        );
        painter.galley(at, suffix, palette.muted);
    }

    if field.pinned {
        ui.interact(row, id.with("pinned"), Sense::hover())
            .on_hover_text(PINNED_HINT);
        return Edit::None;
    }
    if !focused {
        around.on_hover_cursor(egui::CursorIcon::Text);
    }
    if response.lost_focus() {
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            Edit::Revert
        } else {
            Edit::Commit
        }
    } else if response.changed() {
        Edit::Changed
    } else {
        Edit::None
    }
}

/// A setting picked from a menu at the right of `row`. Returns the choice picked.
#[allow(clippy::too_many_arguments)]
fn menu_row(
    ui: &mut egui::Ui,
    palette: &Palette,
    row: Rect,
    name: &str,
    hint: &str,
    value: &str,
    pinned: bool,
    choices: &[(String, bool)],
    align: egui::RectAlign,
) -> Option<usize> {
    let mut painter = ui.painter().clone();
    if pinned {
        painter.multiply_opacity(0.45);
    }
    let button = control_rect(row);
    name_and_hint(
        &painter,
        palette,
        row,
        button.left(),
        name,
        (hint, palette.muted, false),
    );
    let id = ui.id().with(("setting", name));
    let response = ui.interact(
        button,
        id,
        if pinned {
            Sense::hover()
        } else {
            Sense::click()
        },
    );
    let mut picked = None;
    let open = widgets::menu(
        &response,
        align,
        egui::PopupCloseBehavior::CloseOnClick,
        palette,
        |ui| {
            let labels: Vec<_> = choices
                .iter()
                .map(|(label, on)| (label.clone(), *on, ""))
                .collect();
            picked = widgets::pick_menu(ui, MENU_WIDTH, &labels, palette);
        },
    );
    let hover = widgets::fade(ui, id, !pinned && (response.hovered() || open));
    let r = CONTROL.y / 2.0;
    painter.rect_filled(button, r, white(0.06 + 0.06 * hover));
    painter.rect_stroke(
        button,
        r,
        Stroke::new(1.0, white(0.12)),
        egui::StrokeKind::Inside,
    );
    painter.text(
        pos2(button.left() + 12.0, button.center().y),
        Align2::LEFT_CENTER,
        value,
        theme::font(13.0, Weight::Medium),
        palette.text,
    );
    icons::chevron(
        &painter,
        icons::square(pos2(button.right() - 10.0 - 4.0, button.center().y), 8.0),
        palette.muted,
        1.6,
        if open { std::f32::consts::PI } else { 0.0 },
    );
    if pinned {
        ui.interact(row, id.with("pinned"), Sense::hover())
            .on_hover_text(PINNED_HINT);
    } else {
        response.on_hover_cursor(egui::CursorIcon::PointingHand);
    }
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cache_reads_in_tracks_and_sizes() {
        assert_eq!(size_label(8.0), "8 GB");
        assert_eq!(size_label(1.5), "1.5 GB");
        assert_eq!(age_label(30.0), "30 days");
        assert_eq!(age_label(1.0), "1 day");
        assert_eq!(age_label(0.0), "Until full");
        assert_eq!(usage_label(0, 0), "Empty");
        assert_eq!(usage_label(12, 410_000_000), "12 tracks · 410 MB");
        assert_eq!(usage_label(1, 5_240_000_000), "1 track · 5.2 GB");
        assert_eq!(bytes_label(16_000_000_000), "16 GB");
    }

    #[test]
    fn a_typed_length_or_port_is_checked_before_it_applies() {
        assert_eq!(minutes_error("", 71), None, "empty takes the longest");
        assert_eq!(minutes_error("20", 71), None);
        assert_eq!(
            minutes_error("7.5", 71).as_deref(),
            Some("Whole minutes only.")
        );
        assert!(minutes_error("0", 71).is_some() && minutes_error("72", 71).is_some());
        assert_eq!(minutes_text(0.0), "");
        assert_eq!(minutes_text(10.0), "10");

        assert_eq!(port_error("8421"), None);
        assert_eq!(port_error(""), Some("Numbers only."));
        assert_eq!(port_error("80"), Some("Pick 1024 or higher."));
        assert_eq!(port_error("70000"), Some("Ports stop at 65535."));
    }
}
