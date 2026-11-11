//! The log section.
//!
//! A thin client over the same [`LogBuffer`](crate::logbuf::LogBuffer) that
//! `GET /v1/logs` serves, so the window is a second view, never a second logging
//! path. Collapsed by default: it is where you go when something looks wrong,
//! not what the window is for.

use eframe::egui::{self, Color32, Sense, Stroke, pos2, vec2};

use crate::logbuf::LogLine;

use super::icons;
use super::theme::{self, Brushes, Palette, Weight, white};
use super::widgets::{self, Pill};

/// How tall the log area gets before it scrolls.
pub const VISIBLE_HEIGHT: f32 = 168.0;

/// Log rows are set tight: this is a tail you scan, and the useful thing is how
/// many lines fit at once.
const ROW: f32 = 8.0;
const CONTROL_TEXT: f32 = 12.0;
const CONTROL_HEIGHT: f32 = 26.0;
/// Most recent matching lines drawn. Older ones stay in the buffer and the API.
const SHOWN: usize = 400;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelFilter {
    All,
    Info,
    Warn,
}

impl LevelFilter {
    const ALL: [Self; 3] = [Self::All, Self::Info, Self::Warn];

    pub const fn label(self) -> &'static str {
        match self {
            Self::All => "everything",
            Self::Info => "info and up",
            Self::Warn => "warnings only",
        }
    }

    fn admits(self, level: &str) -> bool {
        match self {
            Self::All => true,
            Self::Info => !matches!(level, "DEBUG" | "TRACE"),
            Self::Warn => matches!(level, "WARN" | "ERROR"),
        }
    }
}

/// What the window remembers about how the log is being read.
pub struct View {
    pub min_level: LevelFilter,
    pub filter: String,
    /// Layout pass the pane was last drawn in, to tell when it has just opened.
    drawn_at: u64,
}

impl Default for View {
    fn default() -> Self {
        Self {
            min_level: LevelFilter::Info,
            filter: String::new(),
            drawn_at: 0,
        }
    }
}

/// Draw the log section. Returns true if the buffer should be emptied: the
/// caller owns it, so the button reports rather than acts.
#[must_use]
pub fn show(
    ui: &mut egui::Ui,
    palette: &Palette,
    brushes: &Brushes,
    view: &mut View,
    lines: &[LogLine],
) -> bool {
    let pass = ui.ctx().cumulative_pass_nr();
    let opened = view.drawn_at + 1 != pass;
    view.drawn_at = pass;

    let mut clear = false;
    ui.horizontal(|ui| {
        level_menu(ui, palette, view);
        ui.add(
            egui::TextEdit::singleline(&mut view.filter)
                .hint_text("filter")
                .font(theme::font(CONTROL_TEXT, Weight::Regular))
                .margin(vec2(10.0, 5.0))
                .desired_width(ui.available_width() - 70.0),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (rect, _) = ui.allocate_exact_size(vec2(56.0, CONTROL_HEIGHT), Sense::hover());
            clear = widgets::pill(
                ui,
                rect,
                ui.id().with("clear-logs"),
                "Clear",
                Pill::Ghost,
                !lines.is_empty(),
                palette,
                brushes,
            )
            .on_hover_text("Clears the logs")
            .clicked();
        });
    });
    ui.add_space(4.0);

    let visible = visible_lines(view, lines);
    let font = egui::FontId::monospace(ROW);
    egui::ScrollArea::vertical()
        // Follows new lines while scrolled to the bottom, and stops while
        // scrolled up.
        .stick_to_bottom(true)
        .max_height(VISIBLE_HEIGHT)
        // Shrinks to the lines it has, up to the cap.
        .auto_shrink([false, true])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = vec2(6.0, 0.0);
            ui.spacing_mut().interact_size.y = 0.0;
            let skip = visible.len().saturating_sub(SHOWN);
            for line in &visible[skip..] {
                ui.horizontal_top(|ui| {
                    ui.label(
                        egui::RichText::new(format!("{:<5}", line.level))
                            .font(font.clone())
                            .color(level_colour(palette, &line.level)),
                    );
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&line.message)
                                .font(font.clone())
                                .color(palette.text),
                        )
                        .wrap(),
                    );
                });
            }
            if opened {
                ui.scroll_to_cursor(Some(egui::Align::BOTTOM));
            }
        });

    clear
}

/// The level floor as a pill button with a frosted menu.
fn level_menu(ui: &mut egui::Ui, palette: &Palette, view: &mut View) {
    let font = theme::font(CONTROL_TEXT, Weight::Medium);
    let label = view.min_level.label();
    let text = ui
        .painter()
        .layout_no_wrap(label.to_owned(), font.clone(), palette.text)
        .size()
        .x;
    let (rect, response) = ui.allocate_exact_size(
        vec2(12.0 + text + 6.0 + 8.0 + 10.0, CONTROL_HEIGHT),
        Sense::click(),
    );
    let open = widgets::menu(
        &response,
        egui::RectAlign::BOTTOM_START,
        egui::PopupCloseBehavior::CloseOnClick,
        palette,
        |ui| {
            let labels: Vec<_> = LevelFilter::ALL
                .iter()
                .map(|l| (l.label().to_owned(), *l == view.min_level, ""))
                .collect();
            if let Some(n) = widgets::pick_menu(ui, 150.0, &labels, palette) {
                view.min_level = LevelFilter::ALL[n];
            }
        },
    );
    let hover = widgets::fade(ui, response.id, response.hovered() || open);
    let painter = ui.painter();
    let r = CONTROL_HEIGHT / 2.0;
    painter.rect_filled(rect, r, white(0.06 + 0.06 * hover));
    painter.rect_stroke(
        rect,
        r,
        Stroke::new(1.0, white(0.12)),
        egui::StrokeKind::Inside,
    );
    painter.text(
        pos2(rect.left() + 12.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        font,
        palette.text,
    );
    icons::chevron(
        painter,
        icons::square(pos2(rect.right() - 10.0 - 4.0, rect.center().y), 8.0),
        palette.muted,
        1.6,
        if open { std::f32::consts::PI } else { 0.0 },
    );
    response.on_hover_cursor(egui::CursorIcon::PointingHand);
}

/// Buffered lines passing the level and text filters.
fn visible_lines(view: &View, lines: &[LogLine]) -> Vec<LogLine> {
    let needle = view.filter.to_lowercase();
    lines
        .iter()
        .filter(|l| view.min_level.admits(&l.level))
        .filter(|l| {
            needle.is_empty()
                || l.message.to_lowercase().contains(&needle)
                || l.target.to_lowercase().contains(&needle)
        })
        .cloned()
        .collect()
}

fn level_colour(palette: &Palette, level: &str) -> Color32 {
    match level {
        "ERROR" => Color32::from_rgb(0xFF, 0x45, 0x3A),
        "WARN" => Color32::from_rgb(0xFF, 0x9F, 0x0A),
        "INFO" => Color32::from_rgb(0x0A, 0x84, 0xFF),
        _ => palette.muted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_level_filter_is_a_floor_not_an_exact_match() {
        assert!(LevelFilter::All.admits("TRACE"));
        assert!(LevelFilter::Info.admits("INFO") && LevelFilter::Info.admits("ERROR"));
        assert!(!LevelFilter::Info.admits("DEBUG"));
        assert!(LevelFilter::Warn.admits("WARN") && !LevelFilter::Warn.admits("INFO"));
    }

    fn line(level: &str, message: &str) -> LogLine {
        LogLine {
            at_ms: 0,
            level: level.to_owned(),
            target: "stemd_server::api".to_owned(),
            message: message.to_owned(),
        }
    }

    #[test]
    fn the_text_filter_reads_the_target_as_well_as_the_message() {
        let lines = [line("INFO", "queued a track"), line("INFO", "reaping")];
        let seeking = |needle: &str| {
            let view = View {
                filter: needle.to_owned(),
                ..Default::default()
            };
            visible_lines(&view, &lines).len()
        };

        assert_eq!(seeking("queued"), 1);
        // Both lines share a target, so filtering on it keeps both.
        assert_eq!(seeking("api"), 2);
        assert_eq!(seeking("nothing matches this"), 0);
    }

    /// The two filters have to compose, or narrowing by level would be undone
    /// by a text filter that matches more.
    #[test]
    fn the_level_and_text_filters_both_apply() {
        let lines = [
            line("DEBUG", "queued a track"),
            line("WARN", "queued a track"),
        ];
        let view = View {
            min_level: LevelFilter::Warn,
            filter: "queued".to_owned(),
            ..Default::default()
        };
        assert_eq!(visible_lines(&view, &lines).len(), 1);
    }
}
