//! The log section.
//!
//! A thin client over the same [`LogBuffer`](crate::logbuf::LogBuffer) that
//! `GET /v1/logs` serves, so the window is a second view, never a second logging
//! path. Collapsed by default: it is where you go when something looks wrong,
//! not what the window is for.

use eframe::egui;

use crate::logbuf::LogLine;

use super::header::MENU_CAP;
use super::theme::Palette;

/// How tall the log area gets before it scrolls.
pub const VISIBLE_HEIGHT: f32 = 168.0;

/// Log rows are set much tighter than the rest of the window on purpose. This is
/// a tail you scan rather than read, and the useful thing is how many lines fit
/// at once; the window's ordinary spacing turns twenty lines into six.
const ROW: f32 = 8.0;
const ROW_SPACING: f32 = 0.0;
const CONTROL_TEXT: f32 = 11.0;

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
    pub follow: bool,
    pub min_level: LevelFilter,
    pub filter: String,
}

impl Default for View {
    fn default() -> Self {
        Self {
            follow: true,
            min_level: LevelFilter::Info,
            filter: String::new(),
        }
    }
}

/// Draw the log section. Returns true if the buffer should be emptied: the
/// caller owns it, so the button reports rather than acts.
#[must_use]
pub fn show(ui: &mut egui::Ui, palette: &Palette, view: &mut View, lines: &[LogLine]) -> bool {
    let mut clear = false;

    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("log-level")
            .selected_text(egui::RichText::new(view.min_level.label()).size(CONTROL_TEXT))
            .width(104.0)
            // Far above what three items need, so the list cannot scroll.
            .height(MENU_CAP)
            .show_ui(ui, |ui| {
                super::header::borderless_rows(ui);
                for level in LevelFilter::ALL {
                    ui.selectable_value(&mut view.min_level, level, level.label());
                }
            });

        ui.add(
            egui::TextEdit::singleline(&mut view.filter)
                .hint_text("filter")
                .font(egui::FontId::proportional(CONTROL_TEXT))
                .desired_width(ui.available_width() - 96.0),
        );

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            clear = ui
                .add_enabled(
                    !lines.is_empty(),
                    egui::Button::new(egui::RichText::new("Clear").size(CONTROL_TEXT)),
                )
                .on_hover_text("Empty the buffer. GET /v1/logs reads the same one.")
                .clicked();
            ui.checkbox(
                &mut view.follow,
                egui::RichText::new("follow").size(CONTROL_TEXT),
            );
        });
    });
    ui.add_space(4.0);

    let visible = visible_lines(view, lines);
    // Measured from the font, not assumed from the point size. `show_rows` uses
    // this both to decide which rows are on screen and to tell the scroll area
    // how tall its contents are, so a guess that is a point or two under leaves
    // the area sized for less than it draws, which is the gap that appears
    // under a short log.
    let font = egui::FontId::monospace(ROW);
    let row_height = ui.ctx().fonts_mut(|fonts| fonts.row_height(&font));
    //  Scoped, because the row spacing has to be set on the ui `show_rows` is called
    //  on, not on the one it hands back. `show_rows` adds `item_spacing.y` to the row
    //  height given here and from that decides how tall the contents are and which
    //  rows are on screen, so setting it inside the closure is too late.
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(6.0, ROW_SPACING);
        // Without this every line is at least `interact_size.y` tall: 24
        // points, set for buttons, and a 9.5-point line sits in the middle of
        // it. That is where the density went.
        ui.spacing_mut().interact_size.y = 0.0;
        egui::ScrollArea::vertical()
            .stick_to_bottom(view.follow)
            .max_height(VISIBLE_HEIGHT)
            // Shrinks to the lines it has, up to the cap. A fixed height leaves
            // an empty box under a short log, which reads as broken.
            .auto_shrink([false, true])
            .show_rows(ui, row_height, visible.len(), |ui, range| {
                for line in &visible[range] {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!("{:<5}", line.level))
                                .font(font.clone())
                                .color(level_colour(palette, &line.level)),
                        );
                        ui.label(
                            egui::RichText::new(&line.message)
                                .font(font.clone())
                                .color(palette.text),
                        );
                    });
                }
            });
    });

    clear
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

fn level_colour(palette: &Palette, level: &str) -> egui::Color32 {
    match level {
        "ERROR" => palette.bad,
        "WARN" => palette.warn,
        "INFO" => palette.accent,
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
            follow: true,
        };
        assert_eq!(visible_lines(&view, &lines).len(), 1);
    }
}
