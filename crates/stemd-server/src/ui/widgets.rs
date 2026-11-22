//! Painted controls shared across the window: pill buttons and frosted menus.

use eframe::egui::{self, Rect, Sense, Stroke, pos2, vec2};

use super::blur::{self, Frosted};
use super::icons;
use super::theme::{self, Brushes, Palette, Weight, white};

/// Seconds a hover wash takes to come and go.
pub const HOVER_SECS: f32 = 0.15;

/// Inner margin of every menu panel.
const MENU_MARGIN: i8 = 6;

/// 0 to 1 as `on` turns on and off, for hover washes.
pub fn fade(ui: &egui::Ui, id: egui::Id, on: bool) -> f32 {
    ui.ctx()
        .animate_bool_with_time(id.with("fade"), on, HOVER_SECS)
}

/// The bottom of a list panel where its cut row fades out.
pub fn fade_zone(panel: Rect, row_height: f32) -> Rect {
    let part = (panel.height() / row_height).fract();
    let cut = if part < 0.05 { 0.5 } else { part };
    Rect::from_x_y_ranges(
        panel.x_range(),
        panel.bottom() - row_height * cut..=panel.bottom() - 1.0,
    )
}

/// Where the next shape painted on `ui`'s layer will go, for [`fade_out`].
pub fn paint_mark(ui: &egui::Ui) -> egui::layers::ShapeIdx {
    ui.ctx().graphics_mut(|g| g.entry(ui.layer_id()).next_idx())
}

/// Fade what `painter`'s layer received since `from` out to invisible across `zone`.
pub fn fade_out(painter: &egui::Painter, from: egui::layers::ShapeIdx, zone: Rect) {
    let bands = (zone.height().ceil() as usize).max(2);
    let above = Rect::everything_above(zone.top());
    let caught: Vec<egui::epaint::ClippedShape> = painter.ctx().graphics_mut(|g| {
        let list = g.entry(painter.layer_id());
        let caught: Vec<(usize, egui::epaint::ClippedShape)> = list
            .all_entries()
            .enumerate()
            .skip(from.0)
            .filter(|(_, c)| {
                c.clip_rect.intersects(zone) && c.shape.visual_bounding_rect().intersects(zone)
            })
            .map(|(i, c)| (i, c.clone()))
            .collect();
        for (i, _) in &caught {
            list.mutate_shape(egui::layers::ShapeIdx(*i), |c| {
                c.clip_rect = c.clip_rect.intersect(above);
            });
        }
        caught.into_iter().map(|(_, c)| c).collect()
    });
    // Band by band, so shapes keep their order where they overlap.
    for band in 0..bands {
        let (a, b) = (band as f32 / bands as f32, (band + 1) as f32 / bands as f32);
        let strip = Rect::from_x_y_ranges(
            zone.x_range(),
            zone.top() + zone.height() * a..=zone.top() + zone.height() * b,
        );
        let opacity = fade_at(band, bands);
        if opacity <= 0.0 {
            continue;
        }
        for shape in &caught {
            painter
                .with_clip_rect(shape.clip_rect.intersect(strip))
                .add(faded(&shape.shape, opacity));
        }
    }
}

/// `shape` at `opacity`. Text takes it as its opacity factor, so its galley
/// stays shared.
fn faded(shape: &egui::Shape, opacity: f32) -> egui::Shape {
    match shape {
        egui::Shape::Text(text) => {
            let mut text = text.clone();
            text.opacity_factor *= opacity;
            egui::Shape::Text(text)
        }
        egui::Shape::Vec(shapes) => {
            egui::Shape::Vec(shapes.iter().map(|s| faded(s, opacity)).collect())
        }
        other => {
            let mut other = other.clone();
            egui::epaint::shape_transform::adjust_colors(&mut other, move |colour| {
                if *colour != egui::Color32::PLACEHOLDER {
                    *colour = colour.gamma_multiply(opacity);
                }
            });
            other
        }
    }
}

/// Opacity of band `band` of `bands`: 1 on the first, 0 on the last, linear between.
pub fn fade_at(band: usize, bands: usize) -> f32 {
    1.0 - band as f32 / (bands.max(2) - 1) as f32
}

/// How a pill button is drawn.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Pill {
    /// Translucent, with a hairline: the everyday button.
    Ghost,
    /// The same, with its label in the danger colour.
    Danger,
    /// The light pill: the one answer that matters.
    Light,
}

/// Width of a pill that fits `label` at `height`.
pub fn pill_width(ui: &egui::Ui, label: &str, height: f32) -> f32 {
    let galley = ui.painter().layout_no_wrap(
        label.to_owned(),
        pill_font(height, Pill::Ghost),
        egui::Color32::PLACEHOLDER,
    );
    galley.size().x + height
}

fn pill_font(height: f32, kind: Pill) -> egui::FontId {
    let size = if height >= 30.0 { 13.0 } else { 12.0 };
    let weight = if kind == Pill::Light {
        Weight::Semibold
    } else {
        Weight::Medium
    };
    theme::font(size, weight)
}

/// A pill button filling `rect`. Disabled, it is dimmed and does not respond.
#[allow(clippy::too_many_arguments)]
pub fn pill(
    ui: &egui::Ui,
    rect: Rect,
    id: egui::Id,
    label: &str,
    kind: Pill,
    enabled: bool,
    palette: &Palette,
    brushes: &Brushes,
) -> egui::Response {
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let response = ui.interact(rect, id, sense);
    let hover = fade(ui, id, response.hovered() && enabled);
    let pressed = ui.ctx().animate_bool_with_time(
        id.with("press"),
        response.is_pointer_button_down_on() && enabled,
        0.1,
    );
    let drawn = Rect::from_center_size(rect.center(), rect.size() * (1.0 - 0.03 * pressed));
    let mut painter = ui.painter().clone();
    if !enabled {
        painter.multiply_opacity(0.45);
    }
    let r = drawn.height() / 2.0;
    let text = match kind {
        Pill::Light => {
            brushes.pill(&painter, drawn, r, 1.0);
            if hover > 0.0 {
                painter.rect_filled(drawn, r, white(0.08 * hover));
            }
            palette.ink
        }
        Pill::Ghost | Pill::Danger => {
            painter.rect_filled(drawn, r, white(0.08 + 0.08 * hover));
            painter.rect_stroke(
                drawn,
                r,
                Stroke::new(1.0, white(0.14)),
                egui::StrokeKind::Inside,
            );
            painter.add(theme::top_highlight(drawn, r, white(0.2)));
            if kind == Pill::Danger {
                palette.danger
            } else {
                palette.text
            }
        }
    };
    painter.text(
        drawn.center(),
        egui::Align2::CENTER_CENTER,
        label,
        pill_font(rect.height(), kind),
        text,
    );
    if enabled {
        response.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        response
    }
}

/// The panel every menu sits on.
pub const fn menu_look(palette: &Palette) -> Frosted {
    Frosted {
        radius: 14.0,
        margin: MENU_MARGIN,
        fill: palette.frost,
        outline: palette.menu_outline,
    }
}

/// The panel a dialog sits on.
pub const fn dialog_look(palette: &Palette) -> Frosted {
    Frosted {
        radius: 18.0,
        margin: 18,
        fill: palette.frost,
        outline: palette.menu_outline,
    }
}

/// Width of a menu row in a menu `width` wide: the panel's margin and outline
/// come off both sides.
pub fn row_width(width: f32) -> f32 {
    width - 2.0 * f32::from(MENU_MARGIN) - 2.0
}

/// A frosted menu that opens under `button`, toggled by clicking it.
pub fn menu<R>(
    button: &egui::Response,
    align: egui::RectAlign,
    close: egui::PopupCloseBehavior,
    palette: &Palette,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> bool {
    let popup = egui::Popup::from_toggle_button_response(button)
        .id(button.id.with("menu"))
        .anchor(button.rect.expand2(vec2(4.0, 0.0)))
        .gap(10.0)
        .align(align)
        .frame(egui::Frame::NONE)
        .close_behavior(close);
    let open = popup.is_open();
    popup.show(|ui| blur::frosted(ui, &menu_look(palette), add));
    open
}

/// One menu row with a hover wash. Returns the response.
pub fn menu_row(
    ui: &mut egui::Ui,
    width: f32,
    height: f32,
    id: egui::Id,
) -> (Rect, egui::Response) {
    let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
    let response = ui.interact(rect, id, Sense::click());
    let hover = fade(ui, id, response.hovered());
    if hover > 0.0 {
        ui.painter().rect_filled(rect, 9.0, white(0.10 * hover));
    }
    (
        rect,
        response.on_hover_cursor(egui::CursorIcon::PointingHand),
    )
}

/// A list of labels with a tick on the current one, in a menu `width` wide.
/// Each item is a label, whether it is current, and a hover hint (empty for
/// none). Returns the index clicked.
pub fn pick_menu(
    ui: &mut egui::Ui,
    width: f32,
    labels: &[(String, bool, &'static str)],
    palette: &Palette,
) -> Option<usize> {
    ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
    let inner = row_width(width);
    let mut picked = None;
    for (n, (label, on, hint)) in labels.iter().enumerate() {
        let (rect, response) = menu_row(ui, inner, 32.0, ui.id().with(("pick", n)));
        let painter = ui.painter();
        painter.text(
            pos2(rect.left() + 10.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            label,
            theme::font(13.0, Weight::Regular),
            palette.text,
        );
        if *on {
            icons::check(
                painter,
                icons::square(pos2(rect.right() - 8.0 - 5.5, rect.center().y), 11.0),
                palette.text,
                1.6,
            );
        }
        let response = if hint.is_empty() {
            response
        } else {
            response.on_hover_text(*hint)
        };
        if response.clicked() {
            picked = Some(n);
        }
    }
    picked
}

/// Rows stacked top to bottom, each with a gap under it. Rows are added once
/// and their offsets kept, so the block's height and the rows' positions come
/// from the same numbers.
#[derive(Default)]
pub struct Column {
    used: f32,
    last_gap: f32,
}

impl Column {
    /// Add a row; returns its offset from the top.
    pub fn add(&mut self, height: f32, gap_after: f32) -> f32 {
        let at = self.used;
        self.used += height + gap_after;
        self.last_gap = gap_after;
        at
    }

    /// Offset of the next row to be added.
    pub const fn next(&self) -> f32 {
        self.used
    }

    /// Height of the block, without the gap under its last row.
    pub fn height(&self) -> f32 {
        self.used - self.last_gap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_column_places_rows_and_leaves_off_the_last_gap() {
        let mut column = Column::default();
        assert_eq!(column.add(10.0, 4.0), 0.0);
        assert_eq!(column.add(20.0, 6.0), 14.0);
        assert_eq!(column.height(), 34.0);
    }

    #[test]
    fn the_fade_covers_the_part_of_the_cut_row_in_view() {
        let panel = |rows: f32| Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 40.0 * rows));
        assert_eq!(
            fade_zone(panel(2.75), 40.0).y_range(),
            egui::Rangef::new(80.0, 109.0)
        );
        assert_eq!(
            fade_zone(panel(3.0), 40.0).y_range(),
            egui::Rangef::new(100.0, 119.0),
            "whole rows fade over half a row"
        );
    }

    #[test]
    fn the_fade_runs_linearly_from_full_to_nothing() {
        assert_eq!(fade_at(0, 5), 1.0);
        assert_eq!(fade_at(2, 5), 0.5);
        assert_eq!(fade_at(4, 5), 0.0, "the bottom row still shows");
    }
}
