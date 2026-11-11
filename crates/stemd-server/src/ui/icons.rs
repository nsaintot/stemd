//! Stroke icons, painted from SVG paths in a 10 by 10 box.
//!
//! Painted because a glyph missing from the bundled fonts draws nothing at all.

use eframe::egui::{self, Color32, Pos2, Rect, Stroke, vec2};

/// Draw `paths`, given in a `viewbox`-unit square, into `rect`.
fn paths(
    painter: &egui::Painter,
    rect: Rect,
    viewbox: f32,
    paths: &[&[(f32, f32)]],
    stroke: Stroke,
) {
    let k = rect.width().min(rect.height()) / viewbox;
    let origin = rect.center() - vec2(viewbox, viewbox) * k / 2.0;
    for path in paths {
        let points: Vec<Pos2> = path
            .iter()
            .map(|(x, y)| origin + vec2(*x, *y) * k)
            .collect();
        painter.add(egui::Shape::line(points.clone(), stroke));
        // Round caps, as `stroke-linecap: round`.
        for end in [points[0], points[points.len() - 1]] {
            painter.circle_filled(end, stroke.width / 2.0, stroke.color);
        }
    }
}

/// `M2.5 6.4 5 8.8l4.5-5.4` in a 12-unit box.
pub fn check(painter: &egui::Painter, rect: Rect, colour: Color32, width: f32) {
    paths(
        painter,
        rect,
        12.0,
        &[&[(2.5, 6.4), (5.0, 8.8), (9.5, 3.4)]],
        Stroke::new(width, colour),
    );
}

/// An arrow into a tray, in a 10-unit box.
pub fn download(painter: &egui::Painter, rect: Rect, colour: Color32, width: f32) {
    paths(
        painter,
        rect,
        10.0,
        &[
            &[(5.0, 1.0), (5.0, 6.5)],
            &[(2.8, 4.5), (5.0, 6.7), (7.2, 4.5)],
            &[(1.5, 9.0), (8.5, 9.0)],
        ],
        Stroke::new(width, colour),
    );
}

/// A chevron pointing down, turned by `turn` radians clockwise.
pub fn chevron(painter: &egui::Painter, rect: Rect, colour: Color32, width: f32, turn: f32) {
    let (s, c) = turn.sin_cos();
    let rotate = |x: f32, y: f32| {
        let (dx, dy) = (x - 5.0, y - 5.0);
        (5.0 + dx * c - dy * s, 5.0 + dx * s + dy * c)
    };
    paths(
        painter,
        rect,
        10.0,
        &[&[rotate(2.0, 3.5), rotate(5.0, 6.5), rotate(8.0, 3.5)]],
        Stroke::new(width, colour),
    );
}

/// A cross, in a 10-unit box.
pub fn cross(painter: &egui::Painter, rect: Rect, colour: Color32, width: f32) {
    paths(
        painter,
        rect,
        10.0,
        &[&[(2.0, 2.0), (8.0, 8.0)], &[(8.0, 2.0), (2.0, 8.0)]],
        Stroke::new(width, colour),
    );
}

/// Three lines, the last short: the log.
pub fn lines(painter: &egui::Painter, rect: Rect, colour: Color32, width: f32) {
    paths(
        painter,
        rect,
        14.0,
        &[
            &[(2.5, 3.8), (11.5, 3.8)],
            &[(2.5, 7.0), (11.5, 7.0)],
            &[(2.5, 10.2), (8.0, 10.2)],
        ],
        Stroke::new(width, colour),
    );
}

/// A clock face: waiting.
pub fn clock(painter: &egui::Painter, rect: Rect, colour: Color32, width: f32) {
    let k = rect.width().min(rect.height()) / 12.0;
    painter.circle_stroke(rect.center(), 4.6 * k, Stroke::new(width, colour));
    let origin = rect.center() - vec2(6.0, 6.0) * k;
    paths(
        painter,
        Rect::from_min_size(origin, vec2(12.0, 12.0) * k),
        12.0,
        &[&[(6.0, 3.6), (6.0, 6.0), (7.6, 7.2)]],
        Stroke::new(width, colour),
    );
}

/// A faint ring with a quarter arc running round it, `turn` in revolutions.
pub fn spinner(painter: &egui::Painter, rect: Rect, colour: Color32, width: f32, turn: f32) {
    let k = rect.width().min(rect.height()) / 12.0;
    let r = 4.6 * k;
    let centre = rect.center();
    painter.circle_stroke(centre, r, Stroke::new(width, super::theme::white(0.16)));
    let start = -std::f32::consts::FRAC_PI_2 + turn * std::f32::consts::TAU;
    let points: Vec<Pos2> = (0..=12)
        .map(|i| {
            let a = start + std::f32::consts::FRAC_PI_2 * i as f32 / 12.0;
            centre + r * vec2(a.cos(), a.sin())
        })
        .collect();
    painter.add(egui::Shape::line(points, Stroke::new(width, colour)));
}

/// A square of `size` centred on `centre`.
pub fn square(centre: Pos2, size: f32) -> Rect {
    Rect::from_center_size(centre, vec2(size, size))
}
