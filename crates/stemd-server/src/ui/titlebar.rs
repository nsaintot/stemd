//! The window's own title bar on macOS and Windows: the name and version centred
//! and, on Windows, the window buttons on the right. macOS keeps its traffic
//! lights. Linux keeps the system's title bar.

use eframe::egui::{self, Rect, pos2, vec2};

use super::theme::{self, Palette, Weight};

/// Height of the bar. The content starts under it.
pub const HEIGHT: f32 = if cfg!(target_os = "macos") {
    28.0
} else if cfg!(windows) {
    32.0
} else {
    0.0
};

/// Centre line of the name: level with the traffic lights on macOS.
const TEXT_Y: f32 = if cfg!(target_os = "macos") {
    16.0
} else {
    HEIGHT / 2.0
};

/// Draw the bar over the top of the window. Called last, so its controls sit
/// above everything else.
pub fn show(ui: &mut egui::Ui, palette: &Palette) {
    if HEIGHT == 0.0 {
        return;
    }
    let ctx = ui.ctx().clone();
    let screen = ctx.content_rect();
    let bar = Rect::from_min_size(screen.min, vec2(screen.width(), HEIGHT));
    let focused = ctx.input(|i| i.viewport().focused.unwrap_or(true));
    let dim = if focused { 1.0 } else { 0.5 };

    let painter = ui.painter();
    let name = painter.layout_no_wrap(
        "Stemd".to_owned(),
        theme::font(12.0, Weight::Semibold),
        palette.text.gamma_multiply(dim),
    );
    let version = painter.layout_no_wrap(
        format!("v{}", env!("CARGO_PKG_VERSION")),
        theme::font(12.0, Weight::Regular),
        palette.muted.gamma_multiply(dim),
    );
    let y = bar.top() + TEXT_Y;
    let width = name.size().x + 6.0 + version.size().x;
    // Centred, as macOS draws window titles.
    let x = bar.center().x - width / 2.0;
    let version_x = x + name.size().x + 6.0;
    painter.galley(
        pos2(x, y - name.size().y / 2.0),
        name,
        palette.text.gamma_multiply(dim),
    );
    painter.galley(
        pos2(version_x, y - version.size().y / 2.0),
        version,
        palette.muted.gamma_multiply(dim),
    );

    #[cfg(windows)]
    windows::controls(ui, &ctx, bar, screen, palette, dim);
}

#[cfg(windows)]
mod windows {
    use eframe::egui::{
        self, Color32, CursorIcon, PointerButton, Rect, Sense, Stroke, ViewportCommand, pos2, vec2,
        viewport::ResizeDirection,
    };

    use super::super::theme::{Palette, white};
    use super::HEIGHT;

    /// Size of one window button, as Windows 11 draws them.
    const BUTTON: egui::Vec2 = vec2(46.0, HEIGHT);

    /// Width of the resize grip along each edge, and of each corner.
    const EDGE: f32 = 5.0;
    const CORNER: f32 = 12.0;

    /// The red Windows fills the close button with under the pointer.
    const CLOSE_RED: Color32 = Color32::from_rgb(0xC4, 0x2B, 0x1C);

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Glyph {
        Minimize,
        Maximize,
        Restore,
        Close,
    }

    pub fn controls(
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        bar: Rect,
        screen: Rect,
        palette: &Palette,
        dim: f32,
    ) {
        let maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
        let close = Rect::from_min_size(pos2(bar.right() - BUTTON.x, bar.top()), BUTTON);
        let middle = close.translate(vec2(-BUTTON.x, 0.0));
        let minimize = middle.translate(vec2(-BUTTON.x, 0.0));

        // The rest of the bar moves the window; a double click maximizes it.
        let drag = Rect::from_min_max(bar.min, pos2(minimize.left(), bar.bottom()));
        let moved = ui.interact(drag, ui.id().with("title-drag"), Sense::click_and_drag());
        if moved.double_clicked() {
            ctx.send_viewport_cmd(ViewportCommand::Maximized(!maximized));
        } else if moved.drag_started_by(PointerButton::Primary) {
            ctx.send_viewport_cmd(ViewportCommand::StartDrag);
        }

        if button(ui, minimize, Glyph::Minimize, palette, dim) {
            ctx.send_viewport_cmd(ViewportCommand::Minimized(true));
        }
        let glyph = if maximized {
            Glyph::Restore
        } else {
            Glyph::Maximize
        };
        if button(ui, middle, glyph, palette, dim) {
            ctx.send_viewport_cmd(ViewportCommand::Maximized(!maximized));
        }
        if button(ui, close, Glyph::Close, palette, dim) {
            ctx.send_viewport_cmd(ViewportCommand::Close);
        }

        if !maximized {
            resize_edges(ui, ctx, screen);
        }
    }

    fn button(ui: &mut egui::Ui, rect: Rect, glyph: Glyph, palette: &Palette, dim: f32) -> bool {
        let id = ui.id().with(("title-button", glyph as u8));
        let response = ui.interact(rect, id, Sense::click());
        let hover = super::super::widgets::fade(ui, id, response.hovered());
        let pressed = response.is_pointer_button_down_on();
        let painter = ui.painter();
        let mut ink = palette.text.gamma_multiply(dim);
        if glyph == Glyph::Close {
            if hover > 0.0 {
                let alpha = if pressed { 0.85 } else { 1.0 };
                painter.rect_filled(rect, 0.0, CLOSE_RED.gamma_multiply(hover * alpha));
                ink = super::super::theme::mix(ink, Color32::WHITE, hover);
            }
        } else if hover > 0.0 {
            let wash = if pressed { 0.05 } else { 0.08 };
            painter.rect_filled(rect, 0.0, white(wash * hover));
        }
        draw(ui, rect.center(), glyph, ink);
        response.clicked()
    }

    /// The glyphs Windows 11 draws: 10 points across, one pixel wide, on the
    /// pixel grid so they stay sharp.
    fn draw(ui: &egui::Ui, centre: egui::Pos2, glyph: Glyph, ink: Color32) {
        let ppp = ui.ctx().pixels_per_point();
        let snap = |v: f32| (v * ppp).round() / ppp + 0.5 / ppp;
        let c = pos2(snap(centre.x), snap(centre.y));
        let stroke = Stroke::new(1.0, ink);
        let painter = ui.painter();
        let h = 5.0;
        match glyph {
            Glyph::Minimize => {
                painter.line_segment([c + vec2(-h, 0.0), c + vec2(h, 0.0)], stroke);
            }
            Glyph::Maximize => {
                painter.rect_stroke(
                    Rect::from_center_size(c, vec2(2.0 * h, 2.0 * h)),
                    1.0,
                    stroke,
                    egui::StrokeKind::Middle,
                );
            }
            Glyph::Restore => {
                let front = Rect::from_min_size(c + vec2(-h, -h + 2.0), vec2(8.0, 8.0));
                painter.rect_stroke(front, 1.0, stroke, egui::StrokeKind::Middle);
                painter.line_segment([c + vec2(-h + 2.0, -h), c + vec2(h, -h)], stroke);
                painter.line_segment([c + vec2(h, -h), c + vec2(h, h - 2.0)], stroke);
            }
            Glyph::Close => {
                painter.line_segment([c + vec2(-h, -h), c + vec2(h, h)], stroke);
                painter.line_segment([c + vec2(-h, h), c + vec2(h, -h)], stroke);
            }
        }
    }

    /// Grips along the edges and corners. Without the system frame the window
    /// has no other way to be resized.
    fn resize_edges(ui: &mut egui::Ui, ctx: &egui::Context, s: Rect) {
        use CursorIcon as C;
        use ResizeDirection as D;
        let zones = [
            (
                Rect::from_min_max(s.left_top(), pos2(s.left() + EDGE, s.bottom())),
                D::West,
                C::ResizeWest,
            ),
            (
                Rect::from_min_max(pos2(s.right() - EDGE, s.top()), s.right_bottom()),
                D::East,
                C::ResizeEast,
            ),
            (
                Rect::from_min_max(s.left_top(), pos2(s.right(), s.top() + EDGE)),
                D::North,
                C::ResizeNorth,
            ),
            (
                Rect::from_min_max(pos2(s.left(), s.bottom() - EDGE), s.right_bottom()),
                D::South,
                C::ResizeSouth,
            ),
            (
                Rect::from_min_size(s.left_top(), vec2(CORNER, CORNER)),
                D::NorthWest,
                C::ResizeNorthWest,
            ),
            (
                Rect::from_min_size(pos2(s.right() - CORNER, s.top()), vec2(CORNER, CORNER)),
                D::NorthEast,
                C::ResizeNorthEast,
            ),
            (
                Rect::from_min_size(pos2(s.left(), s.bottom() - CORNER), vec2(CORNER, CORNER)),
                D::SouthWest,
                C::ResizeSouthWest,
            ),
            (
                Rect::from_min_size(
                    s.right_bottom() - vec2(CORNER, CORNER),
                    vec2(CORNER, CORNER),
                ),
                D::SouthEast,
                C::ResizeSouthEast,
            ),
        ];
        let pressed = ctx.input(|i| i.pointer.primary_pressed());
        for (n, (rect, direction, cursor)) in zones.into_iter().enumerate() {
            let response = ui
                .interact(rect, ui.id().with(("resize", n)), Sense::drag())
                .on_hover_cursor(cursor);
            if response.hovered() && pressed {
                ctx.send_viewport_cmd(ViewportCommand::BeginResize(direction));
            }
        }
    }
}
