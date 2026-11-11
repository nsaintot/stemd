//! Colour, type, the window backdrop and the glass surfaces everything sits on.
//!
//! One appearance: the navy backdrop is the design, so the window does not follow
//! the system's light mode.

use std::sync::Arc;

use eframe::egui::{
    self, Color32, CornerRadius, FontFamily, FontId, Pos2, Rect, Stroke, StrokeKind, TextureHandle,
    TextureOptions, pos2, vec2,
};
use egui::epaint::text::{FontTweak, VariationCoords};
use egui::epaint::{RectShape, Shadow};

/// Instrument Sans, variable on `wght` and `wdth`. SIL Open Font License, see
/// `resources/fonts/InstrumentSans-OFL.txt`.
const INSTRUMENT_SANS: &[u8] = include_bytes!("../../../../resources/fonts/InstrumentSans.ttf");

/// Separator inside a line of text.
pub const DOT: &str = " · ";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weight {
    Regular,
    Medium,
    Semibold,
}

impl Weight {
    const ALL: [Self; 3] = [Self::Regular, Self::Medium, Self::Semibold];

    const fn font_name(self) -> &'static str {
        match self {
            Self::Regular => "instrument-400",
            Self::Medium => "instrument-500",
            Self::Semibold => "instrument-600",
        }
    }

    const fn wght(self) -> f32 {
        match self {
            Self::Regular => 400.0,
            Self::Medium => 500.0,
            Self::Semibold => 600.0,
        }
    }

    fn family(self) -> FontFamily {
        match self {
            Self::Regular => FontFamily::Proportional,
            Self::Medium => FontFamily::Name("medium".into()),
            Self::Semibold => FontFamily::Name("semibold".into()),
        }
    }
}

pub fn font(size: f32, weight: Weight) -> FontId {
    FontId::new(size, weight.family())
}

pub struct Palette {
    pub text: Color32,
    pub muted: Color32,
    /// Unselected segment labels and secondary tags.
    pub soft: Color32,
    /// Text on a light pill.
    pub ink: Color32,
    /// The equator band, check boxes and rings.
    pub band: Color32,
    pub red: Color32,
    pub green: Color32,
    pub blue: Color32,
    pub good: Color32,
    pub danger: Color32,
    /// Tooltip fill.
    pub menu: Color32,
    /// Fill over the blurred backdrop of a menu or dialog.
    pub frost: Color32,
    pub menu_outline: Color32,
    pub hairline: Color32,
    pub divider: Color32,
}

impl Palette {
    pub fn navy() -> Self {
        Self {
            text: Color32::from_rgb(0xEE, 0xF0, 0xF6),
            muted: Color32::from_rgb(0xB4, 0xBB, 0xCC),
            soft: Color32::from_rgb(0xC9, 0xCE, 0xDB),
            ink: Color32::from_rgb(0x12, 0x1A, 0x33),
            band: Color32::from_rgb(0xE9, 0xEB, 0xF1),
            red: Color32::from_rgb(0xD4, 0x68, 0x5F),
            green: Color32::from_rgb(0x6F, 0xB4, 0x7C),
            blue: Color32::from_rgb(0x5B, 0x84, 0xD8),
            good: Color32::from_rgb(0x8F, 0xD1, 0x9B),
            danger: Color32::from_rgb(0xF0, 0xA4, 0x9C),
            menu: Color32::from_rgba_unmultiplied(30, 38, 70, 235),
            frost: Color32::from_rgba_unmultiplied(24, 31, 58, 150),
            menu_outline: white(0.14),
            hairline: white(0.08),
            divider: white(0.14),
        }
    }

    pub fn of(_ctx: &egui::Context) -> Self {
        Self::navy()
    }

    /// The colours drawn with less than full alpha.
    #[cfg(test)]
    fn translucent(&self) -> [Color32; 5] {
        [
            self.menu,
            self.frost,
            self.menu_outline,
            self.hairline,
            self.divider,
        ]
    }
}

/// White at `alpha` (0 to 1), premultiplied for normal blending.
pub fn white(alpha: f32) -> Color32 {
    Color32::from_white_alpha(unit(alpha))
}

pub fn black(alpha: f32) -> Color32 {
    Color32::from_black_alpha(unit(alpha))
}

fn unit(alpha: f32) -> u8 {
    (alpha.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// `from` to `to` by `t`, in gamma space like CSS.
pub fn mix(from: Color32, to: Color32, t: f32) -> Color32 {
    from.lerp_to_gamma(to, t.clamp(0.0, 1.0))
}

/// A glass surface: vertical white gradient, hairline, top highlight, soft shadow.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Glass {
    /// Header controls and first-run rows.
    Panel,
    /// The separated list and the log.
    List,
}

/// Alphas of a glass surface's white gradient, outline and highlight, and of
/// its black shadow.
struct GlassSpec {
    top: f32,
    bottom: f32,
    outline: f32,
    highlight: f32,
    shadow: f32,
}

impl Glass {
    const fn spec(self) -> GlassSpec {
        match self {
            Self::Panel => GlassSpec {
                top: 0.10,
                bottom: 0.04,
                outline: 0.12,
                highlight: 0.18,
                shadow: 0.22,
            },
            Self::List => GlassSpec {
                top: 0.08,
                bottom: 0.03,
                outline: 0.10,
                highlight: 0.14,
                shadow: 0.20,
            },
        }
    }
}

/// Texture brushes and baked images, built once per context.
///
/// `RectShape` fills with one colour; a texture sampled across it is how a rounded
/// rectangle gets a gradient that follows its corners.
pub struct Brushes {
    backdrop: TextureHandle,
    panel: TextureHandle,
    list: TextureHandle,
    pill: TextureHandle,
    sheen: TextureHandle,
    spectrum: TextureHandle,
    gloss: TextureHandle,
    pub shade: TextureHandle,
    pub shadow: TextureHandle,
    pub halo: TextureHandle,
}

/// Samples a 1x2 texture from the first texel centre to the second, so the
/// gradient spans the whole shape.
const VERTICAL_UV: Rect = Rect {
    min: Pos2 { x: 0.5, y: 0.25 },
    max: Pos2 { x: 0.5, y: 0.75 },
};

impl Brushes {
    pub fn new(ctx: &egui::Context) -> Self {
        let linear = TextureOptions::LINEAR;
        let repeat = TextureOptions {
            wrap_mode: egui::TextureWrapMode::Repeat,
            ..TextureOptions::LINEAR
        };
        let column = |name: &str, top: Color32, bottom: Color32| {
            ctx.load_texture(
                name,
                egui::ColorImage::new([1, 2], vec![top, bottom]),
                linear,
            )
        };
        Self {
            backdrop: ctx.load_texture("stemd-backdrop", backdrop(256), linear),
            panel: column(
                "stemd-glass-panel",
                white(Glass::Panel.spec().top),
                white(Glass::Panel.spec().bottom),
            ),
            list: column(
                "stemd-glass-list",
                white(Glass::List.spec().top),
                white(Glass::List.spec().bottom),
            ),
            pill: column(
                "stemd-pill",
                Color32::from_rgb(0xEE, 0xF0, 0xF6),
                Color32::from_rgb(0xC2, 0xC7, 0xD4),
            ),
            sheen: ctx.load_texture("stemd-sheen", sheen(64), repeat),
            spectrum: ctx.load_texture("stemd-spectrum", spectrum(), linear),
            gloss: ctx.load_texture("stemd-gloss", gloss(16), linear),
            shade: ctx.load_texture("stemd-shade", shade(128), linear),
            shadow: ctx.load_texture("stemd-shadow", shadow_disc(64), linear),
            halo: ctx.load_texture("stemd-halo", halo(128), linear),
        }
    }

    /// The navy backdrop, stretched over `rect`.
    pub fn backdrop(&self, painter: &egui::Painter, rect: Rect) {
        painter.image(
            self.backdrop.id(),
            rect,
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    /// A glass surface.
    pub fn glass(&self, painter: &egui::Painter, rect: Rect, radius: f32, glass: Glass) {
        painter.add(self.glass_shape(rect, radius, glass));
    }

    /// A glass surface as one shape, for putting behind content drawn first.
    pub fn glass_shape(&self, rect: Rect, radius: f32, glass: Glass) -> egui::Shape {
        let texture = match glass {
            Glass::Panel => &self.panel,
            Glass::List => &self.list,
        };
        let spec = glass.spec();
        egui::Shape::Vec(vec![
            drop_shadow(rect, radius, 6.0, 16.0, black(spec.shadow)),
            RectShape::filled(rect, radius, Color32::WHITE)
                .with_texture(texture.id(), VERTICAL_UV)
                .into(),
            RectShape::stroke(
                rect,
                radius,
                Stroke::new(1.0, white(spec.outline)),
                StrokeKind::Inside,
            )
            .into(),
            top_highlight(rect, radius, white(spec.highlight)),
        ])
    }

    /// The light pill: a selected segment, an open menu button, a primary button.
    pub fn pill(&self, painter: &egui::Painter, rect: Rect, radius: f32, alpha: f32) {
        painter.add(drop_shadow(rect, radius, 1.0, 3.0, black(0.3 * alpha)));
        painter.add(
            RectShape::filled(rect, radius, white(alpha)).with_texture(self.pill.id(), VERTICAL_UV),
        );
    }

    /// The Quality thumb: a pastel red, green and blue gradient sliding left.
    /// `offset` is in widths of the pill, and wraps.
    pub fn sheen(&self, painter: &egui::Painter, rect: Rect, radius: f32, offset: f32) {
        painter.add(drop_shadow(
            rect,
            radius,
            0.0,
            14.0,
            Color32::from_rgba_unmultiplied(140, 170, 240, 140),
        ));
        painter.add(drop_shadow(rect, radius, 1.0, 3.0, black(0.3)));
        // The gradient is 2.2 pill widths long.
        let span = 1.0 / 2.2;
        let u = offset.rem_euclid(1.0);
        painter.add(
            RectShape::filled(rect, radius, Color32::WHITE).with_texture(
                self.sheen.id(),
                Rect::from_min_max(pos2(u, 0.5), pos2(u + span, 0.5)),
            ),
        );
    }

    /// Red, green and blue left to right across `track`, drawn only over `fill`.
    /// The gradient stays anchored to the track as `fill` grows.
    pub fn spectrum(
        &self,
        painter: &egui::Painter,
        track: Rect,
        fill: Rect,
        radius: f32,
        tint: Color32,
    ) {
        if fill.width() <= 0.0 {
            return;
        }
        let u = |x: f32| 1.0 / 6.0 + (x - track.left()) / track.width() * (2.0 / 3.0);
        painter.add(RectShape::filled(fill, radius, tint).with_texture(
            self.spectrum.id(),
            Rect::from_min_max(pos2(u(fill.left()), 0.5), pos2(u(fill.right()), 0.5)),
        ));
    }

    /// White fading to nothing 60% of the way down: the gloss on a progress fill.
    pub fn gloss(&self, painter: &egui::Painter, rect: Rect, radius: f32) {
        painter.add(
            RectShape::filled(rect, radius, Color32::WHITE).with_texture(
                self.gloss.id(),
                Rect::from_min_max(pos2(0.5, 0.0), pos2(0.5, 1.0)),
            ),
        );
    }
}

/// `box-shadow: 0 <dy> <blur> <colour>`.
pub fn drop_shadow(rect: Rect, radius: f32, dy: f32, blur: f32, colour: Color32) -> egui::Shape {
    let shadow = Shadow {
        offset: [0, dy.round() as i8],
        blur: blur.round() as u8,
        spread: 0,
        color: colour,
    };
    shadow.as_shape(rect, CornerRadius::from(radius)).into()
}

/// `box-shadow: inset 0 1px 0 <colour>`: a one-point line under the top edge,
/// following the corners partway round.
pub fn top_highlight(rect: Rect, radius: f32, colour: Color32) -> egui::Shape {
    let inset = 1.5;
    let r = (radius - inset).max(0.0);
    let left = pos2(rect.left() + radius, rect.top() + radius);
    let right = pos2(rect.right() - radius, rect.top() + radius);
    let mut points = Vec::with_capacity(20);
    let steps = 8;
    for i in 0..=steps {
        let a = std::f32::consts::PI * (1.25 + 0.25 * i as f32 / steps as f32);
        points.push(left + r * vec2(a.cos(), a.sin()));
    }
    for i in 0..=steps {
        let a = std::f32::consts::PI * (1.5 + 0.25 * i as f32 / steps as f32);
        points.push(right + r * vec2(a.cos(), a.sin()));
    }
    egui::Shape::line(points, Stroke::new(1.0, colour))
}

/// The outline of a rounded rectangle, clockwise from the top-left corner.
pub fn rounded_outline(rect: Rect, radius: f32, steps: usize) -> Vec<Pos2> {
    use std::f32::consts::{FRAC_PI_2, PI};
    let r = radius.min(rect.width() / 2.0).min(rect.height() / 2.0);
    let corners = [
        (pos2(rect.left() + r, rect.top() + r), PI),
        (pos2(rect.right() - r, rect.top() + r), PI + FRAC_PI_2),
        (pos2(rect.right() - r, rect.bottom() - r), 0.0),
        (pos2(rect.left() + r, rect.bottom() - r), FRAC_PI_2),
    ];
    let mut points = Vec::with_capacity(4 * (steps + 1) + 1);
    for (centre, start) in corners {
        for i in 0..=steps {
            let a = start + FRAC_PI_2 * i as f32 / steps as f32;
            points.push(centre + r * vec2(a.cos(), a.sin()));
        }
    }
    points.push(points[0]);
    points
}

/// `box-shadow: inset 0 0 <depth> <colour>`: a glow fading inward from the edge.
pub fn inner_glow(painter: &egui::Painter, rect: Rect, radius: f32, depth: f32, colour: Color32) {
    if colour.a() == 0 {
        return;
    }
    let steps = 10;
    let outer = rounded_outline(rect, radius, steps);
    let inner_rect = rect.shrink(depth);
    let inner = rounded_outline(inner_rect, (radius - depth).max(0.0), steps);
    // Both outlines have the same number of points when the radius is not
    // clamped differently, which holds for the drop zone's sizes.
    if outer.len() != inner.len() {
        return;
    }
    let mut mesh = egui::Mesh::default();
    for (o, i) in outer.iter().zip(&inner) {
        mesh.colored_vertex(*o, colour);
        mesh.colored_vertex(*i, Color32::TRANSPARENT);
    }
    for k in 0..(outer.len() as u32 - 1) {
        let a = 2 * k;
        mesh.add_triangle(a, a + 1, a + 2);
        mesh.add_triangle(a + 1, a + 3, a + 2);
    }
    painter.add(mesh);
}

/// A dashed rounded rectangle, as `border: dashed`.
pub fn dashed_outline(painter: &egui::Painter, rect: Rect, radius: f32, stroke: Stroke) {
    let points = rounded_outline(rect.shrink(stroke.width / 2.0), radius, 12);
    painter.extend(egui::Shape::dashed_line(&points, stroke, 5.0, 4.0));
}

/// CSS `cubic-bezier(x1, y1, x2, y2)` at `t`.
pub fn cubic_bezier(x1: f32, y1: f32, x2: f32, y2: f32, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    let curve = |a: f32, b: f32, s: f32| {
        let u = 1.0 - s;
        3.0 * u * u * s * a + 3.0 * u * s * s * b + s * s * s
    };
    // Solve x(s) = t by bisection: monotone in s for valid CSS curves.
    let (mut lo, mut hi) = (0.0_f32, 1.0_f32);
    for _ in 0..24 {
        let mid = (lo + hi) / 2.0;
        if curve(x1, x2, mid) < t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    curve(y1, y2, (lo + hi) / 2.0)
}

/// A springy ease, `cubic-bezier(.3, .8, .3, 1)`.
pub fn ease_spring(t: f32) -> f32 {
    cubic_bezier(0.3, 0.8, 0.3, 1.0, t)
}

/// `ease-out`.
pub fn ease_out(t: f32) -> f32 {
    cubic_bezier(0.0, 0.0, 0.58, 1.0, t)
}

/// `ease-in`.
pub fn ease_in(t: f32) -> f32 {
    cubic_bezier(0.42, 0.0, 1.0, 1.0, t)
}

/// `ease-in-out`.
pub fn ease_in_out(t: f32) -> f32 {
    cubic_bezier(0.42, 0.0, 0.58, 1.0, t)
}

/// Frame interval for animation while a track separates: 60 fps.
const PROCESSING_FRAME: std::time::Duration = std::time::Duration::from_micros(16_667);

fn processing_id() -> egui::Id {
    egui::Id::new("stemd-processing")
}

/// Record whether a separation is running, which caps animation at 60 fps.
pub fn set_processing(ctx: &egui::Context, on: bool) {
    ctx.data_mut(|d| d.insert_temp(processing_id(), on));
}

/// Ask for the next animation frame: at the display's rate, or 60 fps while a
/// track separates.
pub fn request_frame(ctx: &egui::Context) {
    if ctx
        .data(|d| d.get_temp::<bool>(processing_id()))
        .unwrap_or(false)
    {
        ctx.request_repaint_after(PROCESSING_FRAME);
    } else {
        ctx.request_repaint();
    }
}

/// A value easing from where it was to a new target over a fixed time.
///
/// `ctx.animate_value_with_time` is linear; this takes any easing and restarts
/// from the current position when the target moves mid-flight.
#[derive(Clone, Copy, Default)]
pub struct Tween {
    from: f32,
    to: f32,
    start: f64,
    duration: f32,
    set: bool,
}

impl Tween {
    /// The value now. Requests a repaint while moving.
    pub fn follow(
        ctx: &egui::Context,
        id: egui::Id,
        target: f32,
        duration: f32,
        ease: fn(f32) -> f32,
    ) -> f32 {
        let now = ctx.input(|i| i.time);
        let mut tween = ctx.data(|d| d.get_temp::<Self>(id)).unwrap_or_default();
        if !tween.set {
            tween = Self {
                from: target,
                to: target,
                start: now,
                duration,
                set: true,
            };
        } else if (tween.to - target).abs() > f32::EPSILON {
            tween = Self {
                from: tween.value(now, ease),
                to: target,
                start: now,
                duration,
                set: true,
            };
        }
        ctx.data_mut(|d| d.insert_temp(id, tween));
        let value = tween.value(now, ease);
        if (value - target).abs() > 1e-3 {
            request_frame(ctx);
        }
        value
    }

    fn value(&self, now: f64, ease: fn(f32) -> f32) -> f32 {
        let t = if self.duration > 0.0 {
            ((now - self.start) as f32 / self.duration).clamp(0.0, 1.0)
        } else {
            1.0
        };
        self.from + (self.to - self.from) * ease(t)
    }
}

/// Bind Instrument Sans at three weights, with egui's own fonts behind it for
/// any glyph it lacks.
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let fallbacks = fonts
        .families
        .get(&FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    for weight in Weight::ALL {
        let name = weight.font_name();
        fonts.font_data.insert(
            name.to_owned(),
            Arc::new(
                egui::FontData::from_static(INSTRUMENT_SANS).tweak(FontTweak {
                    coords: VariationCoords::new([("wght", weight.wght())]),
                    ..Default::default()
                }),
            ),
        );
        let mut chain = vec![name.to_owned()];
        chain.extend(fallbacks.iter().cloned());
        fonts.families.insert(weight.family(), chain);
    }
    ctx.set_fonts(fonts);
}

/// Fonts once per context, style every frame.
pub fn install(ctx: &egui::Context) {
    let flag = egui::Id::new("stemd-fonts-installed");
    if !ctx.data(|d| d.get_temp::<bool>(flag).unwrap_or(false)) {
        install_fonts(ctx);
        ctx.data_mut(|d| d.insert_temp(flag, true));
    }
    ctx.set_theme(egui::ThemePreference::Dark);

    let palette = Palette::navy();
    let mut style = (*ctx.style_of(egui::Theme::Dark)).clone();

    style.text_styles = [
        (egui::TextStyle::Small, font(11.0, Weight::Regular)),
        (egui::TextStyle::Body, font(13.0, Weight::Regular)),
        (egui::TextStyle::Button, font(13.0, Weight::Medium)),
        (egui::TextStyle::Heading, font(18.0, Weight::Semibold)),
        (egui::TextStyle::Monospace, FontId::monospace(11.0)),
    ]
    .into();
    style.spacing.item_spacing = vec2(8.0, 8.0);
    style.spacing.button_padding = vec2(14.0, 6.0);
    style.spacing.window_margin = egui::Margin::same(16);
    style.spacing.menu_margin = egui::Margin::same(6);
    style.spacing.interact_size.y = 26.0;

    let visuals = &mut style.visuals;
    visuals.dark_mode = true;
    visuals.panel_fill = Color32::TRANSPARENT;
    visuals.window_fill = palette.menu;
    visuals.extreme_bg_color = black(0.3);
    visuals.faint_bg_color = white(0.04);
    visuals.override_text_color = Some(palette.text);
    visuals.hyperlink_color = palette.blue;
    visuals.selection.bg_fill = Color32::from_rgba_unmultiplied(91, 132, 216, 0x66);
    visuals.selection.stroke = Stroke::new(1.0, palette.text);
    visuals.window_corner_radius = CornerRadius::same(14);
    visuals.menu_corner_radius = CornerRadius::same(14);
    visuals.window_stroke = Stroke::new(1.0, palette.menu_outline);
    visuals.window_shadow = Shadow {
        offset: [0, 16],
        blur: 36,
        spread: 0,
        color: black(0.5),
    };
    visuals.popup_shadow = visuals.window_shadow;

    for widget in [
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
        &mut visuals.widgets.open,
        &mut visuals.widgets.noninteractive,
    ] {
        widget.corner_radius = CornerRadius::same(9);
        widget.bg_stroke = Stroke::new(1.0, white(0.14));
        widget.fg_stroke.color = palette.text;
    }
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, palette.hairline);
    visuals.widgets.inactive.weak_bg_fill = white(0.08);
    visuals.widgets.inactive.bg_fill = white(0.08);
    visuals.widgets.hovered.weak_bg_fill = white(0.16);
    visuals.widgets.hovered.bg_fill = white(0.16);
    visuals.widgets.active.weak_bg_fill = white(0.20);
    visuals.widgets.active.bg_fill = white(0.20);
    visuals.widgets.open.weak_bg_fill = white(0.16);

    ctx.set_style_of(egui::Theme::Dark, style);
}

// ---------------------------------------------------------------------------
// Baked images.

#[derive(Clone, Copy)]
struct Rgba {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Rgba {
    const fn hex(rgb: u32, a: f32) -> Self {
        Self {
            r: ((rgb >> 16) & 0xFF) as f32 / 255.0,
            g: ((rgb >> 8) & 0xFF) as f32 / 255.0,
            b: (rgb & 0xFF) as f32 / 255.0,
            a,
        }
    }

    fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// Source-over onto an opaque `below`.
    fn over(self, below: Self) -> Self {
        Self {
            r: self.r * self.a + below.r * (1.0 - self.a),
            g: self.g * self.a + below.g * (1.0 - self.a),
            b: self.b * self.a + below.b * (1.0 - self.a),
            a: 1.0,
        }
    }

    fn colour(self) -> Color32 {
        let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        Color32::from_rgba_unmultiplied(c(self.r), c(self.g), c(self.b), c(self.a))
    }
}

/// Position `t` along gradient `stops` (offset, colour), clamped at both ends.
fn gradient(stops: &[(f32, Rgba)], t: f32) -> Rgba {
    let (first, last) = (stops[0], stops[stops.len() - 1]);
    if t <= first.0 {
        return first.1;
    }
    for pair in stops.windows(2) {
        let ((a, ca), (b, cb)) = (pair[0], pair[1]);
        if t <= b {
            return ca.lerp(cb, (t - a) / (b - a));
        }
    }
    last.1
}

/// The window background: a navy radial gradient under three faint colour
/// glows (red top left, blue top right, green from below).
///
/// Sized in fractions of the window, so it stretches with the window.
fn backdrop(size: usize) -> egui::ColorImage {
    let base = [
        (0.0, Rgba::hex(0x27335B, 1.0)),
        (0.46, Rgba::hex(0x161D38, 1.0)),
        (1.0, Rgba::hex(0x0A0E1B, 1.0)),
    ];
    // Centre, radii, colour and alpha, bottom layer first.
    let glows = [
        (0.50, 1.12, 0.80, 0.50, 0x6FB47C, 0.12),
        (0.96, 0.04, 0.70, 0.50, 0x5B84D8, 0.22),
        (0.06, 0.00, 0.70, 0.50, 0xD4685F, 0.16),
    ];
    let ellipse = |u: f32, v: f32, cx: f32, cy: f32, rx: f32, ry: f32| {
        (((u - cx) / rx).powi(2) + ((v - cy) / ry).powi(2)).sqrt()
    };
    let mut pixels = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32;
            let v = (y as f32 + 0.5) / size as f32;
            let mut colour = gradient(&base, ellipse(u, v, 0.5, 0.18, 1.2, 0.85));
            for (cx, cy, rx, ry, rgb, alpha) in glows {
                let t = ellipse(u, v, cx, cy, rx, ry) / 0.7;
                let a = alpha * (1.0 - t).max(0.0);
                colour = Rgba {
                    a,
                    ..Rgba::hex(rgb, a)
                }
                .over(colour);
            }
            pixels.push(colour.colour());
        }
    }
    egui::ColorImage::new([size, size], pixels)
}

/// Pastel red, green, blue and back to red, so it tiles.
fn sheen(width: usize) -> egui::ColorImage {
    let stops = [
        (0.0, Rgba::hex(0xF3B9B3, 1.0)),
        (0.33, Rgba::hex(0xC4E8CB, 1.0)),
        (0.66, Rgba::hex(0xBBCEF7, 1.0)),
        (1.0, Rgba::hex(0xF3B9B3, 1.0)),
    ];
    let pixels = (0..width)
        .map(|x| gradient(&stops, x as f32 / width as f32).colour())
        .collect();
    egui::ColorImage::new([width, 1], pixels)
}

/// Three texels: the slice colours, sampled from the first centre to the last.
fn spectrum() -> egui::ColorImage {
    let palette = Palette::navy();
    egui::ColorImage::new([3, 1], vec![palette.red, palette.green, palette.blue])
}

/// `linear-gradient(180deg, rgba(255,255,255,.4) 0%, transparent 60%)`.
fn gloss(height: usize) -> egui::ColorImage {
    let pixels = (0..height)
        .map(|y| {
            let t = (y as f32 + 0.5) / height as f32;
            white(0.4 * (1.0 - t / 0.6).max(0.0))
        })
        .collect();
    egui::ColorImage::new([1, height], pixels)
}

/// The globe's shading over the bounding square of its circle: white top left,
/// clear through the middle, darker at the far edge.
///
/// SVG `radialGradient cx=.36 cy=.26 r=.85` in that square.
fn shade(size: usize) -> egui::ColorImage {
    let mut pixels = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32;
            let v = (y as f32 + 0.5) / size as f32;
            let t = ((u - 0.36).powi(2) + (v - 0.26).powi(2)).sqrt() / 0.85;
            pixels.push(if t < 0.5 {
                white(0.34 * (1.0 - t / 0.5))
            } else {
                black(0.3 * ((t - 0.5) / 0.5).min(1.0))
            });
        }
    }
    egui::ColorImage::new([size, size], pixels)
}

/// The standard normal CDF, close enough for a blur edge.
fn phi(x: f32) -> f32 {
    // Abramowitz and Stegun 7.1.26 on erf.
    let z = x / std::f32::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z.abs());
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_7 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let erf = 1.0 - poly * (-z * z).exp();
    0.5 * (1.0 + erf.copysign(z))
}

/// Fraction of the shadow image's half-size that the unblurred disc covers.
pub const SHADOW_DISC: f32 = 0.6;

/// A black disc of radius `SHADOW_DISC` with a Gaussian edge filling the rest
/// of the image over three deviations.
fn shadow_disc(size: usize) -> egui::ColorImage {
    let half = size as f32 / 2.0;
    let sigma = (1.0 - SHADOW_DISC) / 3.0;
    let mut pixels = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let dx = (x as f32 + 0.5 - half) / half;
            let dy = (y as f32 + 0.5 - half) / half;
            let d = (dx * dx + dy * dy).sqrt();
            pixels.push(black(phi((SHADOW_DISC - d) / sigma)));
        }
    }
    egui::ColorImage::new([size, size], pixels)
}

/// Fraction of the halo image's half-size that the unblurred disc covers.
pub const HALO_DISC: f32 = 0.53;

/// A conic red, green, blue gradient in a disc, blurred at the edge. Full
/// alpha: the caller fades it.
fn halo(size: usize) -> egui::ColorImage {
    let half = size as f32 / 2.0;
    let sigma = (1.0 - HALO_DISC) / 3.0;
    let stops = [
        (0.0, Rgba::hex(0xD4685F, 1.0)),
        (1.0 / 3.0, Rgba::hex(0x6FB47C, 1.0)),
        (2.0 / 3.0, Rgba::hex(0x5B84D8, 1.0)),
        (1.0, Rgba::hex(0xD4685F, 1.0)),
    ];
    let mut pixels = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let dx = (x as f32 + 0.5 - half) / half;
            let dy = (y as f32 + 0.5 - half) / half;
            let d = (dx * dx + dy * dy).sqrt();
            // Clockwise from twelve o'clock, as CSS `conic-gradient`.
            let turn = (dx.atan2(-dy) / std::f32::consts::TAU).rem_euclid(1.0);
            let colour = gradient(&stops, turn);
            let a = phi((HALO_DISC - d) / sigma);
            pixels.push(Rgba { a, ..colour }.colour());
        }
    }
    egui::ColorImage::new([size, size], pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Color32` whose channels exceed its alpha is additive in epaint, not
    /// translucent: it adds its own brightness on top of whatever is behind it.
    /// `from_rgba_unmultiplied`, `white` and `black` spell the intended meaning.
    #[test]
    fn no_palette_colour_is_secretly_additive() {
        let palette = Palette::navy();
        let glass = [Glass::Panel, Glass::List]
            .into_iter()
            .map(Glass::spec)
            .flat_map(|g| [g.top, g.bottom, g.outline, g.highlight])
            .map(white);
        for colour in palette.translucent().into_iter().chain(glass) {
            let alpha = colour.a();
            for channel in [colour.r(), colour.g(), colour.b()] {
                assert!(
                    channel <= alpha,
                    "{colour:?} has a channel above its alpha, so it adds light \
                     instead of blending"
                );
            }
        }
    }

    /// The baked images carry the same guarantee, texel by texel.
    #[test]
    fn no_baked_texel_is_additive() {
        for (name, image) in [
            ("gloss", gloss(16)),
            ("shade", shade(32)),
            ("shadow", shadow_disc(32)),
            ("halo", halo(32)),
        ] {
            for texel in &image.pixels {
                assert!(
                    texel.r() <= texel.a() && texel.g() <= texel.a() && texel.b() <= texel.a(),
                    "{name}: {texel:?} is additive"
                );
            }
        }
    }

    /// Text has to stay readable on the darkest and lightest part of the backdrop.
    #[test]
    fn text_is_not_the_colour_of_what_it_sits_on() {
        let palette = Palette::navy();
        let image = backdrop(32);
        for texel in &image.pixels {
            for text in [palette.text, palette.muted] {
                let gap = i32::from(text.g()) - i32::from(texel.g());
                assert!(gap > 100, "{text:?} on {texel:?} is too close to it");
            }
        }
    }

    #[test]
    fn the_backdrop_is_lightest_near_the_top_centre() {
        let image = backdrop(64);
        let at = |x: usize, y: usize| image.pixels[y * 64 + x];
        let top = at(32, 11);
        let bottom = at(32, 63);
        assert!(
            top.b() > bottom.b(),
            "{top:?} is not lighter than {bottom:?}"
        );
    }

    #[test]
    fn css_easings_hit_their_ends_and_known_points() {
        for ease in [ease_spring, ease_out, ease_in, ease_in_out] {
            assert!(ease(0.0).abs() < 1e-3);
            assert!((ease(1.0) - 1.0).abs() < 1e-3);
        }
        // ease-in-out is symmetric about its middle.
        assert!((ease_in_out(0.5) - 0.5).abs() < 1e-3);
        // The spring is well past half by a third of the way.
        assert!(ease_spring(0.33) > 0.7);
    }

    #[test]
    fn a_rounded_outline_closes_and_stays_inside_its_rect() {
        let rect = Rect::from_min_size(pos2(10.0, 20.0), vec2(200.0, 80.0));
        let points = rounded_outline(rect, 22.0, 8);
        assert_eq!(points.first(), points.last());
        for p in &points {
            assert!(rect.expand(0.01).contains(*p), "{p:?} is outside {rect:?}");
        }
    }

    #[test]
    fn the_normal_cdf_is_a_cdf() {
        assert!((phi(0.0) - 0.5).abs() < 1e-4);
        assert!(phi(3.0) > 0.998 && phi(-3.0) < 0.002);
    }
}
