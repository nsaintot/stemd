//! Colour, shape and the two painting helpers everything else is built from.
//!
//! The palette is Apple's system colours rather than invented ones, so the
//! window sits next to native apps instead of near them, and it comes in both
//! appearances because a Mac that switches to dark at sunset expects this to
//! follow.

use eframe::egui::{
    self, Color32, CornerRadius, Rect, Stroke, StrokeKind, TextureHandle, TextureOptions,
};

/// Corner radii, in the two sizes this window uses.
pub const CARD_RADIUS: u8 = 10;
pub const ZONE_RADIUS: u8 = 16;

pub struct Palette {
    /// Behind everything.
    pub window: Color32,
    /// Raised surfaces: the drop zone, a row in the list.
    pub card: Color32,
    /// The same, one step lighter, for a hover.
    pub card_hover: Color32,
    /// Hairline around a surface. Low alpha on purpose: an outline you notice
    /// is an outline that is too strong.
    pub outline: Color32,
    pub text: Color32,
    pub muted: Color32,
    pub accent: Color32,
    pub good: Color32,
    pub warn: Color32,
    pub bad: Color32,
    /// Top and bottom of the gradient on a raised surface.
    pub sheen: (Color32, Color32),
}

impl Palette {
    pub fn dark() -> Self {
        Self {
            window: Color32::from_rgb(0x1C, 0x1C, 0x1E),
            card: Color32::from_rgb(0x2A, 0x2A, 0x2E),
            card_hover: Color32::from_rgb(0x33, 0x33, 0x38),
            outline: Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 0x14),
            text: Color32::from_rgb(0xEC, 0xEC, 0xEE),
            muted: Color32::from_rgb(0x98, 0x98, 0x9F),
            accent: Color32::from_rgb(0x0A, 0x84, 0xFF),
            good: Color32::from_rgb(0x30, 0xD1, 0x58),
            warn: Color32::from_rgb(0xFF, 0x9F, 0x0A),
            bad: Color32::from_rgb(0xFF, 0x45, 0x3A),
            sheen: (
                Color32::from_rgba_unmultiplied(0xFF, 0xFF, 0xFF, 0x0C),
                Color32::TRANSPARENT,
            ),
        }
    }

    pub fn light() -> Self {
        Self {
            window: Color32::from_rgb(0xF2, 0xF2, 0xF7),
            card: Color32::from_rgb(0xFF, 0xFF, 0xFF),
            card_hover: Color32::from_rgb(0xFA, 0xFA, 0xFC),
            outline: Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, 0x1A),
            text: Color32::from_rgb(0x1C, 0x1C, 0x1E),
            muted: Color32::from_rgb(0x6E, 0x6E, 0x73),
            accent: Color32::from_rgb(0x00, 0x7A, 0xFF),
            good: Color32::from_rgb(0x34, 0xC7, 0x59),
            warn: Color32::from_rgb(0xFF, 0x95, 0x00),
            bad: Color32::from_rgb(0xFF, 0x3B, 0x30),
            // Light from above means the *bottom* darkens here, so the top is
            // nothing at all rather than white-at-zero, which is additive white
            // however transparent it claims to be.
            sheen: (
                Color32::TRANSPARENT,
                Color32::from_rgba_unmultiplied(0x00, 0x00, 0x00, 0x0A),
            ),
        }
    }

    pub fn of(ctx: &egui::Context) -> Self {
        if is_dark(ctx) {
            Self::dark()
        } else {
            Self::light()
        }
    }

    /// The same colour at a lower alpha, for a tint behind text of that colour.
    pub fn wash(colour: Color32, alpha: u8) -> Color32 {
        Color32::from_rgba_unmultiplied(colour.r(), colour.g(), colour.b(), alpha)
    }
}

/// A two-pixel texture sampled across a shape, which is how a rounded rectangle
/// gets a gradient.
///
/// `RectShape` fills with one flat colour, and a gradient painted as a mesh behind
/// it would square off the corners the rounding just made. A brush is tessellated
/// with the shape, so the gradient is clipped to the rounded outline exactly.
pub struct Sheen {
    texture: TextureHandle,
    /// The appearance it was built for. Rebuilt when the system switches.
    dark: bool,
}

impl Sheen {
    pub fn new(ctx: &egui::Context) -> Self {
        let dark = is_dark(ctx);
        Self {
            texture: build(ctx, dark),
            dark,
        }
    }

    /// The texture for the current appearance, rebuilding it if it changed.
    fn texture(&mut self, ctx: &egui::Context) -> &TextureHandle {
        let dark = is_dark(ctx);
        if dark != self.dark {
            self.texture = build(ctx, dark);
            self.dark = dark;
        }
        &self.texture
    }

    /// A raised surface: flat fill, gradient over it, hairline around it.
    pub fn card(&mut self, ui: &egui::Ui, rect: Rect, radius: u8, fill: Color32, stroke: Stroke) {
        let painter = ui.painter();
        painter.rect_filled(rect, radius, fill);
        painter.add(
            egui::epaint::RectShape::filled(rect, radius, Color32::WHITE).with_texture(
                self.texture(ui.ctx()).id(),
                Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            ),
        );
        painter.rect_stroke(rect, radius, stroke, StrokeKind::Inside);
    }
}

/// Which appearance the system is asking for.
fn is_dark(ctx: &egui::Context) -> bool {
    ctx.theme() == egui::Theme::Dark
}

/// Top row transparent-to-white, bottom row transparent: light from above.
fn build(ctx: &egui::Context, dark: bool) -> TextureHandle {
    let palette = if dark {
        Palette::dark()
    } else {
        Palette::light()
    };
    let (top, bottom) = palette.sheen;
    ctx.load_texture(
        "stemd-sheen",
        egui::ColorImage::new([1, 2], vec![top, bottom]),
        TextureOptions::LINEAR,
    )
}

/// Make egui look like it belongs on this platform.
///
/// egui's defaults are square, high-contrast and tight. Rounding, spacing and a
/// palette that is not pure black do most of the work of making it not look like
/// a debug window.
pub fn install(ctx: &egui::Context) {
    let palette = Palette::of(ctx);
    let theme = ctx.theme();
    let mut style = (*ctx.style_of(theme)).clone();

    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    style.spacing.window_margin = egui::Margin::same(12);
    style.spacing.menu_margin = egui::Margin::same(8);
    style.spacing.interact_size.y = 24.0;

    let visuals = &mut style.visuals;
    visuals.panel_fill = palette.window;
    visuals.window_fill = palette.card;
    visuals.extreme_bg_color = palette.window;
    visuals.override_text_color = Some(palette.text);
    visuals.hyperlink_color = palette.accent;
    visuals.selection.bg_fill = Palette::wash(palette.accent, 0x55);
    visuals.selection.stroke = Stroke::new(1.0, palette.text);
    visuals.window_corner_radius = CornerRadius::same(CARD_RADIUS);
    visuals.menu_corner_radius = CornerRadius::same(CARD_RADIUS);
    visuals.window_stroke = Stroke::new(1.0, palette.outline);
    visuals.window_shadow = egui::epaint::Shadow {
        offset: [0, 8],
        blur: 24,
        spread: 0,
        color: Color32::from_black_alpha(if visuals.dark_mode { 0x66 } else { 0x22 }),
    };
    visuals.popup_shadow = visuals.window_shadow;

    for widget in [
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
        &mut visuals.widgets.open,
        &mut visuals.widgets.noninteractive,
    ] {
        widget.corner_radius = CornerRadius::same(7);
        widget.bg_stroke = Stroke::new(1.0, palette.outline);
        widget.fg_stroke.color = palette.text;
    }
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, palette.outline);
    visuals.widgets.inactive.weak_bg_fill = palette.card;
    visuals.widgets.hovered.weak_bg_fill = palette.card_hover;
    visuals.widgets.active.weak_bg_fill = palette.card_hover;
    visuals.widgets.open.weak_bg_fill = palette.card_hover;

    ctx.set_style_of(theme, style);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Color32` whose channels exceed its alpha is additive in epaint, not
    /// translucent: it adds its own brightness on top of whatever is behind it.
    ///
    /// `from_rgba_premultiplied(0xFF, 0xFF, 0xFF, 0x0E)` painted the drop zone as a
    /// near-solid white ramp, because full-intensity white was added at every pixel.
    /// `from_rgba_unmultiplied` spells the intended meaning, and nothing about the
    /// wrong version looks wrong in the source.
    #[test]
    fn no_palette_colour_is_secretly_additive() {
        for (name, palette) in [("dark", Palette::dark()), ("light", Palette::light())] {
            let (top, bottom) = palette.sheen;
            for colour in [palette.outline, top, bottom] {
                let alpha = colour.a();
                for channel in [colour.r(), colour.g(), colour.b()] {
                    assert!(
                        channel <= alpha,
                        "{name}: {colour:?} has a channel above its alpha, so it \
                         adds light instead of blending — use from_rgba_unmultiplied"
                    );
                }
            }
        }
    }

    /// Both appearances have to define every colour: a `None` here would be a
    /// hole in one theme that the other hides.
    #[test]
    fn the_two_appearances_are_genuinely_different() {
        let (dark, light) = (Palette::dark(), Palette::light());
        assert_ne!(dark.window, light.window);
        assert_ne!(dark.text, light.text);
        assert_ne!(dark.card, light.card);
    }

    /// Text has to stay readable on the surface it sits on. Not a contrast
    /// audit, just a guard against a palette edit that swaps a pair.
    #[test]
    fn text_is_not_the_colour_of_what_it_sits_on() {
        for palette in [Palette::dark(), Palette::light()] {
            for surface in [palette.window, palette.card, palette.card_hover] {
                let gap = i32::from(palette.text.r()) - i32::from(surface.r());
                assert!(gap.abs() > 60, "text on {surface:?} is too close to it");
            }
        }
    }

    /// `Color32` stores premultiplied in eight bits, so a wash cannot be exact:
    /// a dark colour at low alpha is quantised on the way in and does not
    /// unmultiply back to itself. Close is the guarantee, and all a tint needs.
    #[test]
    fn a_wash_keeps_the_colour_and_only_drops_the_alpha() {
        let colour = Color32::from_rgb(10, 20, 30);
        assert_eq!(
            Palette::wash(colour, 0xFF),
            colour,
            "opaque changes nothing"
        );

        let faint = Palette::wash(colour, 0x40);
        assert_eq!(faint.a(), 0x40, "the alpha asked for is the alpha stored");
        let back = faint.to_srgba_unmultiplied();
        for (channel, want) in back.iter().zip([10, 20, 30]) {
            let drift = i32::from(*channel) - want;
            assert!(drift.abs() <= 4, "{back:?} is not the colour that went in");
        }
    }
}
