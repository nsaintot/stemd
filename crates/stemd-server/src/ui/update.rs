//! The version button and the update slot in the title bar, and the install
//! question. See [`crate::update`].
//!
//! The slot sits in the corner away from the window buttons. A status slides
//! in from below and the one it replaces leaves upwards.

use std::f32::consts::TAU;
use std::mem::discriminant;
use std::sync::Arc;
use std::time::Duration;

use eframe::egui::{
    self, Align2, Color32, CursorIcon, Galley, Pos2, Rect, Sense, Shape, Stroke, Vec2, pos2, vec2,
};

use crate::update::{self, Status};

use super::theme::{self, Brushes, Palette, Weight, white};
use super::widgets::{self, Pill};

/// Size of the version's text, and of the slot's.
const VERSION_TEXT: f32 = 12.0;
const SLOT_TEXT: f32 = 12.5;

/// The version button: height, side padding, and its icon's size and gap.
const BUTTON_HEIGHT: f32 = 20.0;
const PAD: f32 = 6.0;
const ICON: f32 = 10.0;
const ICON_GAP: f32 = 5.0;

/// Gap between a status's text and its icon.
const SLOT_GAP: f32 = 6.0;

/// A status change: fade time and slide distance.
const CHANGE_SECS: f64 = 0.28;
const SLIDE: f32 = 7.0;

/// Colours of the slot's icons, and the stops of "New update available!".
const ARROW: Color32 = Color32::from_rgb(0xBB, 0xCE, 0xF7);
const SHEEN: [Color32; 4] = [
    Color32::from_rgb(0xF3, 0xB9, 0xB3),
    Color32::from_rgb(0xC4, 0xE8, 0xCB),
    Color32::from_rgb(0xBB, 0xCE, 0xF7),
    Color32::from_rgb(0xF3, 0xB9, 0xB3),
];

/// Frame interval of the slot's looping animations.
const FRAME: Duration = Duration::from_millis(33);

/// How long the gradient and arrow move after an update appears: 7 cycles of
/// 3 s and 15 of 1.4 s, so both stop where they started.
const SHEEN_SECS: f64 = 21.0;

fn version_galley(ui: &egui::Ui) -> Arc<Galley> {
    ui.painter().layout_no_wrap(
        format!("v{}", update::VERSION),
        theme::font(VERSION_TEXT, Weight::Regular),
        Color32::PLACEHOLDER,
    )
}

/// Width of the version button.
pub fn version_width(ui: &egui::Ui) -> f32 {
    let icon = if update::present() {
        ICON_GAP + ICON
    } else {
        0.0
    };
    PAD + version_galley(ui).size().x + icon + PAD
}

/// Draw the version button with its left edge at `left`, centred on `y`. A
/// click checks for updates. `dim` fades it with an unfocused window.
pub fn version(ui: &mut egui::Ui, left: f32, y: f32, status: &Status, palette: &Palette, dim: f32) {
    let galley = version_galley(ui);
    let rect = Rect::from_min_size(
        pos2(left, y - BUTTON_HEIGHT / 2.0),
        vec2(version_width(ui), BUTTON_HEIGHT),
    );
    let can = update::can_check();
    let busy = *status == Status::Checking;
    let id = ui.id().with("update-version");
    let response = ui.interact(rect, id, if can { Sense::click() } else { Sense::hover() });
    let hovered = can && response.hovered();
    let hover = widgets::fade(ui, id, hovered);
    let lit = if busy { 1.0 } else { hover };
    let painter = ui.painter();
    painter.rect_filled(rect, BUTTON_HEIGHT / 2.0, white(0.09 * hover * dim));
    let ink = theme::mix(palette.muted, Color32::WHITE, lit).gamma_multiply(dim);
    let text_pos = pos2(rect.left() + PAD, y - galley.size().y / 2.0);
    let text_right = text_pos.x + galley.size().x;
    painter.galley(text_pos, galley, ink);
    if update::present() {
        let time = ui.input(|i| i.time);
        let turn = ui
            .ctx()
            .animate_bool_with_time(id.with("turn"), hovered, 0.35);
        let angle = if busy {
            ui.ctx().request_repaint_after(FRAME);
            (time as f32 / 0.8).fract() * TAU
        } else {
            ease(turn) * TAU / 2.0
        };
        let centre = pos2(text_right + ICON_GAP + ICON / 2.0, y + 1.0);
        let alpha = 0.32 + 0.68 * lit;
        refresh_icon(painter, centre, angle, ink.gamma_multiply(alpha));
    }
    if can {
        let response = response
            .on_hover_text("Check for updates")
            .on_hover_cursor(CursorIcon::PointingHand);
        if response.clicked() {
            update::check();
        }
    }
}

/// What the slot shows, and what it showed before the last change.
#[derive(Clone)]
struct Shown {
    now: Status,
    before: Status,
    since: f64,
}

/// Draw the update slot aligned to `anchor`. A click on an update acts on
/// it.
pub fn slot(ui: &mut egui::Ui, anchor: Pos2, align: Align2, palette: &Palette, dim: f32) {
    let status = update::status();
    let id = ui.id().with("update-slot");
    let time = ui.input(|i| i.time);
    let mut shown = ui.data(|d| d.get_temp::<Shown>(id)).unwrap_or(Shown {
        now: Status::Idle,
        before: Status::Idle,
        since: f64::NEG_INFINITY,
    });
    if discriminant(&status) != discriminant(&shown.now) {
        shown.before = std::mem::replace(&mut shown.now, status);
        shown.since = time;
    } else {
        shown.now = status;
    }
    ui.data_mut(|d| d.insert_temp(id, shown.clone()));

    let t = ((time - shown.since) / CHANGE_SECS).clamp(0.0, 1.0) as f32;
    let e = ease(t);
    if t < 1.0 {
        ui.ctx().request_repaint();
        let leaving = vec2(0.0, -SLIDE * e);
        item(
            ui,
            &shown.before,
            anchor + leaving,
            align,
            (1.0 - e) * dim,
            f64::INFINITY,
            palette,
            None,
        );
    }
    let entering = vec2(0.0, SLIDE * (1.0 - e));
    item(
        ui,
        &shown.now,
        anchor + entering,
        align,
        e * dim,
        time - shown.since,
        palette,
        Some(id),
    );
}

/// Width of `status` in the slot.
fn item_width(ui: &egui::Ui, status: &Status) -> f32 {
    let Some((text, icon)) = parts(status) else {
        return 0.0;
    };
    let text = ui
        .painter()
        .layout_no_wrap(text, slot_font(), Color32::PLACEHOLDER)
        .size()
        .x;
    let dots = if *status == Status::Checking {
        dots_width(ui)
    } else {
        0.0
    };
    text + dots + icon.map_or(0.0, |icon| SLOT_GAP + icon.size())
}

/// The icons of the slot.
#[derive(Clone, Copy, PartialEq)]
enum Icon {
    Tick,
    Cross,
    Arrow,
}

impl Icon {
    fn size(self) -> f32 {
        match self {
            Self::Arrow => 11.0,
            Self::Tick | Self::Cross => 12.0,
        }
    }

    /// Whether the icon comes before the text.
    fn leads(self) -> bool {
        self != Self::Arrow
    }
}

/// The text of `status` and its icon. `None` for an empty slot.
fn parts(status: &Status) -> Option<(String, Option<Icon>)> {
    Some(match status {
        Status::Idle => return None,
        Status::Checking => ("Checking".to_owned(), None),
        Status::UpToDate => ("Up to date".to_owned(), Some(Icon::Tick)),
        Status::Failed { title, .. } => ((*title).to_owned(), Some(Icon::Cross)),
        Status::Available(_) => ("New update available!".to_owned(), Some(Icon::Arrow)),
        Status::Downloading(f) => (
            format!("Downloading {}", super::status::percent(*f)),
            Some(Icon::Arrow),
        ),
        Status::Ready => ("Update ready".to_owned(), Some(Icon::Arrow)),
    })
}

fn slot_font() -> egui::FontId {
    theme::font(SLOT_TEXT, Weight::Semibold)
}

/// The tooltip, where there is one.
fn hint(status: &Status) -> Option<String> {
    match status {
        Status::Failed { error, .. } => Some(error.clone()),
        Status::Available(version) => Some(if cfg!(target_os = "macos") {
            format!("Show Stemd {version}")
        } else if cfg!(windows) {
            format!("Download and install Stemd {version}")
        } else {
            format!("Open the Stemd {version} release page")
        }),
        Status::Ready => Some("Installs once nothing is running, or when Stemd quits".to_owned()),
        _ => None,
    }
}

/// Draw one status at `alpha`, shown for `age` seconds. `click` is the id to
/// sense clicks with; the status leaving the slot gets none.
#[allow(clippy::too_many_arguments)]
fn item(
    ui: &mut egui::Ui,
    status: &Status,
    anchor: Pos2,
    align: Align2,
    alpha: f32,
    age: f64,
    palette: &Palette,
    click: Option<egui::Id>,
) {
    let Some((text, icon)) = parts(status) else {
        return;
    };
    let size = vec2(item_width(ui, status), BUTTON_HEIGHT);
    let rect = align.anchor_size(anchor, size);
    let time = ui.input(|i| i.time);
    let y = rect.center().y;
    let sheen = status.sheen();
    let hovered = click.is_some() && ui.rect_contains_pointer(rect.expand(3.0));
    // Move for SHEEN_SECS, then only under the pointer, so an update left for
    // days does not keep the window repainting.
    let clock = if age < SHEEN_SECS {
        Some(age)
    } else if hovered {
        Some(time)
    } else {
        None
    };
    let moving = sheen && clock.is_some();
    if moving || *status == Status::Checking {
        ui.ctx().request_repaint_after(FRAME);
    }

    let galley = ui
        .painter()
        .layout_no_wrap(text, slot_font(), Color32::PLACEHOLDER);
    let galley = if sheen {
        let phase = clock.map_or(0.0, |t| (t / 3.0).fract()) as f32;
        sheened(&galley, phase, alpha)
    } else {
        galley
    };
    let mut x = rect.left();
    if let Some(icon) = icon.filter(|icon| icon.leads()) {
        draw_icon(
            ui.painter(),
            pos2(x, y - icon.size() / 2.0),
            icon,
            palette,
            alpha,
        );
        x += icon.size() + SLOT_GAP;
    }
    let text_pos = pos2(x, y - galley.size().y / 2.0);
    x += galley.size().x;
    ui.painter()
        .galley(text_pos, galley, palette.text.gamma_multiply(alpha));

    if *status == Status::Checking {
        dots(
            ui,
            pos2(x, text_pos.y),
            time,
            palette.text.gamma_multiply(alpha),
        );
    }
    if let Some(icon) = icon.filter(|icon| !icon.leads()) {
        let nudge = match clock.filter(|_| sheen) {
            // Up 1 px to down 2 px and back, every 1.4 s.
            Some(t) => -1.0 + 3.0 * (0.5 - 0.5 * (t as f32 / 1.4 * TAU).cos()),
            None => -1.0,
        };
        let origin = pos2(x + SLOT_GAP, y - icon.size() / 2.0 + nudge);
        draw_icon(ui.painter(), origin, icon, palette, alpha);
    }

    if let Some(id) = click {
        let opens = status.opens();
        let mut response = ui.interact(
            rect.expand(3.0),
            id,
            if opens {
                Sense::click()
            } else {
                Sense::hover()
            },
        );
        if let Some(hint) = hint(status) {
            response = response.on_hover_text(hint);
        }
        if opens {
            response = response.on_hover_cursor(CursorIcon::PointingHand);
            if response.clicked() {
                update::open();
            }
        }
    }
}

/// `galley` coloured with the moving gradient of an update: a gradient 2.2
/// times the text's width, scrolled by `phase` (0 to 1) of its length.
fn sheened(galley: &Galley, phase: f32, alpha: f32) -> Arc<Galley> {
    let mut galley = galley.clone();
    let width = galley.size().x.max(1.0);
    for placed in &mut galley.rows {
        let left = placed.pos.x;
        let row = Arc::make_mut(&mut placed.row);
        for vertex in &mut row.visuals.mesh.vertices {
            let u = ((left + vertex.pos.x) / width / 2.2 + phase).fract();
            vertex.color = gradient(u).gamma_multiply(alpha);
        }
    }
    Arc::new(galley)
}

/// The sheen's colour at `u`, 0 to 1. Stops at 0, 1/3, 2/3 and 1.
fn gradient(u: f32) -> Color32 {
    let scaled = u.clamp(0.0, 1.0) * 3.0;
    let i = (scaled as usize).min(2);
    theme::mix(SHEEN[i], SHEEN[i + 1], scaled - i as f32)
}

fn dots_width(ui: &egui::Ui) -> f32 {
    ui.painter()
        .layout_no_wrap("...".to_owned(), slot_font(), Color32::PLACEHOLDER)
        .size()
        .x
}

/// The three dots after "Checking", each brightening in turn every 1.2 s.
fn dots(ui: &egui::Ui, mut pos: Pos2, time: f64, colour: Color32) {
    for n in 0..3 {
        let p = ((time - 0.15 * n as f64) / 1.2).rem_euclid(1.0) as f32;
        // Keyframes: 0.25 at 0%, 1 at 30%, 0.25 from 60%.
        let rise = if p < 0.3 {
            ease_in_out(p / 0.3)
        } else if p < 0.6 {
            1.0 - ease_in_out((p - 0.3) / 0.3)
        } else {
            0.0
        };
        let galley = ui
            .painter()
            .layout_no_wrap(".".to_owned(), slot_font(), Color32::PLACEHOLDER);
        let advance = galley.size().x;
        ui.painter()
            .galley(pos, galley, colour.gamma_multiply(0.25 + 0.75 * rise));
        pos.x += advance;
    }
}

/// The refresh icon on a 10 by 10 grid, turned by `angle`: a circle open at
/// the top right, with an arrowhead there.
fn refresh_icon(painter: &egui::Painter, centre: Pos2, angle: f32, colour: Color32) {
    let (sin, cos) = angle.sin_cos();
    let at = |x: f32, y: f32| {
        let (dx, dy) = (x - 5.0, y - 5.0);
        centre + vec2(dx * cos - dy * sin, dx * sin + dy * cos)
    };
    let stroke = Stroke::new(0.9, colour);
    // From 3 o'clock, clockwise, to the arrowhead at about 1:30.
    let end = 313.3_f32.to_radians();
    let arc: Vec<Pos2> = (0..=32)
        .map(|n| {
            let a = end * n as f32 / 32.0;
            at(5.0 + 3.5 * a.cos(), 5.0 + 3.5 * a.sin())
        })
        .collect();
    painter.add(Shape::line(arc, stroke));
    painter.add(Shape::line(
        vec![at(8.7, 1.3), at(8.7, 3.6), at(6.4, 3.6)],
        stroke,
    ));
}

/// Draw `icon` from its 12 by 12 grid, scaled to its size, at `origin`.
fn draw_icon(painter: &egui::Painter, origin: Pos2, icon: Icon, palette: &Palette, alpha: f32) {
    let scale = icon.size() / 12.0;
    let colour = match icon {
        Icon::Tick => palette.good,
        Icon::Cross => palette.danger,
        Icon::Arrow => ARROW,
    };
    let stroke = Stroke::new(1.8 * scale, colour.gamma_multiply(alpha));
    let paths: &[&[(f32, f32)]] = match icon {
        Icon::Tick => &[&[(2.5, 6.4), (5.0, 8.8), (9.5, 3.4)]],
        Icon::Cross => &[&[(3.0, 3.0), (9.0, 9.0)], &[(9.0, 3.0), (3.0, 9.0)]],
        Icon::Arrow => &[
            &[(6.0, 1.8), (6.0, 8.8)],
            &[(3.0, 6.2), (6.0, 9.2), (9.0, 6.2)],
        ],
    };
    for path in paths {
        let points: Vec<Pos2> = path
            .iter()
            .map(|&(x, y)| origin + Vec2::new(x, y) * scale)
            .collect();
        // Round caps.
        for end in [points[0], points[points.len() - 1]] {
            painter.circle_filled(end, stroke.width / 2.0, stroke.color);
        }
        painter.add(Shape::line(points, stroke));
    }
}

/// CSS's cubic-bezier(.3, .8, .3, 1), near enough: a fast start that settles.
fn ease(t: f32) -> f32 {
    1.0 - (1.0 - t.clamp(0.0, 1.0)).powi(3)
}

fn ease_in_out(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The install question, while it is asked, and closing the window once Setup
/// has started. `idle` says nothing is running, queued or downloading.
pub fn dialog(ctx: &egui::Context, idle: bool, palette: &Palette, brushes: &Brushes) {
    if update::tick(idle) {
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }
    let Some(version) = update::asking() else {
        return;
    };
    let mut answer = None;
    egui::Modal::new(egui::Id::new("stemd-update"))
        .frame(egui::Frame::NONE)
        .show(ctx, |ui| {
            super::blur::frosted(ui, &widgets::dialog_look(palette), |ui| {
                answer = question(ui, palette, brushes, &version);
            });
        });
    if let Some(install) = answer {
        update::answer(install);
    }
}

/// The question's contents. Returns true to install, false for not now.
fn question(
    ui: &mut egui::Ui,
    palette: &Palette,
    brushes: &Brushes,
    version: &str,
) -> Option<bool> {
    ui.set_max_width(320.0);
    ui.label(
        egui::RichText::new(format!("Stemd {version}"))
            .font(theme::font(17.0, Weight::Semibold))
            .color(palette.text),
    );
    ui.add_space(6.0);
    ui.label(
        egui::RichText::new(
            "Download the installer and run it? Once the download is done and \
             nothing is running, Stemd quits, installs the update and starts \
             again.",
        )
        .font(theme::font(13.0, Weight::Regular))
        .color(palette.muted),
    );
    ui.add_space(16.0);
    let mut answer = None;
    ui.horizontal(|ui| {
        for (label, kind, install) in [
            ("Not now", Pill::Ghost, false),
            ("Download and install", Pill::Light, true),
        ] {
            let width = widgets::pill_width(ui, label, 30.0);
            let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 30.0), Sense::hover());
            let id = ui.id().with(("update-answer", label));
            if widgets::pill(ui, rect, id, label, kind, true, palette, brushes).clicked() {
                answer = Some(install);
            }
        }
    });
    answer
}

/// Gap kept on each side of the slot in the status bar.
#[cfg(all(unix, not(target_os = "macos")))]
const SPACE: f32 = 14.0;

/// Whether `status` fits between `left` and `right` in the status bar, and
/// where it starts if so.
#[cfg(all(unix, not(target_os = "macos")))]
pub fn fits(ui: &egui::Ui, status: &Status, left: f32, right: f32) -> Option<f32> {
    (left + SPACE + item_width(ui, status) + SPACE <= right).then_some(left + SPACE)
}
