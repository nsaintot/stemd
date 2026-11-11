//! What the window shows until the server is ready: on a first run the choice of
//! model and its download, then the model loading, or why startup failed.

use eframe::egui::{self, Rect, Sense, Stroke, pos2, vec2};

use crate::launch::{Launch, Stage};
use crate::models::Preset;

use super::globe::{self, Motion};
use super::icons;
use super::theme::{self, Brushes, Glass, Palette, Weight, white};
use super::widgets::{self, Column, Pill};

const ROW_HEIGHT: f32 = 68.0;
const ROW_GAP: f32 = 8.0;
const ROW_RADIUS: f32 = 18.0;
const QUIT_HEIGHT: f32 = 32.0;

/// State the first-run view keeps between frames.
#[derive(Default)]
pub struct View {
    motion: Motion,
    /// Set once the choice has been offered, so the rows stay while the picked
    /// model downloads and loads.
    offered: bool,
    picked: Option<Preset>,
    /// The last stage drawn, for the main window to fade out over itself.
    pub(super) last: Option<Stage>,
    /// Where the globe sat and how far its slices were spread, for the main
    /// window's globe to start from.
    pub(super) globe: Option<(Rect, f32)>,
    /// The main window draws the globe now; this view draws everything else.
    pub(super) globe_handed_off: bool,
}

impl View {
    /// The globe's motion, so the main window carries on from it.
    pub(super) fn take_motion(&mut self) -> Motion {
        std::mem::take(&mut self.motion)
    }
}

pub fn show(ui: &mut egui::Ui, launch: &Launch, view: &mut View, brushes: &Brushes) {
    let ctx = ui.ctx().clone();
    super::theme::install(&ctx);
    let palette = Palette::of(&ctx);
    let stage = launch.stage();

    brushes.backdrop(ui.painter(), ctx.content_rect());

    let busy = matches!(
        stage,
        Stage::Downloading { .. } | Stage::Verifying(_) | Stage::Loading(_)
    );
    let drive = globe::Drive {
        variant: busy.then_some(0),
        speed: 1.0,
        spin: busy.then_some(3.0),
    };
    let (dt, now) = (globe::frame_dt(&ctx), ctx.input(|i| i.time));
    if view.motion.step(dt, now, &drive) {
        ctx.request_repaint();
    } else {
        ctx.request_repaint_after(super::REPAINT_INTERVAL);
    }

    if let Some(preset) = body(ui, &stage, &palette, brushes, view) {
        launch.pick(preset);
    }
}

/// The preset `stage` is about, if any.
fn preset_of(stage: &Stage) -> Option<Preset> {
    match stage {
        Stage::Downloading { preset, .. } | Stage::Verifying(preset) => Some(*preset),
        Stage::Loading(name) => Preset::ALL.into_iter().find(|p| p.label() == name),
        _ => None,
    }
}

const fn tag(preset: Preset) -> &'static str {
    match preset {
        Preset::Fast => "speed",
        Preset::Balanced => "middle ground",
        Preset::Quality => "best separation",
    }
}

fn megabytes(bytes: u64) -> String {
    format!("{:.0} MB", bytes as f64 / 1e6)
}

/// Draw `stage` into the whole of `ui`. Returns the preset picked this frame.
pub(super) fn body(
    ui: &mut egui::Ui,
    stage: &Stage,
    palette: &Palette,
    brushes: &Brushes,
    view: &mut View,
) -> Option<Preset> {
    if matches!(stage, Stage::Choosing(_)) {
        view.offered = true;
    }
    if let Some(preset) = preset_of(stage) {
        view.picked = Some(preset);
    }
    view.last = Some(stage.clone());
    let area = ui.max_rect();
    let inner = Rect::from_min_max(area.min + vec2(24.0, 26.0), area.max - vec2(24.0, 24.0));
    let mid = inner.center().x;
    let ctx = ui.ctx().clone();
    let now = ctx.input(|i| i.time);
    let busy = matches!(
        stage,
        Stage::Downloading { .. } | Stage::Verifying(_) | Stage::Loading(_)
    );

    let picked = view.picked;
    let (title, sub) = match stage {
        Stage::Starting => ("Starting Stemd".to_owned(), "Checking the GPU".to_owned()),
        Stage::Choosing(installed) if installed.is_empty() => (
            "Choose a model".to_owned(),
            "No model is installed yet. Pick one to download.".to_owned(),
        ),
        Stage::Choosing(_) => (
            "Choose a model".to_owned(),
            "Pick the model to start with.".to_owned(),
        ),
        Stage::Downloading {
            preset,
            file,
            done,
            total,
        } => {
            let fraction = if *total > 0 {
                *done as f32 / *total as f32
            } else {
                0.0
            };
            let dot = theme::DOT;
            (
                format!("Downloading the {} model", preset.label()),
                format!(
                    "{file}{dot}{}{dot}{:.0} / {}",
                    super::status::percent(fraction),
                    *done as f64 / 1e6,
                    megabytes(*total)
                ),
            )
        }
        Stage::Verifying(preset) => (
            format!("Checking downloaded {} model", preset.label()),
            "Verifying local data".to_owned(),
        ),
        Stage::Loading(name) => (
            format!("Loading {name} onto the GPU"),
            "A few seconds".to_owned(),
        ),
        Stage::Failed(message) => ("Stemd could not start".to_owned(), message.clone()),
    };

    let failed = matches!(stage, Stage::Failed(_));
    let title_colour = if failed { palette.danger } else { palette.text };
    let painter = ui.painter();
    let title_galley =
        painter.layout_no_wrap(title, theme::font(22.0, Weight::Semibold), title_colour);
    let title_height = title_galley.size().y;
    let sub_galley = painter.layout(
        sub,
        theme::font(12.5, Weight::Regular),
        palette.muted,
        inner.width(),
    );
    let sub_height = sub_galley.size().y;

    // Each row's height and the gap under it, once. Without the model rows,
    // the block sits in the middle.
    let mut column = Column::default();
    let stage_at = column.add(globe::STAGE.y, 6.0);
    let title_at = column.add(title_height, 6.0);
    let sub_at = column.add(sub_height, 22.0);
    let rows_at = column.next();
    if failed && !view.offered {
        column.add(QUIT_HEIGHT, 0.0);
    }
    let top = if view.offered {
        inner.top()
    } else {
        (inner.center().y - column.height() / 2.0).max(inner.top())
    };
    let stage_rect = Rect::from_min_size(
        pos2(mid - globe::STAGE.x / 2.0, top + stage_at),
        globe::STAGE,
    );
    let spread = ctx.animate_bool_with_time(ui.id().with("first-run-spread"), busy, 0.5);
    view.globe = Some((stage_rect, spread));
    if !view.globe_handed_off {
        globe::paint(
            painter,
            brushes,
            palette,
            stage_rect,
            &view.motion,
            globe::Look {
                spread,
                lift: 0.0,
                halo: 0.0,
            },
            now,
        );
    }

    painter.galley(
        pos2(mid - title_galley.size().x / 2.0, top + title_at),
        title_galley,
        title_colour,
    );
    painter.galley(
        pos2(mid - sub_galley.size().x / 2.0, top + sub_at),
        sub_galley,
        palette.muted,
    );
    let mut y = top + rows_at;

    let mut chosen = None;
    if view.offered {
        let installed: &[Preset] = match stage {
            Stage::Choosing(installed) => installed,
            _ => &[],
        };
        let locked = !matches!(stage, Stage::Choosing(_));
        for preset in Preset::ALL {
            let rect = Rect::from_min_size(pos2(inner.left(), y), vec2(inner.width(), ROW_HEIGHT));
            y += ROW_HEIGHT + ROW_GAP;
            let mine = picked == Some(preset) && locked;
            if option_row(
                ui, rect, preset, stage, mine, locked, picked, installed, palette, brushes, now,
            ) {
                chosen = Some(preset);
            }
        }
    }

    let footer = if view.offered {
        Rect::from_min_max(pos2(inner.left(), y), inner.max)
    } else {
        Rect::from_min_size(pos2(inner.left(), y), vec2(inner.width(), QUIT_HEIGHT))
    };
    match stage {
        Stage::Choosing(_) => {
            ui.painter().text(
                footer.center(),
                egui::Align2::CENTER_CENTER,
                "You can switch later from the model menu.",
                theme::font(12.0, Weight::Regular),
                palette.muted,
            );
        }
        Stage::Failed(_) => {
            let rect = Rect::from_center_size(footer.center(), vec2(88.0, QUIT_HEIGHT));
            let quit = widgets::pill(
                ui,
                rect,
                ui.id().with("quit"),
                "Quit",
                Pill::Light,
                true,
                palette,
                brushes,
            );
            if quit.clicked() {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        _ => {}
    }

    if let Some(preset) = chosen {
        view.picked = Some(preset);
    }
    chosen
}

/// One preset as a glass row. Returns whether it was clicked.
#[allow(clippy::too_many_arguments)]
fn option_row(
    ui: &egui::Ui,
    rect: Rect,
    preset: Preset,
    stage: &Stage,
    mine: bool,
    locked: bool,
    picked: Option<Preset>,
    installed: &[Preset],
    palette: &Palette,
    brushes: &Brushes,
    now: f64,
) -> bool {
    let ctx = ui.ctx();
    let id = ui.id().with(("first-run", preset.label()));
    let response = ui.interact(
        rect,
        id,
        if locked {
            Sense::hover()
        } else {
            Sense::click()
        },
    );
    let dim = ctx.animate_bool_with_time(id.with("dim"), locked && !mine && picked.is_some(), 0.3);
    let alpha = 1.0 - 0.6 * dim;
    let hover = ctx.animate_bool_with_time(id.with("hover"), response.hovered() && !locked, 0.15);
    let pressed = ctx.animate_bool_with_time(
        id.with("press"),
        response.is_pointer_button_down_on() && !locked,
        0.1,
    );
    let rect = Rect::from_center_size(rect.center(), rect.size() * (1.0 - 0.01 * pressed));

    let mut painter = ui.painter().clone();
    painter.multiply_opacity(alpha);
    brushes.glass(&painter, rect, ROW_RADIUS, Glass::Panel);
    if hover > 0.0 {
        painter.rect_filled(rect, ROW_RADIUS, white(0.05 * hover));
    }
    if mine {
        painter.rect_stroke(
            rect,
            ROW_RADIUS,
            Stroke::new(1.0, white(0.34)),
            egui::StrokeKind::Inside,
        );
    }

    let fraction = match stage {
        Stage::Downloading { done, total, .. } if mine && *total > 0 => {
            *done as f32 / *total as f32
        }
        Stage::Verifying(_) | Stage::Loading(_) if mine => 1.0,
        _ => 0.0,
    };
    let shown = ctx.animate_value_with_time(id.with("fill"), fraction, 0.16);
    if shown > 0.0 {
        let fill = Rect::from_min_size(rect.min, vec2(rect.width() * shown, rect.height()));
        brushes.spectrum(&painter, rect, fill, ROW_RADIUS, white(0.30));
    }

    let left = rect.left() + 18.0;
    let label = painter.layout_no_wrap(
        preset.label().to_owned(),
        theme::font(15.0, Weight::Semibold),
        palette.text,
    );
    let label_size = label.size();
    let top = rect.center().y - (label_size.y + 3.0 + 16.0) / 2.0;
    painter.galley(pos2(left, top), label, palette.text);
    painter.text(
        pos2(left + label_size.x + 8.0, top + label_size.y - 2.0),
        egui::Align2::LEFT_BOTTOM,
        tag(preset),
        theme::font(12.0, Weight::Regular),
        palette.soft,
    );
    painter.text(
        pos2(left, top + label_size.y + 3.0),
        egui::Align2::LEFT_TOP,
        preset.detail(),
        theme::font(12.5, Weight::Regular),
        palette.muted,
    );

    let right = rect.right() - 18.0;
    let y = rect.center().y;
    let small = theme::font(12.5, Weight::Regular);
    if !mine {
        let is_installed = installed.contains(&preset);
        let (text, icon) = if is_installed {
            ("installed".to_owned(), false)
        } else {
            (megabytes(preset.total_bytes()), picked.is_none())
        };
        let x = if icon {
            icons::download(
                &painter,
                icons::square(pos2(right - 6.5, y), 13.0),
                palette.soft,
                1.2,
            );
            right - 13.0 - 8.0
        } else {
            right
        };
        painter.text(
            pos2(x, y),
            egui::Align2::RIGHT_CENTER,
            text,
            small,
            palette.soft,
        );
    } else {
        match stage {
            Stage::Downloading { .. } => {
                painter.text(
                    pos2(right, y),
                    egui::Align2::RIGHT_CENTER,
                    super::status::percent(fraction),
                    small,
                    palette.soft,
                );
            }
            _ => icons::spinner(
                &painter,
                icons::square(pos2(right - 6.5, y), 13.0),
                palette.text,
                1.6,
                (now as f32).rem_euclid(1.0),
            ),
        }
    }

    !locked
        && response
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .clicked()
}
