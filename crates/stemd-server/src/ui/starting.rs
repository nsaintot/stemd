//! What the window shows until the server is ready: on a first run the choice of
//! model to download and its progress, then the model loading, or why startup
//! failed.

use eframe::egui;

use crate::launch::{Launch, Stage};
use crate::models::Preset;

use super::header::{DIALOG_WIDTH, progress};
use super::theme::Palette;

pub fn show(ui: &mut egui::Ui, launch: &Launch) {
    let ctx = ui.ctx().clone();
    ctx.request_repaint_after(super::REPAINT_INTERVAL);
    super::theme::install(&ctx);
    let palette = Palette::of(&ctx);
    let stage = launch.stage();

    egui::CentralPanel::default().show(ui, |_| {});
    egui::Window::new("Starting stemd")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(&ctx, |ui| {
            ui.set_max_width(DIALOG_WIDTH);
            ui.add_space(4.0);
            if let Some(preset) = body(ui, &stage, &palette) {
                launch.pick(preset);
            }
            ui.add_space(4.0);
        });
}

/// Draw `stage`. Returns the preset picked this frame, if any.
pub(super) fn body(ui: &mut egui::Ui, stage: &Stage, palette: &Palette) -> Option<Preset> {
    match stage {
        Stage::Starting => {
            ui.label("Checking the GPU");
            ui.add_space(8.0);
            ui.spinner();
        }
        Stage::Choosing(installed) => return choose(ui, installed, palette),
        Stage::Downloading {
            preset,
            file,
            done,
            total,
        } => {
            ui.label(format!("Downloading the {} model", preset.label()));
            ui.label(
                egui::RichText::new(preset.detail())
                    .size(11.0)
                    .color(palette.muted),
            );
            ui.add_space(8.0);
            progress(ui, file, *done, *total, palette);
        }
        Stage::Verifying(preset) => {
            ui.label(format!("Checking the {} model", preset.label()));
            ui.add_space(8.0);
            ui.spinner();
        }
        Stage::Loading(model) => {
            ui.label(format!("Loading the {model} model"));
            ui.add_space(8.0);
            ui.spinner();
        }
        Stage::Failed(message) => {
            ui.label(egui::RichText::new("stemd could not start").color(palette.bad));
            ui.add_space(6.0);
            ui.label(egui::RichText::new(message).size(11.0));
            ui.add_space(8.0);
            if ui.button("Quit").clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }
    None
}

/// One row per preset: a button to pick it, its speed, and its download size.
fn choose(ui: &mut egui::Ui, installed: &[Preset], palette: &Palette) -> Option<Preset> {
    let small = |text: String| egui::RichText::new(text).size(10.5).color(palette.muted);
    ui.label("Choose a model");
    ui.label(small("You can switch later from the model menu.".into()));
    ui.add_space(8.0);
    let mut picked = None;
    for preset in Preset::ALL {
        ui.horizontal(|ui| {
            if ui
                .add_sized([84.0, 28.0], egui::Button::new(preset.label()))
                .clicked()
            {
                picked = Some(preset);
            }
            ui.vertical(|ui| {
                ui.label(small(preset.detail().into()));
                ui.label(small(if installed.contains(&preset) {
                    "installed".into()
                } else {
                    format!("{:.0} MB download", preset.total_bytes() as f64 / 1e6)
                }));
            });
        });
        ui.add_space(4.0);
    }
    picked
}
