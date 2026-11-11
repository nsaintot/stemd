//! The window: drop a track on it, and watch the server behind it.
//!
//! ```text
//! header    the model, and what the stems come out as
//! dropzone  the globe, the drop target, and progress while it works
//! recents   what has been separated, and where the stems went
//! logs      the same lines GET /v1/logs serves
//! status    server state and the cache, along the bottom
//! ```
//!
//! A drop and a `POST /v1/jobs` become the same job in the same queue; see
//! [`crate::drops`]. The server runs headless without any of this.

mod blur;
mod dropzone;
mod globe;
mod header;
mod icons;
mod logs;
#[cfg(test)]
mod probe;
mod quit;
mod recents;
mod section;
mod starting;
mod status;
mod theme;
mod titlebar;
mod widgets;

use std::net::SocketAddr;
use std::sync::Arc;

use eframe::egui;

use crate::api::AppState;
use crate::drops::{AUDIO_EXTENSIONS, Dropped, State, looks_like_audio};
use crate::launch::Launch;
use crate::models::Preset;
use crate::switch::SwitchState;

use theme::{Brushes, Glass, Palette};

const WINDOW_SIZE: [f32; 2] = [560.0, 600.0 + titlebar::HEIGHT];
const MIN_WINDOW_SIZE: [f32; 2] = [480.0, 520.0 + titlebar::HEIGHT];

/// Height the drop zone keeps; the Separated list and the log take the rest.
const ZONE_HEIGHT: f32 = 320.0;

/// The zone's height in a window `content` points tall: [`ZONE_HEIGHT`], less
/// when the window is too short to also show one row of the list.
fn zone_height(content: f32) -> f32 {
    // Header, status bar and their margins; the zone's and the sections' gaps;
    // the "Separated" heading and one row.
    let around = f32::from(TOP)
        + header::HEIGHT
        + status::HEIGHT
        + f32::from(PAD)
        + 3.0 * f32::from(GAP)
        + section::HEIGHT
        + 6.0
        + recents::ROW_HEIGHT;
    (content - around).clamp(dropzone::MIN_HEIGHT, ZONE_HEIGHT)
}

/// Space around the window's contents, and between its regions.
const PAD: i8 = 14;

/// Space above the header, clearing the title bar where the window draws its own.
const TOP: i8 = if titlebar::HEIGHT > 0.0 {
    titlebar::HEIGHT as i8 + 6
} else {
    PAD
};
const GAP: i8 = 10;

/// Poll interval at rest, for state written by other threads. While anything
/// moves, the window repaints every frame.
const REPAINT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// The window's own icon, compiled in.
///
/// Two of them: macOS icons are a rounded body inset in a transparent square,
/// Windows icons are the square. A bundled `.app` takes its icon from
/// `Contents/Resources/stemd.icns`, but the app is not always bundled.
#[cfg(target_os = "macos")]
const WINDOW_ICON: &[u8] = include_bytes!("../../../../resources/stemd-icon-mac-256.png");
#[cfg(not(target_os = "macos"))]
const WINDOW_ICON: &[u8] = include_bytes!("../../../../resources/stemd-icon-win-256.png");

pub fn run(launch: Arc<Launch>, bind: SocketAddr) -> Result<(), eframe::Error> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size(WINDOW_SIZE)
        .with_min_inner_size(MIN_WINDOW_SIZE)
        .with_title("Stemd");
    // The backdrop runs under the title bar, which keeps only its buttons; see
    // `titlebar`.
    #[cfg(target_os = "macos")]
    {
        viewport = viewport
            .with_fullsize_content_view(true)
            .with_titlebar_shown(false)
            .with_title_shown(false);
    }
    // No system frame: `titlebar` draws the bar, its buttons and the resize grips.
    #[cfg(windows)]
    {
        viewport = viewport.with_decorations(false);
    }

    // A window that will not decode its own icon is a window without one, not
    // a reason to refuse to open.
    match eframe::icon_data::from_png_bytes(WINDOW_ICON) {
        Ok(icon) => viewport = viewport.with_icon(icon),
        Err(err) => tracing::warn!("the window icon will not decode ({err}); running without one"),
    }

    // eframe saves the window's position and size here, and puts it back on a
    // monitor if that one is gone. Centred until there is a saved position.
    let saved_in = crate::models::support_dir()
        .ok()
        .map(|dir| dir.join("window.ron"));
    let first_launch = saved_in.as_ref().is_none_or(|file| !file.exists());
    #[allow(unused_mut)]
    let mut options = eframe::NativeOptions {
        viewport,
        centered: first_launch,
        persistence_path: saved_in,
        ..Default::default()
    };

    //  Ask for X11: winit implements file drag and drop on macOS, Windows and X11 and
    //  has no Wayland implementation, so on a Wayland session `dropped_files` never
    //  fills. Xwayland speaks XDND. Only when there is an X server to ask for: a
    //  session without Xwayland has no DISPLAY, and forcing X11 there would trade a
    //  window that cannot accept drops for no window at all.
    #[cfg(all(unix, not(target_os = "macos")))]
    if std::env::var_os("DISPLAY").is_some_and(|display| !display.is_empty()) {
        use winit::platform::x11::EventLoopBuilderExtX11 as _;
        options.event_loop_builder = Some(Box::new(|builder| {
            builder.with_x11();
        }));
    }

    eframe::run_native(
        "Stemd",
        options,
        Box::new(move |cc| {
            theme::install(&cc.egui_ctx);
            Ok(Box::new(App {
                launch,
                bind,
                brushes: Brushes::new(&cc.egui_ctx),
                first_run: starting::View::default(),
                window: None,
            }))
        }),
    )
}

/// The startup view until the server is ready, then [`Window`].
struct App {
    launch: Arc<Launch>,
    bind: SocketAddr,
    brushes: Brushes,
    first_run: starting::View,
    window: Option<Window>,
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.window.is_none()
            && let Some(state) = self.launch.ready()
        {
            let motion = self.first_run.take_motion();
            let globe = self.first_run.globe;
            self.window = Some(Window::new(Arc::clone(state), self.bind, motion, globe));
            self.first_run.globe_handed_off = true;
        }
        match &mut self.window {
            Some(window) => {
                window.ui(ui, &self.brushes);
                // The startup screen fades out over the window as it fades in.
                if let (Some(t), Some(stage)) = (window.reveal(), self.first_run.last.clone()) {
                    let ctx = ui.ctx().clone();
                    let mut fading =
                        ui.new_child(egui::UiBuilder::new().max_rect(ctx.content_rect()));
                    fading.multiply_opacity(1.0 - t);
                    let palette = Palette::of(&ctx);
                    let _ = starting::body(
                        &mut fading,
                        &stage,
                        &palette,
                        &self.brushes,
                        &mut self.first_run,
                    );
                }
            }
            None => starting::show(ui, &self.launch, &mut self.first_run, &self.brushes),
        }
        titlebar::show(ui, &Palette::of(ui.ctx()));
    }

    /// eframe saves only the window's position and size.
    fn persist_egui_memory(&self) -> bool {
        false
    }

    /// The only shutdown the window ever gets.
    ///
    /// Cmd-Q goes `[NSApp terminate:]` to `applicationWillTerminate:` to winit's
    /// `LoopExiting` to here, and AppKit calls `exit()` as soon as that returns. It is
    /// the last point at which the worker can be taken out of the model.
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        crate::shutdown::quit(&self.launch, "window closed")
    }
}

struct Window {
    state: Arc<AppState>,
    bind: SocketAddr,
    logs: logs::View,
    /// Whether the log pane is open.
    show_logs: bool,
    /// Height the log pane took last frame, 0 when closed. Opening it grows
    /// the window by that much instead of squeezing the list.
    logs_height: Option<f32>,
    /// The drop the zone is currently showing. Held so the zone keeps showing
    /// one track while a second is queued behind it.
    current: Option<Arc<Dropped>>,
    /// Drops since the window was last idle, for the per-track bars and the
    /// count when they finish.
    batch: Vec<Arc<Dropped>>,
    /// Whether any drop in the batch was running last frame.
    batch_running: bool,
    /// Text shown under the title in place of the hint, and when it expires.
    note: Option<(String, f64)>,
    motion: globe::Motion,
    /// The preset the header showed last frame, to fire its effect on a change.
    shown: Option<Preset>,
    /// Least window width the header fits in, as last sent to the viewport.
    min_width: f32,
    /// Whether a close is being held for a question. See [`quit`].
    quit: quit::Guard,
    /// The hand-over from the startup screen, while it plays.
    intro: Option<Intro>,
    /// Its eased progress this frame.
    reveal: Option<f32>,
}

/// Where the startup screen's globe was, and when the hand-over began.
struct Intro {
    from: egui::Rect,
    spread: f32,
    start: Option<f64>,
}

/// Seconds the hand-over from the startup screen takes.
const INTRO_SECS: f32 = 0.8;

impl Window {
    fn new(
        state: Arc<AppState>,
        bind: SocketAddr,
        motion: globe::Motion,
        globe: Option<(egui::Rect, f32)>,
    ) -> Self {
        Self {
            state,
            bind,
            logs: logs::View::default(),
            show_logs: false,
            logs_height: None,
            current: None,
            batch: Vec::new(),
            batch_running: false,
            note: None,
            motion,
            shown: None,
            min_width: 0.0,
            quit: quit::Guard::default(),
            intro: globe.map(|(from, spread)| Intro {
                from,
                spread,
                start: None,
            }),
            reveal: None,
        }
    }
}

/// Phase speed and seconds per meridian turn, by preset: Fast spins quick,
/// Quality slow.
const fn pace(preset: Option<Preset>) -> (f32, f32) {
    match preset {
        Some(Preset::Fast) => (1.6, 1.6),
        Some(Preset::Quality) => (0.6, 7.0),
        Some(Preset::Balanced) | None => (1.0, 4.5),
    }
}

const fn fx_for(preset: Preset) -> globe::Fx {
    match preset {
        Preset::Fast => globe::Fx::Fast,
        Preset::Balanced => globe::Fx::Balanced,
        Preset::Quality => globe::Fx::Quality,
    }
}

impl Window {
    fn ui(&mut self, ui: &mut egui::Ui, brushes: &Brushes) {
        let ctx = ui.ctx().clone();
        theme::install(&ctx);
        let palette = Palette::of(&ctx);
        let (dt, now) = (globe::frame_dt(&ctx), ctx.input(|i| i.time));
        brushes.backdrop(ui.painter(), ctx.content_rect());

        let reveal = self.advance_intro(&ctx, now);

        self.take_drops(&ctx);
        self.follow_batch(now);

        // Before the rest of the frame: a held close should put its question up
        // now rather than a frame later, and the answer can end the process.
        quit::guard(&mut self.quit, &self.state, &ctx, &palette, brushes);

        let shown = header::shown_preset(&self.state.switcher);
        if shown != self.shown {
            // A switch landed, after its download and check. A failed one falls
            // back to the old model, which is not a landing.
            let failed = matches!(self.state.switcher.state(), SwitchState::Failed { .. });
            if let (Some(_), Some(preset), false) = (self.shown, shown, failed) {
                let fx = fx_for(preset);
                self.motion.fire(fx, now);
                self.note = Some((preset.detail().to_owned(), now + f64::from(fx.duration())));
            }
            self.shown = shown;
        }
        let switch_state = self.state.switcher.state();
        let running = self.state.queue.running();
        theme::set_processing(&ctx, running.is_some());
        let working = self.current.as_ref().is_some_and(|d| d.is_running());
        let switching = match &switch_state {
            SwitchState::Downloading { .. } | SwitchState::Verifying(_) => true,
            SwitchState::Loading(_) => running.is_none(),
            SwitchState::Idle | SwitchState::Failed { .. } => false,
        };
        let variant = running
            .as_deref()
            .map(globe::variant_of)
            .or((working || switching).then_some(0));
        let (speed, period) = pace(shown);
        let drive = globe::Drive {
            variant,
            speed,
            spin: variant.is_some().then_some(period),
        };
        if self.motion.step(dt, now, &drive) {
            theme::request_frame(&ctx);
        } else {
            ctx.request_repaint_after(REPAINT_INTERVAL);
        }

        let margin = |top: i8, bottom: i8| egui::Margin {
            left: PAD,
            right: PAD,
            top,
            bottom,
        };

        egui::Panel::top("header")
            .frame(egui::Frame::NONE.inner_margin(margin(TOP, 0)))
            .show_separator_line(false)
            .resizable(false)
            .show(ui, |ui| {
                if let Some(t) = reveal {
                    ui.multiply_opacity(t);
                }
                let needed = header::row(ui, &self.state, &palette, brushes, self.motion.clock);
                self.fit_width(&ctx, needed + 2.0 * f32::from(PAD));
            });

        egui::Panel::bottom("status")
            .frame(egui::Frame::NONE.inner_margin(margin(0, PAD)))
            .show_separator_line(false)
            .resizable(false)
            .show(ui, |ui| {
                if let Some(t) = reveal {
                    ui.multiply_opacity(t);
                }
                match status::bar(ui, &self.state, &palette, self.bind.port(), self.show_logs) {
                    Some(status::Action::ToggleLogs) => self.show_logs = !self.show_logs,
                    Some(status::Action::ToggleServer) => {
                        let on = !self.state.serving.is_on();
                        self.state.serving.set(on, &self.state.queue.info().model);
                        self.state.settings.set_serving(on);
                        // Clients get 503 from now on, so nobody could collect
                        // or cancel these.
                        if !on {
                            let cancelled = self.state.queue.cancel_api_jobs();
                            if cancelled > 0 {
                                tracing::info!("cancelled {cancelled} API job(s)");
                                let noun = if cancelled == 1 { "job" } else { "jobs" };
                                let until = ui.input(|i| i.time) + 4.0;
                                self.note =
                                    Some((format!("Stopped {cancelled} client {noun}"), until));
                            }
                        }
                    }
                    None => {}
                }
            });

        // The drop zone keeps a fixed height and the sections take the rest,
        // so the ratio between them holds whatever the list holds.
        let zone = zone_height(ctx.content_rect().height());
        egui::Panel::top("zone")
            .frame(egui::Frame::NONE.inner_margin(margin(GAP, 0)))
            .exact_size(zone + f32::from(GAP))
            .show_separator_line(false)
            .resizable(false)
            .show(ui, |ui| {
                let armed = !ui.ctx().input(|i| i.raw.hovered_files.is_empty());
                let note = self
                    .note
                    .as_ref()
                    .filter(|(_, until)| now < *until)
                    .map(|(text, _)| text.as_str());
                let switch = match &switch_state {
                    SwitchState::Downloading {
                        preset,
                        file,
                        done,
                        total,
                    } => Some(dropzone::Switch::Downloading {
                        preset: *preset,
                        file,
                        done: *done,
                        total: *total,
                    }),
                    SwitchState::Verifying(preset) => Some(dropzone::Switch::Verifying(*preset)),
                    SwitchState::Loading(preset) => Some(dropzone::Switch::Loading {
                        preset: *preset,
                        waiting: running.is_some(),
                    }),
                    SwitchState::Failed { preset, message } => Some(dropzone::Switch::Failed {
                        preset: *preset,
                        message,
                    }),
                    SwitchState::Idle => None,
                };
                let action = dropzone::Zone {
                    palette: &palette,
                    brushes,
                    motion: &self.motion,
                    armed,
                    current: self.current.as_deref(),
                    parts: self.state.settings.get().drop_parts,
                    batch: dropzone::Batch {
                        finished: self.batch.iter().map(|d| !d.is_running()).collect(),
                        shown: self.current.as_ref().and_then(|current| {
                            self.batch.iter().position(|d| Arc::ptr_eq(d, current))
                        }),
                    },
                    note,
                    maxed: shown == Some(Preset::Quality),
                    switch,
                    intro: self
                        .intro
                        .as_ref()
                        .zip(reveal)
                        .map(|(intro, t)| dropzone::Intro {
                            from: intro.from,
                            spread: intro.spread,
                            t,
                        }),
                }
                .show(ui);
                match action {
                    Some(dropzone::Action::ChooseFile) => self.pick_a_file(),
                    Some(dropzone::Action::KeepModel) => self.state.switcher.dismiss_error(),
                    None => {}
                }
            });

        let logs_height = egui::CentralPanel::default()
            .frame(egui::Frame::NONE.inner_margin(margin(GAP, GAP)))
            .show(ui, |ui| {
                if let Some(t) = reveal {
                    ui.multiply_opacity(t);
                }
                self.sections(ui, &palette, brushes)
            })
            .inner;
        self.match_window_to(&ctx, logs_height);
    }

    /// Step the hand-over from the startup screen. Returns its eased progress
    /// while it plays.
    fn advance_intro(&mut self, ctx: &egui::Context, now: f64) -> Option<f32> {
        let intro = self.intro.as_mut()?;
        let start = *intro.start.get_or_insert(now);
        let t = ((now - start) as f32 / INTRO_SECS).clamp(0.0, 1.0);
        if t >= 1.0 {
            self.intro = None;
            self.reveal = None;
            return None;
        }
        theme::request_frame(ctx);
        self.reveal = Some(theme::ease_in_out(t));
        self.reveal
    }

    /// Eased progress of the hand-over from the startup screen, while it plays.
    const fn reveal(&self) -> Option<f32> {
        self.reveal
    }

    /// Keep the window at least as wide as the header needs.
    fn fit_width(&mut self, ctx: &egui::Context, needed: f32) {
        let width = needed.max(MIN_WINDOW_SIZE[0]).ceil();
        if (width - self.min_width).abs() < 1.0 {
            return;
        }
        self.min_width = width;
        ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
            width,
            MIN_WINDOW_SIZE[1],
        )));
    }

    /// The Separated list and, when open, the log. Returns the log pane's
    /// height, 0 when closed.
    fn sections(&mut self, ui: &mut egui::Ui, palette: &Palette, brushes: &Brushes) -> f32 {
        // The glass's drop shadow reaches past the panel; let it.
        ui.set_clip_rect(ui.ctx().content_rect());
        ui.spacing_mut().item_spacing = egui::vec2(0.0, 6.0);
        let recent = self.state.drops.recent();
        let count = (!recent.is_empty()).then(|| recent.len().to_string());
        section::heading(ui, palette, "Separated", count);
        // Room left once the log, as tall as it was last frame, is set aside.
        let logs = if self.show_logs {
            self.logs_height.unwrap_or(0.0) + 4.0 + 6.0
        } else {
            0.0
        };
        let room = ui.available_height() - logs;
        if let Some(item) = recents::list(ui, palette, brushes, &self.state.drops, &recent, room) {
            self.state.drops.stop(&self.state, &item);
        }

        if !self.show_logs {
            return 0.0;
        }
        ui.add_space(4.0);
        let behind = ui.painter().add(egui::Shape::Noop);
        let mut clear_logs = false;
        let frame = egui::Frame::NONE.inner_margin(10).show(ui, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
            let lines = self.state.logs.recent(crate::logbuf::CAPACITY);
            clear_logs = logs::show(ui, palette, brushes, &mut self.logs, &lines);
        });
        ui.painter().set(
            behind,
            brushes.glass_shape(frame.response.rect, 16.0, Glass::List),
        );
        // Not logged: emptying the log and then writing to it would undo the click.
        if clear_logs {
            self.state.logs.clear();
        }
        frame.response.rect.height()
    }

    /// Grow or shrink the window by however much the log pane changed, so
    /// opening it does not squeeze the list. The first frame only records it.
    fn match_window_to(&mut self, ctx: &egui::Context, logs: f32) {
        let Some(previous) = self.logs_height.replace(logs) else {
            return;
        };
        let delta = logs - previous;
        // Heights wobble by a fraction of a point as text is laid out, and a window
        // that resizes by half a pixel every frame never settles.
        if delta.abs() < 1.0 {
            return;
        }
        let Some(size) = ctx.input(|i| i.viewport().inner_rect.map(|r| r.size())) else {
            return;
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
            size.x,
            size.y + delta,
        )));
    }

    /// Hand anything dropped on the window to [`crate::drops`].
    ///
    /// Everything that is not audio is refused here rather than after a decoder
    /// has been pointed at it, so dropping a folder of artwork says what was
    /// wrong instead of producing one failure per file.
    fn take_drops(&mut self, ctx: &egui::Context) {
        let dropped: Vec<std::path::PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });

        for path in dropped {
            if !looks_like_audio(&path) {
                tracing::warn!("{} is not audio this can decode", path.display());
                continue;
            }
            let dropped = self.state.drops.accept(&self.state, path);
            if !self.current.as_ref().is_some_and(|d| d.is_running()) {
                self.current = Some(dropped);
            }
        }

        // Once the shown track is finished, the next running one takes the zone,
        // oldest first; the finished one is still in the list below.
        if !self.current.as_ref().is_some_and(|d| d.is_running()) {
            self.current = self
                .state
                .drops
                .recent()
                .into_iter()
                .rev()
                .find(|d| d.is_running());
        }
    }

    /// Fold new drops into the batch and fire the start and finish effects.
    ///
    /// Reads the drop list directly, so a track chosen with the file picker on
    /// another thread counts too.
    fn follow_batch(&mut self, now: f64) {
        let fresh: Vec<Arc<Dropped>> = self
            .state
            .drops
            .recent()
            .into_iter()
            .rev()
            .filter(|d| d.is_running() && !self.batch.iter().any(|b| Arc::ptr_eq(b, d)))
            .collect();
        if !fresh.is_empty() && !self.batch_running {
            self.batch.clear();
            self.note = None;
            self.motion.fire(globe::Fx::Start, now);
        }
        self.batch.extend(fresh);

        let running = self.batch.iter().any(|d| d.is_running());
        if self.batch_running && !running {
            let done = self
                .batch
                .iter()
                .filter(|d| matches!(d.state(), State::Done(_)))
                .count();
            if done > 0 {
                self.motion.fire(globe::Fx::Cheer, now);
                let noun = if done == 1 { "track" } else { "tracks" };
                self.note = Some((
                    format!("{done} {noun} separated"),
                    now + f64::from(globe::Fx::Cheer.duration()),
                ));
            }
        }
        self.batch_running = running;
    }

    /// The click-to-choose path, for a track that is not convenient to drag.
    ///
    /// An in-process panel would need a crate and, on macOS, the main thread;
    /// shelling out to the system's own dialog keeps this short and off the
    /// event loop. See [`choose_file`].
    fn pick_a_file(&self) {
        let drops = Arc::clone(&self.state.drops);
        let state = Arc::clone(&self.state);
        std::thread::Builder::new()
            .name("stemd-picker".into())
            .spawn(move || {
                for path in choose_files() {
                    // The same gate a drop goes through: the dialog can still be
                    // typed into, and the two ways in should refuse the same files.
                    if looks_like_audio(&path) {
                        drops.accept(&state, path);
                    } else {
                        tracing::warn!("{} is not audio this can decode", path.display());
                    }
                }
            })
            .expect("spawning the file picker");
    }
}

/// Ask the system for one or more files, blocking the thread it is called on.
///
/// Every platform's answer is the same shape: hand the job to a program that
/// already owns a file panel and read one path per line off its stdout. Empty
/// means cancelled, which all three report as a non-zero exit or empty output.
fn choose_files() -> Vec<std::path::PathBuf> {
    let Some(output) = picker().and_then(|mut cmd| cmd.output().ok()) else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    paths_of(&String::from_utf8_lossy(&output.stdout))
}

/// One path per non-empty line.
fn paths_of(text: &str) -> Vec<std::path::PathBuf> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(std::path::PathBuf::from)
        .collect()
}

/// The type list is `public.audio` and every extension this accepts.
/// `public.audio` alone greyed out `.mp4`: an MPEG-4 file conforms to
/// `public.movie` however little video is in it, while it is a container this
/// decodes and accepts on a drop.
#[cfg(target_os = "macos")]
fn picker() -> Option<std::process::Command> {
    let types = std::iter::once("public.audio")
        .chain(AUDIO_EXTENSIONS)
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let choose = format!(
        "set picked to choose file with prompt \"Choose tracks to separate\" \
         of type {{{types}}} with multiple selections allowed"
    );
    let mut cmd = std::process::Command::new("osascript");
    cmd.args([
        "-e",
        &choose,
        "-e",
        "set text item delimiters to linefeed",
        "-e",
        "set paths to {}",
        "-e",
        "repeat with f in picked",
        "-e",
        "set end of paths to POSIX path of f",
        "-e",
        "end repeat",
        "-e",
        "return paths as text",
    ]);
    Some(cmd)
}

/// `OpenFileDialog` out of WinForms, driven by PowerShell.
///
/// `-STA` is not optional: the common file dialog is an apartment-threaded COM
/// object and PowerShell's default MTA makes it throw. The filter is one line of
/// `label|patterns|...`, from the same extension list a drop is checked against.
#[cfg(windows)]
fn picker() -> Option<std::process::Command> {
    use std::os::windows::process::CommandExt;

    //  `CREATE_NO_WINDOW`, and it is not cosmetic. The crate links for the windows
    //  subsystem, so a launch from Explorer has no console, and starting a console
    //  program from a process that has none hands the child a new console window that
    //  would sit beside the file panel. The chosen path comes back over a pipe.
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let patterns = AUDIO_EXTENSIONS
        .iter()
        .map(|e| format!("*.{e}"))
        .collect::<Vec<_>>()
        .join(";");
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms; \
         $d = New-Object System.Windows.Forms.OpenFileDialog; \
         $d.Title = 'Choose tracks to separate'; \
         $d.Multiselect = $true; \
         $d.Filter = 'Audio|{patterns}|All files|*.*'; \
         if ($d.ShowDialog() -eq 'OK') {{ [Console]::Out.Write($d.FileNames -join [char]10) }} else {{ exit 1 }}"
    );
    let mut cmd = std::process::Command::new("powershell");
    cmd.args(["-NoProfile", "-STA", "-Command", &script])
        .creation_flags(CREATE_NO_WINDOW);
    Some(cmd)
}

/// zenity, which is what GTK desktops ship and what KDE's `kdialog` mirrors.
///
/// `None` when neither is installed, which is a real possibility on a server
/// install and is why this returns an `Option` rather than a `Command`: the
/// caller treats it exactly like a cancelled dialog. Drag and drop still works.
#[cfg(target_os = "linux")]
fn picker() -> Option<std::process::Command> {
    let patterns = AUDIO_EXTENSIONS
        .iter()
        .map(|e| format!("*.{e}"))
        .collect::<Vec<_>>()
        .join(" ");
    if which("zenity") {
        let mut cmd = std::process::Command::new("zenity");
        cmd.args([
            "--file-selection",
            "--multiple",
            "--separator=\n",
            "--title=Choose tracks to separate",
            &format!("--file-filter=Audio | {patterns}"),
            "--file-filter=All files | *",
        ]);
        return Some(cmd);
    }
    if which("kdialog") {
        let mut cmd = std::process::Command::new("kdialog");
        cmd.args([
            "--getopenfilename",
            ".",
            &format!("{patterns}|Audio"),
            "--multiple",
            "--separate-output",
        ]);
        return Some(cmd);
    }
    tracing::warn!("no zenity or kdialog to open a file panel with; drag a track in instead");
    None
}

/// Whether a program is on `PATH`.
#[cfg(target_os = "linux")]
fn which(program: &str) -> bool {
    std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).any(|dir| dir.join(program).is_file()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picker_answer_is_one_path_per_line() {
        assert_eq!(
            paths_of("/a/one.wav\n/b/two words.mp3\n\n"),
            vec![
                std::path::PathBuf::from("/a/one.wav"),
                std::path::PathBuf::from("/b/two words.mp3"),
            ]
        );
        assert!(
            paths_of("").is_empty(),
            "a cancelled panel picked something"
        );
    }
}
