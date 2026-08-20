//! The window: drop a track on it, and watch the server behind it.
//!
//! ```text
//! header    which model, and what the stems come out as
//! dropzone  the main region: drop target, and progress while it works
//! recents   what has been separated, and where the stems went
//! logs      the same lines GET /v1/logs serves
//! status    server state, along the bottom
//! ```
//!
//! A drop and a `POST /v1/jobs` become the same job in the same queue; see
//! [`crate::drops`]. The server runs headless without any of this.

mod dropzone;
mod header;
mod logs;
#[cfg(test)]
mod probe;
mod quit;
mod recents;
mod section;
mod status;
mod theme;

use std::net::SocketAddr;
use std::sync::Arc;

use eframe::egui;

use crate::api::AppState;
use crate::drops::{AUDIO_EXTENSIONS, Dropped, looks_like_audio};

use theme::{Palette, Sheen};

/// Small on purpose: this is a drop target that reports, not a console. It grows
/// to fit if both sections are opened, and the minimum still shows a whole drop
/// zone.
const WINDOW_SIZE: [f32; 2] = [460.0, 470.0];
const MIN_WINDOW_SIZE: [f32; 2] = [380.0, 360.0];
const PAD: f32 = 12.0;

/// Nothing here is event-driven, the server writes from other threads, so the
/// window polls at a rate that looks live without spinning a core.
const REPAINT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(250);

/// The window's own icon, compiled in.
///
/// Two of them: macOS icons are a rounded body inset in a transparent square,
/// Windows icons are the square. A bundled `.app` takes its icon from
/// `Contents/Resources/stemd.icns`, but the app is not always bundled.
///
/// A PNG rather than raw pixels: eframe already depends on `image`, and 256 is the
/// largest size anything asks of a window icon.
#[cfg(target_os = "macos")]
const WINDOW_ICON: &[u8] = include_bytes!("../../../../resources/stemd-icon-mac-256.png");
#[cfg(not(target_os = "macos"))]
const WINDOW_ICON: &[u8] = include_bytes!("../../../../resources/stemd-icon-win-256.png");

pub fn run(state: Arc<AppState>, bind: SocketAddr) -> Result<(), eframe::Error> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size(WINDOW_SIZE)
        .with_min_inner_size(MIN_WINDOW_SIZE)
        .with_title("stemd");

    // A window that will not decode its own icon is a window without one, not
    // a reason to refuse to open.
    match eframe::icon_data::from_png_bytes(WINDOW_ICON) {
        Ok(icon) => viewport = viewport.with_icon(icon),
        Err(err) => tracing::warn!("the window icon will not decode ({err}); running without one"),
    }

    #[allow(unused_mut)]
    let mut options = eframe::NativeOptions {
        viewport,
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
        "stemd",
        options,
        Box::new(move |cc| {
            theme::install(&cc.egui_ctx);
            Ok(Box::new(Window::new(state, bind, &cc.egui_ctx)))
        }),
    )
}

struct Window {
    state: Arc<AppState>,
    bind: SocketAddr,
    sheen: Sheen,
    logs: logs::View,
    /// Which sections are open. Held here rather than in egui's memory so the
    /// defaults are stated in one place.
    show_recents: bool,
    show_logs: bool,
    /// Height the sections panel had last frame, so a change can be handed to
    /// the window instead of taken out of the drop zone.
    sections_height: Option<f32>,
    /// The drop the zone is currently showing. Held so the zone keeps showing
    /// one track while a second is queued behind it.
    current: Option<Arc<Dropped>>,
    /// Whether a close is being held for a question. See [`quit`].
    quit: quit::Guard,
}

impl Window {
    fn new(state: Arc<AppState>, bind: SocketAddr, ctx: &egui::Context) -> Self {
        Self {
            state,
            bind,
            sheen: Sheen::new(ctx),
            logs: logs::View::default(),
            show_recents: true,
            show_logs: false,
            sections_height: None,
            current: None,
            quit: quit::Guard::default(),
        }
    }
}

impl eframe::App for Window {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        ctx.request_repaint_after(REPAINT_INTERVAL);
        // The appearance can change under a running window, and the style was
        // installed once at startup.
        theme::install(&ctx);
        let palette = Palette::of(&ctx);

        self.take_drops(&ctx);

        // Before the rest of the frame: a held close should put its question up
        // now rather than a frame later, and the answer can end the process.
        quit::guard(&mut self.quit, &self.state, &ctx, &palette);

        egui::Panel::top("header").show(ui, |ui| {
            ui.add_space(6.0);
            header::row(ui, &self.state, &palette);
            ui.add_space(6.0);
        });

        egui::Panel::bottom("status").show(ui, |ui| {
            status::bar(ui, &self.state, &palette, self.bind);
        });

        // The sections have their own panel, and the window grows to fit it
        // rather than the drop zone shrinking to make room: opening the log to
        // read it should not shrink the thing you are dropping tracks on.
        let sections = egui::Panel::bottom("sections")
            .show(ui, |ui| {
                ui.add_space(2.0);
                self.sections(ui, &palette);
                ui.add_space(2.0);
            })
            .response
            .rect
            .height();
        self.match_window_to(&ctx, sections);

        egui::CentralPanel::default().show(ui, |ui| {
            ui.add_space(PAD);
            let armed = !ui.ctx().input(|i| i.raw.hovered_files.is_empty());
            let clicked = dropzone::Zone {
                palette: &palette,
                sheen: &mut self.sheen,
                armed,
                current: self.current.as_deref(),
            }
            .show(ui);
            if clicked {
                self.pick_a_file();
            }
            ui.add_space(PAD);
        });

        header::switch_dialog(&ctx, &self.state, &palette);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.shutdown()
    }
}

impl Window {
    fn sections(&mut self, ui: &mut egui::Ui, palette: &Palette) {
        let recent = self.state.drops.recent();
        let count = (!recent.is_empty()).then(|| recent.len().to_string());
        section::show(
            ui,
            palette,
            "Separated",
            count,
            &mut self.show_recents,
            section::UNDER_LABEL,
            |ui| recents::list(ui, palette, &mut self.sheen, &self.state.drops, &recent),
        );

        let mut clear_logs = false;
        section::show(
            ui,
            palette,
            "Logs",
            None,
            &mut self.show_logs,
            section::FLUSH,
            |ui| {
                let lines = self.state.logs.recent(crate::logbuf::CAPACITY);
                clear_logs = logs::show(ui, palette, &mut self.logs, &lines);
            },
        );
        // Not logged. Emptying the log and then writing to it is a joke at the
        // reader's expense, and the window has just shown the result.
        if clear_logs {
            self.state.logs.clear();
        }
    }

    /// Grow or shrink the window by however much the sections panel changed.
    ///
    /// Driven by the panel's measured height rather than by the toggle, so it follows
    /// a section whose contents grew. The first frame only records the height.
    fn match_window_to(&mut self, ctx: &egui::Context, sections: f32) {
        let Some(previous) = self.sections_height.replace(sections) else {
            return;
        };
        let delta = sections - previous;
        // A threshold, not zero: heights wobble by a fraction of a point as text
        // is laid out, and a window that resizes by half a pixel every frame
        // never settles.
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
            self.current = Some(self.state.drops.accept(&self.state, path));
        }

        // Once the shown track is finished, let the next running one take the
        // zone; the finished one is still in the list below.
        if self.current.as_ref().is_some_and(|d| !d.is_running()) {
            self.current = self
                .state
                .drops
                .recent()
                .into_iter()
                .find(|d| d.is_running());
        }
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
                let Some(path) = choose_file() else { return };
                // The same gate a drop goes through: the dialog can still be
                // typed into, and the two ways in should refuse the same files.
                if !looks_like_audio(&path) {
                    tracing::warn!("{} is not audio this can decode", path.display());
                    return;
                }
                drops.accept(&state, path);
            })
            .expect("spawning the file picker");
    }

    /// The only shutdown the window ever gets.
    ///
    /// Cmd-Q goes `[NSApp terminate:]` to `applicationWillTerminate:` to winit's
    /// `LoopExiting` to here, and AppKit calls `exit()` as soon as that returns. It is
    /// the last point at which the worker can be taken out of the model.
    fn shutdown(&self) -> ! {
        crate::shutdown::now(&self.state, "window closed")
    }
}

/// Ask the system for a file, blocking the thread it is called on.
///
/// Every platform's answer is the same shape: hand the job to a program that
/// already owns a file panel and read a path off its stdout. `None` means
/// cancelled, which all three report as a non-zero exit or empty output.
fn choose_file() -> Option<std::path::PathBuf> {
    let output = picker()?.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8(output.stdout).ok()?;
    let path = path.trim();
    (!path.is_empty()).then(|| std::path::PathBuf::from(path))
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
    let script = format!(
        "POSIX path of (choose file with prompt \"Choose a track to separate\" \
         of type {{{types}}})"
    );
    let mut cmd = std::process::Command::new("osascript");
    cmd.args(["-e", &script]);
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
         $d.Title = 'Choose a track to separate'; \
         $d.Filter = 'Audio|{patterns}|All files|*.*'; \
         if ($d.ShowDialog() -eq 'OK') {{ [Console]::Out.Write($d.FileName) }} else {{ exit 1 }}"
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
            "--title=Choose a track to separate",
            &format!("--file-filter=Audio | {patterns}"),
            "--file-filter=All files | *",
        ]);
        return Some(cmd);
    }
    if which("kdialog") {
        let mut cmd = std::process::Command::new("kdialog");
        cmd.args(["--getopenfilename", ".", &format!("{patterns}|Audio")]);
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
