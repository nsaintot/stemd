//! Asking before quitting mid-separation.
//!
//! The process already ends correctly under a close: [`crate::shutdown::now`]
//! takes the worker out of the model, gives it ten seconds, then withdraws the
//! mDNS advertisement and leaves. What was missing is being asked: a quality
//! separation is four minutes, and closing the window a minute in threw that
//! minute away silently.
//!
//! **This covers closing the window, not `Cmd-Q`.** The close button raises a
//! request egui can veto, while `Cmd-Q` is `[NSApp terminate:]`, which calls
//! `exit()` as soon as the delegate returns, so there is no point in that path
//! where a window can say "wait". `Cmd-Q` mid-separation is still safe and still
//! cancels the job; it just does not ask.

use std::sync::Arc;

use eframe::egui;

use crate::api::AppState;
use crate::ui::theme::Palette;

/// What the person chose, once they choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// Throw the separation away and go.
    Quit,
    /// Keep the window open and let it finish.
    Stay,
}

/// Whether a close should be held for a question, and what to say in it.
///
/// Held separately from drawing so the decision can be tested without a
/// viewport: the interesting cases are "a job is running" and "one is not",
/// and neither needs a window to check.
#[derive(Default)]
pub struct Guard {
    /// The job being asked about, if the question is on screen.
    asking: Option<String>,
    /// Set once the answer is yes, so the close that follows is not caught
    /// again by the same guard and turned into a second question.
    agreed: bool,
}

impl Guard {
    /// Decide what to do about a close request.
    ///
    /// Returns `true` when the close should proceed. A running job with no
    /// answer yet returns `false` and puts the question on screen.
    pub fn intercept(&mut self, running: Option<String>, closing: bool) -> bool {
        if !closing {
            return false;
        }
        if self.agreed {
            return true;
        }
        match running {
            Some(job) => {
                self.asking = Some(job);
                false
            }
            // Nothing to lose, so nothing to ask about. Closing an idle window
            // should never cost a click.
            None => true,
        }
    }

    /// The job the question is about, or `None` when it is not being asked.
    pub fn asking(&self) -> Option<&str> {
        self.asking.as_deref()
    }

    /// Record an answer. `Quit` arms the guard so the next close goes through.
    pub fn answer(&mut self, choice: Choice) {
        self.asking = None;
        self.agreed = choice == Choice::Quit;
    }
}

/// The question itself.
///
/// Takes the track's name rather than any state, so a probe can draw it.
pub fn dialog(ui: &mut egui::Ui, palette: &Palette, track: &str) -> Option<Choice> {
    ui.set_max_width(320.0);
    ui.heading("Still separating");
    ui.add_space(6.0);
    ui.label(
        egui::RichText::new(format!(
            "{track} is not finished. Quitting now throws away the work done \
             so far — nothing is written until a separation completes."
        ))
        .color(palette.muted),
    );
    ui.add_space(14.0);

    let mut choice = None;
    ui.horizontal(|ui| {
        // The safe option first and focused, because this dialog appears when
        // someone is already leaving and the destructive answer is the one
        // muscle memory will hit.
        if ui.button("Keep separating").clicked() {
            choice = Some(Choice::Stay);
        }
        if ui
            .button(egui::RichText::new("Quit anyway").color(palette.bad))
            .clicked()
        {
            choice = Some(Choice::Quit);
        }
    });
    choice
}

/// Hold the close, ask, and act on the answer.
///
/// Everything the window does about quitting, in one call.
pub fn guard(guard: &mut Guard, state: &Arc<AppState>, ctx: &egui::Context, palette: &Palette) {
    let closing = ctx.input(|i| i.viewport().close_requested());
    if closing && !guard.intercept(state.queue.running(), true) {
        ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
    }

    let Some(track) = guard.asking().map(str::to_owned) else {
        return;
    };
    let mut answered = None;
    egui::Modal::new(egui::Id::new("stemd-quit")).show(ctx, |ui| {
        answered = dialog(ui, palette, &track);
    });
    if let Some(choice) = answered {
        guard.answer(choice);
        if choice == Choice::Quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An idle window closes on the first ask, with no dialog.
    #[test]
    fn closing_with_nothing_running_does_not_ask() {
        let mut guard = Guard::default();
        assert!(guard.intercept(None, true), "an idle close was held up");
        assert_eq!(guard.asking(), None);
    }

    /// A close while separating is held, and says which track.
    #[test]
    fn closing_mid_separation_asks_first() {
        let mut guard = Guard::default();
        assert!(
            !guard.intercept(Some("track.wav".into()), true),
            "the close went through while a job was running"
        );
        assert_eq!(guard.asking(), Some("track.wav"));
    }

    /// Saying yes lets the close through, and only once asked.
    ///
    /// The second `intercept` is the part that matters. Answering `Quit` sends
    /// a fresh close request, which arrives back here on the next frame, and
    /// without `agreed` it would find the job still running and ask again,
    /// forever.
    #[test]
    fn agreeing_lets_the_next_close_through() {
        let mut guard = Guard::default();
        assert!(!guard.intercept(Some("track.wav".into()), true));
        guard.answer(Choice::Quit);
        assert_eq!(guard.asking(), None, "the dialog stayed up after an answer");
        assert!(
            guard.intercept(Some("track.wav".into()), true),
            "it asked a second time about a question already answered"
        );
    }

    /// Saying no dismisses the dialog and keeps the window.
    #[test]
    fn declining_keeps_the_window_and_can_be_asked_again() {
        let mut guard = Guard::default();
        assert!(!guard.intercept(Some("track.wav".into()), true));
        guard.answer(Choice::Stay);
        assert_eq!(guard.asking(), None);
        // And a later close is a fresh question rather than a remembered yes.
        assert!(!guard.intercept(Some("track.wav".into()), true));
        assert_eq!(guard.asking(), Some("track.wav"));
    }

    /// A frame with no close request neither asks nor closes.
    #[test]
    fn an_ordinary_frame_does_nothing() {
        let mut guard = Guard::default();
        assert!(!guard.intercept(Some("track.wav".into()), false));
        assert_eq!(guard.asking(), None);
    }
}
