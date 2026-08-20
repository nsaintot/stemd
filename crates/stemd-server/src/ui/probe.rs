//! A headless egui harness, so a window bug can be reproduced without a window.
//!
//! [`egui::Context::run_ui`] takes synthetic input and returns the shapes it would
//! have drawn. It reads shapes rather than pixels, so it says nothing about how
//! the window looks; what it pins is geometry.

use eframe::egui;

/// A 460x470 window, matching [`super::WINDOW_SIZE`].
fn screen() -> egui::Rect {
    egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(460.0, 470.0))
}

/// Run one frame with the pointer at `pointer`, optionally clicking, and return
/// the shapes egui produced.
fn frame(
    ctx: &egui::Context,
    pointer: Option<egui::Pos2>,
    click: bool,
    mut build: impl FnMut(&mut egui::Ui),
) -> Vec<egui::epaint::ClippedShape> {
    let mut input = egui::RawInput {
        screen_rect: Some(screen()),
        ..Default::default()
    };
    if let Some(pos) = pointer {
        input.events.push(egui::Event::PointerMoved(pos));
        if click {
            for pressed in [true, false] {
                input.events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::default(),
                });
            }
        }
    }
    ctx.run_ui(input, |ui| build(ui)).shapes
}

/// Rects narrow enough to be a scroll bar and tall enough to be worth drawing.
///
/// egui's floating bars are about a point wide and animate in and out, so this
/// deliberately catches a partly-faded one: a bar on its way in is still a bar
/// that appeared.
fn scroll_bars(shapes: &[egui::epaint::ClippedShape]) -> Vec<egui::Rect> {
    fn walk(shape: &egui::epaint::Shape, out: &mut Vec<egui::Rect>) {
        match shape {
            egui::epaint::Shape::Rect(r) if r.rect.width() <= 12.0 && r.rect.height() >= 20.0 => {
                out.push(r.rect);
            }
            egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    for clipped in shapes {
        walk(&clipped.shape, &mut out);
    }
    out
}

/// A dropdown built the way [`super::header`] builds its three.
fn dropdown(ui: &mut egui::Ui, borderless: bool) {
    egui::ComboBox::from_id_salt("probe")
        .selected_text("MP3 320")
        .width(88.0)
        .height(super::header::MENU_CAP)
        .show_ui(ui, |ui| {
            if borderless {
                super::header::borderless_rows(ui);
            }
            for label in ["FLAC", "WAV", "MP3 320", "PCM 16", "PCM 32"] {
                ui.selectable_label(false, label)
                    .on_hover_text("320 kbps, and the only lossy option.");
            }
        });
}

/// Open the dropdown, then walk the pointer down the open list, counting the
/// positions at which a scroll bar is drawn.
///
/// A sweep rather than a hand-picked coordinate: row heights come out of the
/// theme, so a guessed position lands between rows the moment the theme changes
/// and the test silently stops testing anything.
fn positions_growing_a_bar(borderless: bool) -> usize {
    let ctx = egui::Context::default();
    let build = |ui: &mut egui::Ui| dropdown(ui, borderless);

    // A frame before `install`, which reads the context's current theme.
    frame(&ctx, None, false, build);
    super::theme::install(&ctx);
    frame(&ctx, None, false, build);

    let button = egui::pos2(40.0, 10.0);
    frame(&ctx, Some(button), true, build);
    frame(&ctx, Some(button), false, build);

    (0..60u8)
        .filter(|step| {
            let at = egui::pos2(40.0, 4.0 * f32::from(*step));
            // Twice: egui settles a frame behind its own input.
            frame(&ctx, Some(at), false, build);
            !scroll_bars(&frame(&ctx, Some(at), false, build)).is_empty()
        })
        .count()
}

/// A list of five items in a box that caps at 600 points has nothing to scroll,
/// and must not say otherwise.
#[test]
fn no_dropdown_grows_a_scroll_bar() {
    assert_eq!(
        positions_growing_a_bar(true),
        0,
        "a dropdown grew a scroll bar with nothing to scroll to"
    );
}

/// The above passing means nothing unless it can fail. This is the same sweep
/// over the same list with the row borders left on: the bug as it shipped.
#[test]
fn the_sweep_catches_the_bar_it_is_meant_to_catch() {
    assert!(
        positions_growing_a_bar(false) > 0,
        "the sweep found no bar even with the borders that caused it, so it is \
         not testing anything"
    );
}

/// The vertical extent of every log line drawn, given `total` buffered lines.
fn log_text_extent(total: usize) -> egui::Rect {
    use crate::logbuf::LogLine;
    let lines: Vec<LogLine> = (0..total)
        .map(|i| LogLine {
            at_ms: 0,
            level: "INFO".to_owned(),
            target: "stemd_server::api".to_owned(),
            message: format!("line number {i} of the log"),
        })
        .collect();
    let ctx = egui::Context::default();
    frame(&ctx, None, false, |_| {});
    super::theme::install(&ctx);
    let palette = super::theme::Palette::of(&ctx);
    let mut view = super::logs::View::default();
    let mut build = |ui: &mut egui::Ui| {
        let _ = super::logs::show(ui, &palette, &mut view, &lines);
    };
    frame(&ctx, None, false, &mut build);
    let shapes = frame(&ctx, None, false, &mut build);

    let mut extent = egui::Rect::NOTHING;
    for c in &shapes {
        if let egui::epaint::Shape::Text(t) = &c.shape
            && t.galley.text().starts_with("line number")
        {
            extent = extent.union(egui::Rect::from_min_size(t.pos, t.galley.size()));
        }
    }
    extent
}

/// A log with far more lines than fit must draw enough of them to fill the box.
///
/// `show_rows` decides how many rows to hand out by adding `item_spacing.y` to the
/// row height it is given, so spacing set inside its closure comes too late and the
/// rows draw short of the box. The tolerance is one row, not zero: the last row may
/// be partly clipped.
#[test]
fn a_full_log_fills_the_box_it_is_given() {
    let drawn = log_text_extent(200).height();
    let room = super::logs::VISIBLE_HEIGHT;
    assert!(
        drawn > room - 12.0,
        "the log drew {drawn:.1} points of lines into a {room:.0}-point box, \
         leaving {:.1} empty",
        room - drawn
    );
}

/// And the other direction: a handful of lines must not be stretched to fill it.
#[test]
fn a_short_log_stays_short() {
    let drawn = log_text_extent(4).height();
    assert!(
        drawn < super::logs::VISIBLE_HEIGHT / 2.0,
        "four lines took up {drawn:.1} points"
    );
}

/// Text drawn outside the region it is clipped to, with what it says.
///
/// This is the shape of a whole family of window bugs: a string put somewhere
/// too small for it, which renders as ribbons rather than as an error. Reading
/// the galley's own rect against the clip rect catches it without anyone
/// looking.
fn clipped_text(shapes: &[egui::epaint::ClippedShape]) -> Vec<(String, f32)> {
    fn walk(shape: &egui::epaint::Shape, clip: egui::Rect, out: &mut Vec<(String, f32)>) {
        match shape {
            egui::epaint::Shape::Text(text) => {
                let drawn = text.galley.rect.translate(text.pos.to_vec2());
                // A point of slack: egui rounds glyph rects and a descender
                // grazing the edge is not the bug being looked for.
                let overflow = (drawn.height() - clip.height()).max(0.0);
                if overflow > 1.0 {
                    out.push((text.galley.text().to_owned(), overflow));
                }
            }
            egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, clip, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    for clipped in shapes {
        walk(&clipped.shape, clipped.clip_rect, &mut out);
    }
    out
}

/// The model-download dialog draws its byte count somewhere it fits.
///
/// It did not: the count was the progress bar's own text, and the bar is six
/// points tall to match the hairlines elsewhere in the window, so egui clipped
/// a fourteen-point string into it. The fix moves the count out of the bar,
/// and this is what says it stayed out.
#[test]
fn the_download_dialog_does_not_clip_its_own_text() {
    let ctx = egui::Context::default();
    let palette = super::theme::Palette::of(&ctx);

    let shapes = frame(&ctx, None, false, |ui| {
        super::header::downloading(
            ui,
            crate::models::Preset::Quality,
            "bs_roformer_viperx.safetensors",
            6_000_000,
            639_109_056,
            &palette,
        );
    });

    let clipped = clipped_text(&shapes);
    assert!(
        clipped.is_empty(),
        "text is drawn taller than the region it is clipped to: {clipped:?}"
    );

    // And the count is actually on screen, rather than absent and therefore
    // trivially unclipped.
    let drawn: Vec<String> = shapes
        .iter()
        .filter_map(|c| match &c.shape {
            egui::epaint::Shape::Text(t) => Some(t.galley.text().to_owned()),
            _ => None,
        })
        .collect();
    assert!(
        drawn.iter().any(|t| t.contains("639 MB")),
        "the byte count is not drawn at all: {drawn:?}"
    );
    assert!(
        drawn.iter().any(|t| t.contains("0%")),
        "the percentage is not drawn at all: {drawn:?}"
    );
}

/// The figure beside a bar never reads 100% before the work is done, and never
/// runs past three characters.
///
/// Both are about the same thing: the room reserved for it beside the bar is
/// fixed, and a bar that says it has finished while it is still going is worse
/// than one that says 99%.
#[test]
fn the_percentage_beside_a_bar_is_honest_and_fits() {
    use super::status::percent;

    assert_eq!(percent(0.0), "0%");
    assert_eq!(percent(1.0), "100%");
    // Anything short of finished must round down, not to nearest.
    assert_eq!(percent(0.999), "99%");
    assert_eq!(percent(0.995), "99%");
    // A fraction outside the range is a bug elsewhere, not a reason to draw
    // "-3%" or "31400%" into a 34-point gap.
    assert_eq!(percent(-0.5), "0%");
    assert_eq!(percent(2.5), "100%");

    for step in 0..=1000 {
        let text = percent(step as f32 / 1000.0);
        assert!(
            text.len() <= 4,
            "{text} is wider than the space kept for it"
        );
    }
}

/// The horizontal extent of everything drawn, ignoring the backdrop egui paints
/// across the whole panel.
fn drawn_extent(shapes: &[egui::epaint::ClippedShape], within: egui::Rect) -> Option<(f32, f32)> {
    fn walk(shape: &egui::epaint::Shape, within: egui::Rect, out: &mut Vec<egui::Rect>) {
        match shape {
            egui::epaint::Shape::Rect(r) => {
                // Anything spanning the full width is the background, not
                // content, and would make every layout look centred.
                if r.rect.width() < within.width() - 1.0 && r.rect.width() > 0.0 {
                    out.push(r.rect);
                }
            }
            egui::epaint::Shape::Text(t) => out.push(t.galley.rect.translate(t.pos.to_vec2())),
            egui::epaint::Shape::Vec(v) => v.iter().for_each(|s| walk(s, within, out)),
            _ => {}
        }
    }
    let mut rects = Vec::new();
    for clipped in shapes {
        walk(&clipped.shape, within, &mut rects);
    }
    let left = rects.iter().map(|r| r.left()).fold(f32::MAX, f32::min);
    let right = rects.iter().map(|r| r.right()).fold(f32::MIN, f32::max);
    (left <= right).then_some((left, right))
}

/// The drop zone's bar and its figure sit centred, as one row.
///
/// Wrapping the pair in a plain `ui.horizontal` gives it the whole available width,
/// so its contents start hard against the left edge while the title and caption
/// stay centred. A test that only asks whether the text fits cannot see that.
#[test]
fn the_drop_zone_progress_row_is_centred() {
    let ctx = egui::Context::default();
    let palette = super::theme::Palette::of(&ctx);

    let shapes = frame(&ctx, None, false, |ui| {
        let width = ui.available_width();
        let mut centred = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(egui::Rect::from_min_size(
                    ui.min_rect().min,
                    egui::vec2(width, 60.0),
                ))
                .layout(egui::Layout::top_down(egui::Align::Center)),
        );
        super::dropzone::progress_row(&mut centred, Some(0.31), &palette);
    });

    let bounds = screen();
    let (left, right) = drawn_extent(&shapes, bounds).expect("the row drew something");
    let before = left - bounds.left();
    let after = bounds.right() - right;

    //  The row is allocated at exactly its own width and centred by the enclosing
    //  layout, so what is left is ink: the figure keeps a fixed 34-point box so the bar
    //  does not twitch as the number goes from one digit to three.
    //
    //  Measured: 79 and 85 with the row allocated, 0 and 164 without.
    let slack = super::dropzone::PERCENT_WIDTH / 2.0;
    assert!(
        (before - after).abs() <= slack,
        "the row is not centred: {before:.0} points of space on the left, \
         {after:.0} on the right"
    );
}

/// The quit confirmation draws both answers and names the track.
///
/// The failure this guards against is not cosmetic. A dialog that appears with
/// no way to say no is worse than no dialog: the close has already been vetoed
/// by the time it is drawn, so a missing "Keep separating" leaves a window that
/// cannot be closed at all except by quitting the process another way.
#[test]
fn the_quit_confirmation_offers_both_answers() {
    let ctx = egui::Context::default();
    super::theme::install(&ctx);
    let palette = super::theme::Palette::of(&ctx);

    let shapes = frame(&ctx, None, false, |ui| {
        super::quit::dialog(ui, &palette, "Bells Of Eternity.flac");
    });

    let drawn: Vec<String> = shapes
        .iter()
        .filter_map(|c| match &c.shape {
            egui::epaint::Shape::Text(t) => Some(t.galley.text().to_owned()),
            _ => None,
        })
        .collect();

    for wanted in ["Keep separating", "Quit anyway"] {
        assert!(
            drawn.iter().any(|t| t.contains(wanted)),
            "the dialog has no {wanted:?}: {drawn:?}"
        );
    }
    assert!(
        drawn.iter().any(|t| t.contains("Bells Of Eternity")),
        "the dialog does not say which track is running: {drawn:?}"
    );

    // Nothing is clipped away: this text is long and the modal is narrow, and a
    // truncated warning is the second time that has happened in this window.
    let clipped = clipped_text(&shapes);
    assert!(
        clipped.is_empty(),
        "the dialog clips its own text: {clipped:?}"
    );
}

/// The height the separated list takes in the layout, given `rows` finished drops.
///
/// The layout height is the thing under test rather than what the rows draw: the
/// window measures the sections panel and resizes itself by the difference, so a
/// list that reports twelve rows tall grows the window whether or not it paints
/// them.
fn recents_height(rows: usize) -> f32 {
    let ctx = egui::Context::default();
    frame(&ctx, None, false, |_| {});
    super::theme::install(&ctx);
    let palette = super::theme::Palette::of(&ctx);
    let mut sheen = super::theme::Sheen::new(&ctx);

    let drops = crate::drops::Drops::default();
    let dir = std::path::Path::new("/Music/track-stems");
    let items: Vec<_> = (0..rows)
        .map(|i| crate::drops::Dropped::finished(&format!("track {i}"), dir, 4))
        .collect();

    let mut height = 0.0;
    let mut build = |ui: &mut egui::Ui| {
        height = ui
            .scope(|ui| super::recents::list(ui, &palette, &mut sheen, &drops, &items))
            .response
            .rect
            .height();
    };
    // Twice: the scroll area sizes itself from what it measured last frame.
    frame(&ctx, None, false, &mut build);
    frame(&ctx, None, false, &mut build);
    height
}

/// A full list must stop growing at the cap.
///
/// It had no cap at all: every separated track added forty points to the
/// sections panel, `match_window_to` handed each one to the window, and twelve
/// drops, the most the list remembers, took a 470-point window past 1000 and
/// off the bottom of the screen.
#[test]
fn a_long_separated_list_stops_at_the_cap() {
    let drawn = recents_height(crate::drops::REMEMBERED);
    assert!(
        drawn <= visible_cap() + 1.0,
        "twelve rows took {drawn:.1} points, past the {:.1}-point cap",
        visible_cap()
    );
}

/// And the other direction: two rows must not be stretched to fill the cap.
#[test]
fn a_short_separated_list_stays_short() {
    let drawn = recents_height(2);
    assert!(
        drawn < visible_cap(),
        "two rows took up {drawn:.1} points, the whole {:.1}-point box",
        visible_cap()
    );
}

/// The cap [`super::recents`] applies, read through the same theme the tests
/// draw under.
fn visible_cap() -> f32 {
    let ctx = egui::Context::default();
    frame(&ctx, None, false, |_| {});
    super::theme::install(&ctx);
    let mut cap = 0.0;
    frame(&ctx, None, false, |ui| {
        cap = super::recents::visible_height(ui)
    });
    cap
}
