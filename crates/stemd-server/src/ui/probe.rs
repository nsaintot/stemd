//! A headless egui harness, so a window bug can be reproduced without a window.
//!
//! [`egui::Context::run_ui`] takes synthetic input and returns the shapes it would
//! have drawn. It reads shapes rather than pixels, so it says nothing about how
//! the window looks; what it pins is geometry.

use eframe::egui;

/// A 560x600 window, matching [`super::WINDOW_SIZE`].
fn screen() -> egui::Rect {
    egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(560.0, 600.0))
}

/// A context with the window's fonts and style in place.
///
/// Fonts are bound at the start of the frame after they are set, and the
/// weighted families do not exist before that, so this runs one empty frame.
fn themed() -> egui::Context {
    let ctx = egui::Context::default();
    frame(&ctx, None, false, |_| {});
    super::theme::install(&ctx);
    frame(&ctx, None, false, |_| {});
    ctx
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
    let ctx = themed();
    let palette = super::theme::Palette::of(&ctx);
    let brushes = super::theme::Brushes::new(&ctx);
    let mut view = super::logs::View::default();
    let mut build = |ui: &mut egui::Ui| {
        let _ = super::logs::show(ui, &palette, &brushes, &mut view, &lines);
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
/// The tolerance is one row: the last row may be partly clipped.
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

/// A model download shows in the drop zone, with its byte count and figure,
/// and offers no file button while it runs.
#[test]
fn a_model_download_shows_in_the_drop_zone() {
    let ctx = themed();
    let palette = super::theme::Palette::of(&ctx);
    let brushes = super::theme::Brushes::new(&ctx);
    let motion = super::globe::Motion::default();

    let mut build = |ui: &mut egui::Ui| {
        let _ = super::dropzone::Zone {
            palette: &palette,
            brushes: &brushes,
            motion: &motion,
            armed: false,
            current: None,
            parts: crate::settings::DropParts::DEFAULT,
            batch: super::dropzone::Batch::default(),
            note: None,
            maxed: false,
            switch: Some(super::dropzone::Switch::Downloading {
                preset: crate::models::Preset::Quality,
                file: "stemd_roformer.safetensors",
                done: 6_000_000,
                total: 639_109_056,
            }),
            intro: None,
        }
        .show(ui);
    };
    frame(&ctx, None, false, &mut build);
    let shapes = frame(&ctx, None, false, &mut build);

    let clipped = clipped_text(&shapes);
    assert!(clipped.is_empty(), "clipped text: {clipped:?}");
    let drawn = drawn_text(&shapes);
    for want in ["Downloading the Quality model", "639 MB", "0%"] {
        assert!(
            drawn.iter().any(|t| t.contains(want)),
            "{want:?} is not drawn: {drawn:?}"
        );
    }
    assert!(
        !drawn.iter().any(|t| t == "Choose a file"),
        "the file button shows during a download"
    );
}

/// Every text the window draws in `shapes`.
fn drawn_text(shapes: &[egui::epaint::ClippedShape]) -> Vec<String> {
    shapes
        .iter()
        .filter_map(|c| match &c.shape {
            egui::epaint::Shape::Text(t) => Some(t.galley.text().to_owned()),
            _ => None,
        })
        .collect()
}

/// The first-run download shows its model, its progress, and fits.
#[test]
fn the_startup_download_shows_its_progress() {
    let ctx = themed();
    let palette = super::theme::Palette::of(&ctx);
    let brushes = super::theme::Brushes::new(&ctx);
    let mut view = super::starting::View::default();
    let stage = crate::launch::Stage::Downloading {
        preset: crate::models::Preset::Fast,
        file: "htdemucs.safetensors".into(),
        done: 84_000_000,
        total: 168_000_000,
    };

    let shapes = frame(&ctx, None, false, |ui| {
        let _ = super::starting::body(ui, &stage, &palette, &brushes, &mut view);
    });

    let clipped = clipped_text(&shapes);
    assert!(clipped.is_empty(), "clipped text: {clipped:?}");
    let drawn = drawn_text(&shapes);
    for want in ["Downloading the Fast model", "50%", "168 MB"] {
        assert!(
            drawn.iter().any(|t| t.contains(want)),
            "{want:?} is not drawn: {drawn:?}"
        );
    }
}

/// Without the model rows, the globe and the text under it sit in the middle.
#[test]
fn checking_the_model_at_startup_is_centred() {
    let ctx = themed();
    let palette = super::theme::Palette::of(&ctx);
    let brushes = super::theme::Brushes::new(&ctx);
    let mut view = super::starting::View::default();
    let stage = crate::launch::Stage::Verifying(crate::models::Preset::Quality);

    let shapes = frame(&ctx, None, false, |ui| {
        let _ = super::starting::body(ui, &stage, &palette, &brushes, &mut view);
    });
    let title = shapes
        .iter()
        .find_map(|c| match &c.shape {
            egui::epaint::Shape::Text(t)
                if t.galley.text() == "Checking downloaded Quality model" =>
            {
                Some(t.pos.y)
            }
            _ => None,
        })
        .expect("the title");
    let height = screen().height();
    assert!(
        title > height * 0.45 && title < height * 0.65,
        "the title sits at {title:.0} of {height:.0} points"
    );
}

/// A first run offers every preset with its download size, and a click picks one.
#[test]
fn a_first_run_offers_every_model_and_picks_the_clicked_one() {
    let ctx = themed();
    let palette = super::theme::Palette::of(&ctx);
    let brushes = super::theme::Brushes::new(&ctx);
    let mut view = super::starting::View::default();
    let stage = crate::launch::Stage::Choosing(vec![crate::models::Preset::Balanced]);

    let mut picked = None;
    let shapes = frame(&ctx, None, false, |ui| {
        picked = super::starting::body(ui, &stage, &palette, &brushes, &mut view);
    });
    assert_eq!(picked, None, "nothing was clicked");
    let drawn = drawn_text(&shapes);
    for preset in crate::models::Preset::ALL {
        assert!(
            drawn.iter().any(|t| t == preset.label()),
            "{preset:?} is not offered: {drawn:?}"
        );
    }
    assert!(drawn.iter().any(|t| t == "installed"), "{drawn:?}");
    assert!(drawn.iter().any(|t| t.ends_with(" MB")), "{drawn:?}");

    // Click the middle of the first button, found by its label.
    let at = shapes
        .iter()
        .find_map(|c| match &c.shape {
            egui::epaint::Shape::Text(t) if t.galley.text() == "Fast" => {
                Some(t.pos + t.galley.rect.center().to_vec2())
            }
            _ => None,
        })
        .expect("the Fast button");
    // A press and release need a frame with the pointer already there.
    frame(&ctx, Some(at), false, |ui| {
        let _ = super::starting::body(ui, &stage, &palette, &brushes, &mut view);
    });
    frame(&ctx, Some(at), true, |ui| {
        picked = super::starting::body(ui, &stage, &palette, &brushes, &mut view);
    });
    assert_eq!(picked, Some(crate::models::Preset::Fast));
}

/// A failed startup says why and offers a way out.
#[test]
fn a_failed_startup_shows_the_error_and_a_quit_button() {
    let ctx = themed();
    let palette = super::theme::Palette::of(&ctx);
    let brushes = super::theme::Brushes::new(&ctx);
    let mut view = super::starting::View::default();
    let stage = crate::launch::Stage::Failed("fetching the model: GET https://x: timed out".into());

    let shapes = frame(&ctx, None, false, |ui| {
        let _ = super::starting::body(ui, &stage, &palette, &brushes, &mut view);
    });

    let drawn = drawn_text(&shapes);
    for want in ["Stemd could not start", "timed out", "Quit"] {
        assert!(
            drawn.iter().any(|t| t.contains(want)),
            "{want:?} is not drawn: {drawn:?}"
        );
    }
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

/// Rectangles drawn, at any depth.
fn rects(shapes: &[egui::epaint::ClippedShape]) -> Vec<egui::Rect> {
    fn walk(shape: &egui::epaint::Shape, out: &mut Vec<egui::Rect>) {
        match shape {
            egui::epaint::Shape::Rect(r) => out.push(r.rect),
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

/// A working drop shows its title, its stage and a bar centred under them.
///
/// The bar and its figure are one row: centring the bar alone would push the row
/// right by half the figure's width.
#[test]
fn the_drop_zone_progress_bar_is_centred() {
    let ctx = themed();
    let palette = super::theme::Palette::of(&ctx);
    let brushes = super::theme::Brushes::new(&ctx);
    let motion = super::globe::Motion::default();
    let item = crate::drops::Dropped::working(
        "Bells Of Eternity",
        stemd_core::Progress {
            stage: stemd_core::Stage::Separating,
            completed: 7,
            total: 24,
            fraction: 0.31,
            detail: None,
        },
    );

    let mut zone_rect = egui::Rect::NOTHING;
    let mut build = |ui: &mut egui::Ui| {
        zone_rect = ui.available_rect_before_wrap();
        let _ = super::dropzone::Zone {
            palette: &palette,
            brushes: &brushes,
            motion: &motion,
            armed: false,
            current: Some(&item),
            parts: crate::settings::DropParts::DEFAULT,
            batch: super::dropzone::Batch::default(),
            note: None,
            maxed: false,
            switch: None,
            intro: None,
        }
        .show(ui);
    };
    frame(&ctx, None, false, &mut build);
    let shapes = frame(&ctx, None, false, &mut build);

    let drawn = drawn_text(&shapes);
    for want in [
        "Bells Of Eternity",
        "separating · 7/24",
        "31%",
        "Queue another track",
    ] {
        assert!(
            drawn.iter().any(|t| t == want),
            "{want:?} is not drawn: {drawn:?}"
        );
    }

    let track = rects(&shapes)
        .into_iter()
        .find(|r| (r.width() - 260.0).abs() < 0.5 && (r.height() - 8.0).abs() < 0.5)
        .expect("the progress track");
    let before = track.left() - zone_rect.left();
    let after = zone_rect.right() - track.right();
    assert!(
        (before - after).abs() <= 1.0,
        "the bar is not centred: {before:.1} points on the left, {after:.1} on the right"
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
    let ctx = themed();
    let palette = super::theme::Palette::of(&ctx);
    let brushes = super::theme::Brushes::new(&ctx);

    let shapes = frame(&ctx, None, false, |ui| {
        super::quit::dialog(ui, &palette, &brushes, "Bells Of Eternity.flac");
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
    let ctx = themed();
    let palette = super::theme::Palette::of(&ctx);
    let brushes = super::theme::Brushes::new(&ctx);

    let drops = crate::drops::Drops::default();
    let dir = std::path::Path::new("/Music/track-stems");
    let items: Vec<_> = (0..rows)
        .map(|i| crate::drops::Dropped::finished(&format!("track {i}"), dir, 4))
        .collect();

    let mut height = 0.0;
    let mut build = |ui: &mut egui::Ui| {
        height = ui
            .scope(|ui| {
                super::recents::list(ui, &palette, &brushes, &drops, &items, visible_cap());
            })
            .response
            .rect
            .height();
    };
    // Twice: the scroll area sizes itself from what it measured last frame.
    frame(&ctx, None, false, &mut build);
    frame(&ctx, None, false, &mut build);
    height
}

/// A full list stops at the room it is given.
#[test]
fn a_long_separated_list_stops_at_the_cap() {
    let drawn = recents_height(crate::drops::REMEMBERED);
    assert!(
        drawn <= visible_cap() + 1.0,
        "twelve rows took {drawn:.1} points, past the {:.1}-point cap",
        visible_cap()
    );
}

/// And two rows are not stretched to fill it.
#[test]
fn a_short_separated_list_stays_short() {
    let drawn = recents_height(2);
    assert!(
        drawn < visible_cap(),
        "two rows took up {drawn:.1} points, the whole {:.1}-point box",
        visible_cap()
    );
}

/// The room the list is given in these probes: three and a half rows.
fn visible_cap() -> f32 {
    super::recents::ROW_HEIGHT * 3.5
}

/// Every non-ASCII character the window draws as text has a glyph in the fonts
/// it binds. A missing glyph draws nothing at all, so this fails silently on
/// screen.
#[test]
fn the_marks_drawn_as_text_have_glyphs() {
    use super::theme::{Weight, font};
    let ctx = themed();
    for weight in [Weight::Regular, Weight::Medium, Weight::Semibold] {
        for c in ['·', '…'] {
            let id = font(13.0, weight);
            assert!(
                ctx.fonts_mut(|f| f.has_glyph(&id, c)),
                "{c:?} (U+{:04X}) has no glyph at {weight:?}",
                c as u32
            );
        }
    }
}

/// Milliseconds of CPU per frame for the drop zone, building and tessellating.
///
/// `cargo test --release -p stemd-server frame_cost -- --ignored --nocapture`
#[test]
#[ignore = "a measurement, run by hand"]
fn frame_cost() {
    let ctx = themed();
    let palette = super::theme::Palette::of(&ctx);
    let brushes = super::theme::Brushes::new(&ctx);
    let item = crate::drops::Dropped::working(
        "Bells Of Eternity",
        stemd_core::Progress {
            stage: stemd_core::Stage::Separating,
            completed: 7,
            total: 24,
            fraction: 0.31,
            detail: None,
        },
    );
    for (name, working) in [("idle", false), ("working", true)] {
        let mut motion = super::globe::Motion::default();
        let drive = super::globe::Drive {
            variant: working.then_some(2),
            speed: 1.0,
            spin: working.then_some(4.5),
        };
        let frames = 600;
        let mut spent = std::time::Duration::ZERO;
        for n in 0..frames {
            let now = f64::from(n) / 60.0;
            motion.step(1.0 / 60.0, now, &drive);
            let input = egui::RawInput {
                screen_rect: Some(screen()),
                time: Some(now),
                ..Default::default()
            };
            let start = std::time::Instant::now();
            let output = ctx.run_ui(input, |ui| {
                brushes.backdrop(ui.painter(), ui.max_rect());
                let _ = super::dropzone::Zone {
                    palette: &palette,
                    brushes: &brushes,
                    motion: &motion,
                    armed: false,
                    current: working.then_some(&*item),
                    parts: crate::settings::DropParts::DEFAULT,
                    batch: super::dropzone::Batch::default(),
                    note: None,
                    maxed: true,
                    switch: None,
                    intro: None,
                }
                .show(ui);
            });
            let primitives = ctx.tessellate(output.shapes, output.pixels_per_point);
            if n >= 60 {
                spent += start.elapsed();
            }
            if n == frames - 1 {
                let vertices: usize = primitives
                    .iter()
                    .map(|p| match &p.primitive {
                        egui::epaint::Primitive::Mesh(m) => m.vertices.len(),
                        egui::epaint::Primitive::Callback(_) => 0,
                    })
                    .sum();
                println!(
                    "{name}: {} primitives, {vertices} vertices",
                    primitives.len()
                );
            }
        }
        println!(
            "{name}: {:.3} ms per frame",
            spent.as_secs_f64() * 1000.0 / f64::from(frames - 60)
        );
    }
}
