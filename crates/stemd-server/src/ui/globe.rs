//! The globe: three slices of a sphere in the stem colours, and what moves them.
//!
//! Geometry is in the design's SVG units: a circle of radius 52 at (60, 60),
//! cut into vertical strips at x 0..43, 47..73 and 77..120, drawn in a
//! 114 by 127 point stage whose view box starts at (-8, -12). Each slice is
//! clipped on the CPU and transformed point by point, because slices rotate and
//! egui only clips to axis-aligned rectangles.

use std::f32::consts::{FRAC_PI_2, TAU};

use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2, pos2, vec2};

use super::theme::{self, Brushes, Palette};

/// The stage the globe is drawn in, in points.
pub const STAGE: Vec2 = Vec2 { x: 114.0, y: 127.0 };

/// Points per SVG unit.
const K: f32 = 114.0 / 136.0;
const VIEW_MIN: Vec2 = Vec2 { x: -8.0, y: -12.0 };

const CENTRE: Pos2 = Pos2 { x: 60.0, y: 60.0 };
const RADIUS: f32 = 52.0;

/// What CSS `transform-origin: 50% 50%` resolves to for the whole globe.
const GLOBE_ORIGIN: Pos2 = Pos2 { x: 60.0, y: 63.5 };

/// Left and right edge of each slice.
const STRIPS: [(f32, f32); 3] = [(0.0, 43.0), (47.0, 73.0), (77.0, 120.0)];
const PARALLELS: [f32; 6] = [21.0, 34.0, 47.0, 73.0, 86.0, 99.0];
const BAND: (f32, f32) = (56.5, 63.5);

/// One-shot effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fx {
    /// A separation started: the globe flinches and a ring goes out.
    Start,
    /// The batch finished: a wobble and three coloured rings.
    Cheer,
    /// Picked Fast: the meridians whip round and streaks fly past.
    Fast,
    /// Picked Balanced: the globe tilts back and forth.
    Balanced,
    /// Picked Quality: the globe swells.
    Quality,
}

impl Fx {
    /// How long the effect, and any note shown with it, lasts.
    pub const fn duration(self) -> f32 {
        match self {
            Self::Start => 0.9,
            Self::Cheer => 2.4,
            Self::Fast | Self::Balanced | Self::Quality => 1.7,
        }
    }
}

/// What drives the motion this frame.
pub struct Drive {
    /// `Some` while separating: which of the five motions to play.
    pub variant: Option<u8>,
    /// How fast the motion runs: 1.6 on Fast, 1 on Balanced, 0.6 on Quality.
    pub speed: f32,
    /// Seconds per meridian turn while working, `None` at rest.
    pub spin: Option<f32>,
}

/// Where [`Motion::clock`] wraps: a multiple of every period drawn from it
/// (spinner 1 s, shimmer 1.6 s, sheen 2.5 s, halo 7 s), so nothing jumps.
const CLOCK_WRAP: f32 = 280.0;

/// Seconds since the previous frame, measured on egui's clock.
///
/// egui's `stable_dt` reports a fixed 1/60 s after a frame that asked for a
/// delayed repaint, which runs animations fast on a faster display.
pub fn frame_dt(ctx: &egui::Context) -> f32 {
    let id = egui::Id::new("stemd-frame-dt");
    let now = ctx.input(|i| i.time);
    let last = ctx.data(|d| d.get_temp::<f64>(id));
    ctx.data_mut(|d| d.insert_temp(id, now));
    last.map_or(1.0 / 60.0, |last| (now - last).max(0.0) as f32)
}

/// The animation state, stepped once per frame.
#[derive(Clone, Debug)]
pub struct Motion {
    phase: f32,
    amp: f32,
    speed: f32,
    from: u8,
    to: u8,
    mix: f32,
    /// Meridian position, in turns.
    meridian: f32,
    /// Seconds of ambient time, which only runs while the window is live.
    /// Wraps at [`CLOCK_WRAP`].
    pub clock: f32,
    fx: Option<(Fx, f64)>,
}

impl Default for Motion {
    fn default() -> Self {
        Self {
            phase: 0.0,
            amp: 0.0,
            speed: 1.0,
            from: 0,
            to: 0,
            mix: 1.0,
            meridian: 0.0,
            clock: 0.0,
            fx: None,
        }
    }
}

/// Pick one of the five motions for a job, stable for its id.
pub fn variant_of(id: &str) -> u8 {
    // FNV-1a.
    let hash = id.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    (hash % 5) as u8
}

impl Motion {
    /// Start a one-shot, replacing any in flight.
    pub fn fire(&mut self, fx: Fx, now: f64) {
        self.fx = Some((fx, now));
    }

    /// The live one-shot and seconds since it started.
    pub fn fx(&self, now: f64) -> Option<(Fx, f32)> {
        let (fx, start) = self.fx?;
        let elapsed = (now - start) as f32;
        (elapsed < fx.duration()).then_some((fx, elapsed))
    }

    /// Advance by `dt` seconds. Returns whether anything is moving, which is
    /// whether the window should repaint every frame.
    pub fn step(&mut self, dt: f32, now: f64, drive: &Drive) -> bool {
        let dt = dt.min(0.05);
        if let Some(v) = drive.variant {
            if self.amp < 0.05 {
                self.from = v;
                self.to = v;
                self.mix = 1.0;
            } else if v != self.to {
                self.from = self.to;
                self.to = v;
                self.mix = 0.0;
            }
        }
        let k = 1.0 - (-dt * 3.5).exp();
        let target = if drive.variant.is_some() { 1.0 } else { 0.0 };
        self.amp += (target - self.amp) * k;
        self.speed += (drive.speed - self.speed) * k;
        self.mix = (self.mix + dt / 1.1).min(1.0);
        // Only ever read through sin and cos, so it wraps without a seam.
        self.phase = (self.phase + dt * self.speed * 4.4).rem_euclid(TAU);
        if self.amp <= 0.003 && drive.variant.is_none() {
            self.amp = 0.0;
        }

        let whip = matches!(self.fx(now), Some((Fx::Fast, _)));
        let period = if whip { Some(0.45) } else { drive.spin };
        if let Some(period) = period {
            self.meridian = (self.meridian + dt / period).rem_euclid(1.0);
        }
        if self.fx.is_some() && self.fx(now).is_none() {
            self.fx = None;
        }

        let live = self.amp > 0.0 || self.fx.is_some() || period.is_some();
        if live {
            self.clock = (self.clock + dt).rem_euclid(CLOCK_WRAP);
        }
        live
    }

    /// Translate x, translate y, rotation in degrees, and extra scale x and y,
    /// for slice `i` under motion `v`.
    fn motion(&self, v: u8, i: usize) -> [f32; 5] {
        let a = self.phase - i as f32 * 1.05;
        let sn = a.sin();
        let up = sn.max(0.0);
        let dn = (-sn).max(0.0);
        match v {
            // Wave.
            0 => [0.0, 10.0 * sn, 3.0 * a.cos(), 0.0, 0.0],
            // Accordion.
            1 => {
                let c = 0.5 - 0.5 * self.phase.cos();
                if i == 1 {
                    [0.0, 2.0 - 8.0 * c, 0.0, 0.07 * c, 0.07 * c]
                } else {
                    let side = if i == 0 { -1.0 } else { 1.0 };
                    [side * (9.0 * c - 3.0), 0.0, 0.0, 0.0, 0.0]
                }
            }
            // Hop.
            2 => {
                let h = up.powf(1.6);
                [0.0, -14.0 * h, 0.0, 0.0, 0.07 * h - 0.06 * dn * dn]
            }
            // Swell.
            3 => {
                let h = up * up;
                [0.0, -3.0 * h, 0.0, 0.13 * h, 0.13 * h]
            }
            // Rock.
            _ => [0.0, 0.0, 12.0 * sn, 0.0, 0.0],
        }
    }

    /// Slice `i`'s transform, cross-faded between the old and new motion and
    /// scaled by how far the motion has eased in.
    fn slice(&self, i: usize) -> [f32; 5] {
        let m = self.mix * self.mix * (3.0 - 2.0 * self.mix);
        let a = self.motion(self.from, i);
        let b = self.motion(self.to, i);
        std::array::from_fn(|n| (a[n] * (1.0 - m) + b[n] * m) * self.amp)
    }

    fn meridians(&self) -> [f32; 3] {
        std::array::from_fn(|n| {
            let turn = (self.meridian + n as f32 / 3.0).rem_euclid(1.0);
            (FRAC_PI_2 * turn).sin()
        })
    }
}

/// Everything about the globe's look that is not its motion.
#[derive(Clone, Copy, Default)]
pub struct Look {
    /// 0 at rest, 1 with the outer slices pushed apart.
    pub spread: f32,
    /// Points the globe is raised by, under a hovering pointer.
    pub lift: f32,
    /// Opacity of the Quality halo, 0 to 1.
    pub halo: f32,
}

/// A 2D affine map: `x' = a x + c y + e`, `y' = b x + d y + f`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Affine([f32; 6]);

impl Affine {
    #[cfg(test)]
    const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    fn translate(v: Vec2) -> Self {
        Self([1.0, 0.0, 0.0, 1.0, v.x, v.y])
    }

    fn rotate(degrees: f32) -> Self {
        let (s, c) = degrees.to_radians().sin_cos();
        Self([c, s, -s, c, 0.0, 0.0])
    }

    fn scale(x: f32, y: f32) -> Self {
        Self([x, 0.0, 0.0, y, 0.0, 0.0])
    }

    /// `self` applied after `inner`.
    fn after(self, inner: Self) -> Self {
        let [a, b, c, d, e, f] = self.0;
        let [a2, b2, c2, d2, e2, f2] = inner.0;
        Self([
            a * a2 + c * b2,
            b * a2 + d * b2,
            a * c2 + c * d2,
            b * c2 + d * d2,
            a * e2 + c * f2 + e,
            b * e2 + d * f2 + f,
        ])
    }

    /// `self` with its origin moved to `origin`.
    fn about(self, origin: Pos2) -> Self {
        Self::translate(origin.to_vec2())
            .after(self)
            .after(Self::translate(-origin.to_vec2()))
    }

    fn apply(self, p: Pos2) -> Pos2 {
        let [a, b, c, d, e, f] = self.0;
        pos2(a * p.x + c * p.y + e, b * p.x + d * p.y + f)
    }

    /// How much the map scales lengths, on average.
    fn scale_factor(self) -> f32 {
        let [a, b, c, d, ..] = self.0;
        (a * d - b * c).abs().sqrt()
    }
}

/// Keep the part of convex `polygon` on one side of an axis-aligned line.
fn clip(polygon: &[Pos2], inside: impl Fn(Pos2) -> f32) -> Vec<Pos2> {
    let mut out = Vec::with_capacity(polygon.len() + 2);
    for (i, &p) in polygon.iter().enumerate() {
        let q = polygon[(i + 1) % polygon.len()];
        let (dp, dq) = (inside(p), inside(q));
        if dp >= 0.0 {
            out.push(p);
        }
        if (dp >= 0.0) != (dq >= 0.0) {
            out.push(p + (q - p) * (dp / (dp - dq)));
        }
    }
    out
}

fn clip_x(polygon: &[Pos2], x0: f32, x1: f32) -> Vec<Pos2> {
    clip(&clip(polygon, |p| p.x - x0), |p| x1 - p.x)
}

fn clip_y(polygon: &[Pos2], y0: f32, y1: f32) -> Vec<Pos2> {
    clip(&clip(polygon, |p| p.y - y0), |p| y1 - p.y)
}

fn circle(steps: usize) -> Vec<Pos2> {
    (0..steps)
        .map(|i| {
            let a = TAU * i as f32 / steps as f32;
            CENTRE + RADIUS * vec2(a.cos(), a.sin())
        })
        .collect()
}

/// Split a polyline into the runs with `x0 <= x <= x1`.
fn runs_within(points: &[Pos2], x0: f32, x1: f32) -> Vec<Vec<Pos2>> {
    let inside = |p: Pos2| p.x >= x0 && p.x <= x1;
    let crossing = |p: Pos2, q: Pos2| {
        let x = if (p.x < x0) != (q.x < x0) { x0 } else { x1 };
        p + (q - p) * ((x - p.x) / (q.x - p.x))
    };
    let mut runs = Vec::new();
    let mut run: Vec<Pos2> = Vec::new();
    for pair in points.windows(2) {
        let (p, q) = (pair[0], pair[1]);
        match (inside(p), inside(q)) {
            (true, true) => {
                if run.is_empty() {
                    run.push(p);
                }
                run.push(q);
            }
            (true, false) => {
                if run.is_empty() {
                    run.push(p);
                }
                run.push(crossing(p, q));
                runs.push(std::mem::take(&mut run));
            }
            (false, true) => {
                run.push(crossing(p, q));
                run.push(q);
            }
            (false, false) => {}
        }
    }
    if run.len() > 1 {
        runs.push(run);
    }
    // A closed loop that starts inside the strip: join its last run to its first.
    let closed = points.len() > 2 && points[0].distance(points[points.len() - 1]) < 1e-3;
    if closed && runs.len() > 1 && inside(points[0]) {
        let first = runs.remove(0);
        if let Some(last) = runs.last_mut() {
            last.extend(first.into_iter().skip(1));
        }
    }
    runs
}

/// The globe's transform for the live one-shot: scale and rotation.
fn fx_pose(fx: Option<(Fx, f32)>) -> (f32, f32) {
    /// Offset, scale, rotation.
    type Key = (f32, f32, f32);
    let keys: (&[Key], f32, fn(f32) -> f32) = match fx {
        Some((Fx::Start, _)) => (
            &[(0.0, 1.0, 0.0), (0.4, 0.9, 0.0), (1.0, 1.0, 0.0)],
            0.5,
            theme::ease_spring,
        ),
        Some((Fx::Cheer, _)) => (
            &[
                (0.0, 1.0, 0.0),
                (0.35, 1.14, -4.0),
                (0.7, 0.98, 2.0),
                (1.0, 1.0, 0.0),
            ],
            0.9,
            theme::ease_spring,
        ),
        Some((Fx::Quality, _)) => (
            &[(0.0, 1.0, 0.0), (0.45, 1.16, 0.0), (1.0, 1.0, 0.0)],
            0.8,
            theme::ease_spring,
        ),
        Some((Fx::Balanced, _)) => (
            &[
                (0.0, 1.0, 0.0),
                (0.25, 1.0, -11.0),
                (0.55, 1.0, 7.0),
                (0.8, 1.0, -3.0),
                (1.0, 1.0, 0.0),
            ],
            0.9,
            |t| theme::cubic_bezier(0.3, 0.6, 0.3, 1.0, t),
        ),
        Some((Fx::Fast, _)) | None => return (1.0, 0.0),
    };
    let (stops, duration, ease) = keys;
    let t = fx.map_or(1.0, |(_, elapsed)| (elapsed / duration).min(1.0));
    for pair in stops.windows(2) {
        let ((a, sa, ra), (b, sb, rb)) = (pair[0], pair[1]);
        if t <= b {
            let e = ease((t - a) / (b - a));
            return (sa + (sb - sa) * e, ra + (rb - ra) * e);
        }
    }
    (1.0, 0.0)
}

/// Paint the globe and its effects into `stage`.
pub fn paint(
    painter: &egui::Painter,
    brushes: &Brushes,
    palette: &Palette,
    stage: Rect,
    motion: &Motion,
    look: Look,
    now: f64,
) {
    let fx = motion.fx(now);
    let to_screen = |p: Pos2| stage.min + (p.to_vec2() - VIEW_MIN) * K;
    let globe_centre = to_screen(CENTRE);

    halo(painter, brushes, globe_centre, motion.clock, look.halo);
    rings(painter, palette, globe_centre, fx);
    streaks(painter, palette, stage, fx);

    let (scale, rotation) = fx_pose(fx);
    let globe = Affine::translate(vec2(0.0, -look.lift / K)).after(
        Affine::scale(scale, scale)
            .after(Affine::rotate(rotation))
            .about(GLOBE_ORIGIN),
    );

    // `drop-shadow(0 10px 16px rgba(0,0,0,.4))`, as one blurred disc.
    let shadow_half = RADIUS * K * scale / theme::SHADOW_DISC;
    painter.image(
        brushes.shadow.id(),
        Rect::from_center_size(
            to_screen(globe.apply(CENTRE)) + vec2(0.0, 10.0),
            Vec2::splat(2.0 * shadow_half),
        ),
        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
        theme::white(0.4),
    );

    let disc = circle(96);
    let meridians = motion.meridians();
    let colours = [palette.red, palette.green, palette.blue];
    for (i, &(x0, x1)) in STRIPS.iter().enumerate() {
        let [tx, ty, rot, sx, sy] = motion.slice(i);
        let spread = match i {
            0 => vec2(-7.0 * look.spread, 0.0),
            1 => vec2(0.0, 7.0),
            _ => vec2(7.0 * look.spread, 0.0),
        };
        let wave = Affine::translate(vec2(tx, ty))
            .after(Affine::rotate(rot))
            .after(Affine::scale(1.0 + sx, 1.0 + sy))
            .about(CENTRE);
        let local = globe.after(Affine::translate(spread)).after(wave);
        let map = |p: Pos2| to_screen(local.apply(p));
        let grid = Stroke::new(0.7 * K * local.scale_factor(), theme::white(0.5));

        let outline = clip_x(&disc, x0, x1);
        painter.add(egui::Shape::convex_polygon(
            outline.iter().map(|p| map(*p)).collect(),
            colours[i],
            Stroke::NONE,
        ));

        for rx in meridians {
            let rx = RADIUS * rx;
            if rx < 0.5 {
                continue;
            }
            let ellipse: Vec<Pos2> = (0..=72)
                .map(|n| {
                    let a = TAU * n as f32 / 72.0;
                    pos2(CENTRE.x + rx * a.cos(), CENTRE.y + RADIUS * a.sin())
                })
                .collect();
            for run in runs_within(&ellipse, x0, x1) {
                painter.add(egui::Shape::line(
                    run.into_iter().map(map).collect(),
                    Stroke::new(0.7, theme::white(0.5)),
                ));
            }
        }
        for y in PARALLELS {
            let half = (RADIUS * RADIUS - (y - CENTRE.y).powi(2)).sqrt();
            let (a, b) = ((CENTRE.x - half).max(x0), (CENTRE.x + half).min(x1));
            if a < b {
                painter.line_segment([map(pos2(a, y)), map(pos2(b, y))], grid);
            }
        }

        let band = clip_y(&outline, BAND.0, BAND.1);
        if band.len() >= 3 {
            painter.add(egui::Shape::convex_polygon(
                band.iter().map(|p| map(*p)).collect(),
                palette.band,
                Stroke::NONE,
            ));
        }

        let mut shade = egui::Mesh::with_texture(brushes.shade.id());
        let side = 2.0 * RADIUS;
        let corner = CENTRE - Vec2::splat(RADIUS);
        for p in &outline {
            let uv = (*p - corner) / side;
            shade.vertices.push(egui::epaint::Vertex {
                pos: map(*p),
                uv: uv.to_pos2(),
                color: Color32::WHITE,
            });
        }
        for k in 1..outline.len().saturating_sub(1) as u32 {
            shade.add_triangle(0, k, k + 1);
        }
        painter.add(shade);
    }
}

/// The Quality halo: a blurred colour wheel turning once every seven seconds.
fn halo(painter: &egui::Painter, brushes: &Brushes, centre: Pos2, clock: f32, alpha: f32) {
    if alpha <= 0.0 {
        return;
    }
    let half = 75.0 / theme::HALO_DISC;
    let angle = TAU * clock / 7.0;
    let (s, c) = angle.sin_cos();
    let corner = |dx: f32, dy: f32| centre + vec2(dx * c - dy * s, dx * s + dy * c) * half;
    let tint = theme::white(0.55 * alpha);
    let mut mesh = egui::Mesh::with_texture(brushes.halo.id());
    for (pos, uv) in [
        (corner(-1.0, -1.0), pos2(0.0, 0.0)),
        (corner(1.0, -1.0), pos2(1.0, 0.0)),
        (corner(1.0, 1.0), pos2(1.0, 1.0)),
        (corner(-1.0, 1.0), pos2(0.0, 1.0)),
    ] {
        mesh.vertices.push(egui::epaint::Vertex {
            pos,
            uv,
            color: tint,
        });
    }
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);
    painter.add(mesh);
}

/// One ring growing and fading: scale 0.7 to 1.9, opacity 0.75 to 0.
fn ring(painter: &egui::Painter, centre: Pos2, radius: f32, width: f32, colour: Color32, t: f32) {
    if !(0.0..1.0).contains(&t) {
        return;
    }
    let e = theme::ease_out(t);
    let scale = 0.7 + 1.2 * e;
    let alpha = 0.75 * (1.0 - e);
    painter.circle_stroke(
        centre,
        radius * scale,
        Stroke::new(width * scale, colour.gamma_multiply(alpha)),
    );
}

fn rings(painter: &egui::Painter, palette: &Palette, centre: Pos2, fx: Option<(Fx, f32)>) {
    match fx {
        Some((Fx::Start, t)) => ring(painter, centre, 52.75, 1.5, palette.band, t / 0.9),
        Some((Fx::Cheer, t)) => {
            for (n, colour) in [palette.red, palette.green, palette.blue]
                .into_iter()
                .enumerate()
            {
                ring(
                    painter,
                    centre,
                    53.0,
                    2.0,
                    colour,
                    (t - 0.16 * n as f32) / 1.2,
                );
            }
        }
        _ => {}
    }
}

/// Three white streaks flying left to right past the globe.
fn streaks(painter: &egui::Painter, palette: &Palette, stage: Rect, fx: Option<(Fx, f32)>) {
    let Some((Fx::Fast, elapsed)) = fx else {
        return;
    };
    for (n, top) in [30.0, 54.0, 78.0].into_iter().enumerate() {
        let t = (elapsed - 0.08 * n as f32) / 0.5;
        if !(0.0..1.0).contains(&t) {
            continue;
        }
        let e = theme::ease_in(t);
        let x = stage.left() - 60.0 - 40.0 + 270.0 * e;
        let alpha = 0.9 * (1.0 - e);
        let rect = Rect::from_min_size(pos2(x, stage.top() + top), vec2(70.0, 2.0));
        let tip = palette.band.gamma_multiply(alpha);
        let mut mesh = egui::Mesh::default();
        mesh.colored_vertex(rect.left_top(), Color32::TRANSPARENT);
        mesh.colored_vertex(rect.right_top(), tip);
        mesh.colored_vertex(rect.right_bottom(), tip);
        mesh.colored_vertex(rect.left_bottom(), Color32::TRANSPARENT);
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        painter.add(mesh);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moving() -> Motion {
        let mut motion = Motion::default();
        let drive = Drive {
            variant: Some(4),
            speed: 1.0,
            spin: Some(4.5),
        };
        for _ in 0..200 {
            motion.step(1.0 / 60.0, 0.0, &drive);
        }
        motion
    }

    /// At rest every slice sits exactly where the static design puts it.
    #[test]
    fn at_rest_every_slice_transform_is_identity() {
        let motion = Motion::default();
        for v in 0..5 {
            let mut m = motion.clone();
            m.from = v;
            m.to = v;
            for i in 0..3 {
                assert_eq!(m.slice(i), [0.0; 5], "variant {v}, slice {i}");
            }
        }
    }

    /// Easing out of a separation ends at exactly zero, so the window can stop
    /// repainting.
    #[test]
    fn the_motion_settles_to_exactly_zero_and_goes_quiet() {
        let mut motion = moving();
        assert!(motion.amp > 0.9, "{}", motion.amp);
        let idle = Drive {
            variant: None,
            speed: 1.0,
            spin: None,
        };
        let mut live = true;
        for _ in 0..600 {
            live = motion.step(1.0 / 60.0, 100.0, &idle);
        }
        assert_eq!(motion.amp, 0.0);
        assert!(!live, "still asking for frames at rest");
        for i in 0..3 {
            assert_eq!(motion.slice(i), [0.0; 5]);
        }
    }

    #[test]
    fn slice_polygons_stay_inside_the_circle() {
        let disc = circle(96);
        for (x0, x1) in STRIPS {
            let outline = clip_x(&disc, x0, x1);
            assert!(outline.len() >= 3);
            for p in &outline {
                assert!(p.distance(CENTRE) <= RADIUS + 1e-3, "{p:?} is outside");
                assert!(
                    p.x >= x0 - 1e-3 && p.x <= x1 + 1e-3,
                    "{p:?} is outside its strip"
                );
            }
            let band = clip_y(&outline, BAND.0, BAND.1);
            for p in &band {
                assert!(p.y >= BAND.0 - 1e-3 && p.y <= BAND.1 + 1e-3);
            }
        }
    }

    /// The right strip excludes the centre, so its outline must come from clipping.
    #[test]
    fn the_right_slice_has_its_true_corners() {
        let outline = clip_x(&circle(360), 77.0, 120.0);
        let top = outline.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        let expected = CENTRE.y - (RADIUS * RADIUS - 17.0 * 17.0).sqrt();
        assert!(
            (top - expected).abs() < 0.5,
            "top {top}, expected {expected}"
        );
    }

    #[test]
    fn meridian_runs_stay_within_their_strip() {
        let ellipse: Vec<Pos2> = (0..=72)
            .map(|n| {
                let a = TAU * n as f32 / 72.0;
                pos2(60.0 + 40.0 * a.cos(), 60.0 + 52.0 * a.sin())
            })
            .collect();
        let runs = runs_within(&ellipse, 77.0, 120.0);
        assert_eq!(
            runs.len(),
            1,
            "one arc of the ellipse crosses the right strip"
        );
        for p in runs.iter().flatten() {
            assert!(p.x >= 77.0 - 1e-3, "{p:?}");
        }
    }

    #[test]
    fn affine_composition_runs_inner_first() {
        let map = Affine::translate(vec2(10.0, 0.0)).after(Affine::scale(2.0, 2.0));
        assert_eq!(map.apply(pos2(1.0, 1.0)), pos2(12.0, 2.0));
        let turn = Affine::rotate(90.0).about(pos2(1.0, 1.0));
        let p = turn.apply(pos2(2.0, 1.0));
        assert!(
            (p.x - 1.0).abs() < 1e-5 && (p.y - 2.0).abs() < 1e-5,
            "{p:?}"
        );
        assert_eq!(Affine::IDENTITY.apply(pos2(3.0, 4.0)), pos2(3.0, 4.0));
    }

    #[test]
    fn a_one_shot_ends_where_it_started() {
        for fx in [Fx::Start, Fx::Cheer, Fx::Quality, Fx::Balanced] {
            let (scale, rotation) = fx_pose(Some((fx, fx.duration())));
            assert!(
                (scale - 1.0).abs() < 1e-4 && rotation.abs() < 1e-4,
                "{fx:?}"
            );
        }
    }

    #[test]
    fn a_job_keeps_its_motion() {
        assert_eq!(variant_of("job-1"), variant_of("job-1"));
        assert!(
            (0..50)
                .map(|n| variant_of(&format!("job-{n}")))
                .all(|v| v < 5)
        );
    }
}
