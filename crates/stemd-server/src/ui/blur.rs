//! Backdrop blur for menus and dialogs.
//!
//! A paint callback copies what is already drawn under a rect, blurs it with a
//! dual Kawase filter at half, quarter and eighth resolution, and draws it back
//! clipped to a rounded rect. The popup's translucent fill goes on top of it.
//! Costs a handful of small draws, and only while a popup is open.

use std::sync::{Arc, Mutex};

use eframe::egui::{self, Rect};
use eframe::egui_glow;
use eframe::glow::{self, HasContext as _};

/// Points sampled around the rect, so its edges blur with what is beside them.
const MARGIN: f32 = 24.0;
/// Downsampling steps. Each one doubles the blur radius.
const LEVELS: usize = 3;
/// Tap spread, in texels of the level being read.
const SPREAD: f32 = 1.5;

/// The blurred backdrop of `rect` with corner `radius`, as a shape to paint
/// before the popup's own fill.
pub fn shape(rect: Rect, radius: f32) -> egui::Shape {
    let callback = egui_glow::CallbackFn::new(move |info, painter| {
        let mut state = STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let resources = state.get_or_insert_with(|| {
            // SAFETY: called on the GL thread with egui's context current.
            unsafe { Resources::new(painter.gl()) }
                .inspect_err(|err| tracing::warn!("backdrop blur unavailable: {err}"))
                .ok()
        });
        if let Some(resources) = resources {
            // SAFETY: as above; egui restores its own state after the callback.
            unsafe { resources.draw(painter, &info, rect, radius) };
        }
    });
    egui::Shape::Callback(egui::PaintCallback {
        rect: rect.expand(MARGIN),
        callback: Arc::new(callback),
    })
}

/// `None` inside once setting up failed, so it is not retried every frame.
static STATE: Mutex<Option<Option<Resources>>> = Mutex::new(None);

struct Level {
    texture: glow::Texture,
    framebuffer: glow::Framebuffer,
    size: [i32; 2],
}

struct Program {
    program: glow::Program,
    step: Option<glow::UniformLocation>,
    region: Option<glow::UniformLocation>,
    shape: Option<glow::UniformLocation>,
    radius: Option<glow::UniformLocation>,
}

struct Resources {
    down: Program,
    up: Program,
    last: Program,
    vertex_array: glow::VertexArray,
    /// Level 0 holds the copy at full resolution.
    levels: Vec<Level>,
}

const VERTEX: &str = r"
out vec2 v_uv;
void main() {
    vec2 p = vec2(gl_VertexID == 1 ? 2.0 : 0.0, gl_VertexID == 2 ? 2.0 : 0.0);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
";

const DOWN: &str = r"
uniform sampler2D u_tex;
uniform vec2 u_step;
in vec2 v_uv;
out vec4 o_colour;
void main() {
    vec4 sum = texture(u_tex, v_uv) * 4.0;
    sum += texture(u_tex, v_uv - u_step);
    sum += texture(u_tex, v_uv + u_step);
    sum += texture(u_tex, v_uv + vec2(u_step.x, -u_step.y));
    sum += texture(u_tex, v_uv - vec2(u_step.x, -u_step.y));
    o_colour = sum / 8.0;
}
";

const UP: &str = r"
uniform sampler2D u_tex;
uniform vec2 u_step;
in vec2 v_uv;
out vec4 o_colour;
vec4 up(vec2 uv) {
    vec2 s = u_step;
    vec4 sum = texture(u_tex, uv + vec2(-2.0 * s.x, 0.0));
    sum += texture(u_tex, uv + vec2(-s.x, s.y)) * 2.0;
    sum += texture(u_tex, uv + vec2(0.0, 2.0 * s.y));
    sum += texture(u_tex, uv + vec2(s.x, s.y)) * 2.0;
    sum += texture(u_tex, uv + vec2(2.0 * s.x, 0.0));
    sum += texture(u_tex, uv + vec2(s.x, -s.y)) * 2.0;
    sum += texture(u_tex, uv + vec2(0.0, -2.0 * s.y));
    sum += texture(u_tex, uv + vec2(-s.x, -s.y)) * 2.0;
    return sum / 12.0;
}
void main() {
    o_colour = up(v_uv);
}
";

/// The last upsample, drawn onto the screen inside a rounded rect.
const LAST: &str = r"
uniform sampler2D u_tex;
uniform vec2 u_step;
uniform vec4 u_region;
uniform vec4 u_shape;
uniform float u_radius;
in vec2 v_uv;
out vec4 o_colour;
vec4 up(vec2 uv) {
    vec2 s = u_step;
    vec4 sum = texture(u_tex, uv + vec2(-2.0 * s.x, 0.0));
    sum += texture(u_tex, uv + vec2(-s.x, s.y)) * 2.0;
    sum += texture(u_tex, uv + vec2(0.0, 2.0 * s.y));
    sum += texture(u_tex, uv + vec2(s.x, s.y)) * 2.0;
    sum += texture(u_tex, uv + vec2(2.0 * s.x, 0.0));
    sum += texture(u_tex, uv + vec2(s.x, -s.y)) * 2.0;
    sum += texture(u_tex, uv + vec2(0.0, -2.0 * s.y));
    sum += texture(u_tex, uv + vec2(-s.x, -s.y)) * 2.0;
    return sum / 12.0;
}
void main() {
    vec2 uv = (gl_FragCoord.xy - u_region.xy) / u_region.zw;
    vec2 q = abs(gl_FragCoord.xy - u_shape.xy) - u_shape.zw + u_radius;
    float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - u_radius;
    float cover = clamp(0.5 - d, 0.0, 1.0);
    o_colour = vec4(up(uv).rgb, 1.0) * cover;
}
";

impl Resources {
    unsafe fn new(gl: &glow::Context) -> Result<Self, String> {
        let version = egui_glow::ShaderVersion::get(gl);
        if !version.is_new_shader_interface() {
            return Err(format!("{version:?} shaders lack in/out"));
        }
        let header = format!(
            "{}{}",
            version.version_declaration(),
            if version.is_embedded() {
                "precision mediump float;\n"
            } else {
                ""
            }
        );
        unsafe {
            Ok(Self {
                down: Program::new(gl, &header, DOWN)?,
                up: Program::new(gl, &header, UP)?,
                last: Program::new(gl, &header, LAST)?,
                vertex_array: gl.create_vertex_array()?,
                levels: Vec::new(),
            })
        }
    }

    /// Size the levels for a region of `size` pixels, reallocating on change.
    unsafe fn fit(&mut self, gl: &glow::Context, size: [i32; 2]) -> Result<(), String> {
        unsafe {
            if self.levels.len() == LEVELS + 1 && self.levels[0].size == size {
                return Ok(());
            }
            for level in self.levels.drain(..) {
                gl.delete_framebuffer(level.framebuffer);
                gl.delete_texture(level.texture);
            }
            for i in 0..=LEVELS {
                let size = [(size[0] >> i).max(1), (size[1] >> i).max(1)];
                // A level set left short by a failure is rebuilt next frame.
                let texture = gl.create_texture()?;
                gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                for (key, value) in [
                    (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                    (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                    (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                    (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
                ] {
                    gl.tex_parameter_i32(glow::TEXTURE_2D, key, value as i32);
                }
                gl.tex_image_2d(
                    glow::TEXTURE_2D,
                    0,
                    glow::RGBA8 as i32,
                    size[0],
                    size[1],
                    0,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelUnpackData::Slice(None),
                );
                let framebuffer = match gl.create_framebuffer() {
                    Ok(framebuffer) => framebuffer,
                    Err(err) => {
                        gl.delete_texture(texture);
                        return Err(err);
                    }
                };
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(framebuffer));
                gl.framebuffer_texture_2d(
                    glow::FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D,
                    Some(texture),
                    0,
                );
                self.levels.push(Level {
                    texture,
                    framebuffer,
                    size,
                });
            }
            Ok(())
        }
    }

    unsafe fn draw(
        &mut self,
        painter: &egui_glow::Painter,
        info: &egui::PaintCallbackInfo,
        rect: Rect,
        radius: f32,
    ) {
        let gl = painter.gl().clone();
        let ppp = info.pixels_per_point;
        let [width, height] = info.screen_size_px.map(|v| v as f32);
        // Pixel rect, origin bottom left as GL counts it.
        let region = rect.expand(MARGIN);
        let left = (region.left() * ppp).floor().max(0.0);
        let right = (region.right() * ppp).ceil().min(width);
        let top = (region.top() * ppp).floor().max(0.0);
        let bottom = (region.bottom() * ppp).ceil().min(height);
        let size = [(right - left) as i32, (bottom - top) as i32];
        if size[0] < 8 || size[1] < 8 {
            return;
        }
        let origin = [left as i32, (height - bottom) as i32];

        unsafe {
            if let Err(err) = self.fit(&gl, size) {
                tracing::warn!("backdrop blur: {err}");
                return;
            }
            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::BLEND);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_vertex_array(Some(self.vertex_array));

            gl.bind_framebuffer(glow::FRAMEBUFFER, painter.intermediate_fbo());
            gl.bind_texture(glow::TEXTURE_2D, Some(self.levels[0].texture));
            gl.copy_tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                origin[0],
                origin[1],
                size[0],
                size[1],
            );

            let pass = |program: &Program, from: usize, to: Option<usize>| {
                let source = &self.levels[from];
                gl.use_program(Some(program.program));
                gl.uniform_2_f32(
                    program.step.as_ref(),
                    SPREAD / source.size[0] as f32,
                    SPREAD / source.size[1] as f32,
                );
                gl.bind_texture(glow::TEXTURE_2D, Some(source.texture));
                if let Some(to) = to {
                    let target = &self.levels[to];
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(target.framebuffer));
                    gl.viewport(0, 0, target.size[0], target.size[1]);
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                }
            };
            for i in 1..=LEVELS {
                pass(&self.down, i - 1, Some(i));
            }
            for i in (1..LEVELS).rev() {
                pass(&self.up, i + 1, Some(i));
            }

            pass(&self.last, 1, None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, painter.intermediate_fbo());
            gl.viewport(origin[0], origin[1], size[0], size[1]);
            let clip = info.clip_rect_in_pixels();
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(
                clip.left_px,
                clip.from_bottom_px,
                clip.width_px,
                clip.height_px,
            );
            gl.enable(glow::BLEND);
            gl.blend_func_separate(
                glow::ONE,
                glow::ONE_MINUS_SRC_ALPHA,
                glow::ONE_MINUS_DST_ALPHA,
                glow::ONE,
            );
            let centre = rect.center();
            gl.uniform_4_f32(
                self.last.region.as_ref(),
                origin[0] as f32,
                origin[1] as f32,
                size[0] as f32,
                size[1] as f32,
            );
            gl.uniform_4_f32(
                self.last.shape.as_ref(),
                centre.x * ppp,
                height - centre.y * ppp,
                rect.width() * ppp / 2.0,
                rect.height() * ppp / 2.0,
            );
            gl.uniform_1_f32(self.last.radius.as_ref(), radius * ppp);
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.bind_vertex_array(None);
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
    }
}

impl Program {
    unsafe fn new(gl: &glow::Context, header: &str, fragment: &str) -> Result<Self, String> {
        unsafe {
            let program = gl.create_program()?;
            let mut shaders = Vec::new();
            for (kind, source) in [
                (glow::VERTEX_SHADER, VERTEX),
                (glow::FRAGMENT_SHADER, fragment),
            ] {
                let shader = gl.create_shader(kind)?;
                gl.shader_source(shader, &format!("{header}{source}"));
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    return Err(gl.get_shader_info_log(shader));
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                return Err(gl.get_program_info_log(program));
            }
            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }
            gl.use_program(Some(program));
            gl.uniform_1_i32(gl.get_uniform_location(program, "u_tex").as_ref(), 0);
            Ok(Self {
                program,
                step: gl.get_uniform_location(program, "u_step"),
                region: gl.get_uniform_location(program, "u_region"),
                shape: gl.get_uniform_location(program, "u_shape"),
                radius: gl.get_uniform_location(program, "u_radius"),
            })
        }
    }
}

/// Look of a frosted panel: blurred backdrop, tinted fill, hairline, drop shadow.
pub struct Frosted {
    pub radius: f32,
    pub margin: i8,
    pub fill: egui::Color32,
    pub outline: egui::Color32,
}

/// Lay `add` out on a frosted panel. Use inside a popup or modal whose own frame
/// is [`egui::Frame::NONE`].
pub fn frosted<R>(
    ui: &mut egui::Ui,
    look: &Frosted,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    let painter = ui.painter().clone();
    let slots = [(); 4].map(|()| painter.add(egui::Shape::Noop));
    let shown = egui::Frame::NONE.inner_margin(look.margin).show(ui, add);
    let rect = shown.response.rect;
    let radius = look.radius;
    let shadow = egui::epaint::Shadow {
        offset: [0, 16],
        blur: 36,
        spread: 0,
        color: super::theme::black(0.45),
    };
    painter.set(slots[0], shadow.as_shape(rect, radius));
    painter.set(slots[1], shape(rect, radius));
    painter.set(
        slots[2],
        egui::epaint::RectShape::new(
            rect,
            radius,
            look.fill,
            egui::Stroke::new(1.0, look.outline),
            egui::StrokeKind::Inside,
        ),
    );
    painter.set(
        slots[3],
        super::theme::top_highlight(rect, radius, super::theme::white(0.16)),
    );
    shown
}
