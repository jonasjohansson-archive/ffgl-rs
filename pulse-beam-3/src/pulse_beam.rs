use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;
use std::sync::{LazyLock, OnceLock};
use std::time::Instant;

use ffgl_core::{
    handler::simplified::SimpleFFGLInstance,
    info::{PluginInfo, PluginType},
    parameters::{ParamInfo, ParameterTypes, SimpleParamInfo},
    FFGLData, GLInput,
};
use gl::types::*;

use crate::shader;

// ---------------------------------------------------------------------------
// Parameter indices & constants
// ---------------------------------------------------------------------------

const PARAM_TRIGGER: usize = 0;
const PARAM_AUTO_FIRE: usize = 1;    // boolean: continuous periodic auto-fire (for testing)
const PARAM_JITTER: usize = 2;       // 0..1: randomizes auto-fire interval + per-pulse speed
const PARAM_SPEED: usize = 3;
const PARAM_HEAD_WIDTH: usize = 4;   // 0..1: hard-edge head thickness (no smoothstep)
const PARAM_TRAIL_LENGTH: usize = 5;
const PARAM_TRAIL_SOFTNESS: usize = 6;
// Narrow-segment + expand: divide the perpendicular axis into N equal
// segments (Segments = 4 or 10) and pick which one the narrow phase
// occupies (Segment = 0..N-1). The beam stays in that segment until
// the head has traveled past Expand After pixels along the travel
// direction, then hard-snaps to full width.
const PARAM_SEGMENTS: usize = 7;
const PARAM_SEGMENT: usize = 8;
const PARAM_EXPAND_AFTER: usize = 9;
// Expand Segment: which single segment the wide phase fills.
// 0 = full width (default — matches the original "expand everywhere"
// behavior). Any 1..Segments value lights up only that one slice.
const PARAM_EXPAND_SEGMENT: usize = 10;
// HSB+Alpha (instead of RGB+Alpha) is what makes Resolume fold the four
// params into the unified Color picker with PICK/HSB/RGB/Palette tabs.
const PARAM_COLOR_H: usize = 11;
const PARAM_COLOR_S: usize = 12;
const PARAM_COLOR_V: usize = 13;
const PARAM_COLOR_A: usize = 14;
const PARAM_BG_H: usize = 15;
const PARAM_BG_S: usize = 16;
const PARAM_BG_V: usize = 17;
const PARAM_BG_A: usize = 18;
// When on, the Background color is only visible in the narrow-zone
// rectangle (segment × 0..Expand After).
const PARAM_BG_CLIP: usize = 19;
const NUM_PARAMS: usize = 20;
const MAX_PULSES: usize = 8;

// ---------------------------------------------------------------------------
// Pulse
// ---------------------------------------------------------------------------

struct Pulse {
    active: bool,
    start_time: Instant,
    // Per-pulse speed multiplier (1.0 = nominal). Jittered at fire time.
    speed_mul: f32,
}

// ---------------------------------------------------------------------------
// GLSL shaders
// ---------------------------------------------------------------------------

static VS_SRC: &str = "\
#version 150
in vec2 position;
out vec2 v_uv;

void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
";

static FS_SRC: &str = "\
#version 150
in vec2 v_uv;
out vec4 out_color;

uniform float u_pulse_progress[8];
uniform float u_head_width;
uniform float u_trail_length;
uniform float u_trail_softness;
uniform vec4 u_color;
uniform vec4 u_bg_color;
uniform float u_start_min;
uniform float u_start_max;
uniform float u_expand_after;
uniform float u_expand_min;    // perp range the wide phase fills
uniform float u_expand_max;
uniform float u_bg_clip;       // >=0.5: clip BG to the narrow phase only

void main() {
    // Direction is fixed: top → bottom. FFGL's v_uv.y origin is at the
    // top, so the head's coord increases with v_uv.y as the pulse
    // travels down. `perp` is the horizontal axis used to gate the
    // narrow-segment phase.
    float coord = v_uv.y;
    float perp  = v_uv.x;

    vec4 total_color = vec4(0.0);

    for (int i = 0; i < 8; i++) {
        float progress = u_pulse_progress[i];
        if (progress < 0.0) continue;

        // Segment gating, per-fragment. When Expand After = 0 the entire
        // travel stays in the narrow segment (no expansion ever). Above
        // the expand line → narrow segment. Below → expand segment range.
        bool in_narrow = (u_expand_after <= 0.0) || (coord < u_expand_after);
        if (in_narrow) {
            if (perp < u_start_min || perp > u_start_max) continue;
        } else {
            if (perp < u_expand_min || perp > u_expand_max) continue;
        }

        float dist = progress - coord;

        // Hard head + decaying trail. Head is a solid band of length
        // u_head_width right behind the leading edge; trail fades from
        // 1 → 0 over u_trail_length behind that.
        float intensity = 0.0;
        if (dist >= 0.0) {
            if (dist < u_head_width) {
                intensity = 1.0;
            } else if (dist < u_trail_length) {
                float t = dist / max(u_trail_length, 0.001);
                float linear_fade = 1.0 - t;
                float smooth_fade = 1.0 - smoothstep(0.0, 1.0, t);
                intensity = mix(linear_fade, smooth_fade, u_trail_softness);
            }
        }
        if (intensity > 0.001) {
            total_color += vec4(u_color.rgb * intensity, u_color.a * intensity);
        }
    }

    // Composite premultiplied pulse over background. When Clip is on,
    // the background is confined to the narrow segment column. If
    // Expand After > 0, it's also clipped vertically at the expand line.
    vec4 bg = u_bg_color;
    if (u_bg_clip >= 0.5) {
        bool clipped_out = perp < u_start_min || perp > u_start_max;
        if (u_expand_after > 0.0 && coord >= u_expand_after) clipped_out = true;
        if (clipped_out) bg = vec4(0.0);
    }
    vec3 rgb = total_color.rgb + bg.rgb * (1.0 - total_color.a);
    float a  = total_color.a + bg.a * (1.0 - total_color.a);
    out_color = clamp(vec4(rgb, a), 0.0, 1.0);
}
";

// ---------------------------------------------------------------------------
// Static parameter info
// ---------------------------------------------------------------------------

static PARAM_INFOS: LazyLock<[SimpleParamInfo; NUM_PARAMS]> = LazyLock::new(|| {
    [
        // 0 – Trigger
        SimpleParamInfo {
            name: CString::new("Trigger").unwrap(),
            param_type: ParameterTypes::Event,
            ..Default::default()
        },
        // 1 – Auto Fire: continuous periodic firing while enabled (testing).
        SimpleParamInfo {
            name: CString::new("Auto Fire").unwrap(),
            param_type: ParameterTypes::Boolean,
            default: Some(1.0),
            ..Default::default()
        },
        // 2 – Jitter: randomizes the auto-fire interval and per-pulse
        // speed. 0 = perfectly periodic; 1 = ±50% on each.
        SimpleParamInfo {
            name: CString::new("Jitter").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.0),
            ..Default::default()
        },
        // 3 – Speed (0 = slowest, 1 = fastest)
        SimpleParamInfo {
            name: CString::new("Speed").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.5),
            ..Default::default()
        },
        // 4 – Head Width: hard-edge thickness of the bright head
        // (no smoothstep fade). Works with Trail Length=0 to render a
        // pulse with NO trail — just the head as a clean band.
        SimpleParamInfo {
            name: CString::new("Head Width").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.0),
            ..Default::default()
        },
        // 5 – Trail Length
        SimpleParamInfo {
            name: CString::new("Trail Length").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.3),
            ..Default::default()
        },
        // 7 – Trail Softness
        SimpleParamInfo {
            name: CString::new("Trail Softness").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.5),
            ..Default::default()
        },
        // 7 – Segments (any integer count of equal slices)
        SimpleParamInfo {
            name: CString::new("Segments").unwrap(),
            param_type: ParameterTypes::Integer,
            default: Some(1.0),
            min: Some(1.0),
            max: Some(100.0),
            ..Default::default()
        },
        // 8 – Segment (1..N, which slice the narrow beam occupies;
        // e.g. with Segments=5, Segment=3 is the middle slice).
        SimpleParamInfo {
            name: CString::new("Segment").unwrap(),
            param_type: ParameterTypes::Integer,
            default: Some(1.0),
            min: Some(1.0),
            max: Some(100.0),
            ..Default::default()
        },
        // 8 – Expand After (pixels along travel direction;
        // 0 = expand immediately → full width always, no narrowing)
        SimpleParamInfo {
            name: CString::new("Expand After").unwrap(),
            param_type: ParameterTypes::Integer,
            default: Some(0.0),
            min: Some(0.0),
            max: Some(4096.0),
            ..Default::default()
        },
        // 10 – Expand Segment (1..Segments). Single segment the wide
        // phase fills after Expand After. 0 = full width (default).
        SimpleParamInfo {
            name: CString::new("Expand Segment").unwrap(),
            param_type: ParameterTypes::Integer,
            default: Some(0.0),
            min: Some(0.0),
            max: Some(100.0),
            ..Default::default()
        },
        // 11 – Color Hue (first of HSB+A → Resolume unified Color picker)
        SimpleParamInfo {
            name: CString::new("Color").unwrap(),
            param_type: ParameterTypes::Hue,
            default: Some(0.0),
            ..Default::default()
        },
        // 12 – Color Saturation
        SimpleParamInfo {
            name: CString::new("Color S").unwrap(),
            param_type: ParameterTypes::Saturation,
            default: Some(0.0),
            ..Default::default()
        },
        // 13 – Color Brightness
        SimpleParamInfo {
            name: CString::new("Color B").unwrap(),
            param_type: ParameterTypes::Brightness,
            default: Some(1.0),
            ..Default::default()
        },
        // 14 – Color Alpha
        SimpleParamInfo {
            name: CString::new("Color A").unwrap(),
            param_type: ParameterTypes::Alpha,
            default: Some(1.0),
            ..Default::default()
        },
        // 15 – Background Hue
        SimpleParamInfo {
            name: CString::new("Background").unwrap(),
            param_type: ParameterTypes::Hue,
            default: Some(0.0),
            ..Default::default()
        },
        // 16 – Background Saturation
        SimpleParamInfo {
            name: CString::new("Background S").unwrap(),
            param_type: ParameterTypes::Saturation,
            default: Some(0.0),
            ..Default::default()
        },
        // 17 – Background Brightness
        SimpleParamInfo {
            name: CString::new("Background B").unwrap(),
            param_type: ParameterTypes::Brightness,
            default: Some(0.0),
            ..Default::default()
        },
        // 18 – Background Alpha
        SimpleParamInfo {
            name: CString::new("Background A").unwrap(),
            param_type: ParameterTypes::Alpha,
            default: Some(1.0),
            ..Default::default()
        },
        // 19 – Clip: when on, BG only visible in the narrow-zone
        // rectangle (segment × 0..Expand After).
        SimpleParamInfo {
            name: CString::new("Clip").unwrap(),
            param_type: ParameterTypes::Boolean,
            default: Some(1.0),
            ..Default::default()
        },
    ]
});

// ---------------------------------------------------------------------------
// Shared GL resources
// ---------------------------------------------------------------------------
//
// All PulseBeam instances draw the same fullscreen quad with the same
// shader; only uniforms differ per draw call. Compiling the shader and
// allocating VAO/VBO per instance was costing ~100–300 ms each on macOS
// — felt like load lag with even a handful of instances. Sharing one set
// across the process eliminates that.
//
// Initialization happens on first use, which is a new() or draw() call —
// both run on Resolume's render thread with a valid GL context.

struct GlResources {
    vao: GLuint,
    vbo: GLuint,
    program: GLuint,
    u_pulse_progress: GLint,
    u_head_width: GLint,
    u_trail_length: GLint,
    u_trail_softness: GLint,
    u_color: GLint,
    u_bg_color: GLint,
    u_start_min: GLint,
    u_start_max: GLint,
    u_expand_after: GLint,
    u_expand_min: GLint,
    u_expand_max: GLint,
    u_bg_clip: GLint,
}

// Raw GL handles are u32/i32 — naturally Send + Sync, but OnceLock requires
// the explicit marker because the wrapper holds them.
unsafe impl Send for GlResources {}
unsafe impl Sync for GlResources {}

static GL_RESOURCES: OnceLock<GlResources> = OnceLock::new();

fn gl_resources() -> &'static GlResources {
    GL_RESOURCES.get_or_init(|| unsafe {
        gl_loader::init_gl();
        gl::load_with(|s| gl_loader::get_proc_address(s).cast());

        let vs = shader::compile_shader(VS_SRC, gl::VERTEX_SHADER);
        let fs = shader::compile_shader(FS_SRC, gl::FRAGMENT_SHADER);
        let program = shader::link_program(vs, fs);

        let mut vao: GLuint = 0;
        let mut vbo: GLuint = 0;
        gl::GenVertexArrays(1, &mut vao);
        gl::BindVertexArray(vao);
        gl::GenBuffers(1, &mut vbo);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

        static VERTEX_DATA: [GLfloat; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (VERTEX_DATA.len() * std::mem::size_of::<f32>()) as isize,
            VERTEX_DATA.as_ptr().cast(),
            gl::STATIC_DRAW,
        );

        gl::UseProgram(program);
        gl::BindFragDataLocation(program, 0, c"out_color".as_ptr());

        let pos_attr = gl::GetAttribLocation(program, c"position".as_ptr());
        gl::EnableVertexAttribArray(pos_attr as GLuint);
        gl::VertexAttribPointer(
            pos_attr as GLuint, 2, gl::FLOAT, gl::FALSE as GLboolean, 0, ptr::null(),
        );

        let res = GlResources {
            vao,
            vbo,
            program,
            u_pulse_progress: gl::GetUniformLocation(program, c"u_pulse_progress".as_ptr()),
            u_head_width: gl::GetUniformLocation(program, c"u_head_width".as_ptr()),
            u_trail_length: gl::GetUniformLocation(program, c"u_trail_length".as_ptr()),
            u_trail_softness: gl::GetUniformLocation(program, c"u_trail_softness".as_ptr()),
            u_color: gl::GetUniformLocation(program, c"u_color".as_ptr()),
            u_bg_color: gl::GetUniformLocation(program, c"u_bg_color".as_ptr()),
            u_start_min: gl::GetUniformLocation(program, c"u_start_min".as_ptr()),
            u_start_max: gl::GetUniformLocation(program, c"u_start_max".as_ptr()),
            u_expand_after: gl::GetUniformLocation(program, c"u_expand_after".as_ptr()),
            u_expand_min: gl::GetUniformLocation(program, c"u_expand_min".as_ptr()),
            u_expand_max: gl::GetUniformLocation(program, c"u_expand_max".as_ptr()),
            u_bg_clip: gl::GetUniformLocation(program, c"u_bg_clip".as_ptr()),
        };

        gl::BindVertexArray(0);
        gl::UseProgram(0);
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);

        res
    })
}

// ---------------------------------------------------------------------------
// PulseBeam struct
// ---------------------------------------------------------------------------

pub struct PulseBeam {
    pulses: [Pulse; MAX_PULSES],
    speed: f32,            // 0..1, higher = faster
    head_width: f32,
    trail_length: f32,
    trail_softness: f32,
    color: [f32; 4],
    bg_color: [f32; 4],
    bg_clip: f32,                            // 0..1; >=0.5 = enabled
    segments: f32,        // integer count of perpendicular slices
    segment: f32,         // 1-based index of the narrow slice
    expand_after: f32,    // pixels along travel before snap to wide
    expand_segment: f32,  // 1-based single slice; 0 = full width
    auto_fire: f32,                          // 0=off, >=0.5=on (periodic auto-fire)
    jitter: f32,                             // 0..1, randomness in interval + per-pulse speed
    rng_seed: u32,                           // local xorshift state
    last_auto_fire: Option<Instant>,         // last time the auto-fire loop spawned a pulse
    next_auto_interval: f32,                 // jittered seconds until next auto-fire
}

fn hsb_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h.rem_euclid(1.0) * 6.0;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r, g, b) = if h < 1.0 { (c, x, 0.0) }
        else if h < 2.0 { (x, c, 0.0) }
        else if h < 3.0 { (0.0, c, x) }
        else if h < 4.0 { (0.0, x, c) }
        else if h < 5.0 { (x, 0.0, c) }
        else            { (c, 0.0, x) };
    let m = v - c;
    (r + m, g + m, b + m)
}

impl PulseBeam {
    // Tiny xorshift32 — enough for jittering interval + per-pulse speed.
    fn next_rand(&mut self) -> f32 {
        if self.rng_seed == 0 { self.rng_seed = 0x9E3779B9; }
        self.rng_seed ^= self.rng_seed << 13;
        self.rng_seed ^= self.rng_seed >> 17;
        self.rng_seed ^= self.rng_seed << 5;
        (self.rng_seed as f32) / (u32::MAX as f32)
    }

    fn fire_pulse(&mut self) {
        let slot = self
            .pulses
            .iter()
            .position(|p| !p.active)
            .unwrap_or_else(|| {
                self.pulses
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, p)| p.start_time)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });

        // Per-pulse speed jitter: ±50% × jitter strength
        let r = self.next_rand();
        let speed_mul = 1.0 + (r - 0.5) * self.jitter.clamp(0.0, 1.0);
        self.pulses[slot] = Pulse {
            active: true,
            start_time: Instant::now(),
            speed_mul,
        };
    }
}

// ---------------------------------------------------------------------------
// SimpleFFGLInstance impl
// ---------------------------------------------------------------------------

impl SimpleFFGLInstance for PulseBeam {
    fn new(_inst_data: &FFGLData) -> Self {
        // Touch the shared GL setup so the first instance pays the
        // shader-compile cost once for the whole composition.
        let _ = gl_resources();

        let now = Instant::now();

        PulseBeam {
            pulses: std::array::from_fn(|_| Pulse { active: false, start_time: now, speed_mul: 1.0 }),
            speed: 0.5,
            head_width: 0.0,
            trail_length: 0.3,
            trail_softness: 0.5,
            // HSB-A: default Color = white (any H, S=0, V=1, A=1)
            color: [0.0, 0.0, 1.0, 1.0],
            // Default BG = black (V=0)
            bg_color: [0.0, 0.0, 0.0, 1.0],
            bg_clip: 1.0,
            segments: 1.0,      // single slice by default (no narrowing)
            segment: 1.0,       // 1-based
            expand_after: 0.0,  // 0 = no narrowing (full width always)
            expand_segment: 0.0, // 0 = full width during wide phase
            auto_fire: 1.0,     // on by default — easier to verify on drop-in
            jitter: 0.0,
            rng_seed: 0x9E3779B9,
            last_auto_fire: None,
            next_auto_interval: 1.0,
        }
    }

    fn plugin_info() -> PluginInfo {
        PluginInfo {
            unique_id: *b"PLB4",
            name: *b"PulseBeam       ",
            ty: PluginType::Source,
            about: "Triggerable light pulse source".to_string(),
            description: "A line of light with configurable trail".to_string(),
        }
    }

    fn num_params() -> usize {
        NUM_PARAMS
    }

    fn param_info(index: usize) -> &'static dyn ParamInfo {
        &PARAM_INFOS[index]
    }

    fn get_param(&self, index: usize) -> f32 {
        match index {
            PARAM_TRIGGER => 0.0,
            PARAM_SPEED => self.speed,
            PARAM_HEAD_WIDTH => self.head_width,
            PARAM_TRAIL_LENGTH => self.trail_length,
            PARAM_TRAIL_SOFTNESS => self.trail_softness,
            PARAM_COLOR_H => self.color[0],
            PARAM_COLOR_S => self.color[1],
            PARAM_COLOR_V => self.color[2],
            PARAM_COLOR_A => self.color[3],
            PARAM_BG_H => self.bg_color[0],
            PARAM_BG_S => self.bg_color[1],
            PARAM_BG_V => self.bg_color[2],
            PARAM_BG_A => self.bg_color[3],
            PARAM_BG_CLIP => self.bg_clip,
            PARAM_SEGMENTS => self.segments,
            PARAM_SEGMENT => self.segment,
            PARAM_EXPAND_AFTER => self.expand_after,
            PARAM_EXPAND_SEGMENT => self.expand_segment,
            PARAM_AUTO_FIRE => self.auto_fire,
            PARAM_JITTER => self.jitter,
            _ => 0.0,
        }
    }

    fn set_param(&mut self, index: usize, value: f32) {
        match index {
            PARAM_TRIGGER => {
                if value > 0.5 {
                    self.fire_pulse();
                }
            }
            PARAM_SPEED => self.speed = value,
            PARAM_HEAD_WIDTH => self.head_width = value,
            PARAM_TRAIL_LENGTH => self.trail_length = value,
            PARAM_TRAIL_SOFTNESS => self.trail_softness = value,
            PARAM_COLOR_H => self.color[0] = value,
            PARAM_COLOR_S => self.color[1] = value,
            PARAM_COLOR_V => self.color[2] = value,
            PARAM_COLOR_A => self.color[3] = value,
            PARAM_BG_H => self.bg_color[0] = value,
            PARAM_BG_S => self.bg_color[1] = value,
            PARAM_BG_V => self.bg_color[2] = value,
            PARAM_BG_A => self.bg_color[3] = value,
            PARAM_BG_CLIP => self.bg_clip = value,
            PARAM_SEGMENTS => self.segments = value,
            PARAM_SEGMENT => self.segment = value,
            PARAM_EXPAND_AFTER => self.expand_after = value,
            PARAM_EXPAND_SEGMENT => self.expand_segment = value,
            PARAM_AUTO_FIRE => self.auto_fire = value,
            PARAM_JITTER => self.jitter = value,
            _ => {}
        }
    }

    fn get_text_param(&self, _index: usize) -> *const c_char { ptr::null() }
    fn set_text_param(&mut self, _index: usize, _value: &str) {}

    fn draw(&mut self, _inst_data: &FFGLData, _frame_data: GLInput) {
        let now = Instant::now();
        // Speed 0..1 → duration 5.0..0.1 seconds (higher value = faster).
        let duration_secs = (1.0 - self.speed) * 4.9 + 0.1;

        // Auto Fire: spawn a fresh pulse every `duration_secs * 0.7`
        // seconds while enabled (slight overlap → continuous-chase feel,
        // and you get something on screen without external triggers).
        // Each fire picks the NEXT interval with jitter so the chase
        // doesn't tick like a metronome.
        if self.auto_fire >= 0.5 {
            let interval = std::time::Duration::from_secs_f32(self.next_auto_interval.max(0.05));
            let due = self.last_auto_fire.map(|t| now.duration_since(t) >= interval).unwrap_or(true);
            if due {
                self.fire_pulse();
                self.last_auto_fire = Some(now);
                let base = (duration_secs * 0.7).max(0.1);
                let r = self.next_rand();
                self.next_auto_interval = base * (1.0 + (r - 0.5) * self.jitter.clamp(0.0, 1.0));
            }
        } else {
            self.last_auto_fire = None;
        }

        // Segment narrows the beam to one of N equal slices of the
        // perpendicular axis (N = 4 or 10). After the head crosses
        // Expand After pixels along the travel direction, the beam
        // snaps to full width. Expand After = 0 disables narrowing.
        let (_canvas_w, canvas_h) = _inst_data.get_dimensions();
        let travel_dim = canvas_h as f32;
        let divs = (self.segments as i32).max(1);
        let seg = (self.segment as i32).clamp(1, divs); // 1-based
        let start_min_norm = (seg - 1) as f32 / divs as f32;
        let start_max_norm = seg as f32 / divs as f32;
        // Expand Segment: 0 = full width during wide phase; 1..divs
        // confines the wide phase to just that one slice.
        let exp = self.expand_segment as i32;
        let (expand_min_norm, expand_max_norm) = if exp <= 0 {
            (0.0, 1.0)
        } else {
            let e = exp.clamp(1, divs);
            ((e - 1) as f32 / divs as f32, e as f32 / divs as f32)
        };
        let expand_after_norm = if travel_dim > 0.0 { self.expand_after / travel_dim } else { 0.0 };

        let mut progress_values = [-1.0f32; MAX_PULSES];
        for (i, pulse) in self.pulses.iter_mut().enumerate() {
            if pulse.active {
                let elapsed = now.duration_since(pulse.start_time).as_secs_f32() * pulse.speed_mul;
                let p = elapsed / duration_secs;
                if p > 1.0 + self.trail_length {
                    pulse.active = false;
                } else {
                    progress_values[i] = p;
                }
            }
        }

        let r = gl_resources();
        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::UseProgram(r.program);

            gl::Uniform1fv(r.u_pulse_progress, MAX_PULSES as i32, progress_values.as_ptr());
            gl::Uniform1f(r.u_head_width, self.head_width);
            gl::Uniform1f(r.u_trail_length, self.trail_length);
            gl::Uniform1f(r.u_trail_softness, self.trail_softness);
            let (cr, cg, cb) = hsb_to_rgb(self.color[0], self.color[1], self.color[2]);
            gl::Uniform4f(r.u_color, cr, cg, cb, self.color[3]);
            let (br, bgg, bb) = hsb_to_rgb(self.bg_color[0], self.bg_color[1], self.bg_color[2]);
            gl::Uniform4f(r.u_bg_color, br, bgg, bb, self.bg_color[3]);
            gl::Uniform1f(r.u_start_min, start_min_norm);
            gl::Uniform1f(r.u_start_max, start_max_norm);
            gl::Uniform1f(r.u_expand_after, expand_after_norm);
            gl::Uniform1f(r.u_expand_min, expand_min_norm);
            gl::Uniform1f(r.u_expand_max, expand_max_norm);
            gl::Uniform1f(r.u_bg_clip, self.bg_clip);

            gl::BindVertexArray(r.vao);
            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);

            // Restore GL state
            gl::BindVertexArray(0);
            gl::UseProgram(0);
            gl::Disable(gl::BLEND);
        }
    }
}

// PulseBeam owns no per-instance GL resources to clean up. The shared
// GL_RESOURCES set is leaked on purpose: it's freed when the process
// (and its GL context) exits, and there's no safe way to know when the
// last instance is dropped to free it earlier.
