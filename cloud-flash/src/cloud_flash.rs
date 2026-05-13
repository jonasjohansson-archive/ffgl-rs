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
// Parameter indices
// ---------------------------------------------------------------------------
//
// Triggering mirrors PulseBeam exactly so a CloudFlash drops into the
// same Resolume-manager fan-out (/api/pulse/<clip-name>) and the same
// OSC trigger path (/.../video/source/cloudflash/trigger).

const PARAM_TRIGGER:        usize = 0;
const PARAM_AUTO_FIRE:      usize = 1;
const PARAM_JITTER:         usize = 2;
const PARAM_RATE:           usize = 3;  // 0 = slow auto-fire, 1 = fast
const PARAM_STRIKES:        usize = 4;  // integer 1..10 sub-flashes per trigger
const PARAM_STRIKE_SPEED:   usize = 5;  // 0 = long flashes, 1 = brief
const PARAM_DECAY:          usize = 6;  // 0 = linear, 1 = sharp exponential
const PARAM_NOISE_SCALE:    usize = 7;
const PARAM_THRESHOLD:      usize = 8;
const PARAM_SOFTNESS:       usize = 9;
// HSB+Alpha → Resolume folds the four params into one unified picker.
const PARAM_COLOR_H:        usize = 10;
const PARAM_COLOR_S:        usize = 11;
const PARAM_COLOR_V:        usize = 12;
const PARAM_COLOR_A:        usize = 13;
const PARAM_BG_H:           usize = 14;
const PARAM_BG_S:           usize = 15;
const PARAM_BG_V:           usize = 16;
const PARAM_BG_A:           usize = 17;
const NUM_PARAMS:           usize = 18;

const MAX_FLASHES: usize = 8;

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

// Per-pixel: sample fbm noise offset by a per-flash seed, threshold +
// smoothstep it for the lightning-on-cloud silhouette, multiply by the
// flash envelope (fast attack, decay shaped by u_decay). Max across
// active flashes so simultaneous strikes don't double-bright. Output is
// (color * intensity) composited over background.
static FS_SRC: &str = "\
#version 150
in vec2 v_uv;
out vec4 out_color;

uniform float u_time;
uniform float u_flash_start[8];
uniform float u_flash_duration[8];
uniform float u_flash_seed[8];
uniform float u_decay;
uniform float u_noise_scale;
uniform float u_threshold;
uniform float u_softness;
uniform vec4  u_color;
uniform vec4  u_bg_color;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash21(i),                 hash21(i + vec2(1.0, 0.0)), u.x),
        mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

// 4-octave fBm — cloud-like silhouette with enough variation that the
// thresholded mask reads as forked / patchy lightning glow rather than
// a smooth blob.
float fbm(vec2 p) {
    float a = 0.0;
    float w = 0.5;
    for (int i = 0; i < 4; i++) {
        a += w * vnoise(p);
        p *= 2.03;
        w *= 0.5;
    }
    return a;
}

// Envelope: 5% attack ramp, 95% decay shaped by u_decay (0 = linear,
// 1 = exp ^5). Lightning's signature is the asymmetric spike — fast on,
// trail off — so the decay curve carries most of the visual character.
float envelope(float t, float dur, float curve) {
    if (t < 0.0 || t > dur) return 0.0;
    float n = t / dur;
    if (n < 0.05) return n / 0.05;
    float dn = (n - 0.05) / 0.95;
    float e = 1.0 + curve * 4.0;
    return pow(1.0 - dn, e);
}

void main() {
    float total_intensity = 0.0;
    for (int i = 0; i < 8; i++) {
        float dur = u_flash_duration[i];
        if (dur <= 0.0) continue;
        float t = u_time - u_flash_start[i];
        float env = envelope(t, dur, u_decay);
        if (env <= 0.0) continue;
        float seed = u_flash_seed[i];
        vec2  off  = vec2(seed * 7.13, seed * 3.71);
        float n    = fbm(v_uv * u_noise_scale + off);
        float mask = smoothstep(
            u_threshold - max(u_softness, 0.001),
            u_threshold + max(u_softness, 0.001),
            n
        );
        total_intensity = max(total_intensity, env * mask);
    }

    vec3 rgb = mix(u_bg_color.rgb, u_color.rgb, total_intensity);
    float a  = mix(u_bg_color.a,   u_color.a,   total_intensity);
    out_color = clamp(vec4(rgb, a), 0.0, 1.0);
}
";

// ---------------------------------------------------------------------------
// Parameter info
// ---------------------------------------------------------------------------

static PARAM_INFOS: LazyLock<[SimpleParamInfo; NUM_PARAMS]> = LazyLock::new(|| {
    [
        // 0 – Trigger
        SimpleParamInfo {
            name: CString::new("Trigger").unwrap(),
            param_type: ParameterTypes::Event,
            ..Default::default()
        },
        // 1 – Auto Fire
        SimpleParamInfo {
            name: CString::new("Auto Fire").unwrap(),
            param_type: ParameterTypes::Boolean,
            default: Some(1.0),
            ..Default::default()
        },
        // 2 – Jitter (auto-fire interval + per-strike duration)
        SimpleParamInfo {
            name: CString::new("Jitter").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.3),
            ..Default::default()
        },
        // 3 – Rate (auto-fire rate; 0 = slow ~5 s, 1 = fast ~0.1 s)
        SimpleParamInfo {
            name: CString::new("Rate").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.3),
            ..Default::default()
        },
        // 4 – Strikes (1..10 sub-flashes per trigger; multi-strike is
        // what makes a flash read as lightning rather than a blob)
        SimpleParamInfo {
            name: CString::new("Strikes").unwrap(),
            param_type: ParameterTypes::Integer,
            default: Some(3.0),
            min: Some(1.0),
            max: Some(10.0),
            ..Default::default()
        },
        // 5 – Strike Speed (per-strike duration; 0 = ~1 s, 1 = ~50 ms)
        SimpleParamInfo {
            name: CString::new("Strike Speed").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.7),
            ..Default::default()
        },
        // 6 – Decay Curve (0 = linear fade, 1 = sharp exponential)
        SimpleParamInfo {
            name: CString::new("Decay").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.6),
            ..Default::default()
        },
        // 7 – Noise Scale (0..1 → log scale ~1..30 features wide)
        SimpleParamInfo {
            name: CString::new("Noise Scale").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.4),
            ..Default::default()
        },
        // 8 – Threshold (noise cutoff; higher = sparser bright spots)
        SimpleParamInfo {
            name: CString::new("Threshold").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.55),
            ..Default::default()
        },
        // 9 – Softness (smoothstep width around threshold)
        SimpleParamInfo {
            name: CString::new("Softness").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.1),
            ..Default::default()
        },
        // 10 – Color Hue (first of HSB+A → Resolume unified picker)
        SimpleParamInfo {
            name: CString::new("Color").unwrap(),
            param_type: ParameterTypes::Hue,
            default: Some(0.6),  // cool blue-white default
            ..Default::default()
        },
        // 11 – Color Saturation
        SimpleParamInfo {
            name: CString::new("Color S").unwrap(),
            param_type: ParameterTypes::Saturation,
            default: Some(0.15),
            ..Default::default()
        },
        // 12 – Color Brightness
        SimpleParamInfo {
            name: CString::new("Color B").unwrap(),
            param_type: ParameterTypes::Brightness,
            default: Some(1.0),
            ..Default::default()
        },
        // 13 – Color Alpha
        SimpleParamInfo {
            name: CString::new("Color A").unwrap(),
            param_type: ParameterTypes::Alpha,
            default: Some(1.0),
            ..Default::default()
        },
        // 14 – Background Hue
        SimpleParamInfo {
            name: CString::new("Background").unwrap(),
            param_type: ParameterTypes::Hue,
            default: Some(0.0),
            ..Default::default()
        },
        // 15 – Background Saturation
        SimpleParamInfo {
            name: CString::new("Background S").unwrap(),
            param_type: ParameterTypes::Saturation,
            default: Some(0.0),
            ..Default::default()
        },
        // 16 – Background Brightness
        SimpleParamInfo {
            name: CString::new("Background B").unwrap(),
            param_type: ParameterTypes::Brightness,
            default: Some(0.0),
            ..Default::default()
        },
        // 17 – Background Alpha
        SimpleParamInfo {
            name: CString::new("Background A").unwrap(),
            param_type: ParameterTypes::Alpha,
            default: Some(1.0),
            ..Default::default()
        },
    ]
});

// ---------------------------------------------------------------------------
// Shared GL resources
// ---------------------------------------------------------------------------

struct GlResources {
    vao: GLuint,
    #[allow(dead_code)]
    vbo: GLuint,
    program: GLuint,
    u_time:           GLint,
    u_flash_start:    GLint,
    u_flash_duration: GLint,
    u_flash_seed:     GLint,
    u_decay:          GLint,
    u_noise_scale:    GLint,
    u_threshold:      GLint,
    u_softness:       GLint,
    u_color:          GLint,
    u_bg_color:       GLint,
}

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
            u_time:           gl::GetUniformLocation(program, c"u_time".as_ptr()),
            u_flash_start:    gl::GetUniformLocation(program, c"u_flash_start".as_ptr()),
            u_flash_duration: gl::GetUniformLocation(program, c"u_flash_duration".as_ptr()),
            u_flash_seed:     gl::GetUniformLocation(program, c"u_flash_seed".as_ptr()),
            u_decay:          gl::GetUniformLocation(program, c"u_decay".as_ptr()),
            u_noise_scale:    gl::GetUniformLocation(program, c"u_noise_scale".as_ptr()),
            u_threshold:      gl::GetUniformLocation(program, c"u_threshold".as_ptr()),
            u_softness:       gl::GetUniformLocation(program, c"u_softness".as_ptr()),
            u_color:          gl::GetUniformLocation(program, c"u_color".as_ptr()),
            u_bg_color:       gl::GetUniformLocation(program, c"u_bg_color".as_ptr()),
        };

        gl::BindVertexArray(0);
        gl::UseProgram(0);
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);

        res
    })
}

// ---------------------------------------------------------------------------
// Plugin state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Flash {
    active: bool,
    start: f32,     // seconds since plugin epoch
    duration: f32,  // seconds (sampled at fire time)
    seed: f32,      // 0..1, picked at fire time → spatial pattern variety
}

pub struct CloudFlash {
    epoch: Instant,

    flashes: [Flash; MAX_FLASHES],

    auto_fire: f32,
    jitter: f32,
    rate: f32,
    strikes: f32,
    strike_speed: f32,
    decay: f32,
    noise_scale: f32,
    threshold: f32,
    softness: f32,

    color:    [f32; 4],  // HSB+A
    bg_color: [f32; 4],

    rng_seed: u32,
    last_auto: Option<f32>,
    next_auto_interval: f32,
}

// HSB → RGB (h, s, v in 0..1).
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s <= 0.0001 { return (v, v, v); }
    let h = (h.rem_euclid(1.0)) * 6.0;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r, g, b) =
        if      h < 1.0 { (c, x, 0.0) }
        else if h < 2.0 { (x, c, 0.0) }
        else if h < 3.0 { (0.0, c, x) }
        else if h < 4.0 { (0.0, x, c) }
        else if h < 5.0 { (x, 0.0, c) }
        else            { (c, 0.0, x) };
    let m = v - c;
    (r + m, g + m, b + m)
}

impl CloudFlash {
    fn next_rand(&mut self) -> f32 {
        if self.rng_seed == 0 { self.rng_seed = 0x9E3779B9; }
        self.rng_seed ^= self.rng_seed << 13;
        self.rng_seed ^= self.rng_seed >> 17;
        self.rng_seed ^= self.rng_seed << 5;
        (self.rng_seed as f32) / (u32::MAX as f32)
    }

    fn now_secs(&self) -> f32 {
        self.epoch.elapsed().as_secs_f32()
    }

    // Strike duration derived from Strike Speed (0 → 1 s; 1 → 50 ms).
    fn strike_duration(&self) -> f32 {
        (1.0 - self.strike_speed) * 0.95 + 0.05
    }

    // Queue N strikes from a single trigger. Each strike gets its own
    // start time (staggered by ~1.3 * duration so they read as
    // separate sub-flashes, not one long pulse) and its own seed so the
    // spatial pattern varies between strikes.
    fn fire_trigger(&mut self) {
        let now = self.now_secs();
        let dur = self.strike_duration();
        let gap = dur * 1.3;
        let strikes = (self.strikes as i32).clamp(1, MAX_FLASHES as i32) as usize;
        for i in 0..strikes {
            let r1 = self.next_rand();
            let r2 = self.next_rand();
            let dur_j = dur * (1.0 + (r1 - 0.5) * self.jitter.clamp(0.0, 1.0));
            self.spawn_flash(now + (i as f32) * gap, dur_j.max(0.02), r2);
        }
    }

    fn spawn_flash(&mut self, start: f32, duration: f32, seed: f32) {
        // Reuse first inactive slot; otherwise overwrite the oldest.
        let slot = self.flashes.iter().position(|f| !f.active).unwrap_or_else(|| {
            self.flashes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.start.partial_cmp(&b.start).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        });
        self.flashes[slot] = Flash { active: true, start, duration, seed };
    }
}

// ---------------------------------------------------------------------------
// SimpleFFGLInstance impl
// ---------------------------------------------------------------------------

impl SimpleFFGLInstance for CloudFlash {
    fn new(_inst_data: &FFGLData) -> Self {
        let _ = gl_resources();
        CloudFlash {
            epoch: Instant::now(),
            flashes: [Flash { active: false, start: 0.0, duration: 0.0, seed: 0.0 }; MAX_FLASHES],
            auto_fire: 1.0,
            jitter: 0.3,
            rate: 0.3,
            strikes: 3.0,
            strike_speed: 0.7,
            decay: 0.6,
            noise_scale: 0.4,
            threshold: 0.55,
            softness: 0.1,
            color:    [0.6, 0.15, 1.0, 1.0],
            bg_color: [0.0, 0.0,  0.0, 1.0],
            rng_seed: 0x9E3779B9,
            last_auto: None,
            next_auto_interval: 1.0,
        }
    }

    fn plugin_info() -> PluginInfo {
        PluginInfo {
            unique_id: *b"CLFL",
            name: *b"CloudFlash      ",
            ty: PluginType::Source,
            about: "Triggerable lightning-flash generator".to_string(),
            description: "Noise-based flash that mimics lightning illuminating a cloud.".to_string(),
        }
    }

    fn num_params() -> usize { NUM_PARAMS }
    fn param_info(index: usize) -> &'static dyn ParamInfo { &PARAM_INFOS[index] }

    fn get_param(&self, index: usize) -> f32 {
        match index {
            PARAM_TRIGGER       => 0.0,
            PARAM_AUTO_FIRE     => self.auto_fire,
            PARAM_JITTER        => self.jitter,
            PARAM_RATE          => self.rate,
            PARAM_STRIKES       => self.strikes,
            PARAM_STRIKE_SPEED  => self.strike_speed,
            PARAM_DECAY         => self.decay,
            PARAM_NOISE_SCALE   => self.noise_scale,
            PARAM_THRESHOLD     => self.threshold,
            PARAM_SOFTNESS      => self.softness,
            PARAM_COLOR_H => self.color[0],
            PARAM_COLOR_S => self.color[1],
            PARAM_COLOR_V => self.color[2],
            PARAM_COLOR_A => self.color[3],
            PARAM_BG_H => self.bg_color[0],
            PARAM_BG_S => self.bg_color[1],
            PARAM_BG_V => self.bg_color[2],
            PARAM_BG_A => self.bg_color[3],
            _ => 0.0,
        }
    }

    fn set_param(&mut self, index: usize, value: f32) {
        match index {
            PARAM_TRIGGER => {
                if value > 0.5 { self.fire_trigger(); }
            }
            PARAM_AUTO_FIRE     => self.auto_fire = value,
            PARAM_JITTER        => self.jitter = value,
            PARAM_RATE          => self.rate = value,
            PARAM_STRIKES       => self.strikes = value,
            PARAM_STRIKE_SPEED  => self.strike_speed = value,
            PARAM_DECAY         => self.decay = value,
            PARAM_NOISE_SCALE   => self.noise_scale = value,
            PARAM_THRESHOLD     => self.threshold = value,
            PARAM_SOFTNESS      => self.softness = value,
            PARAM_COLOR_H => self.color[0] = value,
            PARAM_COLOR_S => self.color[1] = value,
            PARAM_COLOR_V => self.color[2] = value,
            PARAM_COLOR_A => self.color[3] = value,
            PARAM_BG_H => self.bg_color[0] = value,
            PARAM_BG_S => self.bg_color[1] = value,
            PARAM_BG_V => self.bg_color[2] = value,
            PARAM_BG_A => self.bg_color[3] = value,
            _ => {}
        }
    }

    fn get_text_param(&self, _index: usize) -> *const c_char { ptr::null() }
    fn set_text_param(&mut self, _index: usize, _value: &str) {}

    fn draw(&mut self, _inst_data: &FFGLData, _frame_data: GLInput) {
        let now = self.now_secs();

        // Auto fire: spawn fresh trigger sequences at the chosen rate
        // so the plugin shows life without external OSC. Jittered so
        // it doesn't tick like a metronome.
        if self.auto_fire >= 0.5 {
            let base_interval = (1.0 - self.rate) * 4.9 + 0.1;
            let due = self.last_auto
                .map(|t| now - t >= self.next_auto_interval.max(0.05))
                .unwrap_or(true);
            if due {
                self.fire_trigger();
                self.last_auto = Some(now);
                let r = self.next_rand();
                self.next_auto_interval =
                    base_interval * (1.0 + (r - 0.5) * self.jitter.clamp(0.0, 1.0));
            }
        } else {
            self.last_auto = None;
        }

        // Retire flashes whose envelope has fully decayed; otherwise
        // their stale uniforms keep painting at near-zero intensity.
        for f in self.flashes.iter_mut() {
            if !f.active { continue; }
            if now - f.start > f.duration { f.active = false; }
        }

        // Pack uniform arrays. Inactive slots get duration = 0 so the
        // shader's early-out skips them.
        let mut starts    = [0.0f32; MAX_FLASHES];
        let mut durations = [0.0f32; MAX_FLASHES];
        let mut seeds     = [0.0f32; MAX_FLASHES];
        for (i, f) in self.flashes.iter().enumerate() {
            if f.active {
                starts[i]    = f.start;
                durations[i] = f.duration;
                seeds[i]     = f.seed;
            }
        }

        let (cr, cg, cb) = hsv_to_rgb(self.color[0], self.color[1], self.color[2]);
        let (br, bg_g, bb) = hsv_to_rgb(self.bg_color[0], self.bg_color[1], self.bg_color[2]);
        // Noise scale 0..1 → 1..30 features (log-ish so low end is usable)
        let noise_scale_px = 1.0 + self.noise_scale * self.noise_scale * 29.0;

        let r = gl_resources();
        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::UseProgram(r.program);

            gl::Uniform1f(r.u_time, now);
            gl::Uniform1fv(r.u_flash_start,    MAX_FLASHES as i32, starts.as_ptr());
            gl::Uniform1fv(r.u_flash_duration, MAX_FLASHES as i32, durations.as_ptr());
            gl::Uniform1fv(r.u_flash_seed,     MAX_FLASHES as i32, seeds.as_ptr());
            gl::Uniform1f(r.u_decay,        self.decay);
            gl::Uniform1f(r.u_noise_scale,  noise_scale_px);
            gl::Uniform1f(r.u_threshold,    self.threshold);
            gl::Uniform1f(r.u_softness,     self.softness);
            gl::Uniform4f(r.u_color,    cr, cg, cb, self.color[3]);
            gl::Uniform4f(r.u_bg_color, br, bg_g, bb, self.bg_color[3]);

            gl::BindVertexArray(r.vao);
            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);

            gl::BindVertexArray(0);
            gl::UseProgram(0);
            gl::Disable(gl::BLEND);
        }
    }
}
