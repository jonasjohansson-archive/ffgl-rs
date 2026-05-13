use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;
use std::sync::{LazyLock, OnceLock};

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

const PARAM_SEGMENTS: usize = 0;
const PARAM_SEGMENT: usize = 1;       // 1-based start of the visible range
const PARAM_SEGMENT_END: usize = 2;   // 1-based end; 0 = single segment
const PARAM_DIRECTION: usize = 3;
const NUM_PARAMS: usize = 4;

const DIR_VERTICAL: f32 = 0.0;
const DIR_HORIZONTAL: f32 = 1.0;

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

// Sample the input texture but pass it through ONLY where v_uv lies in
// the picked segment along the chosen direction. Everywhere else: fully
// transparent.
static FS_SRC: &str = "\
#version 150
in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_input;
uniform vec2 u_uv_scale;        // (Width/HardwareWidth, Height/HardwareHeight)
uniform float u_seg_lo;         // 0..1 (already converted to perp-axis fraction)
uniform float u_seg_hi;         // 0..1
uniform float u_direction;      // 0 = vertical (split along x), 1 = horizontal (split along y)

void main() {
    float perp = (u_direction < 0.5) ? v_uv.x : v_uv.y;
    if (perp < u_seg_lo || perp > u_seg_hi) {
        out_color = vec4(0.0);
        return;
    }
    out_color = texture(u_input, v_uv * u_uv_scale);
}
";

// ---------------------------------------------------------------------------
// Static parameter info
// ---------------------------------------------------------------------------

static PARAM_INFOS: LazyLock<[SimpleParamInfo; NUM_PARAMS]> = LazyLock::new(|| {
    [
        // 0 – Segments (any integer count of equal slices)
        SimpleParamInfo {
            name: CString::new("Segments").unwrap(),
            param_type: ParameterTypes::Integer,
            default: Some(4.0),
            min: Some(1.0),
            max: Some(100.0),
            ..Default::default()
        },
        // 1 – Segment (1..N, start of the visible range)
        SimpleParamInfo {
            name: CString::new("Segment").unwrap(),
            param_type: ParameterTypes::Integer,
            default: Some(1.0),
            min: Some(1.0),
            max: Some(100.0),
            ..Default::default()
        },
        // 2 – Segment End (1..N, last segment of the visible range;
        // 0 = single-segment mode: show only `Segment`).
        SimpleParamInfo {
            name: CString::new("Segment End").unwrap(),
            param_type: ParameterTypes::Integer,
            default: Some(0.0),
            min: Some(0.0),
            max: Some(100.0),
            ..Default::default()
        },
        // 3 – Direction (vertical / horizontal)
        SimpleParamInfo {
            name: CString::new("Direction").unwrap(),
            param_type: ParameterTypes::Option,
            default: Some(0.0),
            min: Some(0.0),
            max: Some(1.0),
            elements: Some(vec![
                (CString::new("Vertical").unwrap(), 0.0),
                (CString::new("Horizontal").unwrap(), 1.0),
            ]),
            ..Default::default()
        },
    ]
});

// ---------------------------------------------------------------------------
// Shared GL resources
// ---------------------------------------------------------------------------

struct GlResources {
    vao: GLuint,
    vbo: GLuint,
    program: GLuint,
    u_input: GLint,
    u_uv_scale: GLint,
    u_seg_lo: GLint,
    u_seg_hi: GLint,
    u_direction: GLint,
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
            u_input: gl::GetUniformLocation(program, c"u_input".as_ptr()),
            u_uv_scale: gl::GetUniformLocation(program, c"u_uv_scale".as_ptr()),
            u_seg_lo: gl::GetUniformLocation(program, c"u_seg_lo".as_ptr()),
            u_seg_hi: gl::GetUniformLocation(program, c"u_seg_hi".as_ptr()),
            u_direction: gl::GetUniformLocation(program, c"u_direction".as_ptr()),
        };

        gl::BindVertexArray(0);
        gl::UseProgram(0);
        gl::DeleteShader(vs);
        gl::DeleteShader(fs);

        res
    })
}

// ---------------------------------------------------------------------------
// Segmenter struct
// ---------------------------------------------------------------------------

pub struct Segmenter {
    segments: f32,
    segment: f32,
    segment_end: f32,
    direction: f32,
}

impl SimpleFFGLInstance for Segmenter {
    fn new(_inst_data: &FFGLData) -> Self {
        let _ = gl_resources();
        Segmenter {
            segments: 4.0,
            segment: 1.0,
            segment_end: 0.0,  // 0 = single-segment mode
            direction: DIR_VERTICAL,
        }
    }

    fn plugin_info() -> PluginInfo {
        PluginInfo {
            unique_id: *b"SGMT",
            name: *b"Segmenter       ",
            ty: PluginType::Effect,
            about: "Slice input into N segments, output one".to_string(),
            description: "Splits the perpendicular axis into N equal segments and renders only the chosen one — the rest is transparent. Stack multiple instances to assemble per-segment content from different sources.".to_string(),
        }
    }

    fn num_params() -> usize { NUM_PARAMS }
    fn param_info(index: usize) -> &'static dyn ParamInfo { &PARAM_INFOS[index] }

    fn get_param(&self, index: usize) -> f32 {
        match index {
            PARAM_SEGMENTS => self.segments,
            PARAM_SEGMENT => self.segment,
            PARAM_SEGMENT_END => self.segment_end,
            PARAM_DIRECTION => self.direction,
            _ => 0.0,
        }
    }

    fn set_param(&mut self, index: usize, value: f32) {
        match index {
            PARAM_SEGMENTS => self.segments = value,
            PARAM_SEGMENT => self.segment = value,
            PARAM_SEGMENT_END => self.segment_end = value,
            PARAM_DIRECTION => self.direction = value,
            _ => {}
        }
    }

    fn get_text_param(&self, _index: usize) -> *const c_char { ptr::null() }
    fn set_text_param(&mut self, _index: usize, _value: &str) {}

    fn draw(&mut self, _inst_data: &FFGLData, frame_data: GLInput) {
        // Need at least one input texture to passthrough; otherwise output nothing.
        if frame_data.textures.is_empty() {
            unsafe {
                gl::ClearColor(0.0, 0.0, 0.0, 0.0);
                gl::Clear(gl::COLOR_BUFFER_BIT);
            }
            return;
        }
        let tex = &frame_data.textures[0];
        let uv_scale_x = if tex.HardwareWidth > 0 {
            tex.Width as f32 / tex.HardwareWidth as f32
        } else { 1.0 };
        let uv_scale_y = if tex.HardwareHeight > 0 {
            tex.Height as f32 / tex.HardwareHeight as f32
        } else { 1.0 };

        let r = gl_resources();
        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::UseProgram(r.program);

            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, tex.Handle);
            gl::Uniform1i(r.u_input, 0);
            gl::Uniform2f(r.u_uv_scale, uv_scale_x, uv_scale_y);
            // Resolve visible range. End=0 (or End<Start) → single-segment.
            let n = (self.segments as i32).max(1);
            let start = (self.segment as i32).clamp(1, n);
            let end_raw = self.segment_end as i32;
            let end = if end_raw <= 0 || end_raw < start { start } else { end_raw.min(n) };
            let lo = (start - 1) as f32 / n as f32;
            let hi = end as f32 / n as f32;
            gl::Uniform1f(r.u_seg_lo, lo);
            gl::Uniform1f(r.u_seg_hi, hi);
            gl::Uniform1f(r.u_direction, self.direction);

            gl::BindVertexArray(r.vao);
            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);

            gl::BindVertexArray(0);
            gl::BindTexture(gl::TEXTURE_2D, 0);
            gl::UseProgram(0);
            gl::Disable(gl::BLEND);
        }
    }
}
