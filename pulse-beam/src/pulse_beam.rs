use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;
use std::sync::{atomic::AtomicBool, Arc, LazyLock};
use std::time::Instant;

use ffgl_core::{
    handler::simplified::SimpleFFGLInstance,
    info::{PluginInfo, PluginType},
    parameters::{ParamInfo, ParameterTypes, SimpleParamInfo},
    FFGLData, GLInput,
};
use gl::types::*;

use crate::mqtt::MqttHandle;
use crate::shader;

// ---------------------------------------------------------------------------
// Parameter indices & constants
// ---------------------------------------------------------------------------

const PARAM_TRIGGER: usize = 0;
const PARAM_DURATION: usize = 1;
const PARAM_DIRECTION: usize = 2;
const PARAM_LINE_WIDTH: usize = 3;
const PARAM_TRAIL_LENGTH: usize = 4;
const PARAM_TRAIL_SOFTNESS: usize = 5;
const PARAM_START_R: usize = 6;
const PARAM_START_G: usize = 7;
const PARAM_START_B: usize = 8;
const PARAM_START_A: usize = 9;
const PARAM_END_R: usize = 10;
const PARAM_END_G: usize = 11;
const PARAM_END_B: usize = 12;
const PARAM_END_A: usize = 13;
const PARAM_MQTT_HOST: usize = 14;
const PARAM_MQTT_PORT: usize = 15;
const PARAM_MQTT_TOPIC: usize = 16;
const NUM_PARAMS: usize = 17;
const MAX_PULSES: usize = 8;

// Direction values
const DIR_UP: f32 = 0.0;
const DIR_DOWN: f32 = 1.0;
const DIR_LEFT: f32 = 2.0;
const DIR_RIGHT: f32 = 3.0;

// ---------------------------------------------------------------------------
// Pulse
// ---------------------------------------------------------------------------

struct Pulse {
    active: bool,
    start_time: Instant,
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
uniform float u_direction;
uniform float u_line_width;
uniform float u_trail_length;
uniform float u_trail_softness;
uniform vec4 u_start_color;
uniform vec4 u_end_color;

void main() {
    // Map direction to axis: 0=up, 1=down, 2=left, 3=right
    float coord;
    int dir = int(u_direction + 0.5);
    if (dir == 0)      coord = v_uv.y;         // up: bottom to top
    else if (dir == 1) coord = 1.0 - v_uv.y;   // down: top to bottom
    else if (dir == 2) coord = 1.0 - v_uv.x;   // left: right to left
    else               coord = v_uv.x;          // right: left to right

    vec4 total_color = vec4(0.0);

    for (int i = 0; i < 8; i++) {
        float progress = u_pulse_progress[i];
        if (progress < 0.0) continue;

        float dist = progress - coord;

        // Leading edge: bright core
        float line = 1.0 - smoothstep(0.0, u_line_width, abs(dist));

        // Trail behind the line
        float trail = 0.0;
        float grad_t = 0.0;
        if (dist > 0.0 && dist < u_trail_length) {
            float t = dist / max(u_trail_length, 0.001);
            float linear_fade = 1.0 - t;
            float smooth_fade = 1.0 - smoothstep(0.0, 1.0, t);
            trail = mix(linear_fade, smooth_fade, u_trail_softness);
            grad_t = t;
        }

        // Combine: intensity envelope separate from color gradient
        float intensity = max(line, trail);
        if (intensity > 0.001) {
            float color_t = 0.0;
            if (trail >= line) color_t = grad_t;
            vec4 color = mix(u_start_color, u_end_color, color_t);
            total_color += vec4(color.rgb * intensity, color.a * intensity);
        }
    }

    out_color = clamp(total_color, 0.0, 1.0);
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
        // 1 – Duration
        SimpleParamInfo {
            name: CString::new("Duration").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.18),
            ..Default::default()
        },
        // 2 – Direction
        SimpleParamInfo {
            name: CString::new("Direction").unwrap(),
            param_type: ParameterTypes::Option,
            default: Some(0.0),
            min: Some(0.0),
            max: Some(3.0),
            elements: Some(vec![
                (CString::new("Up").unwrap(), 0.0),
                (CString::new("Down").unwrap(), 1.0),
                (CString::new("Left").unwrap(), 2.0),
                (CString::new("Right").unwrap(), 3.0),
            ]),
            ..Default::default()
        },
        // 3 – Line Width
        SimpleParamInfo {
            name: CString::new("Line Width").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.095),
            ..Default::default()
        },
        // 4 – Trail Length
        SimpleParamInfo {
            name: CString::new("Trail Length").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.3),
            ..Default::default()
        },
        // 5 – Trail Softness
        SimpleParamInfo {
            name: CString::new("Trail Softness").unwrap(),
            param_type: ParameterTypes::Standard,
            default: Some(0.5),
            ..Default::default()
        },
        // 6 – Start Color R
        SimpleParamInfo {
            name: CString::new("Start R").unwrap(),
            param_type: ParameterTypes::Red,
            default: Some(1.0),
            group: Some("Start Color".to_string()),
            ..Default::default()
        },
        // 7 – Start Color G
        SimpleParamInfo {
            name: CString::new("Start G").unwrap(),
            param_type: ParameterTypes::Green,
            default: Some(1.0),
            group: Some("Start Color".to_string()),
            ..Default::default()
        },
        // 8 – Start Color B
        SimpleParamInfo {
            name: CString::new("Start B").unwrap(),
            param_type: ParameterTypes::Blue,
            default: Some(1.0),
            group: Some("Start Color".to_string()),
            ..Default::default()
        },
        // 9 – Start Color A
        SimpleParamInfo {
            name: CString::new("Start A").unwrap(),
            param_type: ParameterTypes::Alpha,
            default: Some(1.0),
            group: Some("Start Color".to_string()),
            ..Default::default()
        },
        // 10 – End Color R
        SimpleParamInfo {
            name: CString::new("End R").unwrap(),
            param_type: ParameterTypes::Red,
            default: Some(0.0),
            group: Some("End Color".to_string()),
            ..Default::default()
        },
        // 11 – End Color G
        SimpleParamInfo {
            name: CString::new("End G").unwrap(),
            param_type: ParameterTypes::Green,
            default: Some(0.0),
            group: Some("End Color".to_string()),
            ..Default::default()
        },
        // 12 – End Color B
        SimpleParamInfo {
            name: CString::new("End B").unwrap(),
            param_type: ParameterTypes::Blue,
            default: Some(0.0),
            group: Some("End Color".to_string()),
            ..Default::default()
        },
        // 13 – End Color A
        SimpleParamInfo {
            name: CString::new("End A").unwrap(),
            param_type: ParameterTypes::Alpha,
            default: Some(1.0),
            group: Some("End Color".to_string()),
            ..Default::default()
        },
        // 14 – MQTT Host
        SimpleParamInfo {
            name: CString::new("MQTT Host").unwrap(),
            param_type: ParameterTypes::Text,
            default_string: Some(CString::new("127.0.0.1").unwrap()),
            ..Default::default()
        },
        // 15 – MQTT Port
        SimpleParamInfo {
            name: CString::new("MQTT Port").unwrap(),
            param_type: ParameterTypes::Text,
            default_string: Some(CString::new("1883").unwrap()),
            ..Default::default()
        },
        // 16 – MQTT Topic
        SimpleParamInfo {
            name: CString::new("MQTT Topic").unwrap(),
            param_type: ParameterTypes::Text,
            default_string: Some(CString::new("pulsebeam/trigger").unwrap()),
            ..Default::default()
        },
    ]
});

// ---------------------------------------------------------------------------
// PulseBeam struct
// ---------------------------------------------------------------------------

pub struct PulseBeam {
    vao: GLuint,
    vbo: GLuint,
    program: GLuint,
    u_pulse_progress: GLint,
    u_direction: GLint,
    u_line_width: GLint,
    u_trail_length: GLint,
    u_trail_softness: GLint,
    u_start_color: GLint,
    u_end_color: GLint,
    pulses: [Pulse; MAX_PULSES],
    duration: f32,
    direction: f32,
    line_width: f32,
    trail_length: f32,
    trail_softness: f32,
    start_color: [f32; 4],
    end_color: [f32; 4],
    mqtt_host: CString,
    mqtt_port: CString,
    mqtt_topic: CString,
    mqtt_trigger: Arc<AtomicBool>,
    mqtt_handle: Option<MqttHandle>,
}

impl PulseBeam {
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

        self.pulses[slot] = Pulse {
            active: true,
            start_time: Instant::now(),
        };
    }
}

// ---------------------------------------------------------------------------
// SimpleFFGLInstance impl
// ---------------------------------------------------------------------------

impl SimpleFFGLInstance for PulseBeam {
    fn new(_inst_data: &FFGLData) -> Self {
        unsafe {
            gl_loader::init_gl();
            gl::load_with(|s| gl_loader::get_proc_address(s).cast());

            // Compile & link shaders
            let vs = shader::compile_shader(VS_SRC, gl::VERTEX_SHADER);
            let fs = shader::compile_shader(FS_SRC, gl::FRAGMENT_SHADER);
            let program = shader::link_program(vs, fs);

            // Full-screen quad
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

            // Bind frag data location
            let out_frag_name = c"out_color";
            gl::BindFragDataLocation(program, 0, out_frag_name.as_ptr());

            // Vertex attribute
            let vert_pos_name = c"position";
            let pos_attr = gl::GetAttribLocation(program, vert_pos_name.as_ptr());
            gl::EnableVertexAttribArray(pos_attr as GLuint);
            gl::VertexAttribPointer(
                pos_attr as GLuint,
                2,
                gl::FLOAT,
                gl::FALSE as GLboolean,
                0,
                ptr::null(),
            );

            // Uniform locations
            let u_pulse_progress =
                gl::GetUniformLocation(program, c"u_pulse_progress".as_ptr());
            let u_direction = gl::GetUniformLocation(program, c"u_direction".as_ptr());
            let u_line_width = gl::GetUniformLocation(program, c"u_line_width".as_ptr());
            let u_trail_length =
                gl::GetUniformLocation(program, c"u_trail_length".as_ptr());
            let u_trail_softness =
                gl::GetUniformLocation(program, c"u_trail_softness".as_ptr());
            let u_start_color = gl::GetUniformLocation(program, c"u_start_color".as_ptr());
            let u_end_color = gl::GetUniformLocation(program, c"u_end_color".as_ptr());

            // Unbind
            gl::BindVertexArray(0);
            gl::UseProgram(0);

            // Clean up individual shader objects
            gl::DeleteShader(vs);
            gl::DeleteShader(fs);

            let now = Instant::now();

            let mqtt_trigger = Arc::new(AtomicBool::new(false));
            let mqtt_host = CString::new("127.0.0.1").unwrap();
            let mqtt_port = CString::new("1883").unwrap();
            let mqtt_topic = CString::new("pulsebeam/trigger").unwrap();

            let port: u16 = mqtt_port.to_str().unwrap_or("1883").parse().unwrap_or(1883);
            let mqtt_handle = Some(MqttHandle::new(
                mqtt_host.to_str().unwrap_or("127.0.0.1"),
                port,
                mqtt_topic.to_str().unwrap_or("pulsebeam/trigger"),
                mqtt_trigger.clone(),
            ));

            PulseBeam {
                vao,
                vbo,
                program,
                u_pulse_progress,
                u_direction,
                u_line_width,
                u_trail_length,
                u_trail_softness,
                u_start_color,
                u_end_color,
                pulses: std::array::from_fn(|_| Pulse { active: false, start_time: now }),
                duration: 0.18,
                direction: DIR_UP,
                line_width: 0.095,
                trail_length: 0.3,
                trail_softness: 0.5,
                start_color: [1.0, 1.0, 1.0, 1.0],
                end_color: [0.0, 0.0, 0.0, 1.0],
                mqtt_host,
                mqtt_port,
                mqtt_topic,
                mqtt_trigger,
                mqtt_handle,
            }
        }
    }

    fn plugin_info() -> PluginInfo {
        PluginInfo {
            unique_id: *b"PLBM",
            name: *b"PulseBeam       ",
            ty: PluginType::Source,
            about: "Triggerable light pulse source".to_string(),
            description: "A line of light with configurable trail and MQTT trigger".to_string(),
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
            PARAM_DURATION => self.duration,
            PARAM_DIRECTION => self.direction,
            PARAM_LINE_WIDTH => self.line_width,
            PARAM_TRAIL_LENGTH => self.trail_length,
            PARAM_TRAIL_SOFTNESS => self.trail_softness,
            PARAM_START_R => self.start_color[0],
            PARAM_START_G => self.start_color[1],
            PARAM_START_B => self.start_color[2],
            PARAM_START_A => self.start_color[3],
            PARAM_END_R => self.end_color[0],
            PARAM_END_G => self.end_color[1],
            PARAM_END_B => self.end_color[2],
            PARAM_END_A => self.end_color[3],
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
            PARAM_DURATION => self.duration = value,
            PARAM_DIRECTION => self.direction = value,
            PARAM_LINE_WIDTH => self.line_width = value,
            PARAM_TRAIL_LENGTH => self.trail_length = value,
            PARAM_TRAIL_SOFTNESS => self.trail_softness = value,
            PARAM_START_R => self.start_color[0] = value,
            PARAM_START_G => self.start_color[1] = value,
            PARAM_START_B => self.start_color[2] = value,
            PARAM_START_A => self.start_color[3] = value,
            PARAM_END_R => self.end_color[0] = value,
            PARAM_END_G => self.end_color[1] = value,
            PARAM_END_B => self.end_color[2] = value,
            PARAM_END_A => self.end_color[3] = value,
            _ => {}
        }
    }

    fn get_text_param(&self, index: usize) -> *const c_char {
        match index {
            PARAM_MQTT_HOST => self.mqtt_host.as_ptr(),
            PARAM_MQTT_PORT => self.mqtt_port.as_ptr(),
            PARAM_MQTT_TOPIC => self.mqtt_topic.as_ptr(),
            _ => ptr::null(),
        }
    }

    fn set_text_param(&mut self, index: usize, value: &str) {
        if let Ok(cstr) = CString::new(value) {
            match index {
                PARAM_MQTT_HOST => self.mqtt_host = cstr,
                PARAM_MQTT_PORT => self.mqtt_port = cstr,
                PARAM_MQTT_TOPIC => self.mqtt_topic = cstr,
                _ => return,
            }

            // Reconnect MQTT with updated parameters
            self.mqtt_handle = None;
            let port: u16 = self
                .mqtt_port
                .to_str()
                .unwrap_or("1883")
                .parse()
                .unwrap_or(1883);
            self.mqtt_handle = Some(MqttHandle::new(
                self.mqtt_host.to_str().unwrap_or("127.0.0.1"),
                port,
                self.mqtt_topic.to_str().unwrap_or("pulsebeam/trigger"),
                self.mqtt_trigger.clone(),
            ));
        }
    }

    fn draw(&mut self, _inst_data: &FFGLData, _frame_data: GLInput) {
        // Check MQTT trigger
        if self
            .mqtt_trigger
            .swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            self.fire_pulse();
        }

        let now = Instant::now();
        let duration_secs = self.duration * 4.9 + 0.1;

        let mut progress_values = [-1.0f32; MAX_PULSES];
        for (i, pulse) in self.pulses.iter_mut().enumerate() {
            if pulse.active {
                let elapsed = now.duration_since(pulse.start_time).as_secs_f32();
                let p = elapsed / duration_secs;
                if p > 1.0 + self.trail_length {
                    pulse.active = false;
                } else {
                    progress_values[i] = p;
                }
            }
        }

        unsafe {
            gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::UseProgram(self.program);

            gl::Uniform1fv(self.u_pulse_progress, MAX_PULSES as i32, progress_values.as_ptr());
            gl::Uniform1f(self.u_direction, self.direction);
            gl::Uniform1f(self.u_line_width, self.line_width * 0.199 + 0.001);
            gl::Uniform1f(self.u_trail_length, self.trail_length);
            gl::Uniform1f(self.u_trail_softness, self.trail_softness);
            gl::Uniform4f(
                self.u_start_color,
                self.start_color[0],
                self.start_color[1],
                self.start_color[2],
                self.start_color[3],
            );
            gl::Uniform4f(
                self.u_end_color,
                self.end_color[0],
                self.end_color[1],
                self.end_color[2],
                self.end_color[3],
            );

            gl::BindVertexArray(self.vao);
            gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);

            // Restore GL state
            gl::BindVertexArray(0);
            gl::UseProgram(0);
            gl::Disable(gl::BLEND);
        }
    }
}

// ---------------------------------------------------------------------------
// Drop – clean up GL resources
// ---------------------------------------------------------------------------

impl Drop for PulseBeam {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.vbo);
            gl::DeleteVertexArrays(1, &self.vao);
            gl::DeleteProgram(self.program);
        }
    }
}
