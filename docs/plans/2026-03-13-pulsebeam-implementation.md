# PulseBeam Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a triggerable light pulse FFGL source plugin with MQTT support for Resolume.

**Architecture:** Custom Rust plugin using `SimpleFFGLInstance` with raw OpenGL. Pulse state (4 slots) managed in Rust, GLSL fragment shader renders full-screen quad. Background thread for MQTT via `rumqttc`.

**Tech Stack:** Rust, `ffgl-core`, `gl` crate, raw OpenGL/GLSL 150, `rumqttc` for MQTT

---

### Task 1: Scaffold the crate

**Files:**
- Create: `pulse-beam/Cargo.toml`
- Create: `pulse-beam/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)

**Step 1: Add pulse-beam to workspace**

In the root `Cargo.toml`, add `"pulse-beam"` to the `members` array:

```toml
members = [
    "example-raw",
    "ffgl-glium",
    "ffgl-core",
    "ffgl-isf",
    "build-common",
    "example-sdfer", "shadertoy",
    "pulse-beam",
]
```

**Step 2: Create pulse-beam/Cargo.toml**

```toml
[package]
name = "pulse-beam"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
ffgl-core = { path = "../ffgl-core" }
gl = "0.14.0"
gl_loader = "0.1.0"
rumqttc = "0.24"
```

**Step 3: Create minimal pulse-beam/src/lib.rs**

A minimal stub that compiles as a valid FFGL source plugin with no params and a black screen:

```rust
mod shader;
mod pulse_beam;

use ffgl_core::{self, handler::simplified::SimpleFFGLHandler};

ffgl_core::plugin_main!(SimpleFFGLHandler<pulse_beam::PulseBeam>);
```

Create `pulse-beam/src/pulse_beam.rs` with a minimal `SimpleFFGLInstance` impl that just clears to transparent.

Create `pulse-beam/src/shader.rs` copying the `compile_shader` and `link_program` helpers from `example-raw/src/shader_helper.rs`.

**Step 4: Verify it compiles**

Run: `cargo build --release -p pulse-beam`
Expected: Compiles successfully, produces `target/release/libpulse_beam.dylib`

**Step 5: Commit**

```bash
git add pulse-beam/ Cargo.toml
git commit -m "feat: scaffold pulse-beam FFGL plugin crate"
```

---

### Task 2: Define parameters and state

**Files:**
- Modify: `pulse-beam/src/pulse_beam.rs`

**Step 1: Define parameter constants and Pulse struct**

```rust
use std::ffi::{CStr, CString};
use std::time::Instant;

// Parameter indices
const PARAM_TRIGGER: usize = 0;
const PARAM_DURATION: usize = 1;
const PARAM_ROTATION: usize = 2;
const PARAM_LINE_WIDTH: usize = 3;
const PARAM_TRAIL_LENGTH: usize = 4;
const PARAM_TRAIL_SOFTNESS: usize = 5;
const PARAM_COLOR_R: usize = 6;
const PARAM_COLOR_G: usize = 7;
const PARAM_COLOR_B: usize = 8;
const PARAM_COLOR_A: usize = 9;
const PARAM_MQTT_HOST: usize = 10;
const PARAM_MQTT_PORT: usize = 11;
const PARAM_MQTT_TOPIC: usize = 12;
const NUM_PARAMS: usize = 13;

const MAX_PULSES: usize = 4;

#[derive(Clone)]
struct Pulse {
    active: bool,
    start_time: Instant,
}
```

**Step 2: Define static parameter info array**

Use `std::sync::LazyLock` (stable since Rust 1.80) to create a static array of `SimpleParamInfo`:

```rust
use std::sync::LazyLock;
use ffgl_core::parameters::info::{SimpleParamInfo, ParameterTypes};

static PARAM_INFOS: LazyLock<[SimpleParamInfo; NUM_PARAMS]> = LazyLock::new(|| {
    [
        SimpleParamInfo { name: CString::new("Trigger").unwrap(), param_type: ParameterTypes::Event, ..Default::default() },
        SimpleParamInfo { name: CString::new("Duration").unwrap(), param_type: ParameterTypes::Standard, default: Some(0.18), min: Some(0.0), max: Some(1.0), ..Default::default() },
        // Duration: 0.0 maps to 0.1s, 1.0 maps to 5.0s. Default 0.18 ≈ 1.0s
        SimpleParamInfo { name: CString::new("Rotation").unwrap(), param_type: ParameterTypes::Standard, default: Some(0.0), ..Default::default() },
        // Rotation: 0.0-1.0 maps to 0-360 degrees
        SimpleParamInfo { name: CString::new("Line Width").unwrap(), param_type: ParameterTypes::Standard, default: Some(0.095), ..Default::default() },
        // Line Width: 0.0-1.0 maps to 0.001-0.2. Default 0.095 ≈ 0.02
        SimpleParamInfo { name: CString::new("Trail Length").unwrap(), param_type: ParameterTypes::Standard, default: Some(0.3), ..Default::default() },
        SimpleParamInfo { name: CString::new("Trail Softness").unwrap(), param_type: ParameterTypes::Standard, default: Some(0.5), ..Default::default() },
        SimpleParamInfo { name: CString::new("Color R").unwrap(), param_type: ParameterTypes::Red, default: Some(1.0), group: Some("Color".to_string()), ..Default::default() },
        SimpleParamInfo { name: CString::new("Color G").unwrap(), param_type: ParameterTypes::Green, default: Some(1.0), group: Some("Color".to_string()), ..Default::default() },
        SimpleParamInfo { name: CString::new("Color B").unwrap(), param_type: ParameterTypes::Blue, default: Some(1.0), group: Some("Color".to_string()), ..Default::default() },
        SimpleParamInfo { name: CString::new("Color A").unwrap(), param_type: ParameterTypes::Alpha, default: Some(1.0), group: Some("Color".to_string()), ..Default::default() },
        SimpleParamInfo { name: CString::new("MQTT Host").unwrap(), param_type: ParameterTypes::Text, default_string: Some(CString::new("127.0.0.1").unwrap()), ..Default::default() },
        SimpleParamInfo { name: CString::new("MQTT Port").unwrap(), param_type: ParameterTypes::Text, default_string: Some(CString::new("1883").unwrap()), ..Default::default() },
        SimpleParamInfo { name: CString::new("MQTT Topic").unwrap(), param_type: ParameterTypes::Text, default_string: Some(CString::new("pulsebeam/trigger").unwrap()), ..Default::default() },
    ]
});
```

**Step 3: Add param storage to PulseBeam struct**

```rust
pub struct PulseBeam {
    // OpenGL handles
    vao: gl::types::GLuint,
    vbo: gl::types::GLuint,
    program: gl::types::GLuint,
    // Uniform locations
    u_pulse_progress: [gl::types::GLint; MAX_PULSES],
    u_rotation: gl::types::GLint,
    u_line_width: gl::types::GLint,
    u_trail_length: gl::types::GLint,
    u_trail_softness: gl::types::GLint,
    u_color: gl::types::GLint,
    // State
    pulses: [Pulse; MAX_PULSES],
    // Params
    duration: f32,       // normalized 0-1
    rotation: f32,       // normalized 0-1
    line_width: f32,     // normalized 0-1
    trail_length: f32,
    trail_softness: f32,
    color: [f32; 4],
    // MQTT
    mqtt_host: CString,
    mqtt_port: CString,
    mqtt_topic: CString,
    mqtt_trigger: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
```

**Step 4: Implement get_param/set_param/get_text_param/set_text_param**

Wire up all 13 params to the struct fields. For text params, return CString pointers. For float params, return/set the normalized values.

**Step 5: Verify it compiles**

Run: `cargo build --release -p pulse-beam`

**Step 6: Commit**

```bash
git add pulse-beam/
git commit -m "feat: define PulseBeam parameters and state"
```

---

### Task 3: Write the GLSL shader

**Files:**
- Create: `pulse-beam/src/shaders.rs` (inline shader strings)

**Step 1: Write the vertex shader**

Full-screen quad vertex shader:

```glsl
#version 150
in vec2 position;
out vec2 v_uv;

void main() {
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
```

**Step 2: Write the fragment shader**

```glsl
#version 150
in vec2 v_uv;
out vec4 out_color;

uniform float u_pulse_progress[4];
uniform float u_rotation;
uniform float u_line_width;
uniform float u_trail_length;
uniform float u_trail_softness;
uniform vec4 u_color;

void main() {
    float angle = u_rotation * 6.28318530718;
    vec2 center = v_uv - 0.5;
    float rotated = dot(center, vec2(-sin(angle), cos(angle))) + 0.5;

    float total_intensity = 0.0;

    for (int i = 0; i < 4; i++) {
        float progress = u_pulse_progress[i];
        if (progress < 0.0) continue;

        float dist = progress - rotated;

        // Leading edge (line width)
        float line = 1.0 - smoothstep(0.0, u_line_width, abs(dist));

        // Trail behind the line
        float trail = 0.0;
        if (dist > 0.0 && dist < u_trail_length) {
            float t = dist / max(u_trail_length, 0.001);
            // Interpolate between linear and smoothstep based on softness
            float linear_fade = 1.0 - t;
            float smooth_fade = 1.0 - smoothstep(0.0, 1.0, t);
            trail = mix(linear_fade, smooth_fade, u_trail_softness);
        }

        total_intensity += max(line, trail);
    }

    total_intensity = clamp(total_intensity, 0.0, 1.0);
    out_color = u_color * total_intensity;
}
```

**Step 3: Verify it compiles**

Run: `cargo build --release -p pulse-beam`

**Step 4: Commit**

```bash
git add pulse-beam/
git commit -m "feat: add PulseBeam GLSL shaders"
```

---

### Task 4: OpenGL initialization and rendering

**Files:**
- Modify: `pulse-beam/src/pulse_beam.rs`

**Step 1: Implement `new()` — OpenGL setup**

- Call `gl_loader::init_gl()` and `gl::load_with()`
- Compile vertex + fragment shaders using helpers from `shader.rs`
- Create full-screen quad VAO/VBO with vertices: `[-1,-1, 1,-1, -1,1, 1,1]` (triangle strip)
- Get all uniform locations
- Initialize pulse slots as inactive (progress = -1.0)

**Step 2: Implement `draw()` — per-frame rendering**

```rust
fn draw(&mut self, data: &FFGLData, _frame_data: GLInput) {
    // Check MQTT trigger
    if self.mqtt_trigger.swap(false, Ordering::Relaxed) {
        self.fire_pulse();
    }

    // Update pulse progress
    let now = Instant::now();
    let duration_secs = self.duration * 4.9 + 0.1; // map 0-1 to 0.1-5.0s

    let mut progress_values = [-1.0f32; MAX_PULSES];
    for (i, pulse) in self.pulses.iter_mut().enumerate() {
        if pulse.active {
            let elapsed = now.duration_since(pulse.start_time).as_secs_f32();
            let p = elapsed / duration_secs;
            if p > 1.0 + self.trail_length {
                pulse.active = false;
                progress_values[i] = -1.0;
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

        // Upload uniforms
        gl::Uniform1fv(self.u_pulse_progress[0], 4, progress_values.as_ptr());
        gl::Uniform1f(self.u_rotation, self.rotation);
        gl::Uniform1f(self.u_line_width, self.line_width * 0.199 + 0.001);
        gl::Uniform1f(self.u_trail_length, self.trail_length);
        gl::Uniform1f(self.u_trail_softness, self.trail_softness);
        gl::Uniform4f(self.u_color, self.color[0], self.color[1], self.color[2], self.color[3]);

        gl::BindVertexArray(self.vao);
        gl::DrawArrays(gl::TRIANGLE_STRIP, 0, 4);

        // Restore GL state (FFGL requirement)
        gl::BindVertexArray(0);
        gl::UseProgram(0);
        gl::Disable(gl::BLEND);
    }
}
```

**Step 3: Implement `fire_pulse()` helper**

```rust
fn fire_pulse(&mut self) {
    // Find oldest inactive slot, or oldest active slot
    let slot = self.pulses.iter()
        .position(|p| !p.active)
        .unwrap_or_else(|| {
            self.pulses.iter()
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
```

**Step 4: Wire up trigger in set_param**

When `PARAM_TRIGGER` is set to 1.0, call `self.fire_pulse()`.

**Step 5: Implement Drop to clean up GL resources**

**Step 6: Verify it compiles**

Run: `cargo build --release -p pulse-beam`

**Step 7: Deploy and test in Resolume**

Run: `./deploy_bundle.sh pulse_beam PulseBeam`

Manual test: Load PulseBeam as source in Resolume, press Trigger, verify a white line sweeps across with trail. Adjust rotation, line width, trail params.

**Step 8: Commit**

```bash
git add pulse-beam/
git commit -m "feat: implement PulseBeam rendering and trigger"
```

---

### Task 5: MQTT integration

**Files:**
- Create: `pulse-beam/src/mqtt.rs`
- Modify: `pulse-beam/src/pulse_beam.rs`

**Step 1: Create mqtt.rs module**

```rust
use rumqttc::{MqttOptions, Client, QoS, Event, Packet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

pub struct MqttHandle {
    shutdown: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl MqttHandle {
    pub fn new(
        host: &str,
        port: u16,
        topic: &str,
        trigger: Arc<AtomicBool>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let topic = topic.to_string();

        let mut opts = MqttOptions::new("pulsebeam", host, port);
        opts.set_keep_alive(Duration::from_secs(30));

        let (client, mut connection) = Client::new(opts, 10);
        let _ = client.subscribe(&topic, QoS::AtLeastOnce);

        let thread = thread::spawn(move || {
            for notification in connection.iter() {
                if shutdown_clone.load(Ordering::Relaxed) {
                    break;
                }
                if let Ok(Event::Incoming(Packet::Publish(_))) = notification {
                    trigger.store(true, Ordering::Relaxed);
                }
            }
        });

        MqttHandle {
            shutdown,
            thread: Some(thread),
        }
    }
}

impl Drop for MqttHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
```

**Step 2: Integrate into PulseBeam**

- Add `mqtt_handle: Option<MqttHandle>` to `PulseBeam`
- In `new()`, spawn the MQTT handle using default host/port/topic
- The `mqtt_trigger: Arc<AtomicBool>` is shared between MQTT thread and main struct
- In `set_text_param()`, when host/port/topic changes, drop old handle and create new one

**Step 3: Add mqtt module to lib.rs**

```rust
mod shader;
mod pulse_beam;
mod mqtt;
```

**Step 4: Verify it compiles**

Run: `cargo build --release -p pulse-beam`

**Step 5: Test MQTT**

Deploy to Resolume: `./deploy_bundle.sh pulse_beam PulseBeam`

Test with mosquitto_pub:
```bash
mosquitto_pub -h 127.0.0.1 -t pulsebeam/trigger -m "1"
```

Verify a pulse fires in Resolume.

**Step 6: Commit**

```bash
git add pulse-beam/
git commit -m "feat: add MQTT trigger support to PulseBeam"
```

---

### Task 6: Final polish and deploy

**Files:**
- Modify: `pulse-beam/src/pulse_beam.rs`

**Step 1: Set proper plugin metadata**

```rust
fn plugin_info() -> PluginInfo {
    PluginInfo {
        unique_id: *b"PLBM",
        name: *b"PulseBeam       ",  // exactly 16 chars
        ty: PluginType::Source,
        about: "Triggerable light pulse source with MQTT".to_string(),
        description: "A line of light that sweeps across the canvas with configurable trail, rotation, and MQTT trigger support.".to_string(),
    }
}
```

**Step 2: Full build and deploy**

```bash
cargo build --release -p pulse-beam
./deploy_bundle.sh pulse_beam PulseBeam
```

**Step 3: Manual test checklist**

- [ ] Plugin loads as Source in Resolume
- [ ] Trigger button fires a pulse
- [ ] Multiple pulses can overlap (up to 4)
- [ ] Duration slider changes sweep speed
- [ ] Rotation rotates the line direction
- [ ] Line Width changes the bright core width
- [ ] Trail Length changes fade distance
- [ ] Trail Softness transitions between sharp and smooth
- [ ] Color RGBA controls work
- [ ] MQTT trigger fires from mosquitto_pub
- [ ] Background is fully transparent (layers below show through)

**Step 4: Commit**

```bash
git add pulse-beam/
git commit -m "feat: finalize PulseBeam plugin metadata and polish"
```
