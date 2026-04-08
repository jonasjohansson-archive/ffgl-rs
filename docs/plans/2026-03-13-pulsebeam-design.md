# PulseBeam — FFGL Source Plugin Design

## Overview

PulseBeam is a triggerable light pulse source plugin for Resolume. Each trigger fires a line that sweeps across the canvas with a configurable trail. Up to 4 overlapping pulses can be active simultaneously. Fully transparent background for layer blending. Supports MQTT for remote triggering.

**Type**: Source (no input image)
**Framework**: `SimpleFFGLInstance` from `ffgl-core` with raw OpenGL

## Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| Trigger | Event | — | — | Fire a new pulse |
| Duration | Float | 0.1–5.0s | 1.0 | Time for pulse to cross canvas |
| Rotation | Float | 0.0–360.0° | 0.0 | Line angle (0°=horizontal moving up) |
| Line Width | Float | 0.001–0.2 | 0.02 | Width of leading edge (normalized) |
| Trail Length | Float | 0.0–1.0 | 0.3 | Fade distance behind the line |
| Trail Softness | Float | 0.0–1.0 | 0.5 | 0=sharp cutoff, 1=soft gaussian falloff |
| Color R | Red | 0.0–1.0 | 1.0 | |
| Color G | Green | 0.0–1.0 | 1.0 | |
| Color B | Blue | 0.0–1.0 | 1.0 | |
| Color A | Alpha | 0.0–1.0 | 1.0 | |
| MQTT Host | Text | — | 127.0.0.1 | Broker address |
| MQTT Port | Text | — | 1883 | Broker port |
| MQTT Topic | Text | — | pulsebeam/trigger | Subscribe topic |

## Architecture

### Pulse State (Rust)

```rust
struct Pulse {
    active: bool,
    start_time: Instant,
}

struct PulseBeam {
    pulses: [Pulse; 4],
    // params, shader handles, MQTT state...
}
```

- On trigger: find oldest/inactive slot, reset its start_time, mark active
- Each frame: compute `progress = elapsed / duration`, deactivate when `progress > 1.0 + trail_length`

### MQTT (Background Thread)

- Uses `rumqttc` crate
- Spawned on instance creation
- Connects to configured broker, subscribes to topic
- On message: sets `AtomicBool` trigger flag
- Main thread checks flag each frame, fires pulse if set
- Auto-reconnects on disconnect
- Host/port/topic params applied on next reconnect cycle

### GLSL Shader (Fragment)

- Full-screen quad rendering
- Uniforms: 4x pulse progress, rotation, line width, trail length, trail softness, RGBA color
- Per fragment:
  1. Rotate UV by rotation angle
  2. For each active pulse: compute distance from leading edge along travel axis
  3. Apply line width (bright core) + trail fade behind
  4. Trail softness interpolates between linear and smoothstep falloff
  5. Composite all 4 pulses additively
  6. Output `color * intensity` with `alpha = intensity` (transparent background)

### Lifecycle

- `new()`: compile shader, init OpenGL buffers, spawn MQTT thread
- `draw()`: update pulse states, upload uniforms, render full-screen quad
- `drop()`: signal MQTT thread shutdown

## Approach Decision

Chose custom Rust plugin (Approach B) over pure ISF or hybrid because:
- Pulse state management (4 overlapping timers) is clean in Rust
- MQTT integrates naturally as a background thread
- Shader stays simple — just receives uniforms and renders
