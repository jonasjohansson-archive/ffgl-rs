mod shader;
mod pulse_beam;

use ffgl_core::{self, handler::simplified::SimpleFFGLHandler};

ffgl_core::plugin_main!(SimpleFFGLHandler<pulse_beam::PulseBeam>);
