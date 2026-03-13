use ffgl_core::{handler::simplified::SimpleFFGLInstance, info::PluginInfo, info::PluginType, FFGLData, GLInput};

pub struct PulseBeam;

impl SimpleFFGLInstance for PulseBeam {
    fn new(_inst_data: &FFGLData) -> Self {
        PulseBeam
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

    fn draw(&mut self, _inst_data: &FFGLData, _frame_data: GLInput) {
        // Stub - will be implemented in Task 4
    }
}
