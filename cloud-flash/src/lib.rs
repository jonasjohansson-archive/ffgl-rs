mod shader;
mod cloud_flash;

use ffgl_core::{self, handler::simplified::SimpleFFGLHandler};

ffgl_core::plugin_main!(SimpleFFGLHandler<cloud_flash::CloudFlash>);
