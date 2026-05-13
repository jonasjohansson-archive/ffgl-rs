mod shader;
mod segmenter;

use ffgl_core::{self, handler::simplified::SimpleFFGLHandler};

ffgl_core::plugin_main!(SimpleFFGLHandler<segmenter::Segmenter>);
