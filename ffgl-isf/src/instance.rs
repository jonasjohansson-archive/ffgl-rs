use crate::handler::IsfFFGLState;
use crate::param;
use crate::param::TextParamHandler;
use crate::shader;
use crate::shader::IsfShaderLoadError;
use crate::util::MultiUniforms;

use ffgl_core::parameters::builtin::OverlayParams;
use ffgl_core::parameters::handler::ParamValueHandler;
use glium::texture::{RawImage2d, Texture2d};
use glium::uniforms::UniformValue;

use ffgl_core;

use ffgl_core::handler::FFGLInstance;

use std::cmp::max;
use std::collections::HashMap;
use std::fmt::Formatter;
use std::os::raw::c_char;

use std::fmt::Debug;

use ffgl_glium::FFGLGlium;

/// 1D distance transform (Felzenszwalb & Huttenlocher algorithm).
/// Input f[i] = 0 for seed pixels, INF for others.
/// Output d[i] = squared distance to nearest seed.
fn dt_1d(f: &[f32]) -> Vec<f32> {
    let n = f.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![f[0]];
    }

    let inf = 1e20f32;
    let mut d = vec![0.0f32; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f32; n + 1];

    let mut k = 0usize;
    v[0] = 0;
    z[0] = -inf;
    z[1] = inf;

    for q in 1..n {
        let qi = q as f32;
        let s;
        loop {
            let vk = v[k] as f32;
            let candidate = ((f[q] + qi * qi) - (f[v[k]] + vk * vk)) / (2.0 * qi - 2.0 * vk);
            if candidate > z[k] {
                s = candidate;
                break;
            }
            k -= 1; // safe: z[0] = -inf so we always break at k=0
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = inf;
    }

    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let diff = q as f32 - v[k] as f32;
        d[q] = diff * diff + f[v[k]];
    }

    d
}

/// 2D Euclidean Distance Transform.
/// For each pixel marked true (inside), computes the distance to the nearest
/// false (outside) pixel. Outside pixels get distance 0.
fn compute_edt(inside: &[bool], width: usize, height: usize) -> Vec<f32> {
    let inf = (width * width + height * height) as f32;

    // Initialize: inside pixels = INF (distance unknown), outside = 0 (seeds)
    let mut grid: Vec<f32> = inside
        .iter()
        .map(|&b| if b { inf } else { 0.0 })
        .collect();

    // Transform rows
    for y in 0..height {
        let start = y * width;
        let row: Vec<f32> = grid[start..start + width].to_vec();
        let result = dt_1d(&row);
        grid[start..start + width].copy_from_slice(&result);
    }

    // Transform columns
    for x in 0..width {
        let col: Vec<f32> = (0..height).map(|y| grid[y * width + x]).collect();
        let result = dt_1d(&col);
        for y in 0..height {
            grid[y * width + x] = result[y];
        }
    }

    // sqrt to get Euclidean distances
    grid.iter().map(|&d| d.sqrt()).collect()
}

/// Preprocess a mask image for use in shaders:
/// - R channel: mask value (luminance * alpha)
/// - G channel: normalized distance from edge (0 = at edge, 1 = deepest interior)
/// - B: 0, A: 255
fn preprocess_mask_with_distance(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    let num_pixels = width * height;

    // Compute mask values and binary threshold
    let mut mask_values = vec![0.0f32; num_pixels];
    let mut inside = vec![false; num_pixels];

    for i in 0..num_pixels {
        let r = rgba[i * 4] as f32 / 255.0;
        let g = rgba[i * 4 + 1] as f32 / 255.0;
        let b = rgba[i * 4 + 2] as f32 / 255.0;
        let a = rgba[i * 4 + 3] as f32 / 255.0;
        let lum = (0.299 * r + 0.587 * g + 0.114 * b) * a;
        mask_values[i] = lum;
        inside[i] = lum > 0.5;
    }

    // Compute distance field
    let distances = compute_edt(&inside, width, height);

    // Normalize distances so max = 1.0
    let max_dist = distances.iter().cloned().fold(0.0f32, f32::max);
    let norm = if max_dist > 0.0 { 1.0 / max_dist } else { 1.0 };

    // Pack into RGBA
    let mut output = vec![0u8; num_pixels * 4];
    for i in 0..num_pixels {
        output[i * 4] = (mask_values[i].clamp(0.0, 1.0) * 255.0) as u8;
        output[i * 4 + 1] = ((distances[i] * norm).clamp(0.0, 1.0) * 255.0) as u8;
        output[i * 4 + 2] = 0;
        output[i * 4 + 3] = 255;
    }

    output
}

pub struct IsfFFGLInstance {
    pub shader: shader::IsfShader,
    pub state: IsfFFGLState,
    pub glium: FFGLGlium,
    pub file_textures: HashMap<String, Texture2d>,
}

impl Debug for IsfFFGLInstance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IsfFFGLInstance").finish()
    }
}

impl FFGLInstance for IsfFFGLInstance {
    fn get_param(&self, index: usize) -> f32 {
        let _span = self.state.span.enter();
        self.state.inputs.get_param(index)
    }

    fn set_param(&mut self, index: usize, value: f32) {
        let _span = self.state.span.enter();
        self.state.inputs.set_param(index, value)
    }

    fn get_text_param(&self, index: usize) -> *const c_char {
        let _span = self.state.span.enter();
        self.state.inputs.get_text_param(index)
    }

    fn set_text_param(&mut self, index: usize, value: &str) {
        let _span = self.state.span.enter();
        self.state.inputs.set_text_param(index, value)
    }

    fn draw(&mut self, inst_data: &ffgl_core::FFGLData, frame_data: ffgl_core::GLInput) {
        let _span = self.state.span.enter();

        // Reload any file textures that changed
        for input in self.state.inputs.iter_mut() {
            if let param::IsfFFGLParam::FileImage(ref mut fp) = input {
                if fp.needs_reload && !fp.path.is_empty() {
                    fp.needs_reload = false;
                    match image::open(&fp.path) {
                        Ok(img) => {
                            let rgba = img.to_rgba8();
                            let dims = rgba.dimensions();
                            let pixels = rgba.into_raw();
                            let processed = preprocess_mask_with_distance(
                                &pixels,
                                dims.0 as usize,
                                dims.1 as usize,
                            );
                            let raw = RawImage2d {
                                data: std::borrow::Cow::Owned(processed),
                                width: dims.0,
                                height: dims.1,
                                format: glium::texture::ClientFormat::U8U8U8U8,
                            };
                            match Texture2d::new(&self.glium.ctx, raw) {
                                Ok(tex) => {
                                    self.file_textures.insert(fp.name.clone(), tex);
                                }
                                Err(e) => {
                                    tracing::error!("Failed to create texture for {}: {e}", fp.name);
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to load image {}: {e}", fp.path);
                        }
                    }
                }
            }
        }

        let scale = match &self.state.inputs[0] {
            crate::param::IsfFFGLParam::Overlay(OverlayParams::Scale, val) => (*val).powf(2.0),
            _ => 1.0,
        };

        let dest_res = frame_data
            .textures
            .first()
            .map(|t| (t.HardwareWidth, t.HardwareHeight))
            .unwrap_or(inst_data.get_dimensions());

        let render_res = (
            max((dest_res.0 as f32 * scale) as u32, 1),
            max((dest_res.1 as f32 * scale) as u32, 1),
        );

        self.glium
            .draw(render_res, dest_res, frame_data, &mut |target, textures| {
                let mut image_uniforms: Vec<(&str, UniformValue)> = self
                    .state
                    .inputs
                    .iter()
                    .filter_map(|i| match i {
                        param::IsfFFGLParam::Isf(param::IsfShaderParam {
                            ty: isf::InputType::Image,
                            name,
                            ..
                        }) => Some((
                            name.as_str(),
                            UniformValue::Texture2d(textures.first()?, None),
                        )),
                        _ => None,
                    })
                    .collect();

                // Add file-loaded textures
                for input in self.state.inputs.iter() {
                    if let param::IsfFFGLParam::FileImage(ref fp) = input {
                        if let Some(tex) = self.file_textures.get(&fp.name) {
                            image_uniforms.push((
                                &fp.name,
                                UniformValue::Texture2d(tex, None),
                            ));
                        }
                    }
                }

                let uniforms = MultiUniforms {
                    uniforms: image_uniforms,
                    next: &self.state,
                };

                self.shader.try_update_size(&self.glium.ctx, render_res);

                self.shader.draw(target, &uniforms)?;

                Ok(())
            });
        drop(_span)
    }
}

impl IsfFFGLInstance {
    pub(crate) fn new(
        state: &IsfFFGLState,
        inst_data: &ffgl_core::FFGLData,
    ) -> Result<Self, IsfShaderLoadError> {
        tracing::debug!("CREATED INSTANCE");

        let glium = FFGLGlium::new(inst_data);

        let shader = shader::IsfShader::new(
            &glium.ctx,
            &state.info,
            inst_data.get_dimensions(),
            &state.source,
        )?;

        Ok(Self {
            shader,
            state: state.clone(),
            glium,
            file_textures: HashMap::new(),
        })
    }
}
