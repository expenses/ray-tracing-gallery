#![no_std]

type DeviceAddress = u64;

use glam::{Mat4, Vec3};

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Uniforms {
    pub view_inverse: Mat4,
    pub proj_inverse: Mat4,
    pub sun_dir: Vec3,
    pub sun_radius: f32,
    pub blue_noise_texture_index: u32,
    pub frame_index: u32,
    pub _padding: u32,
    pub show_heatmap: bool,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct ModelInfo {
    pub position_buffer_address: DeviceAddress,
    pub normal_buffer_address: DeviceAddress,
    pub uv_buffer_address: DeviceAddress,
    pub geometry_info_address: DeviceAddress,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GeometryInfo {
    pub index_buffer_address: DeviceAddress,
    pub images: GeometryImages,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct GeometryImages {
    pub diffuse_image_index: u32,
    pub metallic_roughness_image_index: u32,
    pub normal_map_image_index: i32,
    pub _padding: u32,
}

#[cfg_attr(not(target_arch = "spirv"), derive(Debug))]
#[derive(Copy, Clone)]
#[repr(C)]
pub struct PushConstantBufferAddresses {
    pub model_info: DeviceAddress,
    pub uniforms: DeviceAddress,
    pub acceleration_structure: DeviceAddress,
}
