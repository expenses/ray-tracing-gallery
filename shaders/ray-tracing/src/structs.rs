use spirv_std::glam::{Mat4, Vec2, Vec3};

type DeviceAddress = u64;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Uniforms {
    pub view_inverse: Mat4,
    pub proj_inverse: Mat4,
    pub sun_dir: Vec3,
    pub show_heatmap: bool,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ModelInfo {
    pub position_buffer_address: DeviceAddress,
    pub normal_buffer_address: DeviceAddress,
    pub uv_buffer_address: DeviceAddress,
    pub geometry_info_address: DeviceAddress,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GeometryInfo {
    pub index_buffer_address: DeviceAddress,
    pub image_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PrimaryRayPayload {
    pub colour: Vec3,
    pub new_ray_origin: Vec3,
    pub new_ray_direction: Vec3,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ShadowRayPayload {
    pub shadowed: bool,
}
