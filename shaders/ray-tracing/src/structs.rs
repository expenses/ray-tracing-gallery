use spirv_std::glam::{const_vec3, IVec3, Mat4, Vec2, Vec3, Vec4};

#[derive(Copy, Clone)]
pub struct Uniforms {
    pub view_inverse: Mat4,
    pub proj_inverse: Mat4,
    pub sun_dir: Vec3,
    pub sun_radius: f32,
}

#[derive(Copy, Clone)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Copy, Clone)]
pub struct ModelInfo {
    pub vertex_buffer_address: u64,
    pub index_buffer_address: u64,
    pub texture_index: u32,
}

#[derive(Copy, Clone)]
pub struct PrimaryRayPayload {
    pub hit_value: Vec3,
}

#[derive(Copy, Clone)]
pub struct ShadowRayPayload {
    pub shadowed: bool,
}
