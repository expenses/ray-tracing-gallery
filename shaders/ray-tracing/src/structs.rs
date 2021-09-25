use spirv_std::glam::{Mat4, Vec2, Vec3};

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

#[repr(C)]
#[derive(Copy, Clone)]
pub struct PrimaryRayPayload {
    pub colour: Vec3,
    pub ray_hit_t: f32,
    pub reflected_direction: Vec3,
}

#[derive(Copy, Clone)]
pub struct ShadowRayPayload {
    pub shadowed: bool,
}
