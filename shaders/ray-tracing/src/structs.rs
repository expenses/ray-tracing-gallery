use spirv_std::glam::{Vec2, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
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
