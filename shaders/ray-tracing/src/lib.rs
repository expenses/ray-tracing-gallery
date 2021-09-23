#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std
)]
#![feature(asm)]

// This needs to be here to provide `#[panic_handler]`.
extern crate spirv_std;

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

use spirv_std::glam::{const_vec3, IVec3, Mat4, Vec2, Vec3, Vec4};

use spirv_std::{
    image::SampledImage,
    ray_tracing::{AccelerationStructure, RayFlags},
    Image, RuntimeArray,
};

pub struct PrimaryRayPayload {
    hit_value: Vec3,
}

pub struct Uniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    sun_dir: Vec3,
    sun_radius: f32,
}

#[spirv(miss)]
pub fn shadow_ray_miss(#[spirv(incoming_ray_payload)] shadowed: &mut bool) {
    *shadowed = false;
}

const SKY_COLOUR: Vec3 = const_vec3!([0.0, 0.0, 0.2]);

#[spirv(miss)]
pub fn primary_ray_miss(#[spirv(incoming_ray_payload)] payload: &mut PrimaryRayPayload) {
    payload.hit_value = SKY_COLOUR;
}

#[spirv(ray_generation)]
pub fn ray_generation(
    #[spirv(ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(descriptor_set = 1, binding = 0)] top_level_as: &AccelerationStructure,
    #[spirv(descriptor_set = 1, binding = 1)] image: &Image!(2D, format = rgba8, sampled = false),
    #[spirv(descriptor_set = 1, binding = 2, uniform)] uniforms: &Uniforms,
    #[spirv(launch_id)] launch_id: IVec3,
    #[spirv(launch_size)] launch_size: IVec3,
) {
    let pixel_center = launch_id.truncate().as_f32() + 0.5;
    let in_uv = pixel_center / launch_size.truncate().as_f32();

    let d = in_uv * 2.0 - 1.0;

    let origin = uniforms.view_inverse * Vec4::new(0.0, 0.0, 0.0, 1.0);
    let target = uniforms.proj_inverse * Vec4::new(d.x, d.y, 1.0, 1.0);

    let direction = uniforms.view_inverse * target.truncate().normalize().extend(0.0);

    let t_min = 0.001;
    let t_max = 10_000.0;

    payload.hit_value = Vec3::splat(0.0);

    unsafe {
        top_level_as.trace_ray(
            RayFlags::OPAQUE,
            0xff,
            0,
            0,
            0,
            origin.truncate(),
            t_min,
            direction.truncate(),
            t_max,
            payload,
        );

        image.write(launch_id.truncate(), payload.hit_value.extend(1.0));
    }
}

fn interpolate(a: Vec3, b: Vec3, c: Vec3, barycentric_coords: Vec3) -> Vec3 {
    a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z
}

pub struct ModelInfo {
    vertex_buffer_address: u64,
    index_buffer_address: u64,
    texture_index: u32,
}

#[derive(Copy, Clone)]
pub struct Vertex {
    pos: Vec3,
    normal: Vec3,
    uv: Vec2,
}

fn get_shadow_terminator_fix_shadow_origin(
    a: Vertex,
    b: Vertex,
    c: Vertex,
    barycentric_coords: Vec3,
    world_to_object: Mat4,
) -> Vec3 {
    let point = interpolate(a.pos, b.pos, c.pos, barycentric_coords);

    fn compute_vector_and_project_onto_tangent_plane(point: Vec3, vert: Vertex) -> Vec3 {
        let vector_to_point = point - vert.pos;

        let dot_product = vector_to_point.dot(vert.normal).min(0.0);

        vector_to_point - (dot_product * vert.normal)
    }

    let offset_a = compute_vector_and_project_onto_tangent_plane(point, a);
    let offset_b = compute_vector_and_project_onto_tangent_plane(point, b);
    let offset_c = compute_vector_and_project_onto_tangent_plane(point, c);

    let intepolated_offset = interpolate(offset_a, offset_b, offset_c, barycentric_coords);

    (world_to_object * (point + intepolated_offset).extend(1.0)).truncate()
}

#[spirv(closest_hit)]
pub fn wip_closest_hit(
    #[spirv(instance_custom_index)] model_index: u32,
    #[spirv(hit_attribute)] hit_attributes: &mut Vec2,
    #[spirv(primitive_id)] primitive_id: u32,
    #[spirv(incoming_ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(ray_payload)] shadowed: &mut bool,
    #[spirv(world_ray_origin)] world_ray_origin: Vec3,
    #[spirv(world_ray_direction)] world_ray_direction: Vec3,
    #[spirv(ray_tmax)] ray_hit_t: f32,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] model_info: &[ModelInfo],
    #[spirv(descriptor_set = 0, binding = 1)] texture: &SampledImage<
        Image!(2D, type=f32, sampled=true),
    >,
    #[spirv(descriptor_set = 1, binding = 0)] top_level_as: &AccelerationStructure,
    #[spirv(descriptor_set = 1, binding = 2, uniform)] uniforms: &Uniforms,
) {
    let info = &model_info[model_index as usize];

    // Missing features:
    //
    // * casting u64 addresses to pointers
    // * `SampledImage`s don't seem to work

    let colour_modulator = {
        let mut components = [1.0, 1.0, 1.0];
        components[info.texture_index as usize % 3] = 0.0;
        Vec3::from(components)
    };

    let barycentric_coords = Vec3::new(
        1.0 - hit_attributes.x - hit_attributes.y,
        hit_attributes.x,
        hit_attributes.y,
    );

    let index_offset = primitive_id * 3;

    payload.hit_value = barycentric_coords * colour_modulator;

    /*
    {
        let uv = Vec2::splat(0.0);
        let colour: Vec4 = unsafe { texture.sample(uv) };
    }
    */

    let t_min = 0.001;
    let t_max = 10_000.0;

    let shadow_origin = world_ray_origin + world_ray_direction * ray_hit_t;

    let shadowed = {
        *shadowed = true;

        unsafe {
            top_level_as.trace_ray(
                RayFlags::OPAQUE
                    | RayFlags::TERMINATE_ON_FIRST_HIT
                    | RayFlags::SKIP_CLOSEST_HIT_SHADER,
                0xff,
                1,
                0,
                1,
                shadow_origin,
                t_min,
                uniforms.sun_dir,
                t_max,
                shadowed,
            );
        }

        *shadowed
    };

    let lighting = !shadowed as u8 as f32;

    payload.hit_value *= (lighting * 0.6) + 0.4;
}
