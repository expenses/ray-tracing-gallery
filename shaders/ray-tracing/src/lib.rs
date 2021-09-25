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

mod structs;

use structs::{ModelInfo, PrimaryRayPayload, ShadowRayPayload, Uniforms, Vertex};

#[spirv(miss)]
pub fn shadow_ray_miss(#[spirv(incoming_ray_payload)] payload: &mut ShadowRayPayload) {
    payload.shadowed = false;
}

const SKY_COLOUR: Vec3 = const_vec3!([0.0, 0.0, 0.2]);

#[spirv(miss)]
pub fn primary_ray_miss(#[spirv(incoming_ray_payload)] payload: &mut PrimaryRayPayload) {
    payload.colour = SKY_COLOUR;
}

struct TraceRayExplicitParams<'a, T> {
    tlas: &'a AccelerationStructure,
    flags: RayFlags,
    cull_mask: i32,
    sbt_offset: i32,
    sbt_stride: i32,
    miss_shader_index: i32,
    origin: Vec3,
    t_min: f32,
    direction: Vec3,
    t_max: f32,
    payload: &'a mut T,
}

fn trace_ray_explicit<T>(params: TraceRayExplicitParams<T>) {
    unsafe {
        params.tlas.trace_ray(
            params.flags,
            params.cull_mask,
            params.sbt_offset,
            params.sbt_stride,
            params.miss_shader_index,
            params.origin,
            params.t_min,
            params.direction,
            params.t_max,
            params.payload,
        )
    }
}

#[spirv(ray_generation)]
pub fn ray_generation(
    #[spirv(ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(descriptor_set = 1, binding = 0)] tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 1, binding = 1)] image: &Image!(2D, format = rgba8, sampled = false),
    #[spirv(descriptor_set = 1, binding = 2, uniform)] uniforms: &Uniforms,
    #[spirv(launch_id)] launch_id: IVec3,
    #[spirv(launch_size)] launch_size: IVec3,
) {
    let launch_id_xy = launch_id.truncate();
    let launch_size_xy = launch_size.truncate();

    let pixel_center = launch_id_xy.as_f32() + 0.5;

    let texture_coordinates = pixel_center / launch_size_xy.as_f32();

    let render_coordinates = texture_coordinates * 2.0 - 1.0;

    // Transform [0, 0, 0] in view-space into world space
    let mut origin = (uniforms.view_inverse * Vec4::new(0.0, 0.0, 0.0, 1.0)).truncate();
    // Transform the render coordinates into project-y space
    let target =
        uniforms.proj_inverse * Vec4::new(render_coordinates.x, render_coordinates.y, 1.0, 1.0);
    let local_direction_vector = target.truncate().normalize();
    // Rotate the location direction vector into a global direction vector.
    let mut direction = (uniforms.view_inverse * local_direction_vector.extend(0.0)).truncate();

    for _ in 0..3 {
        *payload = PrimaryRayPayload {
            colour: Vec3::splat(0.0),
            new_ray_origin: Vec3::splat(0.0),
            new_ray_direction: Vec3::splat(0.0),
        };

        trace_ray_explicit(TraceRayExplicitParams {
            tlas,
            flags: RayFlags::OPAQUE,
            origin,
            direction,
            t_min: 0.001,
            t_max: 10_000.0,
            cull_mask: 0xff,
            sbt_offset: 0,
            sbt_stride: 0,
            miss_shader_index: 0,
            payload,
        });

        if payload.new_ray_direction == Vec3::splat(0.0) {
            // We've hit a non-reflecting object.
            break;
        } else {
            origin = payload.new_ray_origin;
            direction = payload.new_ray_direction;
        }
    }

    unsafe {
        image.write(launch_id_xy, payload.colour.extend(1.0));
    }
}

fn interpolate(a: Vec3, b: Vec3, c: Vec3, barycentric_coords: Vec3) -> Vec3 {
    a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z
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
    #[spirv(ray_payload)] shadow_payload: &mut ShadowRayPayload,
    #[spirv(world_ray_origin)] world_ray_origin: Vec3,
    #[spirv(world_ray_direction)] world_ray_direction: Vec3,
    #[spirv(ray_tmax)] ray_hit_t: f32,
    #[spirv(descriptor_set = 0, binding = 0, storage_buffer)] model_info: &[ModelInfo],
    #[spirv(descriptor_set = 0, binding = 1)] textures: &RuntimeArray<
        SampledImage<Image!(2D, type=f32, sampled=true)>,
    >,
    #[spirv(descriptor_set = 1, binding = 0)] tlas: &AccelerationStructure,
    #[spirv(descriptor_set = 1, binding = 2, uniform)] uniforms: &Uniforms,
) {
    let info = &model_info[model_index as usize];

    // Missing features:
    //
    // * casting u64 addresses to pointers (OpConvertUToPtr)
    //   - https://github.com/EmbarkStudios/rust-gpu/issues/383
    // * NonUniform support:
    //   - https://github.com/EmbarkStudios/rust-gpu/issues/756

    // To check that reading from model_info works.
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

    let uv = Vec2::splat(0.0);
    let texture = unsafe { textures.index(info.texture_index as usize) };
    let colour: Vec4 = unsafe { texture.sample_by_lod(uv, 0.0) };
    let colour = colour.truncate();

    let shadowed = {
        shadow_payload.shadowed = true;

        trace_ray_explicit(TraceRayExplicitParams {
            tlas,
            flags: RayFlags::OPAQUE
                | RayFlags::TERMINATE_ON_FIRST_HIT
                | RayFlags::SKIP_CLOSEST_HIT_SHADER,
            origin: world_ray_origin + world_ray_direction * ray_hit_t,
            direction: uniforms.sun_dir,
            t_min: 0.001,
            t_max: 10_000.0,
            payload: shadow_payload,
            cull_mask: 0xff,
            // Not sure if we need this offset
            sbt_offset: 1,
            sbt_stride: 0,
            miss_shader_index: 1,
        });

        shadow_payload.shadowed
    };

    let lighting = !shadowed as u8 as f32;

    payload.colour = colour * ((lighting * 0.6) + 0.4);
}
