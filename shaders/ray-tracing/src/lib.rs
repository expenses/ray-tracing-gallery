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

use spirv_std::glam::{const_vec3, IVec3, Vec2, Vec3, Vec4};

use spirv_std::{
    memory::Scope,
    ray_tracing::{AccelerationStructure, RayFlags},
    Image,
};
use spirv_std::num_traits::Float;

mod heatmap;
mod pbr;
mod structs;

use core::ops::{Add, Mul};
use heatmap::heatmap_temperature;
use shared_structs::{PushConstantBufferAddresses, Uniforms};
use structs::{PrimaryRayPayload, ShadowRayPayload, Vertex};

#[spirv(miss)]
pub fn shadow_ray_miss(#[spirv(incoming_ray_payload)] payload: &mut ShadowRayPayload) {
    payload.shadowed = false;
}

const SKY_COLOUR: Vec3 = const_vec3!([0.0, 0.0, 0.05]);

#[spirv(miss)]
pub fn primary_ray_miss(
    #[spirv(incoming_ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(descriptor_set = 1, binding = 1, uniform)] uniforms: &Uniforms,
    #[spirv(world_ray_direction)] world_ray_direction: Vec3,
) {
    if world_ray_direction.dot(uniforms.sun_dir.into()) > uniforms.sun_radius.cos() {
        payload.colour = Vec3::splat(1.0);
    } else {
        payload.colour = SKY_COLOUR;
    }
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

// https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB
fn linear_to_srgb_scalar(linear: f32) -> f32 {
    if linear < 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

fn linear_to_srgb(linear: Vec3) -> Vec3 {
    Vec3::new(linear_to_srgb_scalar(linear.x), linear_to_srgb_scalar(linear.y), linear_to_srgb_scalar(linear.z))
}

#[cfg(target_arch = "spirv")]
#[spirv(ray_generation)]
pub fn ray_generation(
    #[spirv(ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(descriptor_set = 1, binding = 0)] image: &Image!(2D, format = rgba8, sampled = false),
    #[spirv(descriptor_set = 1, binding = 1, uniform)] uniforms: &Uniforms,
    #[spirv(launch_id)] launch_id: IVec3,
    #[spirv(launch_size)] launch_size: IVec3,
    #[spirv(push_constant)] buffer_addresses: &PushConstantBufferAddresses,
) {
    let tlas = unsafe { AccelerationStructure::from_u64(buffer_addresses.acceleration_structure) };

    use spirv_std::arch::read_clock_khr;

    let start_time = if uniforms.show_heatmap {
        unsafe { read_clock_khr::<{ Scope::Subgroup as u32 }>() }
    } else {
        0
    };

    let launch_id_xy = launch_id.truncate();
    let launch_size_xy = launch_size.truncate();

    let pixel_center = launch_id_xy.as_vec2() + 0.5;

    let texture_coordinates = pixel_center / launch_size_xy.as_vec2();

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
            tlas: &tlas,
            flags: RayFlags::empty(),
            origin,
            direction,
            t_min: 0.01,
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

    let colour = if uniforms.show_heatmap {
        let end_time = unsafe { read_clock_khr::<{ Scope::Subgroup as u32 }>() };

        let delta_time = end_time - start_time;

        let heatmap_scale = 1_000_000.0;

        let delta_time_scaled = delta_time as f32 / heatmap_scale;

        heatmap_temperature(delta_time_scaled) + (payload.colour * 0.000001)
    } else {
        payload.colour
    };


    unsafe {
        image.write(launch_id_xy, linear_to_srgb(colour).extend(1.0));
    }
}

fn compute_barycentric_coords(hit_attributes: Vec2) -> Vec3 {
    Vec3::new(
        1.0 - hit_attributes.x - hit_attributes.y,
        hit_attributes.x,
        hit_attributes.y,
    )
}

fn interpolate<T: Mul<f32, Output = T> + Add<T, Output = T>>(
    a: T,
    b: T,
    c: T,
    barycentric_coords: Vec3,
) -> T {
    a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z
}

struct Triangle {
    a: Vertex,
    b: Vertex,
    c: Vertex,
}

impl Triangle {
    fn interpolate(&self, barycentric_coords: Vec3) -> Vertex {
        Vertex {
            pos: interpolate(self.a.pos, self.b.pos, self.c.pos, barycentric_coords),
            normal: interpolate(
                self.a.normal,
                self.b.normal,
                self.c.normal,
                barycentric_coords,
            ),
            uv: interpolate(self.a.uv, self.b.uv, self.c.uv, barycentric_coords),
        }
    }
}

/*
#[spirv(closest_hit)]
pub fn wip_closest_hit(
    #[spirv(instance_custom_index)] model_index: u32,
    #[spirv(hit_attribute)] hit_attributes: &mut Vec2,
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

    let triangle = Triangle {
        a: Vertex::default(),
        b: Vertex::default(),
        c: Vertex::default(),
    };

    let barycentric_coords = compute_barycentric_coords(*hit_attributes);

    let interpolated = triangle.interpolate(barycentric_coords);

    let texture = unsafe { textures.index(info.texture_index as usize) };
    let colour: Vec4 = unsafe { texture.sample_by_lod(interpolated.uv, 0.0) };
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
*/

#[spirv(closest_hit)]
pub fn closest_hit_portal(
    #[spirv(incoming_ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(world_ray_origin)] world_ray_origin: Vec3,
    #[spirv(world_ray_direction)] world_ray_direction: Vec3,
    #[spirv(ray_tmax)] ray_hit_t: f32,
) {
    let portal_relative_position = Vec3::new(0.0, 5.0, 0.0);

    payload.new_ray_direction = world_ray_direction;
    payload.new_ray_origin =
        world_ray_origin + world_ray_direction * ray_hit_t + portal_relative_position;
}
