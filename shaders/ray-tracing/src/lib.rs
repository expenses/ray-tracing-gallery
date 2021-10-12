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

use spirv_std::glam::{const_vec3, IVec3, UVec3, Vec2, Vec3, Vec4};

use spirv_std::{
    arch::ignore_intersection,
    image::SampledImage,
    memory::Scope,
    ray_tracing::{AccelerationStructure, RayFlags},
    Image, RuntimeArray,
};

mod heatmap;
mod pbr;
mod structs;

use core::ops::{Add, Mul};
use heatmap::heatmap_temperature;
use shared_structs::{GeometryInfo, ModelInfo, PushConstantBufferAddresses, Uniforms};
use structs::{PrimaryRayPayload, ShadowRayPayload, Vertex};

#[spirv(miss)]
pub fn shadow_ray_miss(#[spirv(incoming_ray_payload)] payload: &mut ShadowRayPayload) {
    payload.shadowed = false;
}

const SKY_COLOUR: Vec3 = const_vec3!([0.0, 0.0, 0.05]);

#[spirv(miss)]
pub fn primary_ray_miss(
    #[spirv(incoming_ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(world_ray_direction)] world_ray_direction: Vec3,
    #[spirv(push_constant)] buffer_addresses: &PushConstantBufferAddresses,
) {
    use spirv_std::arch::resource_from_handle;

    let uniforms: &'static Uniforms = unsafe { resource_from_handle(buffer_addresses.uniforms) };

    if world_ray_direction.dot(uniforms.sun_dir) > 0.998 {
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

#[cfg(target_arch = "spirv")]
#[spirv(ray_generation)]
pub fn ray_generation(
    #[spirv(ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(descriptor_set = 1, binding = 0)] image: &Image!(2D, format = rgba8, sampled = false),
    //#[spirv(descriptor_set = 1, binding = 1, uniform)] uniforms: &Uniforms,
    #[spirv(launch_id)] launch_id: IVec3,
    #[spirv(launch_size)] launch_size: IVec3,
    #[spirv(push_constant)] buffer_addresses: &PushConstantBufferAddresses,
) {
    use spirv_std::arch::{read_clock_khr, resource_from_handle};

    let uniforms: &'static Uniforms = unsafe { resource_from_handle(buffer_addresses.uniforms) };
    let tlas = unsafe { AccelerationStructure::from_u64(buffer_addresses.acceleration_structure) };

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

    let gamma = 2.2;

    let gamma_corrected_colour = colour.powf(1.0 / gamma);

    unsafe {
        image.write(launch_id_xy, gamma_corrected_colour.extend(1.0));
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

#[spirv(any_hit)]
pub fn any_hit_alpha_clip(
    #[spirv(push_constant)] buffer_addresses: &PushConstantBufferAddresses,
    #[spirv(instance_custom_index)] instance_custom_index: u32,
    #[spirv(ray_geometry_index)] geometry_index: u32,
    #[spirv(primitive_id)] primitive_id: u32,
    #[spirv(hit_attribute)] hit_attributes: &mut Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &RuntimeArray<
        SampledImage<Image!(2D, type=f32, sampled=true)>,
    >,
) {
    let model_info: &'static ModelInfo = unsafe {
        load_runtime_array(buffer_addresses.model_info).index(instance_custom_index as usize)
    };

    let geometry_info: &'static GeometryInfo = unsafe {
        load_runtime_array(model_info.geometry_info_address).index(geometry_index as usize)
    };

    let indices = read_indices(geometry_info, primitive_id);

    let interpolated_uv: Vec2 = Triangle::load(model_info.uv_buffer_address, indices)
        .interpolate(compute_barycentric_coords(*hit_attributes));

    // Todo: index needs to be non-uniform.
    let texture = unsafe { textures.index(geometry_info.images.diffuse_image_index as usize) };

    let sample: Vec4 = unsafe { texture.sample_by_lod(interpolated_uv, 0.0) };

    if sample.w < 0.5 {
        unsafe {
            ignore_intersection();
        }
    }
}

unsafe fn load_runtime_array<T>(handle: u64) -> &'static mut RuntimeArray<T> {
    spirv_std::arch::resource_from_handle::<&'static mut RuntimeArray<T>>(handle)
}

fn read_indices(geometry_info: &GeometryInfo, primitive_id: u32) -> UVec3 {
    let runtime_array: &'static mut RuntimeArray<u32> =
        unsafe { load_runtime_array(geometry_info.index_buffer_address) };

    let offset = primitive_id as usize * 3;

    // We need to do 3 reads instead of 1 read as the alignment is wrongly set to 8 bytes instead of 4.
    unsafe {
        UVec3::new(
            *runtime_array.index(offset),
            *runtime_array.index(offset + 1),
            *runtime_array.index(offset + 2),
        )
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

struct Triangle<V> {
    a: V,
    b: V,
    c: V,
}

impl<V: Copy + Add<V, Output = V> + Mul<f32, Output = V> + 'static> Triangle<V> {
    fn load(handle: u64, indices: UVec3) -> Self {
        let array: &'static mut RuntimeArray<V> = unsafe { load_runtime_array(handle) };

        unsafe {
            Self {
                a: *array.index(indices.x as usize),
                b: *array.index(indices.y as usize),
                c: *array.index(indices.z as usize),
            }
        }
    }

    fn interpolate(&self, barycentric_coords: Vec3) -> V {
        interpolate(self.a, self.b, self.c, barycentric_coords)
    }
}
