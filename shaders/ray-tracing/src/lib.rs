#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std,
    feature(asm)
)]

// This needs to be here to provide `#[panic_handler]`.
extern crate spirv_std;

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

use spirv_std::glam::{const_vec3, IVec3, Mat3, Vec2, Vec3, Vec4};

use spirv_std::num_traits::Float;
use spirv_std::{
    arch::ignore_intersection,
    image::SampledImage,
    memory::Scope,
    ray_tracing::{AccelerationStructure, RayFlags},
    Image, RuntimeArray,
    arch::convert_u_to_ptr,
};

mod heatmap;
//mod pbr;
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
    let uniforms = unsafe { &mut *convert_u_to_ptr::<Uniforms>(buffer_addresses.uniforms) };

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
    Vec3::new(
        linear_to_srgb_scalar(linear.x),
        linear_to_srgb_scalar(linear.y),
        linear_to_srgb_scalar(linear.z),
    )
}

#[cfg(target_arch = "spirv")]
#[spirv(ray_generation)]
pub fn ray_generation(
    #[spirv(ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(descriptor_set = 1, binding = 0)] image: &Image!(2D, format = rgba8, sampled = false),
    #[spirv(launch_id)] launch_id: IVec3,
    #[spirv(launch_size)] launch_size: IVec3,
    #[spirv(push_constant)] buffer_addresses: &PushConstantBufferAddresses,
) {
    use spirv_std::arch::read_clock_khr;

    let uniforms = unsafe { &mut *convert_u_to_ptr::<Uniforms>(buffer_addresses.uniforms) };
    let tlas = unsafe { AccelerationStructure::from_u64(buffer_addresses.acceleration_structure) };

    /*
    if launch_id == IVec3::new(0, 0, 0) {
        unsafe {
            spirv_std::macros::debug_printfln!(
                "show_heatmap = %u, frame_index = %u",
                uniforms.show_heatmap,
                uniforms.frame_index
            );
        }
    }
    */

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

type Textures = RuntimeArray<SampledImage<Image!(2D, type=f32, sampled=true)>>;

fn sample_texture(textures: &Textures, index: u32, uv: Vec2) -> Vec4 {
    unsafe {
        let index = index as usize;
        // Todo: index needs to be non-uniform.
        let texture = textures.index(index);
        texture.sample_by_lod(uv, 0.0)
    }
}

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
    #[spirv(instance_custom_index)] instance_custom_index: usize,
    #[spirv(ray_geometry_index)] geometry_index: usize,
    #[spirv(primitive_id)] primitive_id: usize,
    #[spirv(hit_attribute)] hit_attributes: &mut Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
) {
    let (model_info, geometry_info) = load_model_and_geometry_info(buffer_addresses, instance_custom_index, geometry_index);

    let indices = read_indices(&geometry_info, primitive_id);

    let interpolated_uv: Vec2 = TriangleOfComponents::load_vec2(model_info.uv_buffer_address, indices)
        .interpolate(compute_barycentric_coords(*hit_attributes));

    let sample = sample_texture(
        textures,
        geometry_info.images.diffuse_image_index,
        interpolated_uv,
    );

    if sample.w < 0.5 {
        unsafe {
            ignore_intersection();
        }
    }
}

#[spirv(matrix)]
pub struct Mat4x3 {
    x: Vec3,
    y: Vec3,
    z: Vec3,
    _w: Vec3,
}

impl Mat4x3 {
    fn as_mat3(&self) -> Mat3 {
        Mat3::from_cols(self.x, self.y, self.z)
    }
}

fn load_model_and_geometry_info(buffer_addresses: &PushConstantBufferAddresses, instance_custom_index: usize, geometry_index: usize) -> (ModelInfo, GeometryInfo) {
    let model_info = unsafe {
        *load_runtime_array_from_handle::<ModelInfo>(buffer_addresses.model_info).index(instance_custom_index)
    };

    let geometry_info = unsafe {
        *load_runtime_array_from_handle::<GeometryInfo>(model_info.geometry_info_address).index(geometry_index)
    };

    (model_info, geometry_info)
}

#[spirv(closest_hit)]
pub fn closest_hit_mirror(
    #[spirv(push_constant)] buffer_addresses: &PushConstantBufferAddresses,
    #[spirv(instance_custom_index)] instance_custom_index: usize,
    #[spirv(ray_geometry_index)] geometry_index: usize,
    #[spirv(primitive_id)] primitive_id: usize,
    #[spirv(hit_attribute)] hit_attributes: &mut Vec2,
    #[spirv(object_to_world)] object_to_world: Mat4x3,
    #[spirv(incoming_ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(world_ray_origin)] world_ray_origin: Vec3,
    #[spirv(world_ray_direction)] world_ray_direction: Vec3,
    #[spirv(ray_tmax)] ray_hit_t: f32,
) {
    let (model_info, geometry_info) = load_model_and_geometry_info(buffer_addresses, instance_custom_index, geometry_index);

    let indices = read_indices(&geometry_info, primitive_id);

    let interpolated_normal: Vec3 = TriangleOfComponents::load_vec3(model_info.normal_buffer_address, indices)
        .interpolate(compute_barycentric_coords(*hit_attributes));

    let rotated_normal = object_to_world.as_mat3() * interpolated_normal;

    let normal = rotated_normal.normalize();

    payload.new_ray_direction = reflect(world_ray_direction, normal);
    payload.new_ray_origin = world_ray_origin + world_ray_direction * ray_hit_t;
}

#[spirv(closest_hit)]
pub fn closest_hit_textured(
    #[spirv(push_constant)] buffer_addresses: &PushConstantBufferAddresses,
    #[spirv(instance_custom_index)] instance_custom_index: usize,
    #[spirv(ray_geometry_index)] geometry_index: usize,
    #[spirv(primitive_id)] primitive_id: usize,
    #[spirv(hit_attribute)] hit_attributes: &mut Vec2,
    #[spirv(incoming_ray_payload)] payload: &mut PrimaryRayPayload,
    #[spirv(descriptor_set = 0, binding = 0)] textures: &Textures,
) {
    let (model_info, geometry_info) = load_model_and_geometry_info(buffer_addresses, instance_custom_index, geometry_index);

    let uniforms = unsafe { &mut *convert_u_to_ptr::<Uniforms>(buffer_addresses.uniforms) };

    let triangle = FullTriangle::load(&model_info, &geometry_info, primitive_id);

    let barycentric_coords = compute_barycentric_coords(*hit_attributes);

    let interpolated = triangle.interpolate(barycentric_coords);

    let sun_factor = 1.0;

    let material_data = MaterialData::load(&geometry_info, interpolated.uv, textures);

    payload.colour = material_data.colour;
}

struct MaterialData {
    colour: Vec3,
    metallic: f32,
    roughness: f32,
}

impl MaterialData {
    fn load(info: &GeometryInfo, uv: Vec2, textures: &Textures) -> Self {
        let metallic_roughness_sample =
            sample_texture(textures, info.images.metallic_roughness_image_index, uv);

        Self {
            metallic: metallic_roughness_sample.y,
            roughness: metallic_roughness_sample.x,
            colour: sample_texture(textures, info.images.diffuse_image_index, uv).truncate(),
        }
    }
}

struct FullTriangle {
    positions: TriangleOfComponents<Vec3>,
    normals: TriangleOfComponents<Vec3>,
    uvs: TriangleOfComponents<Vec2>,
}

impl FullTriangle {
    pub fn load(model_info: &ModelInfo, geometry_info: &GeometryInfo, primitive_id: usize) -> Self {
        let indices = read_indices(geometry_info, primitive_id);

        Self {
            positions: TriangleOfComponents::load_vec3(model_info.position_buffer_address, indices),
            normals: TriangleOfComponents::load_vec3(model_info.normal_buffer_address, indices),
            uvs: TriangleOfComponents::load_vec2(model_info.uv_buffer_address, indices),
        }
    }

    pub fn interpolate(&self, barycentric_coords: Vec3) -> Vertex {
        Vertex {
            pos: self.positions.interpolate(barycentric_coords),
            normal: self.normals.interpolate(barycentric_coords),
            uv: self.uvs.interpolate(barycentric_coords),
        }
    }
}

fn reflect(incidence: Vec3, normal: Vec3) -> Vec3 {
    incidence - 2.0 * normal.dot(incidence) * normal
}


fn read_indices(geometry_info: &GeometryInfo, primitive_id: usize) -> [usize; 3] {
    let runtime_array =
        unsafe { load_runtime_array_from_handle::<usize>(geometry_info.index_buffer_address) };

    let offset = primitive_id * 3;

    // We need to do 3 reads instead of 1 read as the alignment is wrongly set to 8 bytes instead of 4.
    unsafe {
        [
            *runtime_array.index(offset),
            *runtime_array.index(offset + 1),
            *runtime_array.index(offset + 2),
        ]
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

struct TriangleOfComponents<V> {
    a: V,
    b: V,
    c: V,
}

impl<V: Copy + Add<V, Output = V> + Mul<f32, Output = V> + 'static> TriangleOfComponents<V> {
    fn interpolate(&self, barycentric_coords: Vec3) -> V {
        interpolate(self.a, self.b, self.c, barycentric_coords)
    }
}

unsafe fn load_runtime_array_from_handle<T>(handle: u64) -> &'static mut RuntimeArray<T> {
    &mut *convert_u_to_ptr(handle)
}

impl TriangleOfComponents<Vec3> {
    fn load_vec3(handle: u64, indices: [usize; 3]) -> Self {
        let array = unsafe { load_runtime_array_from_handle::<f32>(handle) };

        let load_vec3 = |index: usize| -> Vec3 {
            let offset = index * 3;
            unsafe {
                Vec3::new(
                    *array.index(offset),
                    *array.index(offset + 1),
                    *array.index(offset + 2),
                )
            }
        };

        Self {
            a: load_vec3(indices[0]),
            b: load_vec3(indices[1]),
            c: load_vec3(indices[2]),
        }
    }
}

impl TriangleOfComponents<Vec2> {
    fn load_vec2(handle: u64, indices: [usize; 3]) -> Self {
        let array = unsafe { load_runtime_array_from_handle::<Vec2>(handle) };

        unsafe {
            Self {
                a: *array.index(indices[0]),
                b: *array.index(indices[1]),
                c: *array.index(indices[2]),
            }
        }
    }
}
