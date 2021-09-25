use crate::util_structs::AccelerationStructure;
use crate::HitShader;
use ash::vk;
use ultraviolet::{Mat4, Vec2, Vec3, Vec4};

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
pub struct RayTracingUniforms {
    pub view_inverse: Mat4,
    pub proj_inverse: Mat4,
    pub sun_dir: Vec3,
    pub sun_radius: f32,
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
pub struct ModelInfo {
    pub vertex_buffer_address: vk::DeviceAddress,
    pub index_buffer_address: vk::DeviceAddress,
    pub image_index: u32,
    pub _padding: u32,
}

// A `bytemuck`'d `vk::AccelerationStructureInstanceKHR`.
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
pub struct AccelerationStructureInstance {
    pub transform: [Vec4; 3],
    pub instance_custom_index_and_mask: u32,
    pub instance_shader_binding_table_record_offset_and_flags: u32,
    pub acceleration_structure_device_address: vk::DeviceAddress,
}

impl AccelerationStructureInstance {
    pub fn new(
        transform: Mat4,
        blas: &AccelerationStructure,
        device: &ash::Device,
        custom_index: u32,
        hit_shader: HitShader,
    ) -> Self {
        fn merge_fields(lower: u32, upper: u8) -> u32 {
            ((upper as u32) << 24) | lower
        }

        let sbt_offset = hit_shader as u32;

        Self {
            transform: transpose_matrix_for_instance(transform),
            instance_custom_index_and_mask: merge_fields(custom_index, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: merge_fields(
                sbt_offset,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_device_address: blas.buffer.device_address(device),
        }
    }
}

pub fn transpose_matrix_for_instance(matrix: Mat4) -> [Vec4; 3] {
    let rows = matrix.transposed().cols;
    [rows[0], rows[1], rows[2]]
}
