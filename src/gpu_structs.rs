use crate::util_structs::{Device, Model};
use crate::HitShader;
use ash::vk;
use ultraviolet::{Mat4, Vec3, Vec4};

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct RayTracingUniforms {
    pub view_inverse: Mat4,
    pub proj_inverse: Mat4,
    pub sun_dir: Vec3,
    pub _padding: u32,
    pub show_heatmap: bool,
}

pub fn bytes_of<T>(reference: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(reference as *const T as *const u8, std::mem::size_of::<T>())
    }
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
pub struct ModelInfo {
    pub position_buffer_address: vk::DeviceAddress,
    pub normal_buffer_address: vk::DeviceAddress,
    pub uv_buffer_address: vk::DeviceAddress,
    pub index_buffer_address: vk::DeviceAddress,
    pub image_index: u32,
    pub _padding: u32,
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
pub struct PushConstantBufferAddresses {
    pub model_info: vk::DeviceAddress,
    pub uniforms: vk::DeviceAddress,
    pub acceleration_structure: vk::DeviceAddress,
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
    pub fn new(transform: Mat4, model: &Model, device: &Device, hit_shader: HitShader) -> Self {
        fn merge_fields(lower: u32, upper: u8) -> u32 {
            ((upper as u32) << 24) | lower
        }

        let sbt_offset = hit_shader as u32;

        Self {
            transform: transpose_matrix_for_instance(transform),
            instance_custom_index_and_mask: merge_fields(model.id, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: merge_fields(
                sbt_offset,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_device_address: model.blas.buffer.device_address(device),
        }
    }
}

pub fn transpose_matrix_for_instance(matrix: Mat4) -> [Vec4; 3] {
    let rows = matrix.transposed().cols;
    [rows[0], rows[1], rows[2]]
}
