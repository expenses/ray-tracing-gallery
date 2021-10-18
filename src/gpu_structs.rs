use crate::util_structs::{Device, Model};
use crate::HitShader;
use ash::vk;
use ultraviolet::{Mat4, Vec4};

pub unsafe fn unsafe_bytes_of<T>(reference: &T) -> &[u8] {
    std::slice::from_raw_parts(reference as *const T as *const u8, std::mem::size_of::<T>())
}

pub unsafe fn unsafe_cast_slice<T>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(
        slice as *const [T] as *const u8,
        slice.len() * std::mem::size_of::<T>(),
    )
}

// A slightly easier to use `vk::AccelerationStructureInstanceKHR`.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct AccelerationStructureInstance {
    pub transform: [Vec4; 3],
    pub instance_custom_index_and_mask: vk::Packed24_8,
    pub instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8,
    pub acceleration_structure_device_address: vk::DeviceAddress,
}

impl AccelerationStructureInstance {
    pub fn new(
        transform: Mat4,
        model: &Model,
        device: &Device,
        hit_shader: HitShader,
        double_sided: bool,
    ) -> Self {
        let flags = if double_sided {
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE
        } else {
            vk::GeometryInstanceFlagsKHR::empty()
        };

        let sbt_offset = hit_shader as u32;

        Self {
            transform: transpose_matrix_for_instance(transform),
            instance_custom_index_and_mask: vk::Packed24_8::new(model.id, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                sbt_offset,
                flags.as_raw() as u8,
            ),
            acceleration_structure_device_address: model.blas.buffer.device_address(device),
        }
    }
}

pub fn transpose_matrix_for_instance(matrix: Mat4) -> [Vec4; 3] {
    let rows = matrix.transposed().cols;
    [rows[0], rows[1], rows[2]]
}
