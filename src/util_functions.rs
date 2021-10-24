use ash::vk;
use ash_opinionated_abstractions::{create_image_from_bytes, InitResources};
use image::GenericImageView;
use std::ffi::CStr;

use crate::util_structs::{AccelerationStructure, Allocator, Buffer, Image, ScratchBuffer};

pub fn load_shader_module_as_stage(
    bytes: &[u8],
    stage: vk::ShaderStageFlags,
    device: &ash::Device,
) -> anyhow::Result<vk::PipelineShaderStageCreateInfo> {
    let builder = ash_opinionated_abstractions::load_shader_module_as_stage(
        bytes,
        stage,
        device,
        CStr::from_bytes_with_nul(b"main\0")?,
    )?;

    Ok(*builder)
}

pub struct ShaderEntryPoint {
    name: &'static str,
    c_string: std::ffi::CString,
}

impl ShaderEntryPoint {
    pub fn new(name: &'static str) -> anyhow::Result<Self> {
        Ok(Self {
            name,
            c_string: std::ffi::CString::new(name)?,
        })
    }
}

pub fn load_rust_shader_module_as_stage<'a>(
    entry_point: &'a ShaderEntryPoint,
    stage: vk::ShaderStageFlags,
    device: &ash::Device,
) -> anyhow::Result<vk::PipelineShaderStageCreateInfoBuilder<'a>> {
    let filename = &format!("shaders/{}.spv", entry_point.name);

    ash_opinionated_abstractions::load_shader_module_as_stage(
        &std::fs::read(filename)?,
        stage,
        device,
        &entry_point.c_string,
    )
}

pub trait ImagePixelFormat: bytemuck::Pod {
    const FORMAT: vk::Format;
}

impl ImagePixelFormat for f32 {
    const FORMAT: vk::Format = vk::Format::R32_SFLOAT;
}

impl ImagePixelFormat for [f32; 4] {
    const FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
}

pub fn create_single_colour_image<P: ImagePixelFormat>(
    pixel: P,
    name: &str,
    command_buffer: vk::CommandBuffer,
    allocator: &mut Allocator,
    buffers_to_cleanup: &mut Vec<Buffer>,
) -> anyhow::Result<Image> {
    let (image, staging_buffer) = create_image_from_bytes(
        bytemuck::bytes_of(&pixel),
        vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
        vk::ImageViewType::TYPE_2D,
        P::FORMAT,
        name,
        &mut InitResources {
            device: &allocator.device,
            allocator: &mut allocator.inner,
            command_buffer,
            debug_utils_loader: Some(&allocator.device.debug_utils_loader),
        },
        &[vk_sync::AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer],
        vk_sync::ImageLayout::Optimal,
    )?;

    buffers_to_cleanup.push(staging_buffer.into());

    Ok(image.into())
}

pub fn load_png_image_from_bytes(
    bytes: &[u8],
    name: &str,
    format: vk::Format,
    command_buffer: vk::CommandBuffer,
    allocator: &mut Allocator,
    buffers_to_cleanup: &mut Vec<Buffer>,
) -> anyhow::Result<Image> {
    let decoded_image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?;

    let rgba_image = decoded_image.to_rgba8();

    let (image, staging_buffer) = create_image_from_bytes(
        &*rgba_image,
        vk::Extent3D {
            width: decoded_image.width(),
            height: decoded_image.height(),
            depth: 1,
        },
        vk::ImageViewType::TYPE_2D,
        format,
        name,
        &mut InitResources {
            device: &allocator.device,
            allocator: &mut allocator.inner,
            command_buffer,
            debug_utils_loader: Some(&allocator.device.debug_utils_loader),
        },
        &[vk_sync::AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer],
        vk_sync::ImageLayout::Optimal,
    )?;

    buffers_to_cleanup.push(staging_buffer.into());

    Ok(image.into())
}

pub enum ShaderGroup {
    TriangleHitGroup {
        closest_hit_shader: u32,
        any_hit_shader: u32,
    },
    General(u32),
}

pub fn info_from_group(group: ShaderGroup) -> vk::RayTracingShaderGroupCreateInfoKHR {
    let mut info = vk::RayTracingShaderGroupCreateInfoKHR {
        general_shader: vk::SHADER_UNUSED_KHR,
        closest_hit_shader: vk::SHADER_UNUSED_KHR,
        any_hit_shader: vk::SHADER_UNUSED_KHR,
        intersection_shader: vk::SHADER_UNUSED_KHR,
        ..Default::default()
    };

    match group {
        ShaderGroup::TriangleHitGroup {
            closest_hit_shader,
            any_hit_shader,
        } => {
            info.ty = vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP;
            info.closest_hit_shader = closest_hit_shader;
            info.any_hit_shader = any_hit_shader;
        }
        ShaderGroup::General(general_shader) => {
            info.ty = vk::RayTracingShaderGroupTypeKHR::GENERAL;
            info.general_shader = general_shader;
        }
    }

    info
}

pub fn sbt_aligned_size(props: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR) -> u32 {
    // Copied from:
    // https://github.com/SaschaWillems/Vulkan/blob/eb11297312a164d00c60b06048100bac1d780bb4/base/VulkanTools.cpp#L383
    fn aligned_size(value: u32, alignment: u32) -> u32 {
        (value + alignment - 1) & !(alignment - 1)
    }

    aligned_size(
        props.shader_group_handle_size,
        props.shader_group_handle_alignment,
    )
}

pub fn build_tlas(
    instances: &Buffer,
    num_instances: u32,
    allocator: &mut Allocator,
    scratch_buffer: &mut ScratchBuffer,
    command_buffer: vk::CommandBuffer,
) -> anyhow::Result<AccelerationStructure> {
    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::builder().data(
        vk::DeviceOrHostAddressConstKHR {
            device_address: instances.device_address(&allocator.device),
        },
    );

    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: *instances,
        });

    let offset =
        vk::AccelerationStructureBuildRangeInfoKHR::builder().primitive_count(num_instances);

    let geometries = &[*geometry];

    let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
        )
        .geometries(geometries);

    let build_sizes = unsafe {
        allocator
            .device
            .as_loader
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &geometry_info,
                &[num_instances],
            )
    };

    let scratch_buffer =
        scratch_buffer.ensure_size_of(build_sizes.build_scratch_size, command_buffer, allocator)?;

    AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        "tlas",
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        allocator,
        scratch_buffer,
        geometry_info,
        &[*offset],
        command_buffer,
    )
}
