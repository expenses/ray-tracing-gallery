use crate::SurfaceLoader;
use ash::vk;
use gpu_allocator::vulkan::AllocationCreateDesc;
use image::GenericImageView;
use std::ffi::CStr;
use std::os::raw::c_char;

use crate::util_structs::{
    AccelerationStructure, Allocator, Buffer, CStrList, Image, ScratchBuffer,
};

pub fn select_physical_device(
    instance: &ash::Instance,
    required_extensions: &CStrList,
    surface_loader: &SurfaceLoader,
    surface: vk::SurfaceKHR,
) -> anyhow::Result<Option<(vk::PhysicalDevice, u32, vk::SurfaceFormatKHR)>> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }?;

    log::info!(
        "Found {} device{}",
        physical_devices.len(),
        if physical_devices.len() == 1 { "" } else { "s" }
    );

    let selection = physical_devices
        .into_iter()
        .filter_map(|physical_device| unsafe {
            let properties = instance.get_physical_device_properties(physical_device);

            log::info!("");
            log::info!(
                "Checking Device: {:?}",
                cstr_from_array(&properties.device_name)
            );

            log::debug!("Api version: {}", properties.api_version);

            let queue_family = instance
                .get_physical_device_queue_family_properties(physical_device)
                .into_iter()
                .enumerate()
                .position(|(i, queue_family_properties)| {
                    queue_family_properties
                        .queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
                        && surface_loader
                            .get_physical_device_surface_support(physical_device, i as u32, surface)
                            .unwrap()
                })
                .map(|queue_family| queue_family as u32);

            log::info!(
                "  Checking for a graphics queue family: {}",
                tick(queue_family.is_some())
            );

            let queue_family = match queue_family {
                Some(queue_family) => queue_family,
                None => return None,
            };

            let surface_formats = surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap();

            let surface_format = surface_formats
                .iter()
                .find(|surface_format| {
                    surface_format.format == vk::Format::B8G8R8A8_UNORM
                        && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .or_else(|| surface_formats.get(0));

            log::info!(
                "  Checking for an appropriate surface format: {}",
                tick(surface_format.is_some())
            );

            let surface_format = match surface_format {
                Some(surface_format) => *surface_format,
                None => return None,
            };

            log::info!("  Checking for required extensions:");

            let supported_device_extensions = instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap();

            let mut has_required_extensions = true;

            for required_extension in &required_extensions.list {
                let device_has_extension = supported_device_extensions.iter().any(|extension| {
                    &cstr_from_array(&extension.extension_name) == required_extension
                });

                log::info!(
                    "    * {:?}: {}",
                    required_extension,
                    tick(device_has_extension)
                );

                has_required_extensions &= device_has_extension;
            }

            if log::log_enabled!(log::Level::Debug) {
                log::debug!("  Supported extensions:");
                supported_device_extensions.iter().for_each(|extension| {
                    log::debug!("    * {:?}", &cstr_from_array(&extension.extension_name));
                });
            }

            if !has_required_extensions {
                return None;
            }

            Some((physical_device, queue_family, surface_format, properties))
        })
        .max_by_key(|(.., properties)| match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 2,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 0,
        });

    log::info!("");

    Ok(match selection {
        Some((physical_device, queue_family, surface_format, properties)) => {
            unsafe {
                log::info!(
                    "Using device {:?}",
                    cstr_from_array(&properties.device_name)
                );
            }

            Some((physical_device, queue_family, surface_format))
        }
        None => None,
    })
}

fn tick(supported: bool) -> &'static str {
    if supported {
        "✔️"
    } else {
        "❌"
    }
}

unsafe fn cstr_from_array(array: &[c_char]) -> &CStr {
    CStr::from_ptr(array.as_ptr())
}

pub fn load_shader_module(bytes: &[u8], device: &ash::Device) -> anyhow::Result<vk::ShaderModule> {
    let spv = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    Ok(unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&spv), None)
    }?)
}

pub fn load_shader_module_as_stage(
    bytes: &[u8],
    stage: vk::ShaderStageFlags,
    device: &ash::Device,
) -> anyhow::Result<vk::PipelineShaderStageCreateInfo> {
    let module = load_shader_module(bytes, device)?;

    Ok(*vk::PipelineShaderStageCreateInfo::builder()
        .module(module)
        .stage(stage)
        .name(CStr::from_bytes_with_nul(b"main\0")?))
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

    let module = load_shader_module(&std::fs::read(filename)?, device)?;

    Ok(vk::PipelineShaderStageCreateInfo::builder()
        .module(module)
        .stage(stage)
        .name(&entry_point.c_string))
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
    create_image_from_bytes(
        bytemuck::bytes_of(&pixel),
        vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        },
        vk::ImageViewType::TYPE_2D,
        P::FORMAT,
        name,
        command_buffer,
        allocator,
        buffers_to_cleanup,
    )
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

    create_image_from_bytes(
        &*rgba_image,
        vk::Extent3D {
            width: decoded_image.width(),
            height: decoded_image.height(),
            depth: 1,
        },
        vk::ImageViewType::TYPE_2D,
        format,
        name,
        command_buffer,
        allocator,
        buffers_to_cleanup,
    )
}

pub fn create_image_from_bytes(
    bytes: &[u8],
    extent: vk::Extent3D,
    view_ty: vk::ImageViewType,
    format: vk::Format,
    name: &str,
    command_buffer: vk::CommandBuffer,
    allocator: &mut Allocator,
    buffers_to_cleanup: &mut Vec<Buffer>,
) -> anyhow::Result<Image> {
    fn ty_from_view_ty(ty: vk::ImageViewType) -> vk::ImageType {
        match ty {
            vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_1D_ARRAY => vk::ImageType::TYPE_1D,
            vk::ImageViewType::TYPE_2D
            | vk::ImageViewType::TYPE_2D_ARRAY
            | vk::ImageViewType::CUBE
            | vk::ImageViewType::CUBE_ARRAY => vk::ImageType::TYPE_2D,
            vk::ImageViewType::TYPE_3D => vk::ImageType::TYPE_3D,
            _ => vk::ImageType::default(),
        }
    }

    let staging_buffer = Buffer::new(
        bytes,
        &format!("{} staging buffer", name),
        vk::BufferUsageFlags::TRANSFER_SRC,
        allocator,
    )?;

    let image = unsafe {
        allocator.device.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(ty_from_view_ty(view_ty))
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
            None,
        )
    }?;

    let requirements = unsafe { allocator.device.get_image_memory_requirements(image) };

    let allocation = allocator.inner.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: gpu_allocator::MemoryLocation::GpuOnly,
        linear: false,
    })?;

    unsafe {
        allocator
            .device
            .bind_image_memory(image, allocation.memory(), allocation.offset())?;

        allocator.device.set_object_name(image, name)?;
    }

    let subresource_range = *vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

    let view = unsafe {
        allocator.device.create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(view_ty)
                .format(format)
                .subresource_range(subresource_range),
            None,
        )
    }?;

    vk_sync::cmd::pipeline_barrier(
        &allocator.device,
        command_buffer,
        None,
        &[],
        &[vk_sync::ImageBarrier {
            previous_accesses: &[vk_sync::AccessType::Nothing],
            next_accesses: &[vk_sync::AccessType::TransferWrite],
            next_layout: vk_sync::ImageLayout::Optimal,
            image,
            range: subresource_range,
            discard_contents: true,
            ..Default::default()
        }],
    );

    unsafe {
        allocator.device.cmd_copy_buffer_to_image(
            command_buffer,
            staging_buffer.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[*vk::BufferImageCopy::builder()
                .buffer_row_length(extent.width)
                .buffer_image_height(extent.height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(extent)],
        );
    }

    vk_sync::cmd::pipeline_barrier(
        &allocator.device,
        command_buffer,
        None,
        &[],
        &[vk_sync::ImageBarrier {
            previous_accesses: &[vk_sync::AccessType::TransferWrite],
            next_accesses: &[
                vk_sync::AccessType::RayTracingShaderReadSampledImageOrUniformTexelBuffer,
            ],
            next_layout: vk_sync::ImageLayout::Optimal,
            image,
            range: subresource_range,
            discard_contents: true,
            ..Default::default()
        }],
    );

    buffers_to_cleanup.push(staging_buffer);

    Ok(Image {
        image,
        view,
        allocation,
    })
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

pub unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let filter_out = (message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        && message_type == vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
        || (message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            && message_type == vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
        || (message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            && message_type == vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION);

    if filter_out {
        return vk::FALSE;
    }

    let level = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
        _ => log::Level::Info,
    };

    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let ty = format!("{:?}", message_type).to_lowercase();
    log::log!(level, "[Debug Msg][{}] {:?}", ty, message);
    vk::FALSE
}
