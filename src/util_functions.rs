use crate::SurfaceLoader;
use crate::Vertex;
use ash::extensions::khr::AccelerationStructure as AccelerationStructureLoader;
use ash::vk;
use gpu_allocator::vulkan::AllocationCreateDesc;
use std::ffi::CStr;
use std::os::raw::c_char;

use crate::util_structs::{
    AccelerationStructure, Allocator, Buffer, CStrList, Image, ImageManager, ModelBuffers,
    ScratchBuffer,
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
                    // todo: srgb?
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

pub fn load_rgba_png_image_from_bytes(
    bytes: &[u8],
    name: &str,
    command_buffer: vk::CommandBuffer,
    allocator: &mut Allocator,
) -> anyhow::Result<(Image, Buffer)> {
    let decoded_image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?;

    let rgba_image = decoded_image.to_rgba8();

    let staging_buffer = Buffer::new(
        &rgba_image,
        &format!("{} staging buffer", name),
        vk::BufferUsageFlags::TRANSFER_SRC,
        allocator,
    )?;

    let extent = vk::Extent3D {
        width: rgba_image.width(),
        height: rgba_image.height(),
        depth: 1,
    };

    let image = unsafe {
        allocator.device.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
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
            .bind_image_memory(image, allocation.memory(), allocation.offset())
    }?;

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

    let view = unsafe {
        allocator.device.create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(*subresource_range),
            None,
        )
    }?;

    unsafe {
        cmd_pipeline_image_memory_barrier_explicit(&PipelineImageMemoryBarrierParams {
            device: &allocator.device,
            buffer: command_buffer,
            // We don't need to block on anything before this.
            src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
            // We're blocking a transfer
            dst_stage: vk::PipelineStageFlags::TRANSFER,
            image_memory_barriers: &[*vk::ImageMemoryBarrier::builder()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .subresource_range(*subresource_range)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
        });

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

        cmd_pipeline_image_memory_barrier_explicit(&PipelineImageMemoryBarrierParams {
            device: &allocator.device,
            buffer: command_buffer,
            // We're blocking on a transfer.
            src_stage: vk::PipelineStageFlags::TRANSFER,
            // We're blocking the use of the texture in ray tracing
            dst_stage: vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            image_memory_barriers: &[*vk::ImageMemoryBarrier::builder()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .subresource_range(*subresource_range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
        });
    }

    Ok((
        Image {
            image,
            view,
            allocation,
        },
        staging_buffer,
    ))
}

pub fn load_gltf(
    bytes: &[u8],
    name: &str,
    fallback_image_index: u32,
    allocator: &mut Allocator,
    image_manager: &mut ImageManager,
    command_buffer: vk::CommandBuffer,
) -> anyhow::Result<(ModelBuffers, Option<Buffer>)> {
    let gltf = gltf::Gltf::from_slice(bytes)?;

    let buffer_blob = gltf.blob.as_ref().unwrap();

    let mut indices = Vec::new();
    let mut vertices = Vec::new();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| {
                debug_assert_eq!(buffer.index(), 0);
                Some(buffer_blob)
            });

            let read_indices = match reader.read_indices().unwrap() {
                gltf::mesh::util::ReadIndices::U16(indices) => indices,
                gltf::mesh::util::ReadIndices::U32(_) => {
                    return Err(anyhow::anyhow!("U32 indices not supported"))
                }
                _ => unreachable!(),
            };

            let num_vertices = vertices.len() as u16;
            indices.extend(read_indices.map(|index| index + num_vertices));

            let positions = reader.read_positions().unwrap();
            let normals = reader.read_normals().unwrap();
            let uvs = reader.read_tex_coords(0).unwrap().into_f32();

            positions
                .zip(normals)
                .zip(uvs)
                .for_each(|((position, normal), uv)| {
                    vertices.push(Vertex {
                        pos: position.into(),
                        normal: normal.into(),
                        uv: uv.into(),
                    });
                })
        }
    }

    let mut image_index = || -> Option<anyhow::Result<(u32, Option<Buffer>)>> {
        let material = gltf.materials().next()?;

        let diffuse_texture = material
            .pbr_metallic_roughness()
            .base_color_texture()?
            .texture();

        let linear_filtering =
            diffuse_texture.sampler().mag_filter() == Some(gltf::texture::MagFilter::Linear);

        let image_view = match diffuse_texture.source().source() {
            gltf::image::Source::View { view, .. } => view,
            _ => return None,
        };

        let image_start = image_view.offset();
        let image_end = image_start + image_view.length();
        let image_bytes = &buffer_blob[image_start..image_end];

        Some(
            load_rgba_png_image_from_bytes(
                image_bytes,
                &format!("{} image", name),
                command_buffer,
                allocator,
            )
            .map(|(image, staging_buffer)| {
                (
                    image_manager.push_image(image, linear_filtering),
                    Some(staging_buffer),
                )
            }),
        )
    };

    let (image_index, staging_buffer) = match image_index() {
        Some(result) => result?,
        None => (fallback_image_index, None),
    };

    Ok((
        ModelBuffers::new(&vertices, &indices, image_index, name, allocator)?,
        staging_buffer,
    ))
}

fn shader_group_for_stage(
    index: u32,
    stage: vk::ShaderStageFlags,
) -> vk::RayTracingShaderGroupCreateInfoKHR {
    let mut info = vk::RayTracingShaderGroupCreateInfoKHR {
        general_shader: vk::SHADER_UNUSED_KHR,
        closest_hit_shader: vk::SHADER_UNUSED_KHR,
        any_hit_shader: vk::SHADER_UNUSED_KHR,
        intersection_shader: vk::SHADER_UNUSED_KHR,
        ..Default::default()
    };

    match stage {
        vk::ShaderStageFlags::CLOSEST_HIT_KHR => {
            info.ty = vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP;
            info.closest_hit_shader = index;
        }
        vk::ShaderStageFlags::ANY_HIT_KHR => {
            info.ty = vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP;
            info.any_hit_shader = index;
        }
        vk::ShaderStageFlags::INTERSECTION_KHR => {
            info.ty = vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP;
            info.intersection_shader = index;
        }
        _ => {
            info.ty = vk::RayTracingShaderGroupTypeKHR::GENERAL;
            info.general_shader = index;
        }
    }

    info
}

pub fn shader_groups_for_stages<const N: usize>(
    stages: &[vk::PipelineShaderStageCreateInfo; N],
) -> [vk::RayTracingShaderGroupCreateInfoKHR; N] {
    let mut groups = [vk::RayTracingShaderGroupCreateInfoKHR::default(); N];

    for (i, stage) in stages.iter().enumerate() {
        groups[i] = shader_group_for_stage(i as u32, stage.stage);
    }

    groups
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

pub struct PipelineImageMemoryBarrierParams<'a> {
    pub device: &'a ash::Device,
    pub buffer: vk::CommandBuffer,
    pub src_stage: vk::PipelineStageFlags,
    pub dst_stage: vk::PipelineStageFlags,
    pub image_memory_barriers: &'a [vk::ImageMemoryBarrier],
}

// `cmd_pipeline_barrier` is one of those cases where it's nice if each param is clear.
pub unsafe fn cmd_pipeline_image_memory_barrier_explicit(
    params: &PipelineImageMemoryBarrierParams,
) {
    params.device.cmd_pipeline_barrier(
        params.buffer,
        params.src_stage,
        params.dst_stage,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        params.image_memory_barriers,
    )
}

pub fn build_blas(
    model_buffers: &ModelBuffers,
    name: &str,
    as_loader: &AccelerationStructureLoader,
    scratch_buffer: &mut ScratchBuffer,
    allocator: &mut Allocator,
    command_buffer: vk::CommandBuffer,
) -> anyhow::Result<AccelerationStructure> {
    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: model_buffers.triangles_data,
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    let offset = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .primitive_count(model_buffers.primitive_count);

    let geometries = &[*geometry];

    let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(geometries);

    let build_sizes = unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_info,
            &[model_buffers.primitive_count],
        )
    };

    let scratch_buffer =
        scratch_buffer.ensure_size_of(build_sizes.build_scratch_size, allocator)?;

    let blas = AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        name,
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        as_loader,
        allocator,
        scratch_buffer,
        geometry_info,
        offset,
        command_buffer,
    )?;

    Ok(blas)
}

pub fn build_tlas(
    instances: &Buffer,
    num_instances: u32,
    as_loader: &AccelerationStructureLoader,
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
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

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
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_info,
            &[num_instances],
        )
    };

    let scratch_buffer =
        scratch_buffer.ensure_size_of(build_sizes.build_scratch_size, allocator)?;

    AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        "tlas",
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        as_loader,
        allocator,
        scratch_buffer,
        geometry_info,
        offset,
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
