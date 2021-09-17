use crate::SurfaceLoader;
use crate::Vertex;
use ash::vk;
use gpu_allocator::vulkan::AllocationCreateDesc;
use std::ffi::CStr;
use std::os::raw::c_char;

use crate::util_structs::{Allocator, Buffer, CStrList, Image, ImageManager, ModelBuffers};

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

pub fn load_shader_module(
    bytes: &[u8],
    stage: vk::ShaderStageFlags,
    device: &ash::Device,
    entry_point: Option<&CStr>,
) -> anyhow::Result<vk::PipelineShaderStageCreateInfo> {
    let spv = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&spv), None)
    }?;

    let entry_point = match entry_point {
        Some(entry_point) => entry_point,
        None => CStr::from_bytes_with_nul(b"main\0")?,
    };

    Ok(*vk::PipelineShaderStageCreateInfo::builder()
        .module(module)
        .stage(stage)
        .name(entry_point))
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
            .map(|(image, staging_buffer)| (image_manager.push_image(image), Some(staging_buffer))),
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

pub fn shader_group_for_type(
    index: u32,
    ty: vk::ShaderGroupShaderKHR,
) -> vk::RayTracingShaderGroupCreateInfoKHR {
    let mut info = vk::RayTracingShaderGroupCreateInfoKHR {
        general_shader: vk::SHADER_UNUSED_KHR,
        closest_hit_shader: vk::SHADER_UNUSED_KHR,
        any_hit_shader: vk::SHADER_UNUSED_KHR,
        intersection_shader: vk::SHADER_UNUSED_KHR,
        ..Default::default()
    };

    match ty {
        vk::ShaderGroupShaderKHR::GENERAL => {
            info.ty = vk::RayTracingShaderGroupTypeKHR::GENERAL;
            info.general_shader = index;
        }
        vk::ShaderGroupShaderKHR::CLOSEST_HIT => {
            info.ty = vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP;
            info.closest_hit_shader = index;
        }
        vk::ShaderGroupShaderKHR::ANY_HIT => {
            info.ty = vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP;
            info.any_hit_shader = index;
        }
        vk::ShaderGroupShaderKHR::INTERSECTION => {
            info.ty = vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP;
            info.intersection_shader = index;
        }
        _ => {}
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
