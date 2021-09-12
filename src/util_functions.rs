use crate::SurfaceLoader;
use crate::Vertex;
use ash::vk;
use gpu_allocator::vulkan::AllocationCreateDesc;
use std::ffi::CStr;
use std::os::raw::c_char;
use ultraviolet::Vec3;

use crate::util_structs::{Allocator, Buffer, CStrList, Image, ModelBuffers};

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
) -> anyhow::Result<vk::PipelineShaderStageCreateInfo> {
    let spv = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&spv), None)
    }?;

    Ok(*vk::PipelineShaderStageCreateInfo::builder()
        .module(module)
        .stage(stage)
        .name(CStr::from_bytes_with_nul(b"main\0")?))
}

pub fn load_rgba_image_from_bytes(
    bytes: &[u8],
    name: &str,
    width: u32,
    height: u32,
    command_buffer: vk::CommandBuffer,
    allocator: &mut Allocator,
) -> anyhow::Result<(Image, Buffer)> {
    let staging_buffer = Buffer::new(
        bytes,
        &format!("{} staging buffer", name),
        vk::BufferUsageFlags::TRANSFER_SRC,
        allocator,
    )?;

    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };

    let image = unsafe {
        allocator.device.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
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
                .format(vk::Format::R8G8B8A8_SRGB)
                .subresource_range(*subresource_range),
            None,
        )
    }?;

    unsafe {
        allocator.device.cmd_pipeline_barrier(
            command_buffer,
            // See
            // https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#synchronization-access-types-supported
            // https://vulkan-tutorial.com/Texture_mapping/Images
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[*vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(*subresource_range)],
        );

        allocator.device.cmd_copy_buffer_to_image(
            command_buffer,
            staging_buffer.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[*vk::BufferImageCopy::builder()
                .buffer_row_length(width)
                .buffer_image_height(height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(extent)],
        );

        allocator.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[*vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .subresource_range(*subresource_range)],
        );
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

// Copied pretty much verbatim from:
// https://github.com/SaschaWillems/Vulkan/blob/eb11297312a164d00c60b06048100bac1d780bb4/base/VulkanTools.cpp#L119-L123
//
// This stuff is super annoying, I wish this image layout to access mask matching was
// in the api itself.
pub fn set_image_layout(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    subresource_range: vk::ImageSubresourceRange,
) {
    let mut image_memory_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(subresource_range);

    // Source layouts (old)
    // Source access mask controls actions that have to be finished on the old layout
    // before it will be transitioned to the new layout
    match old_layout {
        vk::ImageLayout::UNDEFINED => {
            // Image layout is undefined (or does not matter)
            // Only valid as initial layout
            // No flags required, listed only for completeness
            image_memory_barrier.src_access_mask = vk::AccessFlags::empty();
        }
        vk::ImageLayout::PREINITIALIZED => {
            // Image is preinitialized
            // Only valid as initial layout for linear images, preserves memory contents
            // Make sure host writes have been finished
            image_memory_barrier.src_access_mask = vk::AccessFlags::HOST_WRITE;
        }
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
            // Image is a color attachment
            // Make sure any writes to the color buffer have been finished
            image_memory_barrier.src_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        }
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            // Image is a depth/stencil attachment
            // Make sure any writes to the depth/stencil buffer have been finished
            image_memory_barrier.src_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
            // Image is a transfer source
            // Make sure any reads from the image have been finished
            image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        }
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
            // Image is a transfer destination
            // Make sure any writes to the image have been finished
            image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        }

        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
            // Image is read by a shader
            // Make sure any shader reads from the image have been finished
            image_memory_barrier.src_access_mask = vk::AccessFlags::SHADER_READ;
        }
        // Other source layouts aren't handled (yet)
        _ => {}
    }

    // Target layouts (new)
    // Destination access mask controls the dependency for the new image layout
    match new_layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
            // Image will be used as a transfer destination
            // Make sure any writes to the image have been finished
            image_memory_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        }

        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
            // Image will be used as a transfer source
            // Make sure any reads from the image have been finished
            image_memory_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;
        }

        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
            // Image will be used as a color attachment
            // Make sure any writes to the color buffer have been finished
            image_memory_barrier.dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        }

        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            // Image layout will be used as a depth/stencil attachment
            // Make sure any writes to depth/stencil buffer have been finished
            image_memory_barrier.dst_access_mask |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }

        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
            // Image will be read in a shader (sampler, input attachment)
            // Make sure any writes to the image have been finished
            if image_memory_barrier.src_access_mask == vk::AccessFlags::empty() {
                image_memory_barrier.src_access_mask =
                    vk::AccessFlags::HOST_WRITE | vk::AccessFlags::TRANSFER_WRITE;
            }
            image_memory_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
        }
        _ => {
            // Other source layouts aren't handled (yet)
        }
    }

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[*image_memory_barrier],
        )
    }
}

pub fn load_gltf(
    bytes: &[u8],
    name: &str,
    size_modifier: f32,
    allocator: &mut Allocator,
) -> anyhow::Result<ModelBuffers> {
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

            positions.zip(normals).for_each(|(position, normal)| {
                vertices.push(Vertex {
                    pos: Vec3::from(position) * size_modifier,
                    normal: normal.into(),
                });
            })
        }
    }

    ModelBuffers::new(&vertices, &indices, name, allocator)
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
