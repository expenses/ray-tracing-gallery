use crate::SurfaceLoader;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use std::ffi::CStr;
use std::os::raw::c_char;

pub fn select_physical_device(
    instance: &ash::Instance,
    required_extensions: &CStrList,
    surface_loader: &SurfaceLoader,
    surface: vk::SurfaceKHR,
) -> anyhow::Result<Option<(vk::PhysicalDevice, u32, vk::SurfaceFormatKHR)>> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }?;

    println!(
        "\nFound {} device{}:",
        physical_devices.len(),
        if physical_devices.len() == 1 { "" } else { "s" }
    );

    let selection = physical_devices
        .into_iter()
        .filter_map(|physical_device| unsafe {
            let properties = instance.get_physical_device_properties(physical_device);

            println!(
                "\nChecking Device: {:?}",
                cstr_from_array(&properties.device_name)
            );

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

            println!(
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
                    surface_format.format == vk::Format::B8G8R8A8_SRGB
                        && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .or_else(|| surface_formats.get(0));

            println!(
                "  Checking for a surface format: {}",
                tick(surface_format.is_some())
            );

            let surface_format = match surface_format {
                Some(surface_format) => *surface_format,
                None => return None,
            };

            println!("  Checking for required extensions:");

            let supported_device_extensions = instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap();

            let mut has_required_extensions = true;

            for required_extension in &required_extensions.list {
                let device_has_extension = supported_device_extensions.iter().any(|extension| {
                    &cstr_from_array(&extension.extension_name) == required_extension
                });

                println!(
                    "    * {:?}: {}",
                    required_extension,
                    tick(device_has_extension)
                );

                has_required_extensions &= device_has_extension;
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

    println!("");

    Ok(match selection {
        Some((physical_device, queue_family, surface_format, properties)) => {
            unsafe {
                println!(
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

unsafe fn cstr_from_array<'a>(array: &'a [c_char]) -> &'a CStr {
    CStr::from_ptr(array.as_ptr())
}

// A list of C strings and their associated pointers
pub struct CStrList<'a> {
    list: Vec<&'a CStr>,
    pub pointers: Vec<*const c_char>,
}

impl<'a> CStrList<'a> {
    pub fn new(list: Vec<&'a CStr>) -> Self {
        let pointers = list.iter().map(|cstr| cstr.as_ptr()).collect();

        Self { list, pointers }
    }
}

pub fn load_shader_module(bytes: &[u8], device: &ash::Device) -> anyhow::Result<vk::ShaderModule> {
    let spv = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&spv), None) }
        .map_err(|err| err.into())
}

pub struct Buffer {
    pub allocation: Allocation,
    pub buffer: vk::Buffer,
}

impl Buffer {
    pub fn new(
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
        allocator: &mut Allocator,
        queue_family: u32,
        device: &ash::Device,
    ) -> anyhow::Result<Self> {
        let buffer_size = bytes.len() as u64;

        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(buffer_size)
                    .usage(usage)
                    .queue_family_indices(&[queue_family]),
                None,
            )
        }?;

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let mut allocation = allocator.allocate(&AllocationCreateDesc {
            name: "",
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
        })?;

        let memory = unsafe { allocation.memory() };

        let slice = allocation.mapped_slice_mut().unwrap();

        slice.copy_from_slice(bytes);

        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;

        Ok(Self { buffer, allocation })
    }

    pub fn cleanup(self, allocator: &mut Allocator, device: &ash::Device) -> anyhow::Result<()> {
        allocator.free(self.allocation)?;

        unsafe { device.destroy_buffer(self.buffer, None) };

        Ok(())
    }
}

/*
pub struct Image {
    image: vk::Image,
    allocation: vk_mem::Allocation,
    pub view: vk::ImageView,
}

impl Image {
    pub fn new_depth_buffer(
        width: u32,
        height: u32,
        device: &Device,
        allocator: &vk_mem::Allocator,
    ) -> anyhow::Result<Self> {
        let (image, allocation, _allocation_info) = allocator
            .create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::GpuOnly,
                    flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
                    ..Default::default()
                },
            )
            .map_err(|err| anyhow::anyhow!("{:?}", err))?;

        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::D32_SFLOAT)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .level_count(1)
                            .layer_count(1)
                            .build(),
                    ),
                None,
            )
        }?;

        Ok(Self {
            image,
            allocation,
            view,
        })
    }

    pub fn cleanup(&self, device: &Device, allocator: &vk_mem::Allocator) -> anyhow::Result<()> {
        unsafe {
            device.destroy_image_view(self.view, None);
        }

        allocator
            .destroy_image(self.image, &self.allocation)
            .map_err(|err| err.into())
    }
}

pub fn load_rgba_image_from_bytes(
    bytes: &[u8],
    width: u32,
    height: u32,
    command_buffer: vk::CommandBuffer,
    device: &Device,
    allocator: &vk_mem::Allocator,
    queue_family: u32,
) -> anyhow::Result<(Image, Buffer)> {
    let staging_buffer = Buffer::new(
        bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
        &allocator,
        queue_family,
    )?;

    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };

    let (image, allocation, _allocation_info) = allocator.create_image(
        &vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
        &vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        },
    )?;

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .subresource_range(*subresource_range),
            None,
        )
    }?;

    unsafe {
        device.cmd_pipeline_barrier(
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

        device.cmd_copy_buffer_to_image(
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

        device.cmd_pipeline_barrier(
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
*/
