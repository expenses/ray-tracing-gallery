use ash::extensions::khr::AccelerationStructure as AccelerationStructureLoader;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocatorCreateDesc};
use std::ffi::CStr;
use std::os::raw::c_char;

use crate::util_functions::set_image_layout;

// A list of C strings and their associated pointers
pub struct CStrList<'a> {
    pub list: Vec<&'a CStr>,
    pub pointers: Vec<*const c_char>,
}

impl<'a> CStrList<'a> {
    pub fn new(list: Vec<&'a CStr>) -> Self {
        let pointers = list.iter().map(|cstr| cstr.as_ptr()).collect();

        Self { list, pointers }
    }
}

pub struct Allocator {
    pub inner: gpu_allocator::vulkan::Allocator,
    pub device: ash::Device,
    queue_family: u32,
}

impl Allocator {
    pub fn new(
        instance: ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
    ) -> anyhow::Result<Self> {
        Ok(Allocator {
            inner: gpu_allocator::vulkan::Allocator::new(&AllocatorCreateDesc {
                instance,
                device: device.clone(),
                physical_device,
                debug_settings: gpu_allocator::AllocatorDebugSettings {
                    log_memory_information: false,
                    log_leaks_on_shutdown: true,
                    store_stack_traces: false,
                    log_allocations: true,
                    log_frees: true,
                    log_stack_traces: false,
                },
                // Needed for getting buffer device addresses
                buffer_device_address: true,
            })?,
            queue_family,
            device,
        })
    }
}

pub struct AccelerationStructure {
    pub buffer: Buffer,
    pub acceleration_structure: vk::AccelerationStructureKHR,
}

impl AccelerationStructure {
    pub fn new(
        size: vk::DeviceSize,
        name: &str,
        ty: vk::AccelerationStructureTypeKHR,
        loader: &AccelerationStructureLoader,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        log::info!("Creating a {} of {} bytes", name, size);

        let buffer = Buffer::new_of_size(
            size,
            name,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            allocator,
        )?;

        let acceleration_structure = unsafe {
            loader.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::builder()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .size(size)
                    .ty(ty),
                None,
            )
        }?;

        Ok(Self {
            buffer,
            acceleration_structure,
        })
    }
}

pub struct ScratchBuffer {
    pub inner: Buffer,
}

impl ScratchBuffer {
    pub fn new(size: vk::DeviceSize, allocator: &mut Allocator) -> anyhow::Result<Self> {
        log::info!("Creating a scratch buffer of {} bytes", size);

        Ok(Self {
            inner: Buffer::new_of_size(
                size,
                "scratch buffer",
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                allocator,
            )?,
        })
    }

    pub fn ensure_size_of(
        &mut self,
        size: vk::DeviceSize,
        allocator: &mut Allocator,
    ) -> anyhow::Result<()> {
        if self.inner.allocation.size() < size {
            let old = std::mem::replace(self, Self::new(size, allocator)?);

            old.inner.cleanup(allocator)?;
        }

        Ok(())
    }
}

pub struct Buffer {
    pub allocation: Allocation,
    pub buffer: vk::Buffer,
}

impl Buffer {
    pub fn new_of_size(
        size: vk::DeviceSize,
        name: &str,
        usage: vk::BufferUsageFlags,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        let buffer = unsafe {
            allocator.device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(size)
                    .usage(usage)
                    .queue_family_indices(&[allocator.queue_family]),
                None,
            )
        }?;

        let requirements = unsafe { allocator.device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator.inner.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true,
        })?;

        unsafe {
            allocator
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }?;

        Ok(Self { buffer, allocation })
    }

    pub fn new(
        bytes: &[u8],
        name: &str,
        usage: vk::BufferUsageFlags,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        let buffer_size = bytes.len() as vk::DeviceSize;

        let buffer = unsafe {
            allocator.device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(buffer_size)
                    .usage(usage)
                    .queue_family_indices(&[allocator.queue_family]),
                None,
            )
        }?;

        let requirements = unsafe { allocator.device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator.inner.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
        })?;

        Self::from_parts(allocation, buffer, bytes, allocator)
    }

    pub fn new_with_custom_alignment(
        bytes: &[u8],
        name: &str,
        usage: vk::BufferUsageFlags,
        alignment: vk::DeviceSize,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        let buffer_size = bytes.len() as vk::DeviceSize;

        let buffer = unsafe {
            allocator.device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(buffer_size)
                    .usage(usage)
                    .queue_family_indices(&[allocator.queue_family]),
                None,
            )
        }?;

        let mut requirements = unsafe { allocator.device.get_buffer_memory_requirements(buffer) };

        requirements.alignment = alignment;

        let allocation = allocator.inner.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
        })?;

        Self::from_parts(allocation, buffer, bytes, allocator)
    }

    fn from_parts(
        mut allocation: Allocation,
        buffer: vk::Buffer,
        bytes: &[u8],
        allocator: &Allocator,
    ) -> anyhow::Result<Self> {
        let slice = allocation.mapped_slice_mut().unwrap();

        slice[..bytes.len()].copy_from_slice(bytes);

        unsafe {
            allocator
                .device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }?;

        Ok(Self { buffer, allocation })
    }

    pub fn device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        unsafe {
            device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder().buffer(self.buffer),
            )
        }
    }

    pub fn cleanup(self, allocator: &mut Allocator) -> anyhow::Result<()> {
        allocator.inner.free(self.allocation)?;

        unsafe { allocator.device.destroy_buffer(self.buffer, None) };

        Ok(())
    }
}

pub struct Image {
    pub image: vk::Image,
    pub allocation: Allocation,
    pub view: vk::ImageView,
}

impl Image {
    pub fn new_depth_buffer(
        width: u32,
        height: u32,
        name: &str,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        let image = unsafe {
            allocator.device.create_image(
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
                None,
            )
        }?;

        let requirements = unsafe { allocator.device.get_image_memory_requirements(image) };

        let allocation = allocator.inner.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: false,
        })?;

        unsafe {
            allocator
                .device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }?;

        let view = unsafe {
            allocator.device.create_image_view(
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

    pub fn new_storage_image(
        width: u32,
        height: u32,
        format: vk::Format,
        command_buffer: vk::CommandBuffer,
        queue: vk::Queue,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        let image = unsafe {
            allocator.device.create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE),
                None,
            )
        }?;

        let requirements = unsafe { allocator.device.get_image_memory_requirements(image) };

        let allocation = allocator.inner.allocate(&AllocationCreateDesc {
            name: "storage image",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
        })?;

        unsafe {
            allocator
                .device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }?;

        let view = unsafe {
            allocator.device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1)
                            .build(),
                    ),
                None,
            )
        }?;

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        unsafe {
            allocator.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            set_image_layout(
                &allocator.device,
                command_buffer,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                *subresource_range,
            );

            allocator.device.end_command_buffer(command_buffer)?;

            let fence = allocator
                .device
                .create_fence(&vk::FenceCreateInfo::builder(), None)?;

            allocator.device.queue_submit(
                queue,
                &[*vk::SubmitInfo::builder().command_buffers(&[command_buffer])],
                fence,
            )?;

            allocator.device.wait_for_fences(&[fence], true, u64::MAX)?;
            allocator.device.destroy_fence(fence, None);
        }

        Ok(Self {
            image,
            view,
            allocation,
        })
    }

    pub fn cleanup(self, allocator: &mut Allocator) -> anyhow::Result<()> {
        allocator.inner.free(self.allocation)?;

        unsafe { allocator.device.destroy_image_view(self.view, None) };

        unsafe { allocator.device.destroy_image(self.image, None) };

        Ok(())
    }
}
