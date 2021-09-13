use ash::extensions::khr::{
    AccelerationStructure as AccelerationStructureLoader, Swapchain as SwapchainLoader,
};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocatorCreateDesc};
use std::ffi::CStr;
use std::os::raw::c_char;

use crate::util_functions::{sbt_aligned_size, set_image_layout};

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
                    log_allocations: false,
                    log_frees: false,
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
        device: &ash::Device,
        allocator: &mut Allocator,
        scratch_buffer: &Buffer,
        mut geometry_info: vk::AccelerationStructureBuildGeometryInfoKHRBuilder,
        offset: vk::AccelerationStructureBuildRangeInfoKHRBuilder,
        command_buffer_and_queue: &CommandBufferAndQueue,
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

        geometry_info = geometry_info
            .dst_acceleration_structure(acceleration_structure)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.device_address(device),
            });

        command_buffer_and_queue.begin(device)?;

        unsafe {
            loader.cmd_build_acceleration_structures(
                command_buffer_and_queue.buffer,
                &[*geometry_info],
                &[&[*offset]],
            );
        }

        command_buffer_and_queue.finish_block_and_reset(device)?;

        Ok(Self {
            buffer,
            acceleration_structure,
        })
    }
}

enum ScratchBufferInner {
    Unallocated,
    Allocated(Buffer),
}

pub struct ScratchBuffer {
    inner: ScratchBufferInner,
}

impl ScratchBuffer {
    pub fn new() -> Self {
        Self {
            inner: ScratchBufferInner::Unallocated,
        }
    }

    pub fn ensure_size_of(
        &mut self,
        size: vk::DeviceSize,
        allocator: &mut Allocator,
    ) -> anyhow::Result<&Buffer> {
        match &mut self.inner {
            ScratchBufferInner::Unallocated => {
                log::info!("Creating a scratch buffer of {} bytes", size);

                self.inner = ScratchBufferInner::Allocated(Buffer::new_of_size(
                    size,
                    "scratch buffer",
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                    allocator,
                )?);
            }
            ScratchBufferInner::Allocated(buffer) => {
                if buffer.allocation.size() < size {
                    buffer.cleanup(allocator)?;
                    log::info!(
                        "Resizing scratch buffer from {} bytes to {} bytes",
                        buffer.allocation.size(),
                        size
                    );

                    *buffer = Buffer::new_of_size(
                        size,
                        "scratch buffer",
                        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                        allocator,
                    )?;
                }
            }
        }

        match &self.inner {
            ScratchBufferInner::Allocated(buffer) => Ok(buffer),
            _ => unreachable!(),
        }
    }

    pub fn cleanup_and_drop(self, allocator: &mut Allocator) -> anyhow::Result<()> {
        if let ScratchBufferInner::Allocated(buffer) = self.inner {
            buffer.cleanup_and_drop(allocator)?;
        }

        Ok(())
    }
}

pub struct ShaderBindingTable {
    pub buffer: Buffer,
    pub address_region: vk::StridedDeviceAddressRegionKHR,
}

impl ShaderBindingTable {
    pub fn new(
        group_handles: &[u8],
        name: &str,
        props: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
        offset: u32,
        num_shaders: u64,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        let handle_size_aligned = sbt_aligned_size(props);

        let offset = (handle_size_aligned * offset) as usize;
        let size = handle_size_aligned as usize * num_shaders as usize;

        let slice = &group_handles[offset..offset + size];

        let buffer = Buffer::new_with_custom_alignment(
            slice,
            name,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            props.shader_group_base_alignment as u64,
            allocator,
        )?;

        let address_region = vk::StridedDeviceAddressRegionKHR::builder()
            .device_address(buffer.device_address(device))
            .stride(handle_size_aligned as u64)
            .size(handle_size_aligned as u64 * num_shaders);

        Ok(Self {
            buffer,
            address_region: *address_region,
        })
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

    pub fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()> {
        allocator.inner.free(self.allocation.clone())?;

        unsafe { allocator.device.destroy_buffer(self.buffer, None) };

        Ok(())
    }

    // Prefer using this when practical.
    pub fn cleanup_and_drop(self, allocator: &mut Allocator) -> anyhow::Result<()> {
        self.cleanup(allocator)?;
        drop(self);
        Ok(())
    }
}

pub struct Image {
    pub image: vk::Image,
    pub allocation: Allocation,
    pub view: vk::ImageView,
}

impl Image {
    pub fn _new_depth_buffer(
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

    pub fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()> {
        allocator.inner.free(self.allocation.clone())?;

        unsafe { allocator.device.destroy_image_view(self.view, None) };

        unsafe { allocator.device.destroy_image(self.image, None) };

        Ok(())
    }
}

pub struct ModelBuffers {
    pub vertices: Buffer,
    pub indices: Buffer,
    pub triangles_data: vk::AccelerationStructureGeometryTrianglesDataKHR,
    pub primitive_count: u32,
    pub image_index: u32,
}

impl ModelBuffers {
    pub fn new<T: bytemuck::Pod>(
        vertices: &[T],
        indices: &[u16],
        image_index: u32,
        name: &str,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        let num_vertices = vertices.len() as u32;
        let num_indices = indices.len() as u32;
        let primitive_count = num_indices / 3;
        let stride = std::mem::size_of::<T>() as u64;

        let vertices = Buffer::new(
            bytemuck::cast_slice(vertices),
            &format!("{} vertices", name),
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            allocator,
        )?;

        let indices = Buffer::new(
            bytemuck::cast_slice(indices),
            &format!("{} indices", name),
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            allocator,
        )?;

        let triangles_data = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertices.device_address(&allocator.device),
            })
            .vertex_stride(stride)
            .max_vertex(num_vertices)
            .index_type(vk::IndexType::UINT16)
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: indices.device_address(&allocator.device),
            });

        Ok(Self {
            vertices,
            indices,
            triangles_data: *triangles_data,
            primitive_count,
            image_index,
        })
    }

    pub fn model_info(&self, device: &ash::Device) -> crate::ModelInfo {
        crate::ModelInfo {
            vertex_buffer_address: self.vertices.device_address(device),
            index_buffer_address: self.indices.device_address(device),
            image_index: self.image_index,
            _padding: 0,
        }
    }

    pub fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()> {
        self.vertices.cleanup(allocator)?;
        self.indices.cleanup(allocator)?;
        Ok(())
    }
}

pub struct Syncronisation {
    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,
    pub draw_commands_fence: vk::Fence,
}

impl Syncronisation {
    pub fn new(device: &ash::Device) -> anyhow::Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        Ok(Self {
            present_complete_semaphore: unsafe { device.create_semaphore(&semaphore_info, None) }?,
            rendering_complete_semaphore: unsafe {
                device.create_semaphore(&semaphore_info, None)
            }?,
            draw_commands_fence: unsafe { device.create_fence(&fence_info, None) }?,
        })
    }

    unsafe fn _cleanup(&self, device: &ash::Device) {
        device.destroy_semaphore(self.present_complete_semaphore, None);
        device.destroy_semaphore(self.rendering_complete_semaphore, None);
        device.destroy_fence(self.draw_commands_fence, None)
    }
}

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
}

impl Swapchain {
    pub fn new(
        swapchain_loader: &SwapchainLoader,
        info: vk::SwapchainCreateInfoKHR,
    ) -> anyhow::Result<Self> {
        let swapchain = unsafe { swapchain_loader.create_swapchain(&info, None) }?;
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }?;

        Ok(Self { images, swapchain })
    }
}

pub struct ImageManager {
    images: Vec<Image>,
    sampler: vk::Sampler,
    image_infos: Vec<vk::DescriptorImageInfo>,
}

impl ImageManager {
    pub fn new(device: &ash::Device) -> anyhow::Result<Self> {
        Ok(Self {
            sampler: unsafe {
                device.create_sampler(
                    &vk::SamplerCreateInfo::builder()
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .max_lod(vk::LOD_CLAMP_NONE),
                    None,
                )
            }?,
            images: Default::default(),
            image_infos: Default::default(),
        })
    }

    pub fn descriptor_count(&self) -> u32 {
        self.images.len() as u32
    }

    pub fn push_image(&mut self, image: Image) -> u32 {
        let index = self.images.len() as u32;

        self.image_infos.push(
            *vk::DescriptorImageInfo::builder()
                .image_view(image.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(self.sampler),
        );

        self.images.push(image);

        index
    }

    pub fn write_descriptor_set<'a>(
        &'a self,
        set: vk::DescriptorSet,
        binding: u32,
    ) -> vk::WriteDescriptorSetBuilder<'a> {
        vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&self.image_infos)
    }

    pub fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()> {
        for image in &self.images {
            image.cleanup(allocator)?;
        }

        Ok(())
    }
}

pub struct CommandBufferAndQueue {
    pub buffer: vk::CommandBuffer,
    pub queue: vk::Queue,
    pub pool: vk::CommandPool,
}

impl CommandBufferAndQueue {
    pub fn begin(&self, device: &ash::Device) -> anyhow::Result<()> {
        unsafe {
            device.begin_command_buffer(
                self.buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }?;

        Ok(())
    }

    pub fn finish_block_and_reset(&self, device: &ash::Device) -> anyhow::Result<()> {
        unsafe {
            device.end_command_buffer(self.buffer)?;

            let fence = device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

            device.queue_submit(
                self.queue,
                &[*vk::SubmitInfo::builder().command_buffers(&[self.buffer])],
                fence,
            )?;

            device.wait_for_fences(&[fence], true, u64::MAX)?;
            device.destroy_fence(fence, None);

            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;

            Ok(())
        }
    }
}
