use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{
    AccelerationStructure as AccelerationStructureLoader,
    RayTracingPipeline as RayTracingPipelineLoader, Swapchain as SwapchainLoader,
};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocatorCreateDesc};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use crate::gpu_structs::{ModelInfo, Vertex};
use crate::util_functions::{
    cmd_pipeline_image_memory_barrier_explicit, load_rgba_png_image_from_bytes, sbt_aligned_size,
    PipelineImageMemoryBarrierParams,
};

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

// Contains a logical device as well as extension loaders.
#[derive(Clone)]
pub struct Device {
    inner: ash::Device,
    debug_utils_loader: DebugUtilsLoader,
    pub as_loader: AccelerationStructureLoader,
    pub swapchain_loader: SwapchainLoader,
    pub rt_pipeline_loader: RayTracingPipelineLoader,
}

impl Device {
    pub fn wrap_with_extensions(
        device: ash::Device,
        instance: &ash::Instance,
        debug_utils_loader: DebugUtilsLoader,
    ) -> Self {
        Self {
            debug_utils_loader,
            as_loader: AccelerationStructureLoader::new(instance, &device),
            swapchain_loader: SwapchainLoader::new(instance, &device),
            rt_pipeline_loader: RayTracingPipelineLoader::new(instance, &device),
            inner: device,
        }
    }

    pub unsafe fn set_object_name<T: vk::Handle>(
        &self,
        handle: T,
        name: &str,
    ) -> anyhow::Result<()> {
        self.debug_utils_loader.debug_utils_set_object_name(
            self.inner.handle(),
            &*vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(T::TYPE)
                .object_handle(handle.as_raw())
                .object_name(&CString::new(name)?),
        )?;

        Ok(())
    }
}

impl std::ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct Allocator {
    pub inner: gpu_allocator::vulkan::Allocator,
    pub device: Device,
    queue_family: u32,
}

impl Allocator {
    pub fn new(
        instance: ash::Instance,
        device: Device,
        physical_device: vk::PhysicalDevice,
        queue_family: u32,
    ) -> anyhow::Result<Self> {
        Ok(Allocator {
            inner: gpu_allocator::vulkan::Allocator::new(&AllocatorCreateDesc {
                instance,
                device: device.inner.clone(),
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
    pub size: vk::DeviceSize,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    pub ty: vk::AccelerationStructureTypeKHR,
}

impl AccelerationStructure {
    fn build_blas(
        triangles_data: vk::AccelerationStructureGeometryTrianglesDataKHR,
        num_indices: u32,
        name: &str,
        scratch_buffer: &mut ScratchBuffer,
        allocator: &mut Allocator,
        command_buffer: vk::CommandBuffer,
        opaque: bool,
    ) -> anyhow::Result<Self> {
        let num_triangles = num_indices / 3;

        let flags = if opaque {
            vk::GeometryFlagsKHR::OPAQUE
        } else {
            vk::GeometryFlagsKHR::empty()
        };

        let geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: triangles_data,
            })
            .flags(flags);

        let offset =
            vk::AccelerationStructureBuildRangeInfoKHR::builder().primitive_count(num_triangles);

        let geometries = &[*geometry];

        let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(geometries);

        let build_sizes = unsafe {
            allocator
                .device
                .as_loader
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &geometry_info,
                    &[num_triangles],
                )
        };

        let scratch_buffer = scratch_buffer.ensure_size_of(
            build_sizes.build_scratch_size,
            command_buffer,
            allocator,
        )?;

        AccelerationStructure::new(
            build_sizes.acceleration_structure_size,
            name,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            allocator,
            scratch_buffer,
            geometry_info,
            offset,
            command_buffer,
        )
    }

    pub fn new(
        size: vk::DeviceSize,
        name: &str,
        ty: vk::AccelerationStructureTypeKHR,
        allocator: &mut Allocator,
        scratch_buffer: &Buffer,
        mut geometry_info: vk::AccelerationStructureBuildGeometryInfoKHRBuilder,
        offset: vk::AccelerationStructureBuildRangeInfoKHRBuilder,
        command_buffer: vk::CommandBuffer,
    ) -> anyhow::Result<Self> {
        log::info!("Creating a {} of {} bytes", name, size);

        let buffer = Buffer::new_of_size(
            size,
            &format!("{} buffer", name),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            allocator,
        )?;

        let acceleration_structure = unsafe {
            allocator.device.as_loader.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::builder()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .size(size)
                    .ty(ty),
                None,
            )
        }?;

        unsafe {
            allocator
                .device
                .set_object_name(acceleration_structure, name)?;
        }

        geometry_info = geometry_info
            .dst_acceleration_structure(acceleration_structure)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_buffer.device_address(&allocator.device),
            });

        unsafe {
            allocator
                .device
                .as_loader
                .cmd_build_acceleration_structures(
                    command_buffer,
                    &[*geometry_info],
                    &[&[*offset]],
                );
        }

        Ok(Self {
            buffer,
            acceleration_structure,
            size,
            flags: geometry_info.flags,
            ty,
        })
    }

    pub fn update_tlas(
        &mut self,
        instances: &Buffer,
        num_instances: u32,
        command_buffer: vk::CommandBuffer,
        allocator: &mut Allocator,
        scratch_buffer: &mut ScratchBuffer,
    ) -> anyhow::Result<()> {
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

        let geometries = &[*geometry];

        let mut geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
            .flags(self.flags)
            .geometries(geometries)
            // These two are intended to be the same if we're doing an update-in-place.
            // This was super unobvious!
            // https://github.com/nvpro-samples/nvpro_core/blob/07b6dae2d966285aeb4f217f1b1bf6b9e8c769a2/nvvk/raytraceKHR_vk.cpp#L364-L365
            //
            // Ray Tracing Gems II, p 239.
            // https://link.springer.com/content/pdf/10.1007%2F978-1-4842-7185-8.pdf
            .src_acceleration_structure(self.acceleration_structure)
            .dst_acceleration_structure(self.acceleration_structure);

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

        let scratch_buffer = scratch_buffer.ensure_size_of(
            build_sizes.update_scratch_size,
            command_buffer,
            allocator,
        )?;

        geometry_info = geometry_info.scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.device_address(&allocator.device),
        });

        let offset =
            vk::AccelerationStructureBuildRangeInfoKHR::builder().primitive_count(num_instances);

        unsafe {
            allocator
                .device
                .as_loader
                .cmd_build_acceleration_structures(
                    command_buffer,
                    &[*geometry_info],
                    &[&[*offset]],
                );
        }

        Ok(())
    }

    pub fn clone(
        &self,
        command_buffer: vk::CommandBuffer,
        name: &str,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        let new_buffer = Buffer::new_of_size(
            self.size,
            &format!("{} buffer", name),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            allocator,
        )?;

        let new_acceleration_structure = unsafe {
            allocator.device.as_loader.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::builder()
                    .buffer(new_buffer.buffer)
                    .offset(0)
                    .size(self.size)
                    .ty(self.ty),
                None,
            )
        }?;

        unsafe {
            allocator
                .device
                .set_object_name(new_acceleration_structure, name)?;

            allocator.device.as_loader.cmd_copy_acceleration_structure(
                command_buffer,
                &vk::CopyAccelerationStructureInfoKHR::builder()
                    .src(self.acceleration_structure)
                    .dst(new_acceleration_structure)
                    .mode(vk::CopyAccelerationStructureModeKHR::CLONE),
            );
        }

        Ok(Self {
            buffer: new_buffer,
            acceleration_structure: new_acceleration_structure,
            size: self.size,
            flags: self.flags,
            ty: self.ty,
        })
    }

    pub fn rename(&mut self, name: &str, allocator: &mut Allocator) -> anyhow::Result<()> {
        unsafe {
            allocator
                .device
                .set_object_name(self.acceleration_structure, name)
        }?;

        self.buffer.rename(&format!("{} buffer", name), allocator)?;

        Ok(())
    }
}

enum ScratchBufferInner {
    Unallocated,
    Allocated(Buffer),
}

pub struct ScratchBuffer {
    inner: ScratchBufferInner,
    alignment: vk::DeviceSize,
    name: String,
}

impl ScratchBuffer {
    pub fn new(
        name: &str,
        as_props: &vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    ) -> Self {
        Self {
            inner: ScratchBufferInner::Unallocated,
            alignment: as_props.min_acceleration_structure_scratch_offset_alignment
                as vk::DeviceSize,
            name: name.into(),
        }
    }

    pub fn ensure_size_of(
        &mut self,
        size: vk::DeviceSize,
        command_buffer: vk::CommandBuffer,
        allocator: &mut Allocator,
    ) -> anyhow::Result<&Buffer> {
        match &mut self.inner {
            ScratchBufferInner::Unallocated => {
                log::info!("Creating a scratch buffer of {} bytes", size);

                self.inner = ScratchBufferInner::Allocated(Buffer::new_of_size_with_alignment(
                    size,
                    &self.name,
                    vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::STORAGE_BUFFER,
                    self.alignment,
                    allocator,
                )?);
            }
            ScratchBufferInner::Allocated(buffer) => {
                if buffer.allocation.size() < size {
                    buffer.cleanup(allocator)?;
                    log::info!(
                        "Resizing {} from {} bytes to {} bytes",
                        self.name,
                        buffer.allocation.size(),
                        size
                    );

                    *buffer = Buffer::new_of_size_with_alignment(
                        size,
                        &self.name,
                        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                        self.alignment,
                        allocator,
                    )?;
                } else {
                    // This is important! If we're re-using the scratch buffer, then we need to
                    // insert a pipeline barrier so that we don't try and write to it during 2
                    // concurrent acceleration structure builds.
                    //
                    // Atleast, I think this is the case. It seems to crash my gpu sometimes if I don't do this.

                    log::debug!("Inserting a scratch buffer re-use command buffer.");

                    unsafe {
                        allocator.device.cmd_pipeline_barrier(
                            command_buffer,
                            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                            vk::DependencyFlags::empty(),
                            // We don't *seem* to need a memory barrier, but I don't know why not.
                            /*&[*vk::MemoryBarrier::builder()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],*/
                            &[],
                            &[],
                            &[],
                        )
                    }
                }
            }
        }

        match &self.inner {
            ScratchBufferInner::Allocated(buffer) => Ok(buffer),
            _ => unreachable!(),
        }
    }

    pub fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()> {
        if let ScratchBufferInner::Allocated(buffer) = &self.inner {
            buffer.cleanup(allocator)?;
        }

        Ok(())
    }

    pub fn cleanup_and_drop(self, allocator: &mut Allocator) -> anyhow::Result<()> {
        self.cleanup(allocator)?;
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
        range: std::ops::Range<u32>,
        allocator: &mut Allocator,
    ) -> anyhow::Result<Self> {
        let offset = range.start;
        let num_shaders = (range.end - range.start) as u64;

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
            .device_address(buffer.device_address(&allocator.device))
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
            allocator.device.bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )?;

            allocator.device.set_object_name(buffer, name)?;
        };

        Ok(Self { buffer, allocation })
    }

    pub fn new_of_size_with_alignment(
        size: vk::DeviceSize,
        name: &str,
        usage: vk::BufferUsageFlags,
        alignment: vk::DeviceSize,
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

        let mut requirements = unsafe { allocator.device.get_buffer_memory_requirements(buffer) };

        requirements.alignment = alignment;

        let allocation = allocator.inner.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true,
        })?;

        unsafe {
            allocator.device.bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )?;

            allocator.device.set_object_name(buffer, name)?;
        };

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

        Self::from_parts(allocation, buffer, bytes, name, &allocator.device)
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

        Self::from_parts(allocation, buffer, bytes, name, &allocator.device)
    }

    fn from_parts(
        mut allocation: Allocation,
        buffer: vk::Buffer,
        bytes: &[u8],
        name: &str,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let slice = allocation.mapped_slice_mut().unwrap();

        slice[..bytes.len()].copy_from_slice(bytes);

        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;

            device.set_object_name(buffer, name)?;
        };

        Ok(Self { buffer, allocation })
    }

    pub fn device_address(&self, device: &Device) -> vk::DeviceAddress {
        unsafe {
            device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder().buffer(self.buffer),
            )
        }
    }

    pub fn descriptor_buffer_info(&self) -> vk::DescriptorBufferInfo {
        *vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .range(vk::WHOLE_SIZE)
    }

    pub fn write_mapped(&mut self, bytes: &[u8], offset: usize) -> anyhow::Result<()> {
        let slice = self
            .allocation
            .mapped_slice_mut()
            .ok_or_else(|| anyhow::anyhow!("Attempted to write to a buffer that wasn't mapped"))?;
        slice[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    pub fn rename(&mut self, name: &str, allocator: &mut Allocator) -> anyhow::Result<()> {
        unsafe {
            allocator.device.set_object_name(self.buffer, name)?;
        }

        allocator
            .inner
            .rename_allocation(&mut self.allocation, name)?;

        Ok(())
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
    pub fn new_storage_image(
        width: u32,
        height: u32,
        name: &str,
        format: vk::Format,
        command_buffer: vk::CommandBuffer,
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
        };

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
            cmd_pipeline_image_memory_barrier_explicit(&PipelineImageMemoryBarrierParams {
                device: &allocator.device,
                buffer: command_buffer,
                // No need to block on anything before this.
                src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                // We're blocking the use of the texture in ray tracing
                dst_stage: vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                image_memory_barriers: &[*vk::ImageMemoryBarrier::builder()
                    .image(image)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .subresource_range(*subresource_range)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)],
            });
        }

        Ok(Self {
            image,
            view,
            allocation,
        })
    }

    pub fn descriptor_image_info(&self) -> vk::DescriptorImageInfo {
        *vk::DescriptorImageInfo::builder()
            .image_view(self.view)
            .image_layout(vk::ImageLayout::GENERAL)
    }

    pub fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()> {
        allocator.inner.free(self.allocation.clone())?;

        unsafe { allocator.device.destroy_image_view(self.view, None) };

        unsafe { allocator.device.destroy_image(self.image, None) };

        Ok(())
    }
}

pub struct Model {
    pub vertices: Buffer,
    pub indices: Buffer,
    pub blas: AccelerationStructure,
    pub image_index: u32,
    pub id: u32,
}

impl Model {
    pub fn load_gltf(
        bytes: &[u8],
        name: &str,
        fallback_image_index: u32,
        allocator: &mut Allocator,
        image_manager: &mut ImageManager,
        command_buffer: vk::CommandBuffer,
        scratch_buffer: &mut ScratchBuffer,
        opaque: bool,
        model_info: &mut Vec<ModelInfo>,
        buffers_to_cleanup: &mut Vec<Buffer>,
    ) -> anyhow::Result<Self> {
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

        let mut image_index = || -> Option<anyhow::Result<u32>> {
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
                    buffers_to_cleanup,
                )
                .map(|image| image_manager.push_image(image, linear_filtering)),
            )
        };

        let image_index = match image_index() {
            Some(result) => result?,
            None => fallback_image_index,
        };

        let model_id = model_info.len() as u32;

        let model = Self::new(
            &vertices,
            &indices,
            image_index,
            name,
            allocator,
            command_buffer,
            scratch_buffer,
            opaque,
            model_id,
        )?;

        model_info.push(ModelInfo {
            vertex_buffer_address: model.vertices.device_address(&allocator.device),
            index_buffer_address: model.indices.device_address(&allocator.device),
            image_index: model.image_index,
            _padding: 0,
        });

        Ok(model)
    }

    fn new(
        vertices: &[Vertex],
        indices: &[u16],
        image_index: u32,
        name: &str,
        allocator: &mut Allocator,
        command_buffer: vk::CommandBuffer,
        scratch_buffer: &mut ScratchBuffer,
        opaque: bool,
        id: u32,
    ) -> anyhow::Result<Self> {
        let num_vertices = vertices.len() as u32;
        let num_indices = indices.len() as u32;
        let stride = std::mem::size_of::<Vertex>() as u64;

        let buffer_flags = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        let vertices = Buffer::new(
            bytemuck::cast_slice(vertices),
            &format!("{} vertices", name),
            buffer_flags,
            allocator,
        )?;

        let indices = Buffer::new(
            bytemuck::cast_slice(indices),
            &format!("{} indices", name),
            buffer_flags,
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
            image_index,
            blas: AccelerationStructure::build_blas(
                *triangles_data,
                num_indices,
                &format!("{} blas", name),
                scratch_buffer,
                allocator,
                command_buffer,
                opaque,
            )?,
            id,
        })
    }

    pub fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()> {
        self.vertices.cleanup(allocator)?;
        self.indices.cleanup(allocator)?;
        self.blas.buffer.cleanup(allocator)?;
        Ok(())
    }
}

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub fn new(device: &Device, info: vk::SwapchainCreateInfoKHR) -> anyhow::Result<Self> {
        unsafe {
            let swapchain = device.swapchain_loader.create_swapchain(&info, None)?;
            let images = device.swapchain_loader.get_swapchain_images(swapchain)?;

            for (i, image) in images.iter().enumerate() {
                device.set_object_name(*image, &format!("Swapchain image {}", i))?;
            }

            let image_views: Vec<_> = images
                .iter()
                .map(|swapchain_image| {
                    let image_view_info = vk::ImageViewCreateInfo::builder()
                        .image(*swapchain_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(info.image_format)
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1)
                                .build(),
                        );
                    device.create_image_view(&image_view_info, None)
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(Self {
                images,
                swapchain,
                image_views,
            })
        }
    }
}

pub struct ImageManager {
    images: Vec<Image>,
    nearest_sampler: vk::Sampler,
    linear_sampler: vk::Sampler,
    image_infos: Vec<vk::DescriptorImageInfo>,
}

impl ImageManager {
    pub fn new(device: &Device) -> anyhow::Result<Self> {
        Ok(Self {
            nearest_sampler: unsafe {
                device.create_sampler(
                    &vk::SamplerCreateInfo::builder().max_lod(vk::LOD_CLAMP_NONE),
                    None,
                )
            }?,
            linear_sampler: unsafe {
                device.create_sampler(
                    &vk::SamplerCreateInfo::builder()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
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

    pub fn push_image(&mut self, image: Image, linear_filtering: bool) -> u32 {
        let index = self.images.len() as u32;

        let sampler = if linear_filtering {
            self.linear_sampler
        } else {
            self.nearest_sampler
        };

        self.image_infos.push(
            *vk::DescriptorImageInfo::builder()
                .image_view(image.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .sampler(sampler),
        );

        self.images.push(image);

        index
    }

    pub fn fill_with_dummy_images_up_to(&mut self, items: usize) {
        while self.image_infos.len() < items {
            self.image_infos
                .push(*vk::DescriptorImageInfo::builder().sampler(self.nearest_sampler));
        }
    }

    pub fn write_descriptor_set(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
    ) -> vk::WriteDescriptorSetBuilder {
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
    fence: vk::Fence,
}

impl CommandBufferAndQueue {
    pub fn new(
        device: &Device,
        buffer: vk::CommandBuffer,
        queue: vk::Queue,
        pool: vk::CommandPool,
    ) -> anyhow::Result<Self> {
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::builder(), None) }?;

        Ok(Self {
            buffer,
            queue,
            pool,
            fence,
        })
    }

    pub fn begin(&self, device: &Device) -> anyhow::Result<()> {
        unsafe {
            device.begin_command_buffer(
                self.buffer,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }?;

        Ok(())
    }

    pub fn begin_buffer_guard(
        &mut self,
        device: Device,
    ) -> anyhow::Result<CommandBufferAndQueueRaii> {
        self.begin(&device)?;

        Ok(CommandBufferAndQueueRaii {
            inner: self,
            device,
            already_finished: false,
        })
    }

    pub fn finish_block_and_reset(&self, device: &Device) -> anyhow::Result<()> {
        unsafe {
            device.end_command_buffer(self.buffer)?;

            device.queue_submit(
                self.queue,
                &[*vk::SubmitInfo::builder().command_buffers(&[self.buffer])],
                self.fence,
            )?;

            device.wait_for_fences(&[self.fence], true, u64::MAX)?;

            device.reset_fences(&[self.fence])?;
            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;

            Ok(())
        }
    }
}

pub struct CommandBufferAndQueueRaii<'a> {
    inner: &'a mut CommandBufferAndQueue,
    device: Device,
    already_finished: bool,
}

impl<'a> CommandBufferAndQueueRaii<'a> {
    pub fn buffer(&self) -> vk::CommandBuffer {
        self.inner.buffer
    }

    pub fn finish(mut self) -> anyhow::Result<()> {
        self.already_finished = true;
        self.inner.finish_block_and_reset(&self.device)
    }
}

impl<'a> Drop for CommandBufferAndQueueRaii<'a> {
    fn drop(&mut self) {
        log::debug!("Dropping RAII Command Buffer");

        if self.already_finished {
            return;
        }

        log::warn!("RAII Command Buffer not finished explicitly.");

        if let Err(error) = self.inner.finish_block_and_reset(&self.device) {
            log::error!("Error while submitting command buffer to queue: {}", error);
        }
    }
}
