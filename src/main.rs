use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr;
use ash::extensions::khr::{
    AccelerationStructure as AccelerationStructureLoader,
    DeferredHostOperations as DeferredHostOperationsLoader,
    RayTracingPipeline as RayTracingPipelineLoader, Surface as SurfaceLoader,
    Swapchain as SwapchainLoader,
};
use ash::vk;
use std::ffi::CStr;
use ultraviolet::{Mat4, Vec3};
use winit::{event_loop::EventLoop, window::WindowBuilder};

mod utils;

use utils::{
    select_physical_device, AccelerationStructure, Allocator, Buffer, CStrList, ScratchBuffer,
};

fn main() -> anyhow::Result<()> {
    simple_logger::SimpleLogger::new().init()?;

    /*let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Vulkan base")
        .build(&event_loop)?;*/

    let entry = unsafe { ash::Entry::new() }?;

    // Vulkan 1.2, hell yea
    let api_version = vk::make_api_version(0, 1, 2, 0);

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance
    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"Ray Tracing Gallery\0")?)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(api_version);

    let instance_extensions = CStrList::new({
        let mut instance_extensions = vec![khr::Surface::name(), khr::Win32Surface::name()]; //ash_window::enumerate_required_extensions(&window)?;
        instance_extensions.push(DebugUtilsLoader::name());
        instance_extensions
    });

    let validation_layer_name = CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0")?;

    let instance_layers = CStrList::new(vec![validation_layer_name]);

    let device_extensions = CStrList::new(vec![
        SwapchainLoader::name(),
        DeferredHostOperationsLoader::name(),
        AccelerationStructureLoader::name(),
        RayTracingPipelineLoader::name(),
    ]);

    let device_layers = CStrList::new(vec![validation_layer_name]);

    let mut debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(vulkan_debug_utils_callback));

    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions.pointers)
                .enabled_layer_names(&instance_layers.pointers)
                .push_next(&mut debug_messenger_info),
            None,
        )
    }?;

    let debug_utils_loader = DebugUtilsLoader::new(&entry, &instance);
    let debug_messenger =
        unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_messenger_info, None) }?;

    let surface_loader = SurfaceLoader::new(&entry, &instance);

    //let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }?;

    let (physical_device, queue_family, surface_format) =
        match select_physical_device(&instance, &device_extensions, &surface_loader)? {
            Some(selection) => selection,
            None => {
                println!("No suitable device found 💔. Exiting program");
                return Ok(());
            }
        };

    let device = {
        let queue_info = [*vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0])];

        let device_features = vk::PhysicalDeviceFeatures::builder();

        let mut vk_12_features =
            vk::PhysicalDeviceVulkan12Features::builder().buffer_device_address(true);

        // We need this because 'device coherent memory' is one of the memory type bits of
        // `get_buffer_memory_requirements` for some reason, on my machine at-least.
        let mut amd_device_coherent_memory =
            vk::PhysicalDeviceCoherentMemoryFeaturesAMD::builder().device_coherent_memory(true);

        let mut ray_tracing_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder().ray_tracing_pipeline(true);

        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true);

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extensions.pointers)
            .enabled_layer_names(&device_layers.pointers)
            .push_next(&mut vk_12_features)
            .push_next(&mut amd_device_coherent_memory)
            .push_next(&mut ray_tracing_features)
            .push_next(&mut acceleration_structure_features);

        unsafe { instance.create_device(physical_device, &device_info, None) }?
    };

    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let mut allocator = Allocator::new(
        instance.clone(),
        device.clone(),
        physical_device,
        queue_family,
    )?;

    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family),
            None,
        )
    }?;

    let cmd_buf_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }?[0];

    let as_loader = AccelerationStructureLoader::new(&instance, &device);

    let cube_verts_buffer = Buffer::new(
        bytemuck::cast_slice(&vertices()),
        "cube verts",
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        &mut allocator,
    )?;

    let cube_indices_buffer = Buffer::new(
        bytemuck::cast_slice(&indices()),
        "cube indices",
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        &mut allocator,
    )?;

    // Create the BLAS

    let (blas, mut scratch_buffer) = build_blas(
        &cube_verts_buffer,
        3,
        &cube_indices_buffer,
        3,
        &device,
        &as_loader,
        &mut allocator,
        command_buffer,
        queue,
    )?;

    unsafe {
        device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;
    }

    // Allocate the instance buffer

    let instance = vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: identity_3x4_matrix(),
        },
        instance_custom_index_and_mask: 0,
        instance_shader_binding_table_record_offset_and_flags:
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw(),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            host_handle: blas.acceleration_structure,
        },
    };

    let instances = &[instance];

    let instances_buffer = Buffer::new_with_custom_alignment(
        unsafe { instances_as_bytes(instances) },
        "instances buffer",
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        16,
        &mut allocator,
    )?;

    // Create the tlas

    let tlas = build_tlas(
        &instances_buffer,
        instances.len() as u32,
        &device,
        &as_loader,
        &mut allocator,
        &mut scratch_buffer,
        command_buffer,
        queue,
    )?;

    instances_buffer.cleanup(&mut allocator)?;
    blas.buffer.cleanup(&mut allocator)?;
    tlas.buffer.cleanup(&mut allocator)?;
    scratch_buffer.inner.cleanup(&mut allocator)?;
    cube_verts_buffer.cleanup(&mut allocator)?;
    cube_indices_buffer.cleanup(&mut allocator)?;

    Ok(())
}

fn build_blas(
    vertices: &Buffer,
    num_vertices: u32,
    indices: &Buffer,
    num_indices: u32,
    device: &ash::Device,
    as_loader: &AccelerationStructureLoader,
    allocator: &mut Allocator,
    command_buffer: vk::CommandBuffer,
    queue: vk::Queue,
) -> anyhow::Result<(AccelerationStructure, ScratchBuffer)> {
    let primitive_count = num_indices / 3;

    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: vertices.device_address(device),
        })
        .vertex_stride(std::mem::size_of::<Vertex>() as u64)
        .max_vertex(num_vertices)
        .index_type(vk::IndexType::UINT16)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: indices.device_address(device),
        });

    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: *triangles,
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    let offset =
        vk::AccelerationStructureBuildRangeInfoKHR::builder().primitive_count(primitive_count);

    let geometries = &[*geometry];

    let mut geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(geometries);

    let build_sizes = unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_info,
            &[primitive_count],
        )
    };

    let mut scratch_buffer = ScratchBuffer::new(build_sizes.build_scratch_size, allocator)?;

    let blas = AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        "blas",
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        as_loader,
        allocator,
    )?;

    geometry_info = geometry_info
        .dst_acceleration_structure(blas.acceleration_structure)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.inner.device_address(&device),
        });

    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        as_loader.cmd_build_acceleration_structures(
            command_buffer,
            &[*geometry_info],
            &[&[*offset]],
        );

        device.end_command_buffer(command_buffer)?;

        let fence = device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

        device.queue_submit(
            queue,
            &[*vk::SubmitInfo::builder().command_buffers(&[command_buffer])],
            fence,
        )?;

        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.destroy_fence(fence, None);
    }

    Ok((blas, scratch_buffer))
}

fn build_tlas(
    instances: &Buffer,
    num_instances: u32,
    device: &ash::Device,
    as_loader: &AccelerationStructureLoader,
    allocator: &mut Allocator,
    scratch_buffer: &mut ScratchBuffer,
    command_buffer: vk::CommandBuffer,
    queue: vk::Queue,
) -> anyhow::Result<AccelerationStructure> {
    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::builder().data(
        vk::DeviceOrHostAddressConstKHR {
            device_address: instances.device_address(&device),
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

    let mut geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(geometries);

    let build_sizes = unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_info,
            &[num_instances],
        )
    };

    scratch_buffer.ensure_size_of(build_sizes.build_scratch_size, allocator)?;

    let tlas = AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        "tlas",
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        as_loader,
        allocator,
    )?;

    geometry_info = geometry_info
        .dst_acceleration_structure(tlas.acceleration_structure)
        .scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.inner.device_address(&device),
        });

    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        let memory_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR);

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            &[*memory_barrier],
            &[],
            &[],
        );

        as_loader.cmd_build_acceleration_structures(
            command_buffer,
            &[*geometry_info],
            &[&[*offset]],
        );

        device.end_command_buffer(command_buffer)?;

        let fence = device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

        device.queue_submit(
            queue,
            &[*vk::SubmitInfo::builder().command_buffers(&[command_buffer])],
            fence,
        )?;

        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.destroy_fence(fence, None);
    }

    Ok(tlas)
}

// This is lazy because I could really just write a bytemuck'd struct for this.
unsafe fn instances_as_bytes(instances: &[vk::AccelerationStructureInstanceKHR]) -> &[u8] {
    std::slice::from_raw_parts(
        instances.as_ptr() as *const u8,
        instances.len() * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>(),
    )
}

// https://github.com/SaschaWillems/Vulkan/blob/eb11297312a164d00c60b06048100bac1d780bb4/examples/raytracingbasic/raytracingbasic.cpp#L275
#[rustfmt::skip]
fn identity_3x4_matrix() -> [f32; 12] {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
}

#[rustfmt::skip]
fn flatted_matrix(matrix: Mat4) -> [f32; 12] {
    [
        matrix.cols[0].x, matrix.cols[0].y, matrix.cols[0].z, matrix.cols[0].w,
        matrix.cols[1].x, matrix.cols[1].y, matrix.cols[1].z, matrix.cols[1].w,
        matrix.cols[2].x, matrix.cols[2].y, matrix.cols[2].z, matrix.cols[2].w,
    ]
}

#[test]
fn did_i_do_matrices_right() {
    assert_eq!(identity_3x4_matrix(), flatted_matrix(Mat4::identity()));
}

unsafe extern "system" fn vulkan_debug_utils_callback(
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

    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    println!("[Debug Msg][{}][{}] {:?}", severity, ty, message);
    vk::FALSE
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct Vertex {
    pos: Vec3,
}

fn vertices() -> [Vertex; 3] {
    [
        Vertex {
            pos: Vec3::new(1.0, 1.0, 0.0),
        },
        Vertex {
            pos: Vec3::new(-1.0, 1.0, 0.0),
        },
        Vertex {
            pos: Vec3::new(0.0, -1.0, 0.0),
        },
    ]
}

fn indices() -> [u16; 3] {
    [0, 1, 2]
}

/*
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct Vertex {
    position: Vec3,
    colour: Vec3,
}

impl Vertex {
    fn attribute_desc() -> [vk::VertexInputAttributeDescription; 2] {
        [
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::size_of::<Vec3>() as u32),
        ]
    }
}

fn vertex(x: f32, z: f32, y: f32) -> Vertex {
    let corner = Vec3::new(x, y, z);

    Vertex {
        position: corner * 2.0 - Vec3::broadcast(1.0),
        colour: corner,
    }
}

fn cube_verts() -> ([Vertex; 8], [u16; 36]) {
    (
        [
            vertex(0.0, 0.0, 0.0),
            vertex(1.0, 0.0, 0.0),
            vertex(0.0, 1.0, 0.0),
            vertex(1.0, 1.0, 0.0),
            vertex(0.0, 0.0, 1.0),
            vertex(1.0, 0.0, 1.0),
            vertex(0.0, 1.0, 1.0),
            vertex(1.0, 1.0, 1.0),
        ],
        [
            0, 1, 2, 2, 1, 3, // bottom
            3, 1, 5, 3, 5, 7, // front
            0, 2, 4, 4, 2, 6, // back
            1, 0, 4, 1, 4, 5, // left
            2, 3, 6, 6, 3, 7, // right
            5, 4, 6, 5, 6, 7, // top
        ],
    )
}
*/
