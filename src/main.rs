use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{
    AccelerationStructure as AccelerationStructureLoader, Surface as SurfaceLoader,
    Swapchain as SwapchainLoader,
};
use ash::vk;
use byte_strings::c_str;
use ultraviolet::Vec3;
use winit::{event_loop::EventLoop, window::WindowBuilder};

mod utils;

use utils::{select_physical_device, AccelerationStructure, Allocator, Buffer, CStrList};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Vulkan base")
        .build(&event_loop)?;

    let entry = unsafe { ash::Entry::new() }?;

    // Vulkan 1.2, hell yea
    let api_version = vk::make_api_version(0, 1, 2, 0);

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&c_str!("Ray Tracing Gallery"))
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(api_version);

    let instance_extensions = CStrList::new({
        let mut instance_extensions = ash_window::enumerate_required_extensions(&window)?;
        instance_extensions.push(DebugUtilsLoader::name());
        instance_extensions
    });

    let instance_layers = CStrList::new(vec![c_str!("VK_LAYER_KHRONOS_validation")]);

    let device_extensions = CStrList::new(vec![
        SwapchainLoader::name(),
        vk::KhrDeferredHostOperationsFn::name(),
        ash::extensions::khr::AccelerationStructure::name(),
        ash::extensions::khr::RayTracingPipeline::name(),
    ]);

    let device_layers = CStrList::new(vec![c_str!("VK_LAYER_KHRONOS_validation")]);

    let mut debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::all() ^ vk::DebugUtilsMessageTypeFlagsEXT::GENERAL,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback));

    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::builder()
                .push_next(&mut debug_messenger_info)
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions.pointers)
                .enabled_layer_names(&instance_layers.pointers),
            None,
        )
    }?;

    let debug_utils_loader = DebugUtilsLoader::new(&entry, &instance);
    let debug_messenger =
        unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_messenger_info, None) }?;

    let surface_loader = SurfaceLoader::new(&entry, &instance);

    let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }?;

    let (physical_device, queue_family, surface_format) =
        match select_physical_device(&instance, &device_extensions, &surface_loader, surface)? {
            Some(selection) => selection,
            None => {
                println!("No suitable device found 💔. Exiting program");
                return Ok(());
            }
        };

    let queue_info = [*vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];

    let device_features = vk::PhysicalDeviceFeatures::builder();

    let mut vk_12_features =
        vk::PhysicalDeviceVulkan12Features::builder().buffer_device_address(true);

    let mut amd_device_coherent_memory =
        vk::PhysicalDeviceCoherentMemoryFeaturesAMD::builder().device_coherent_memory(true);

    let mut ray_tracing_features =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder().ray_tracing_pipeline(true);

    let mut acceleration_structure_features =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder().acceleration_structure(true);

    let device_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_info)
        .enabled_features(&device_features)
        .enabled_extension_names(&device_extensions.pointers)
        .enabled_layer_names(&device_layers.pointers)
        .push_next(&mut vk_12_features)
        .push_next(&mut amd_device_coherent_memory)
        .push_next(&mut ray_tracing_features)
        .push_next(&mut acceleration_structure_features);

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }?;
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

    let (cube_verts, cube_indices) = cube_verts();

    let primitive_count = cube_indices.len() as u32 / 3;

    let as_triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            host_address: cube_verts.as_ptr() as *const _,
        })
        .vertex_stride(std::mem::size_of::<Vertex>() as u64)
        .max_vertex(cube_verts.len() as u32)
        .index_type(vk::IndexType::UINT16)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            host_address: cube_indices.as_ptr() as *const _,
        });

    let as_geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: *as_triangles,
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
        .build();

    let as_offset =
        vk::AccelerationStructureBuildRangeInfoKHR::builder().primitive_count(primitive_count);

    let as_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(&[as_geometry])
        .build();

    let build_sizes = unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &as_geometry_info,
            &[primitive_count],
        )
    };

    let scratch_buffer = Buffer::new_of_size(
        build_sizes.build_scratch_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        &mut allocator,
    );

    let acceleration_structure = AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        &as_loader,
        &mut allocator,
    )?;

    /*unsafe {
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        as_loader.create_acceleration_structure()

        as_loader.cmd_build_acceleration_structures(
            command_buffer,
            &[as_geometry_info],
            &[&[*as_offset]],
        );

        device.end_command_buffer(command_buffer)?;

        let blas_built_fence = device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

        device.queue_submit(
            queue,
            &[*vk::SubmitInfo::builder().command_buffers(&[command_buffer])],
            blas_built_fence,
        )?;

        device.wait_for_fences(&[blas_built_fence], true, u64::MAX)?;

        device.destroy_fence(blas_built_fence, None);
    }*/

    Ok(())
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    println!("[Debug Msg][{}][{}] {:?}", severity, ty, message);
    vk::FALSE
}

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
