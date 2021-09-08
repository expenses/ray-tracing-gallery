use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{Surface as SurfaceLoader, Swapchain as SwapchainLoader};
use ash::vk;
use byte_strings::c_str;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use winit::{event_loop::EventLoop, window::WindowBuilder};

mod utils;

use utils::{select_physical_device, Buffer, CStrList};

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

    let device_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_info)
        .enabled_features(&device_features)
        .enabled_extension_names(&device_extensions.pointers)
        .enabled_layer_names(&device_layers.pointers)
        .push_next(&mut vk_12_features)
        .push_next(&mut amd_device_coherent_memory);

    let device = unsafe { instance.create_device(physical_device, &device_info, None) }?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let buffer_device_address = {
        let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();

        let mut physical_device_features_2 =
            vk::PhysicalDeviceFeatures2::builder().push_next(&mut bda_features);

        unsafe {
            instance.get_physical_device_features2(physical_device, &mut physical_device_features_2)
        };

        bda_features.buffer_device_address != 0
    };

    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance,
        device: device.clone(),
        physical_device,
        debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
        buffer_device_address,
    })?;

    let buffer = Buffer::new(
        &[0; 12],
        vk::BufferUsageFlags::VERTEX_BUFFER,
        &mut allocator,
        queue_family,
        &device,
    )?;

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
