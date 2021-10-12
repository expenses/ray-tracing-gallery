use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{
    AccelerationStructure as AccelerationStructureLoader,
    DeferredHostOperations as DeferredHostOperationsLoader,
    RayTracingPipeline as RayTracingPipelineLoader, Surface as SurfaceLoader,
    Swapchain as SwapchainLoader,
};
use ash::vk;
use std::f32::consts::PI;
use std::ffi::CStr;
use structopt::StructOpt;
use ultraviolet::{Mat3, Mat4, Vec2, Vec3, Vec4};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::{
    event_loop::{ControlFlow, EventLoop},
    window::Fullscreen,
    window::WindowBuilder,
};

mod command_buffer_recording;
mod gpu_structs;
mod scene;
mod util_functions;
mod util_structs;

use util_structs::{
    Allocator, Buffer, CStrList, CommandBufferAndQueue, Device as DeviceWithExtensions, Image,
    ImageManager, ScratchBuffer, ShaderBindingTable, Swapchain,
};

use util_functions::{
    build_tlas, info_from_group, load_png_image_from_bytes, load_shader_module,
    load_shader_module_as_stage, sbt_aligned_size, select_physical_device,
    vulkan_debug_utils_callback, ShaderGroup,
};

use gpu_structs::{unsafe_bytes_of, unsafe_cast_slice};
use shared_structs::{PushConstantBufferAddresses, Uniforms};

use command_buffer_recording::{GlobalResources, PerFrameResources, ShaderBindingTables};

use crate::scene::{DefaultScene, EitherScene, LoadedModelScene, Scene};

const MAX_BOUND_IMAGES: u32 = 128;

pub enum HitShader {
    Textured = 0,
    Mirror = 1,
    Portal = 2,
}

#[derive(Debug, StructOpt)]
struct Opt {
    model_to_load: Option<String>,
}

fn main() -> anyhow::Result<()> {
    {
        use simplelog::*;

        CombinedLogger::init(vec![
            TermLogger::new(
                LevelFilter::Info,
                Config::default(),
                TerminalMode::Mixed,
                ColorChoice::Auto,
            ),
            WriteLogger::new(
                LevelFilter::Debug,
                Config::default(),
                std::fs::File::create("run.log")?,
            ),
        ])?;
    }

    let opt = Opt::from_args();

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Ray Tracing Gallery")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
        .build(&event_loop)?;

    let entry = ash::Entry::new();

    // Vulkan 1.2, hell yea
    let api_version = vk::API_VERSION_1_2;

    // https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Instance
    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"Ray Tracing Gallery\0")?)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(api_version);

    let instance_extensions = CStrList::new({
        let mut instance_extensions = ash_window::enumerate_required_extensions(&window)?;
        instance_extensions.push(DebugUtilsLoader::name());
        instance_extensions
    });

    let enabled_layers = CStrList::new(vec![CStr::from_bytes_with_nul(
        b"VK_LAYER_KHRONOS_validation\0",
    )?]);

    let device_extensions = CStrList::new(vec![
        SwapchainLoader::name(),
        DeferredHostOperationsLoader::name(),
        AccelerationStructureLoader::name(),
        RayTracingPipelineLoader::name(),
        vk::KhrShaderClockFn::name(),
    ]);

    let mut debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(vulkan_debug_utils_callback));

    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(&instance_extensions.pointers)
                .enabled_layer_names(&enabled_layers.pointers)
                .push_next(&mut debug_messenger_info),
            None,
        )
    }?;

    let debug_utils_loader = DebugUtilsLoader::new(&entry, &instance);
    let _debug_messenger =
        unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_messenger_info, None) }?;

    let surface_loader = SurfaceLoader::new(&entry, &instance);

    let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }?;

    // Pick a physical device

    let (physical_device, queue_family, surface_format) =
        match select_physical_device(&instance, &device_extensions, &surface_loader, surface)? {
            Some(selection) => selection,
            None => {
                log::info!("No suitable device found 💔. Exiting program");
                return Ok(());
            }
        };

    let surface_caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }?;

    // Create the logical device.

    let device = {
        let queue_info = [*vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0])];

        let device_features = vk::PhysicalDeviceFeatures::builder().shader_int64(true);

        let mut vk_12_features = vk::PhysicalDeviceVulkan12Features::builder()
            .shader_int8(true)
            .buffer_device_address(true)
            .storage_buffer8_bit_access(true)
            .runtime_descriptor_array(true)
            .shader_sampled_image_array_non_uniform_indexing(true);

        // We need this because 'device coherent memory' is one of the memory type bits of
        // `get_buffer_memory_requirements` for some reason, on my machine at-least.
        let mut amd_device_coherent_memory =
            vk::PhysicalDeviceCoherentMemoryFeaturesAMD::builder().device_coherent_memory(true);

        let mut ray_tracing_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder().ray_tracing_pipeline(true);

        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true);

        let mut robustness_features =
            vk::PhysicalDeviceRobustness2FeaturesEXT::builder().null_descriptor(true);

        let mut clock_features =
            vk::PhysicalDeviceShaderClockFeaturesKHR::builder().shader_subgroup_clock(true);

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extensions.pointers)
            .enabled_layer_names(&enabled_layers.pointers)
            .push_next(&mut vk_12_features)
            .push_next(&mut amd_device_coherent_memory)
            .push_next(&mut ray_tracing_features)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut robustness_features)
            .push_next(&mut clock_features);

        unsafe { instance.create_device(physical_device, &device_info, None) }?
    };

    let device = DeviceWithExtensions::wrap_with_extensions(device, &instance, debug_utils_loader)?;

    // Get some properties relating to ray tracing and acceleration structures

    let (ray_tracing_props, as_props) = {
        let mut ray_tracing_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();

        let mut device_props_2 = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut ray_tracing_props)
            .push_next(&mut as_props);

        unsafe { instance.get_physical_device_properties2(physical_device, &mut device_props_2) }

        (ray_tracing_props, as_props)
    };

    // Create descriptor set layouts

    let (general_dsl, per_frame_dsl, descriptor_pool) =
        create_descriptor_set_layouts_and_pool(&device)?;

    // Create pipelines

    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&[general_dsl, per_frame_dsl])
                .push_constant_ranges(&[*vk::PushConstantRange::builder()
                    .stage_flags(
                        vk::ShaderStageFlags::ANY_HIT_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                    )
                    .size(std::mem::size_of::<PushConstantBufferAddresses>() as u32)]),
            None,
        )
    }?;

    let ray_tracing_module =
        load_shader_module(include_bytes!("../shaders/ray-tracing.spv"), &device)?;

    let shader_stages = [
        // Ray generation shader
        *vk::PipelineShaderStageCreateInfo::builder()
            .module(ray_tracing_module)
            .stage(vk::ShaderStageFlags::RAYGEN_KHR)
            .name(CStr::from_bytes_with_nul(b"ray_generation\0")?),
        // Miss shaders
        *vk::PipelineShaderStageCreateInfo::builder()
            .module(ray_tracing_module)
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .name(CStr::from_bytes_with_nul(b"primary_ray_miss\0")?),
        *vk::PipelineShaderStageCreateInfo::builder()
            .module(ray_tracing_module)
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .name(CStr::from_bytes_with_nul(b"shadow_ray_miss\0")?),
        // Hit shaders
        *vk::PipelineShaderStageCreateInfo::builder()
            .module(ray_tracing_module)
            .stage(vk::ShaderStageFlags::ANY_HIT_KHR)
            .name(CStr::from_bytes_with_nul(b"any_hit_alpha_clip\0")?),
        load_shader_module_as_stage(
            &std::fs::read("shaders/closest_hit_textured.spv")?,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            &device,
        )?,
        /*load_shader_module_as_stage(
            &std::fs::read("shaders/closest_hit_mirror.spv")?,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            &device,
        )?,*/
        *vk::PipelineShaderStageCreateInfo::builder()
            .module(ray_tracing_module)
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .name(CStr::from_bytes_with_nul(b"closest_hit_mirror\0")?),
        *vk::PipelineShaderStageCreateInfo::builder()
            .module(ray_tracing_module)
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .name(CStr::from_bytes_with_nul(b"closest_hit_portal\0")?),
    ];

    let shader_groups = [
        info_from_group(ShaderGroup::General(0)),
        info_from_group(ShaderGroup::General(1)),
        info_from_group(ShaderGroup::General(2)),
        info_from_group(ShaderGroup::TriangleHitGroup {
            any_hit_shader: 3,
            closest_hit_shader: 4,
        }),
        info_from_group(ShaderGroup::TriangleHitGroup {
            any_hit_shader: 3,
            closest_hit_shader: 5,
        }),
        info_from_group(ShaderGroup::TriangleHitGroup {
            any_hit_shader: 3,
            closest_hit_shader: 6,
        }),
    ];

    let create_pipeline = vk::RayTracingPipelineCreateInfoKHR::builder()
        .stages(&shader_stages)
        .groups(&shader_groups)
        // Nvidia GPUs allow for 31 levels of ray recursion, but AMD GPUs only allow for 1 (*),
        // due to hardware differences. This means that you can have a primary ray that hits
        // an object, and in that closest-hit shader you can spawn a secondary shadow ray.
        // But then that shadow ray MUST NOT call a shader that spawns another ray.
        //
        // This means that instead we have to do ray bounces in the raygen shader, which is fine
        // but does mean that we can't encapsulate behaviour - the raygen shader needs to 'know'
        // that rays can bounce.
        //
        // *:
        // https://vulkan.gpuinfo.org/displayextensionproperty.php?extensionname=VK_KHR_ray_tracing_pipeline&extensionproperty=maxRayRecursionDepth&platform=all
        .max_pipeline_ray_recursion_depth(1)
        .layout(pipeline_layout);

    let pipelines = unsafe {
        device.rt_pipeline_loader.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            &[*create_pipeline],
            None,
        )
    }?;

    let pipeline = pipelines[0];

    // Create an allocator

    let mut allocator = Allocator::new(instance, device.clone(), physical_device, queue_family)?;

    // Shader binding tables

    let sbt_group_handles = {
        let handle_size_aligned = sbt_aligned_size(&ray_tracing_props);
        let num_groups = shader_groups.len() as u32;
        let shader_binding_table_size = num_groups * handle_size_aligned;

        // Fetch the SBT handles and upload them to buffers.

        unsafe {
            device
                .rt_pipeline_loader
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    num_groups,
                    shader_binding_table_size as usize,
                )
        }?
    };

    let shader_binding_tables = ShaderBindingTables {
        raygen: ShaderBindingTable::new(
            &sbt_group_handles,
            "raygen shader binding table",
            &ray_tracing_props,
            0..1,
            &mut allocator,
        )?,
        // 2 miss shaders
        miss: ShaderBindingTable::new(
            &sbt_group_handles,
            "miss shader binding table",
            &ray_tracing_props,
            1..3,
            &mut allocator,
        )?,
        // 3 hit shaders.
        hit: ShaderBindingTable::new(
            &sbt_group_handles,
            "hit shader binding table",
            &ray_tracing_props,
            3..6,
            &mut allocator,
        )?,
    };

    // Now we can start loading models, textures and creating acceleration structures.

    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    let mut command_buffer_and_queue = {
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

        CommandBufferAndQueue::new(&device, command_buffer, queue, command_pool)?
    };

    // Load images and models

    let mut image_manager = ImageManager::new(&device)?;

    let init_command_buffer = command_buffer_and_queue.begin_buffer_guard(device.clone())?;

    let mut buffers_to_cleanup = Vec::new();

    {
        let green_texture = load_png_image_from_bytes(
            include_bytes!("../resources/green.png"),
            "green texture",
            vk::Format::R8G8B8A8_SRGB,
            init_command_buffer.buffer(),
            &mut allocator,
            &mut buffers_to_cleanup,
        )?;

        image_manager.push_image(green_texture, false);

        let pink_texture = load_png_image_from_bytes(
            include_bytes!("../resources/pink.png"),
            "pink texture",
            vk::Format::R8G8B8A8_SRGB,
            init_command_buffer.buffer(),
            &mut allocator,
            &mut buffers_to_cleanup,
        )?;

        image_manager.push_image(pink_texture, false);

        let blue_noise_texture = load_png_image_from_bytes(
            include_bytes!("../resources/blue_noise_64x64.png"),
            "blue noise texture",
            vk::Format::R8G8B8A8_UNORM,
            init_command_buffer.buffer(),
            &mut allocator,
            &mut buffers_to_cleanup,
        )?;

        image_manager.push_image(blue_noise_texture, false);

        let ggx_lut_image = load_png_image_from_bytes(
            include_bytes!("../resources/flipped_ggx_lut.png"),
            "ggx lut texture",
            vk::Format::R8G8B8A8_UNORM,
            init_command_buffer.buffer(),
            &mut allocator,
            &mut buffers_to_cleanup,
        )?;

        image_manager.push_image(ggx_lut_image, true);
    };

    // Load model buffers and blases

    let mut scratch_buffer = ScratchBuffer::new("init scratch buffer", &as_props);

    let (mut scene, instances, model_info) = if let Some(model_to_load) = opt.model_to_load {
        let (scene, instances, model_info) = LoadedModelScene::new(
            &model_to_load,
            init_command_buffer.buffer(),
            &mut scratch_buffer,
            &mut allocator,
            &mut image_manager,
            &mut buffers_to_cleanup,
        )?;
        (EitherScene::SceneA(scene), instances, model_info)
    } else {
        let (scene, instances, model_info) = DefaultScene::new(
            init_command_buffer.buffer(),
            &mut scratch_buffer,
            &mut allocator,
            &mut image_manager,
            &mut buffers_to_cleanup,
        )?;

        (EitherScene::SceneB(scene), instances, model_info)
    };

    // Add dummy image bindings up to the bound limit. This avoids a warning about
    // bidings being using in draws but not having been updated.
    image_manager.fill_with_dummy_images_up_to(MAX_BOUND_IMAGES as usize);

    let model_info_buffer = Buffer::new(
        unsafe { unsafe_cast_slice(&model_info) },
        "model info",
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        &mut allocator,
    )?;

    let mut instances_buffer = Buffer::new_with_custom_alignment(
        unsafe { unsafe_cast_slice(&instances) },
        "instances buffer",
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        16,
        &mut allocator,
    )?;

    let num_instances = instances.len() as u32;

    // Create the tlas

    // Wait for BLAS builds to finish
    unsafe {
        device.cmd_pipeline_barrier(
            init_command_buffer.buffer(),
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            &[*vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)],
            &[],
            &[],
        );
    }

    let mut tlas = build_tlas(
        &instances_buffer,
        num_instances,
        &mut allocator,
        &mut scratch_buffer,
        init_command_buffer.buffer(),
    )?;

    // Wait for TLAS build to finish
    unsafe {
        device.cmd_pipeline_barrier(
            init_command_buffer.buffer(),
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            &[*vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)],
            &[],
            &[],
        );
    }

    // Create descriptor sets

    let general_ds =
        create_general_descriptors_set(&device, &image_manager, general_dsl, descriptor_pool)?;

    // Create frames for multibuffering and their resources.

    let window_size = window.inner_size();

    let mut extent = vk::Extent2D {
        width: window_size.width,
        height: window_size.height,
    };

    let ray_tracing_sets = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&[per_frame_dsl; 2])
                .descriptor_pool(descriptor_pool),
        )
    }?;

    // Uniforms and state

    let mut camera = FirstPersonCamera {
        eye: Vec3::new(0.0, 2.0, -5.0),
        pitch: 0.0,
        yaw: 180.0_f32.to_radians(),
    };

    let mut camera_velocity = Vec3::zero();

    let mut sun_velocity = Vec2::zero();
    let mut sun = Sun {
        pitch: 0.5,
        yaw: 1.0,
    };

    let mut uniforms = Uniforms {
        sun_dir: <[f32; 3]>::from(sun.as_normal()).into(),
        sun_radius: 0.1,
        view_inverse: glam::Mat4::IDENTITY,
        proj_inverse: mat_to_glam(
            ultraviolet::projection::perspective_reversed_infinite_z_vk(
                59.0_f32.to_radians(),
                extent.width as f32 / extent.height as f32,
                0.1,
            )
            .inversed(),
        ),
        show_heatmap: false,
        blue_noise_texture_index: 2,
        ggx_lut_texture_index: 3,
        frame_index: 0,
    };

    let mut multibuffering_frames = unsafe {
        [
            Frame::new(
                &device,
                queue_family,
                PerFrameResources {
                    storage_image: Image::new_storage_image(
                        extent.width,
                        extent.height,
                        "storage image 0",
                        surface_format.format,
                        init_command_buffer.buffer(),
                        &mut allocator,
                    )?,
                    ray_tracing_ds: ray_tracing_sets[0],
                    ray_tracing_uniforms: Buffer::new(
                        unsafe_bytes_of(&uniforms),
                        "ray tracing uniforms 0",
                        vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        &mut allocator,
                    )?,
                    tlas: tlas.clone(init_command_buffer.buffer(), "tlas 0", &mut allocator)?,
                    scratch_buffer: ScratchBuffer::new("scratch buffer 0", &as_props),
                    instances_buffer: Buffer::new_with_custom_alignment(
                        unsafe_cast_slice(&instances),
                        "instances buffer 0",
                        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        16,
                        &mut allocator,
                    )?,
                    num_instances,
                },
            )?,
            Frame::new(
                &device,
                queue_family,
                PerFrameResources {
                    storage_image: Image::new_storage_image(
                        extent.width,
                        extent.height,
                        "storage image 1",
                        surface_format.format,
                        init_command_buffer.buffer(),
                        &mut allocator,
                    )?,
                    ray_tracing_ds: ray_tracing_sets[1],
                    ray_tracing_uniforms: Buffer::new(
                        unsafe_bytes_of(&uniforms),
                        "ray tracing uniforms 1",
                        vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        &mut allocator,
                    )?,
                    tlas: {
                        tlas.rename("tlas 1", &mut allocator)?;
                        tlas
                    },
                    scratch_buffer: ScratchBuffer::new("scratch buffer 1", &as_props),
                    instances_buffer: {
                        instances_buffer.rename("instances buffer 1", &mut allocator)?;
                        instances_buffer
                    },
                    num_instances,
                },
            )?,
        ]
    };

    init_command_buffer.finish()?;

    // Clean up from initialisation.
    {
        drop(instances);

        for staging_buffer in buffers_to_cleanup.into_iter() {
            staging_buffer.cleanup_and_drop(&mut allocator)?;
        }

        scratch_buffer.cleanup_and_drop(&mut allocator)?;
    }

    // Create swapchain

    let mut image_count = (surface_caps.min_image_count + 1).max(3);
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    log::info!("Using {} swapchain images at a time.", image_count);

    let mut swapchain_info = *vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let mut swapchain = Swapchain::new(&device, swapchain_info)?;

    let mut current_frame = 0;

    // main loop

    let mut kbd_state = KbdState::default();

    let mut cursor_grab = false;

    let mut screen_center =
        winit::dpi::LogicalPosition::new(extent.width as f64 / 2.0, extent.height as f64 / 2.0);

    let global_resources = GlobalResources {
        pipeline,
        pipeline_layout,
        shader_binding_tables,
        general_ds,
        model_info_buffer,
    };

    event_loop.run(move |event, _, control_flow| {
        let loop_closure = || -> anyhow::Result<()> {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(size) => {
                        extent.width = size.width as u32;
                        extent.height = size.height as u32;

                        screen_center = winit::dpi::LogicalPosition::new(
                            extent.width as f64 / 2.0,
                            extent.height as f64 / 2.0,
                        );

                        uniforms.proj_inverse = mat_to_glam(
                            ultraviolet::projection::perspective_reversed_infinite_z_vk(
                                59.0_f32.to_radians(),
                                extent.width as f32 / extent.height as f32,
                                0.1,
                            )
                            .inversed(),
                        );

                        unsafe {
                            device.queue_wait_idle(queue)?;
                        }

                        // Replace the swapchain.

                        swapchain_info.image_extent = extent;
                        swapchain_info.old_swapchain = swapchain.swapchain;

                        swapchain = Swapchain::new(&device, swapchain_info)?;

                        let resize_command_buffer =
                            command_buffer_and_queue.begin_buffer_guard(device.clone())?;

                        for (i, frame) in multibuffering_frames.iter_mut().enumerate() {
                            frame.resources.resize(
                                resize_command_buffer.buffer(),
                                extent,
                                surface_format.format,
                                i,
                                &device,
                                &mut allocator,
                            )?;
                        }

                        resize_command_buffer.finish()?;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(key),
                                ..
                            },
                        ..
                    } => {
                        let pressed = state == ElementState::Pressed;

                        match key {
                            VirtualKeyCode::F11 => {
                                if pressed {
                                    if window.fullscreen().is_some() {
                                        window.set_fullscreen(None);
                                    } else {
                                        window.set_fullscreen(Some(Fullscreen::Borderless(None)))
                                    }
                                }
                            }
                            VirtualKeyCode::W => kbd_state.forward = pressed,
                            VirtualKeyCode::S => kbd_state.back = pressed,
                            VirtualKeyCode::A => kbd_state.left = pressed,
                            VirtualKeyCode::D => kbd_state.right = pressed,
                            VirtualKeyCode::G => {
                                if pressed {
                                    cursor_grab = !cursor_grab;

                                    if cursor_grab {
                                        window.set_cursor_position(screen_center)?;
                                    }

                                    window.set_cursor_visible(!cursor_grab);
                                    window.set_cursor_grab(cursor_grab)?;
                                }
                            }
                            VirtualKeyCode::Up => kbd_state.sun_up = pressed,
                            VirtualKeyCode::Right => kbd_state.sun_cw = pressed,
                            VirtualKeyCode::Left => kbd_state.sun_ccw = pressed,
                            VirtualKeyCode::Down => kbd_state.sun_down = pressed,
                            VirtualKeyCode::H if pressed => {
                                uniforms.show_heatmap = !uniforms.show_heatmap
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if cursor_grab {
                            let position = position.to_logical::<f64>(window.scale_factor());

                            let delta = Vec2::new(
                                (position.x - screen_center.x) as f32,
                                (position.y - screen_center.y) as f32,
                            );

                            window.set_cursor_position(screen_center)?;

                            camera.yaw -= delta.x.to_radians() * 0.05;
                            camera.pitch = (camera.pitch - delta.y.to_radians() * 0.05)
                                .min(PI / 2.0)
                                .max(-PI / 2.0);
                        }
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    {
                        let mut local_velocity = Vec3::zero();
                        let acceleration = 0.005;
                        let max_velocity = 0.2;

                        if kbd_state.forward {
                            local_velocity.z -= acceleration * camera.pitch.cos();
                            local_velocity.y += acceleration * camera.pitch.sin();
                        }

                        if kbd_state.back {
                            local_velocity.z += acceleration * camera.pitch.cos();
                            local_velocity.y -= acceleration * camera.pitch.sin();
                        }

                        if kbd_state.left {
                            local_velocity.x -= acceleration;
                        }

                        if kbd_state.right {
                            local_velocity.x += acceleration;
                        }

                        camera_velocity += Mat3::from_rotation_y(camera.yaw) * local_velocity;
                        let magnitude = camera_velocity.mag();
                        if magnitude > max_velocity {
                            let clamped_magnitude = magnitude.min(max_velocity);
                            camera_velocity *= clamped_magnitude / magnitude;
                        }

                        camera.eye += camera_velocity;

                        camera_velocity *= 0.9;
                    }

                    {
                        let acceleration = 0.002;
                        let max_velocity = 0.05;

                        if kbd_state.sun_up {
                            sun_velocity.y += acceleration;
                        }

                        if kbd_state.sun_down {
                            sun_velocity.y -= acceleration;
                        }

                        if kbd_state.sun_cw {
                            sun_velocity.x += acceleration;
                        }

                        if kbd_state.sun_ccw {
                            sun_velocity.x -= acceleration;
                        }

                        let magnitude = sun_velocity.mag();
                        if magnitude > max_velocity {
                            let clamped_magnitude = magnitude.min(max_velocity);
                            sun_velocity *= clamped_magnitude / magnitude;
                        }

                        sun.yaw -= sun_velocity.x;
                        sun.pitch = (sun.pitch + sun_velocity.y).min(PI / 2.0).max(0.0);

                        sun_velocity *= 0.95;
                    }

                    scene.update();

                    window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    let frame = &mut multibuffering_frames[current_frame];

                    unsafe {
                        device.wait_for_fences(&[frame.render_fence], true, u64::MAX)?;

                        device.reset_fences(&[frame.render_fence])?;

                        device.reset_command_pool(
                            frame.command_pool,
                            vk::CommandPoolResetFlags::empty(),
                        )?;
                    }

                    match unsafe {
                        device.swapchain_loader.acquire_next_image(
                            swapchain.swapchain,
                            u64::MAX,
                            frame.present_semaphore,
                            vk::Fence::null(),
                        )
                    } {
                        Ok((swapchain_image_index, _suboptimal)) => {
                            let swapchain_image = swapchain.images[swapchain_image_index as usize];
                            let resources = &mut frame.resources;

                            uniforms.view_inverse = mat_to_glam(camera.as_view_matrix().inversed());
                            uniforms.sun_dir = <[f32; 3]>::from(sun.as_normal()).into();
                            uniforms.frame_index += 1;

                            resources
                                .ray_tracing_uniforms
                                .write_mapped(unsafe { unsafe_bytes_of(&uniforms) }, 0)?;

                            unsafe {
                                device.begin_command_buffer(
                                    frame.command_buffer,
                                    &vk::CommandBufferBeginInfo::builder()
                                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                                )?;

                                scene.write_resources(
                                    resources,
                                    frame.command_buffer,
                                    &mut allocator,
                                )?;

                                resources.record(
                                    frame.command_buffer,
                                    &device,
                                    swapchain_image,
                                    extent,
                                    &global_resources,
                                )?;

                                device.end_command_buffer(frame.command_buffer)?;

                                device.queue_submit(
                                    queue,
                                    &[*vk::SubmitInfo::builder()
                                        .wait_semaphores(&[frame.present_semaphore])
                                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                                        .command_buffers(&[frame.command_buffer])
                                        .signal_semaphores(&[frame.render_semaphore])],
                                    frame.render_fence,
                                )?;

                                device.swapchain_loader.queue_present(
                                    queue,
                                    &vk::PresentInfoKHR::builder()
                                        .wait_semaphores(&[frame.render_semaphore])
                                        .swapchains(&[swapchain.swapchain])
                                        .image_indices(&[swapchain_image_index]),
                                )?;

                                current_frame = (current_frame + 1) % multibuffering_frames.len();
                            }
                        }
                        Err(error) => log::warn!("Next frame error: {:?}", error),
                    }
                }
                Event::LoopDestroyed => {
                    unsafe {
                        device.device_wait_idle()?;
                    }

                    scene.cleanup(&mut allocator)?;

                    global_resources.model_info_buffer.cleanup(&mut allocator)?;

                    let sbts = &global_resources.shader_binding_tables;

                    sbts.raygen.buffer.cleanup(&mut allocator)?;
                    sbts.miss.buffer.cleanup(&mut allocator)?;
                    sbts.hit.buffer.cleanup(&mut allocator)?;

                    image_manager.cleanup(&mut allocator)?;

                    for frame in &mut multibuffering_frames {
                        let resources = &mut frame.resources;

                        resources.storage_image.cleanup(&mut allocator)?;
                        resources.ray_tracing_uniforms.cleanup(&mut allocator)?;
                        resources.tlas.buffer.cleanup(&mut allocator)?;
                        resources.scratch_buffer.cleanup(&mut allocator)?;
                        resources.instances_buffer.cleanup(&mut allocator)?;
                    }
                }
                _ => {}
            }

            Ok(())
        };

        if let Err(loop_closure) = loop_closure() {
            log::error!("Error: {}", loop_closure);
        }
    });
}

pub fn create_descriptor_set_layouts_and_pool(
    device: &ash::Device,
) -> anyhow::Result<(
    vk::DescriptorSetLayout,
    vk::DescriptorSetLayout,
    vk::DescriptorPool,
)> {
    let general_dsl = unsafe {
        device.create_descriptor_set_layout(
            &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(MAX_BOUND_IMAGES)
                    .stage_flags(
                        vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::ANY_HIT_KHR,
                    ),
            ]),
            None,
        )
    }?;

    // TODO: Remove acceleration structure and uniform buffer from these, using
    // buffer device addresses instead. This requires support in rust-gpu tho.
    let per_frame_dsl = unsafe {
        device.create_descriptor_set_layout(
            &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            ]),
            None,
        )
    }?;

    let num_frames = 2;

    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&[
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(num_frames),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(MAX_BOUND_IMAGES),
                ])
                .max_sets(num_frames + 1),
            None,
        )
    }?;

    Ok((general_dsl, per_frame_dsl, descriptor_pool))
}

pub fn create_general_descriptors_set(
    device: &ash::Device,
    image_manager: &ImageManager,
    general_dsl: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
) -> anyhow::Result<vk::DescriptorSet> {
    let descriptor_sets = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&[general_dsl])
                .descriptor_pool(descriptor_pool),
        )
    }?;

    let general_ds = descriptor_sets[0];

    unsafe {
        device.update_descriptor_sets(&[*image_manager.write_descriptor_set(general_ds, 0)], &[]);
    }

    Ok(general_ds)
}

pub struct Frame {
    present_semaphore: vk::Semaphore,
    render_semaphore: vk::Semaphore,
    render_fence: vk::Fence,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    resources: PerFrameResources,
}

impl Frame {
    unsafe fn new(
        device: &ash::Device,
        queue_family: u32,
        resources: PerFrameResources,
    ) -> anyhow::Result<Self> {
        resources.write_descriptor_sets(device);

        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        let command_pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family),
            None,
        )?;

        Ok(Self {
            command_pool,
            present_semaphore: device.create_semaphore(&semaphore_info, None)?,
            render_semaphore: device.create_semaphore(&semaphore_info, None)?,
            render_fence: device.create_fence(&fence_info, None)?,
            command_buffer: device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0],
            resources,
        })
    }
}

#[derive(Default)]
pub struct KbdState {
    left: bool,
    right: bool,
    forward: bool,
    back: bool,
    sun_up: bool,
    sun_down: bool,
    sun_cw: bool,
    sun_ccw: bool,
}

struct Sun {
    pitch: f32,
    yaw: f32,
}

impl Sun {
    fn as_normal(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        )
    }
}

#[derive(Copy, Clone)]
pub struct FirstPersonCamera {
    eye: Vec3,
    pitch: f32,
    yaw: f32,
}

impl FirstPersonCamera {
    // https://www.3dgep.com/understanding-the-view-matrix/#FPS_Camera
    fn as_view_matrix(self) -> Mat4 {
        let Self { eye, pitch, yaw } = self;

        let x_axis = Vec3::new(yaw.cos(), 0.0, -yaw.sin());
        let y_axis = Vec3::new(
            yaw.sin() * pitch.sin(),
            pitch.cos(),
            yaw.cos() * pitch.sin(),
        );
        let z_axis = Vec3::new(
            yaw.sin() * pitch.cos(),
            -pitch.sin(),
            pitch.cos() * yaw.cos(),
        );

        // Create a 4x4 view matrix from the right, up, forward and eye position vectors
        Mat4::new(
            Vec4::new(x_axis.x, y_axis.x, z_axis.x, 0.0),
            Vec4::new(x_axis.y, y_axis.y, z_axis.y, 0.0),
            Vec4::new(x_axis.z, y_axis.z, z_axis.z, 0.0),
            Vec4::new(-x_axis.dot(eye), -y_axis.dot(eye), -z_axis.dot(eye), 1.0),
        )
    }
}

fn mat_to_glam(mat: Mat4) -> glam::Mat4 {
    let mut glam = glam::Mat4::default();
    *glam.as_mut() = *mat.as_array();
    glam
}
