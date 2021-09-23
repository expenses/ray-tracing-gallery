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
use ultraviolet::{Mat3, Mat4, Vec2, Vec3, Vec4};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::{
    event_loop::{ControlFlow, EventLoop},
    window::Fullscreen,
    window::WindowBuilder,
};

mod command_buffer_recording;
mod util_functions;
mod util_structs;

use util_structs::{
    AccelerationStructure, Allocator, Buffer, CStrList, CommandBufferAndQueue, Image, ImageManager,
    ModelBuffers, ScratchBuffer, ShaderBindingTable, Swapchain,
};

use util_functions::{
    load_gltf, load_rgba_png_image_from_bytes, load_shader_module, sbt_aligned_size,
    select_physical_device, shader_group_for_type,
};

use command_buffer_recording::{GlobalResources, PerFrameResources};

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

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Ray Tracing Gallery")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
        .build(&event_loop)?;

    let entry = unsafe { ash::Entry::new() }?;

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

    let (physical_device, queue_family, surface_format) =
        match select_physical_device(&instance, &device_extensions, &surface_loader, surface)? {
            Some(selection) => selection,
            None => {
                log::info!("No suitable device found 💔. Exiting program");
                return Ok(());
            }
        };

    let device = {
        let queue_info = [*vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family)
            .queue_priorities(&[1.0])];

        let device_features = vk::PhysicalDeviceFeatures::builder()
            .shader_int16(true)
            .shader_int64(true);

        let mut storage16_features =
            vk::PhysicalDevice16BitStorageFeatures::builder().storage_buffer16_bit_access(true);

        let mut vk_12_features = vk::PhysicalDeviceVulkan12Features::builder()
            .buffer_device_address(true)
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

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extensions.pointers)
            .enabled_layer_names(&enabled_layers.pointers)
            .push_next(&mut vk_12_features)
            .push_next(&mut amd_device_coherent_memory)
            .push_next(&mut ray_tracing_features)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut storage16_features);

        unsafe { instance.create_device(physical_device, &device_info, None) }?
    };

    let (ray_tracing_props, as_props) = {
        let mut ray_tracing_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();

        let mut device_props_2 = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut ray_tracing_props)
            .push_next(&mut as_props);

        unsafe { instance.get_physical_device_properties2(physical_device, &mut device_props_2) }

        (ray_tracing_props, as_props)
    };

    let mut allocator = Allocator::new(
        instance.clone(),
        device.clone(),
        debug_utils_loader,
        physical_device,
        queue_family,
    )?;

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

    let image_staging_buffers = {
        let (green_texture, green_texture_staging_buffer) = load_rgba_png_image_from_bytes(
            include_bytes!("../resources/green.png"),
            "green texture",
            init_command_buffer.buffer(),
            &mut allocator,
        )?;

        image_manager.push_image(green_texture, false);

        let (pink_texture, pink_texture_staging_buffer) = load_rgba_png_image_from_bytes(
            include_bytes!("../resources/pink.png"),
            "pink texture",
            init_command_buffer.buffer(),
            &mut allocator,
        )?;

        image_manager.push_image(pink_texture, false);

        [green_texture_staging_buffer, pink_texture_staging_buffer]
    };

    let (plane_buffers, plane_staging_buffer) = load_gltf(
        include_bytes!("../resources/plane.glb"),
        "plane",
        0,
        &mut allocator,
        &mut image_manager,
        init_command_buffer.buffer(),
    )?;

    let (tori_buffers, tori_staging_buffer) = load_gltf(
        include_bytes!("../resources/tori.glb"),
        "tori",
        1,
        &mut allocator,
        &mut image_manager,
        init_command_buffer.buffer(),
    )?;

    let (lain_buffers, lain_staging_buffer) = load_gltf(
        include_bytes!("../resources/lain.glb"),
        "lain",
        1,
        &mut allocator,
        &mut image_manager,
        init_command_buffer.buffer(),
    )?;

    // Create the BLASes

    let as_loader = AccelerationStructureLoader::new(&instance, &device);

    let mut scratch_buffer = ScratchBuffer::new(&as_props);

    let plane_blas = build_blas(
        &plane_buffers,
        "plane blas",
        &as_loader,
        &mut scratch_buffer,
        &mut allocator,
        init_command_buffer.buffer(),
    )?;

    let tori_blas = build_blas(
        &tori_buffers,
        "tori blas",
        &as_loader,
        &mut scratch_buffer,
        &mut allocator,
        init_command_buffer.buffer(),
    )?;

    let lain_blas = build_blas(
        &lain_buffers,
        "lain blas",
        &as_loader,
        &mut scratch_buffer,
        &mut allocator,
        init_command_buffer.buffer(),
    )?;

    // Allocate the instance buffer

    let lain_base_transform =
        Mat4::from_translation(Vec3::new(-2.0, 0.0, -1.0)) * Mat4::from_scale(0.5);
    let mut lain_rotation = 150.0_f32.to_radians();
    let lain_instance_offset = std::mem::size_of::<AccelerationStructureInstance>() * 2;

    let mut instances = vec![
        AccelerationStructureInstance::new(Mat4::from_scale(10.0), &plane_blas, &device, 0),
        AccelerationStructureInstance::new(
            Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
            &tori_blas,
            &device,
            1,
        ),
        AccelerationStructureInstance::new(
            lain_base_transform * Mat4::from_rotation_y(lain_rotation),
            &lain_blas,
            &device,
            2,
        ),
    ];

    {
        use rand::Rng;

        let mut rng = rand::thread_rng();

        for _ in 0..10000 {
            instances.push(AccelerationStructureInstance::new(
                Mat4::from_translation(Vec3::new(
                    rng.gen_range(-10.0..10.0),
                    rng.gen_range(0.5..2.5),
                    rng.gen_range(-10.0..10.0),
                )) * Mat4::from_rotation_y(rng.gen_range(0.0..100.0))
                    * Mat4::from_scale(rng.gen_range(0.01..0.1)),
                &tori_blas,
                &device,
                1,
            ))
        }
    }

    let instances_buffer = Buffer::new_with_custom_alignment(
        bytemuck::cast_slice(&instances),
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

    let tlas = build_tlas(
        &instances_buffer,
        num_instances,
        &as_loader,
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

    // Create descriptor sets and set layouts

    let model_info_buffer = Buffer::new(
        bytemuck::cast_slice(&[
            plane_buffers.model_info(&device),
            tori_buffers.model_info(&device),
            lain_buffers.model_info(&device),
        ]),
        "model info",
        vk::BufferUsageFlags::STORAGE_BUFFER,
        &mut allocator,
    )?;

    let (general_dsl, per_frame_dsl, descriptor_pool, general_ds) =
        create_descriptors_sets(&device, &image_manager, &model_info_buffer)?;

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

    let mut uniforms = RayTracingUniforms {
        sun_dir: sun.as_normal(),
        sun_radius: 0.1,
        view_inverse: Mat4::identity(),
        proj_inverse: ultraviolet::projection::perspective_infinite_z_vk(
            59.0_f32.to_radians(),
            extent.width as f32 / extent.height as f32,
            0.1,
        )
        .inversed(),
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
                        bytemuck::bytes_of(&uniforms),
                        "ray tracing uniforms 0",
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        &mut allocator,
                    )?,
                    tlas: tlas.clone(
                        init_command_buffer.buffer(),
                        "tlas 0",
                        &as_loader,
                        &mut allocator,
                    )?,
                    scratch_buffer: ScratchBuffer::new(&as_props),
                    instances_buffer: Buffer::new_with_custom_alignment(
                        bytemuck::cast_slice(&instances),
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
                        bytemuck::bytes_of(&uniforms),
                        "ray tracing uniforms 1",
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        &mut allocator,
                    )?,
                    tlas: {
                        tlas.rename("tlas 1", &allocator)?;
                        tlas
                    },
                    scratch_buffer: ScratchBuffer::new(&as_props),
                    instances_buffer: {
                        instances_buffer.rename("instances buffer 1", &allocator)?;
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

        for staging_buffer in image_staging_buffers {
            staging_buffer.cleanup_and_drop(&mut allocator)?;
        }

        for staging_buffer in [
            plane_staging_buffer,
            tori_staging_buffer,
            lain_staging_buffer,
        ] {
            if let Some(staging_buffer) = staging_buffer {
                staging_buffer.cleanup_and_drop(&mut allocator)?;
            }
        }

        scratch_buffer.cleanup_and_drop(&mut allocator)?;
    }

    // Create pipelines

    let pipeline_loader = RayTracingPipelineLoader::new(&instance, &device);

    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&[general_dsl, per_frame_dsl])
                .push_constant_ranges(&[]),
            None,
        )
    }?;

    let shader_stages = [
        load_shader_module(
            include_bytes!("shaders/raygen.rgen.spv"),
            vk::ShaderStageFlags::RAYGEN_KHR,
            &device,
        )?,
        load_shader_module(
            include_bytes!("shaders/closesthit.rchit.spv"),
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            &device,
        )?,
        load_shader_module(
            include_bytes!("shaders/miss.rmiss.spv"),
            vk::ShaderStageFlags::MISS_KHR,
            &device,
        )?,
        load_shader_module(
            include_bytes!("shaders/shadow.rmiss.spv"),
            vk::ShaderStageFlags::MISS_KHR,
            &device,
        )?,
    ];

    let shader_groups = [
        shader_group_for_type(0, vk::ShaderGroupShaderKHR::GENERAL),
        shader_group_for_type(1, vk::ShaderGroupShaderKHR::CLOSEST_HIT),
        shader_group_for_type(2, vk::ShaderGroupShaderKHR::GENERAL),
        shader_group_for_type(3, vk::ShaderGroupShaderKHR::GENERAL),
    ];

    let create_pipeline = vk::RayTracingPipelineCreateInfoKHR::builder()
        .stages(&shader_stages)
        .groups(&shader_groups)
        .max_pipeline_ray_recursion_depth(1)
        .layout(pipeline_layout);

    let pipelines = unsafe {
        pipeline_loader.create_ray_tracing_pipelines(
            vk::DeferredOperationKHR::null(),
            vk::PipelineCache::null(),
            &[*create_pipeline],
            None,
        )
    }?;

    let pipeline = pipelines[0];

    // Shader binding tables

    let handle_size_aligned = sbt_aligned_size(&ray_tracing_props);
    let num_groups = shader_groups.len() as u32;
    let shader_binding_table_size = num_groups * handle_size_aligned;

    // Fetch the SBT handles and upload them to buffers.

    let group_handles = unsafe {
        pipeline_loader.get_ray_tracing_shader_group_handles(
            pipeline,
            0,
            num_groups,
            shader_binding_table_size as usize,
        )
    }?;

    let raygen_sbt = ShaderBindingTable::new(
        &group_handles,
        "raygen shader binding table",
        &ray_tracing_props,
        0..1,
        &device,
        &mut allocator,
    )?;

    let closest_hit_sbt = ShaderBindingTable::new(
        &group_handles,
        "closest hit shader binding table",
        &ray_tracing_props,
        1..2,
        &device,
        &mut allocator,
    )?;

    // 2 miss shaders
    let miss_sbt = ShaderBindingTable::new(
        &group_handles,
        "miss shader binding table",
        &ray_tracing_props,
        2..4,
        &device,
        &mut allocator,
    )?;

    // Create swapchain

    let surface_caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }?;
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

    let swapchain_loader = SwapchainLoader::new(&instance, &device);
    let mut swapchain = Swapchain::new(&swapchain_loader, swapchain_info, &allocator)?;

    let mut current_frame = 0;

    // main loop

    let mut kbd_state = KbdState::default();

    let mut cursor_grab = false;

    let mut screen_center =
        winit::dpi::LogicalPosition::new(extent.width as f64 / 2.0, extent.height as f64 / 2.0);

    let global_resources = GlobalResources {
        pipeline,
        pipeline_layout,
        raygen_sbt,
        closest_hit_sbt,
        miss_sbt,
        general_ds,
        as_loader,
        pipeline_loader,
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

                        uniforms.proj_inverse = ultraviolet::projection::perspective_infinite_z_vk(
                            59.0_f32.to_radians(),
                            extent.width as f32 / extent.height as f32,
                            0.1,
                        )
                        .inversed();

                        unsafe {
                            device.queue_wait_idle(queue)?;
                        }

                        // Replace the swapchain.

                        swapchain_info.image_extent = extent;
                        swapchain_info.old_swapchain = swapchain.swapchain;

                        swapchain = Swapchain::new(&swapchain_loader, swapchain_info, &allocator)?;

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

                    lain_rotation += 0.05;

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
                        swapchain_loader.acquire_next_image(
                            swapchain.swapchain,
                            u64::MAX,
                            frame.present_semaphore,
                            vk::Fence::null(),
                        )
                    } {
                        Ok((swapchain_image_index, _suboptimal)) => {
                            let swapchain_image = swapchain.images[swapchain_image_index as usize];
                            let resources = &mut frame.resources;

                            uniforms.view_inverse = camera.as_view_matrix().inversed();
                            uniforms.sun_dir = sun.as_normal();

                            resources
                                .ray_tracing_uniforms
                                .write_mapped(bytemuck::bytes_of(&uniforms), 0)?;

                            let lain_instance_transform = transpose_matrix_for_instance(
                                lain_base_transform * Mat4::from_rotation_y(lain_rotation),
                            );

                            resources.instances_buffer.write_mapped(
                                // The transform is the first 48 bytes of the instance.
                                bytemuck::bytes_of(&lain_instance_transform),
                                lain_instance_offset,
                            )?;

                            unsafe {
                                device.begin_command_buffer(
                                    frame.command_buffer,
                                    &vk::CommandBufferBeginInfo::builder()
                                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                                )?;

                                resources.record(
                                    frame.command_buffer,
                                    swapchain_image,
                                    extent,
                                    &global_resources,
                                    &device,
                                    &mut allocator,
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

                                swapchain_loader.queue_present(
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

                    plane_buffers.cleanup(&mut allocator)?;
                    plane_blas.buffer.cleanup(&mut allocator)?;
                    tori_blas.buffer.cleanup(&mut allocator)?;
                    tori_buffers.cleanup(&mut allocator)?;
                    lain_blas.buffer.cleanup(&mut allocator)?;
                    lain_buffers.cleanup(&mut allocator)?;

                    model_info_buffer.cleanup(&mut allocator)?;

                    global_resources.raygen_sbt.buffer.cleanup(&mut allocator)?;
                    global_resources.miss_sbt.buffer.cleanup(&mut allocator)?;
                    global_resources
                        .closest_hit_sbt
                        .buffer
                        .cleanup(&mut allocator)?;

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

pub fn create_descriptors_sets(
    device: &ash::Device,
    image_manager: &ImageManager,
    model_info_buffer: &Buffer,
) -> anyhow::Result<(
    vk::DescriptorSetLayout,
    vk::DescriptorSetLayout,
    vk::DescriptorPool,
    vk::DescriptorSet,
)> {
    let general_dsl = unsafe {
        device.create_descriptor_set_layout(
            &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(image_manager.descriptor_count())
                    .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            ]),
            None,
        )
    }?;

    let per_frame_dsl = unsafe {
        device.create_descriptor_set_layout(
            &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .stage_flags(
                        vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                    ),
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(
                        vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                    ),
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
                        .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(num_frames),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(num_frames),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(num_frames),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(image_manager.descriptor_count()),
                ])
                .max_sets(num_frames + 1),
            None,
        )
    }?;

    let descriptor_sets = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&[general_dsl])
                .descriptor_pool(descriptor_pool),
        )
    }?;

    let general_ds = descriptor_sets[0];

    unsafe {
        device.update_descriptor_sets(
            &[
                *vk::WriteDescriptorSet::builder()
                    .dst_set(general_ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&[model_info_buffer.descriptor_buffer_info()]),
                *image_manager.write_descriptor_set(general_ds, 1),
            ],
            &[],
        );
    }

    Ok((general_dsl, per_frame_dsl, descriptor_pool, general_ds))
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

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct RayTracingUniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
    sun_dir: Vec3,
    sun_radius: f32,
}

fn build_blas(
    model_buffers: &ModelBuffers,
    name: &str,
    as_loader: &AccelerationStructureLoader,
    scratch_buffer: &mut ScratchBuffer,
    allocator: &mut Allocator,
    command_buffer: vk::CommandBuffer,
) -> anyhow::Result<AccelerationStructure> {
    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            triangles: model_buffers.triangles_data,
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    let offset = vk::AccelerationStructureBuildRangeInfoKHR::builder()
        .primitive_count(model_buffers.primitive_count);

    let geometries = &[*geometry];

    let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(geometries);

    let build_sizes = unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_info,
            &[model_buffers.primitive_count],
        )
    };

    let scratch_buffer =
        scratch_buffer.ensure_size_of(build_sizes.build_scratch_size, allocator)?;

    let blas = AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        name,
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        as_loader,
        allocator,
        scratch_buffer,
        geometry_info,
        offset,
        command_buffer,
    )?;

    Ok(blas)
}

fn build_tlas(
    instances: &Buffer,
    num_instances: u32,
    as_loader: &AccelerationStructureLoader,
    allocator: &mut Allocator,
    scratch_buffer: &mut ScratchBuffer,
    command_buffer: vk::CommandBuffer,
) -> anyhow::Result<AccelerationStructure> {
    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::builder().data(
        vk::DeviceOrHostAddressConstKHR {
            device_address: instances.device_address(&allocator.device),
        },
    );

    let geometry = vk::AccelerationStructureGeometryKHR::builder()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: *instances,
        })
        .flags(vk::GeometryFlagsKHR::OPAQUE);

    let offset =
        vk::AccelerationStructureBuildRangeInfoKHR::builder().primitive_count(num_instances);

    let geometries = &[*geometry];

    let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .flags(
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
        )
        .geometries(geometries);

    let build_sizes = unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_info,
            &[num_instances],
        )
    };

    let scratch_buffer =
        scratch_buffer.ensure_size_of(build_sizes.build_scratch_size, allocator)?;

    AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        "tlas",
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        as_loader,
        allocator,
        scratch_buffer,
        geometry_info,
        offset,
        command_buffer,
    )
}

// A `bytemuck`'d `vk::AccelerationStructureInstanceKHR`.
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct AccelerationStructureInstance {
    transform: [Vec4; 3],
    instance_custom_index_and_mask: u32,
    instance_shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_device_address: vk::DeviceAddress,
}

impl AccelerationStructureInstance {
    fn new(
        transform: Mat4,
        blas: &AccelerationStructure,
        device: &ash::Device,
        custom_index: u32,
    ) -> Self {
        fn merge_fields(lower: u32, upper: u8) -> u32 {
            ((upper as u32) << 24) | lower
        }

        Self {
            transform: transpose_matrix_for_instance(transform),
            instance_custom_index_and_mask: merge_fields(custom_index, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: merge_fields(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_device_address: blas.buffer.device_address(device),
        }
    }
}

fn transpose_matrix_for_instance(matrix: Mat4) -> [Vec4; 3] {
    let rows = matrix.transposed().cols;
    [rows[0], rows[1], rows[2]]
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

    let level = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
        _ => log::Level::Info,
    };

    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let ty = format!("{:?}", message_type).to_lowercase();
    log::log!(level, "[Debug Msg][{}] {:?}", ty, message);
    vk::FALSE
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct Vertex {
    pos: Vec3,
    normal: Vec3,
    uv: Vec2,
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
pub struct ModelInfo {
    vertex_buffer_address: vk::DeviceAddress,
    index_buffer_address: vk::DeviceAddress,
    image_index: u32,
    _padding: u32,
}
