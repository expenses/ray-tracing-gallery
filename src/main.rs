use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{
    AccelerationStructure as AccelerationStructureLoader,
    DeferredHostOperations as DeferredHostOperationsLoader,
    RayTracingPipeline as RayTracingPipelineLoader, Surface as SurfaceLoader,
    Swapchain as SwapchainLoader,
};
use ash::vk;
use std::ffi::CStr;
use ultraviolet::{Mat4, Vec3};
use winit::event::{Event, WindowEvent};
use winit::{
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

mod util_functions;
mod util_structs;

use util_structs::{
    AccelerationStructure, Allocator, Buffer, CStrList, Image, ScratchBuffer, ShaderBindingTable,
};

use util_functions::{load_shader_module, select_physical_device, set_image_layout};

fn main() -> anyhow::Result<()> {
    simple_logger::SimpleLogger::new().init()?;

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Ray Tracing Gallery")
        .build(&event_loop)?;

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
        let mut instance_extensions = ash_window::enumerate_required_extensions(&window)?;
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
    let _debug_messenger =
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
        vertices().len() as u32,
        &cube_indices_buffer,
        indices().len() as u32,
        &device,
        &as_loader,
        &mut allocator,
        command_buffer,
        queue,
    )?;

    cube_verts_buffer.cleanup_and_drop(&mut allocator)?;
    cube_indices_buffer.cleanup_and_drop(&mut allocator)?;

    // Allocate the instance buffer

    let instances = &[vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: identity_3x4_matrix(),
        },
        instance_custom_index_and_mask: 0xFF << 24,
        instance_shader_binding_table_record_offset_and_flags:
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() << 24,
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas.buffer.device_address(&device),
        },
    }];

    let instances_buffer = Buffer::new_with_custom_alignment(
        unsafe { instances_as_bytes(instances) },
        "instances buffer",
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        16,
        &mut allocator,
    )?;

    // Create the tlas

    unsafe {
        device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;
    }

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

    instances_buffer.cleanup_and_drop(&mut allocator)?;
    scratch_buffer.inner.cleanup_and_drop(&mut allocator)?;

    // Create storage image

    let window_size = window.inner_size();

    let mut extent = vk::Extent2D {
        width: window_size.width,
        height: window_size.height,
    };

    unsafe {
        device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;
    }

    let mut storage_image = Image::new_storage_image(
        extent.width,
        extent.height,
        surface_format.format,
        command_buffer,
        queue,
        &mut allocator,
    )?;

    // Create the descriptor set

    let descriptor_set_layout = unsafe {
        device.create_descriptor_set_layout(
            &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
                *vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            ]),
            None,
        )
    }?;

    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&[
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(1),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1),
                ])
                .max_sets(1),
            None,
        )
    }?;

    let descriptor_sets = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&[descriptor_set_layout])
                .descriptor_pool(descriptor_pool),
        )
    }?;

    let descriptor_set = descriptor_sets[0];

    let structures = &[tlas.acceleration_structure];

    let mut write_acceleration_structures =
        vk::WriteDescriptorSetAccelerationStructureKHR::builder()
            .acceleration_structures(structures);

    unsafe {
        device.update_descriptor_sets(
            &[
                {
                    let mut write_as = *vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .push_next(&mut write_acceleration_structures);

                    write_as.descriptor_count = 1;

                    write_as
                },
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[*vk::DescriptorImageInfo::builder()
                        .image_view(storage_image.view)
                        .image_layout(vk::ImageLayout::GENERAL)]),
            ],
            &[],
        );
    }

    // Create pipelines

    let pipeline_loader = RayTracingPipelineLoader::new(&instance, &device);

    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&[descriptor_set_layout])
                .push_constant_ranges(&[*vk::PushConstantRange::builder()
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                    .size(std::mem::size_of::<CameraProperties>() as u32)]),
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
            include_bytes!("shaders/miss.rmiss.spv"),
            vk::ShaderStageFlags::MISS_KHR,
            &device,
        )?,
        load_shader_module(
            include_bytes!("shaders/closesthit.rchit.spv"),
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            &device,
        )?,
    ];

    let shader_groups = [
        *vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        *vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(1)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        *vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .closest_hit_shader(2)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
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

    let ray_tracing_props = {
        let mut ray_tracing_props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();

        let mut device_props_2 =
            vk::PhysicalDeviceProperties2::builder().push_next(&mut ray_tracing_props);

        unsafe { instance.get_physical_device_properties2(physical_device, &mut device_props_2) }

        ray_tracing_props
    };

    // Copied from:
    // https://github.com/SaschaWillems/Vulkan/blob/eb11297312a164d00c60b06048100bac1d780bb4/base/VulkanTools.cpp#L383
    fn aligned_size(value: u32, alignment: u32) -> u32 {
        (value + alignment - 1) & !(alignment - 1)
    }

    let handle_size = ray_tracing_props.shader_group_handle_size as usize;
    let handle_size_aligned = aligned_size(
        ray_tracing_props.shader_group_handle_size,
        ray_tracing_props.shader_group_handle_alignment,
    );
    let num_groups = shader_groups.len() as u32;
    let shader_binding_table_size = num_groups * handle_size_aligned;

    // We have to fetch the shader group handles as bytes ... then upload them as buffers.
    // Okay. FIne. Sure.

    let group_handles = unsafe {
        pipeline_loader.get_ray_tracing_shader_group_handles(
            pipeline,
            0,
            num_groups,
            shader_binding_table_size as usize,
        )
    }?;

    let alignment = ray_tracing_props.shader_group_base_alignment as u64;
    let handle_size_aligned = handle_size_aligned as u64;

    let shader_binding_tables = [
        ShaderBindingTable::new(
            &group_handles[..handle_size],
            "raygen shader binding table",
            alignment,
            handle_size_aligned,
            &device,
            &mut allocator,
        )?,
        ShaderBindingTable::new(
            &group_handles[handle_size..handle_size * 2],
            "miss shader binding table",
            alignment,
            handle_size_aligned,
            &device,
            &mut allocator,
        )?,
        ShaderBindingTable::new(
            &group_handles[handle_size * 2..],
            "closest hit shader binding table",
            alignment,
            handle_size_aligned,
            &device,
            &mut allocator,
        )?,
    ];

    // Create swapchain

    let surface_caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }?;
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    let mut swapchain_info = *vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain_loader = SwapchainLoader::new(&instance, &device);
    let mut swapchain = Swapchain::new(swapchain_loader.clone(), swapchain_info)?;

    // main loop

    let syncronisation = Syncronisation::new(&device);

    let mut perspective_matrix = ultraviolet::projection::perspective_infinite_z_vk(
        59.0_f32.to_radians(),
        extent.width as f32 / extent.height as f32,
        0.1,
    );

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(size) => {
                extent.width = size.width as u32;
                extent.height = size.height as u32;

                perspective_matrix = ultraviolet::projection::perspective_infinite_z_vk(
                    59.0_f32.to_radians(),
                    extent.width as f32 / extent.height as f32,
                    0.1,
                );

                unsafe {
                    device.device_wait_idle().unwrap();

                    device
                        .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                        .unwrap();
                }

                // Free the old storage image, create a new one and write it to the descriptor set.

                storage_image.cleanup(&mut allocator).unwrap();

                storage_image = Image::new_storage_image(
                    extent.width,
                    extent.height,
                    surface_format.format,
                    command_buffer,
                    queue,
                    &mut allocator,
                )
                .unwrap();

                unsafe {
                    device.update_descriptor_sets(
                        &[*vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&[*vk::DescriptorImageInfo::builder()
                                .image_view(storage_image.view)
                                .image_layout(vk::ImageLayout::GENERAL)])],
                        &[],
                    );
                }

                // Replace the swapchain.

                swapchain_info.image_extent = extent;

                unsafe {
                    swapchain.cleanup();
                }

                swapchain = Swapchain::new(swapchain_loader.clone(), swapchain_info).unwrap();
            }
            _ => {}
        },
        Event::MainEventsCleared => {
            match unsafe {
                swapchain.swapchain_loader.acquire_next_image(
                    swapchain.swapchain,
                    u64::MAX,
                    syncronisation.present_complete_semaphore,
                    vk::Fence::null(),
                )
            } {
                Ok((image_index, _suboptimal)) => {
                    let swapchain_image = swapchain.images[image_index as usize];

                    unsafe {
                        device
                            .wait_for_fences(&[syncronisation.draw_commands_fence], true, u64::MAX)
                            .unwrap();

                        device
                            .reset_fences(&[syncronisation.draw_commands_fence])
                            .unwrap();

                        device
                            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                            .unwrap();

                        device
                            .begin_command_buffer(
                                command_buffer,
                                &vk::CommandBufferBeginInfo::builder()
                                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                            )
                            .unwrap();

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::RAY_TRACING_KHR,
                            pipeline,
                        );

                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::RAY_TRACING_KHR,
                            pipeline_layout,
                            0,
                            &[descriptor_set],
                            &[],
                        );

                        let view_matrix =
                            Mat4::look_at(Vec3::new(0.0, 2.0, -5.0), Vec3::zero(), Vec3::unit_y());

                        device.cmd_push_constants(
                            command_buffer,
                            pipeline_layout,
                            vk::ShaderStageFlags::RAYGEN_KHR,
                            0,
                            bytemuck::bytes_of(&CameraProperties {
                                view_inverse: view_matrix.inversed(),
                                proj_inverse: perspective_matrix.inversed(),
                            }),
                        );

                        pipeline_loader.cmd_trace_rays(
                            command_buffer,
                            &shader_binding_tables[0].address_region,
                            &shader_binding_tables[1].address_region,
                            &shader_binding_tables[2].address_region,
                            // We don't use callable shaders here
                            &Default::default(),
                            extent.width,
                            extent.height,
                            1,
                        );

                        let subresource = *vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1);

                        let subresource_range = vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1);

                        // prepare image to be a destimation
                        set_image_layout(
                            &device,
                            command_buffer,
                            swapchain_image,
                            vk::ImageLayout::UNDEFINED,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            *subresource_range,
                        );

                        // prepare storage image to be a source
                        set_image_layout(
                            &device,
                            command_buffer,
                            storage_image.image,
                            vk::ImageLayout::GENERAL,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            *subresource_range,
                        );

                        device.cmd_copy_image(
                            command_buffer,
                            storage_image.image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            swapchain_image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[*vk::ImageCopy::builder()
                                .src_subresource(subresource)
                                .dst_subresource(subresource)
                                .extent(vk::Extent3D {
                                    width: extent.width,
                                    height: extent.height,
                                    depth: 1,
                                })],
                        );

                        // Reset image layouts

                        set_image_layout(
                            &device,
                            command_buffer,
                            swapchain_image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::ImageLayout::PRESENT_SRC_KHR,
                            *subresource_range,
                        );

                        set_image_layout(
                            &device,
                            command_buffer,
                            storage_image.image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            vk::ImageLayout::GENERAL,
                            *subresource_range,
                        );

                        device.end_command_buffer(command_buffer).unwrap();

                        device
                            .queue_submit(
                                queue,
                                &[*vk::SubmitInfo::builder()
                                    .wait_semaphores(&[syncronisation.present_complete_semaphore])
                                    .wait_dst_stage_mask(&[
                                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                                    ])
                                    .command_buffers(&[command_buffer])
                                    .signal_semaphores(&[
                                        syncronisation.rendering_complete_semaphore
                                    ])],
                                syncronisation.draw_commands_fence,
                            )
                            .unwrap();

                        swapchain
                            .swapchain_loader
                            .queue_present(
                                queue,
                                &vk::PresentInfoKHR::builder()
                                    .wait_semaphores(&[syncronisation.rendering_complete_semaphore])
                                    .swapchains(&[swapchain.swapchain])
                                    .image_indices(&[image_index]),
                            )
                            .expect("Presenting failed. This is very unlikely to happen.");
                    }
                }
                Err(error) => println!("Next frame error: {:?}", error),
            }
        }
        Event::LoopDestroyed => unsafe {
            device.device_wait_idle().unwrap();

            tlas.buffer.cleanup(&mut allocator).unwrap();
            blas.buffer.cleanup(&mut allocator).unwrap();
            storage_image.cleanup(&mut allocator).unwrap();

            for table in &shader_binding_tables {
                table.buffer.cleanup(&mut allocator).unwrap();
            }
        },
        _ => {}
    })
}

struct Syncronisation {
    present_complete_semaphore: vk::Semaphore,
    rendering_complete_semaphore: vk::Semaphore,
    draw_commands_fence: vk::Fence,
}

impl Syncronisation {
    fn new(device: &ash::Device) -> Self {
        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        Self {
            present_complete_semaphore: unsafe { device.create_semaphore(&semaphore_info, None) }
                .unwrap(),
            rendering_complete_semaphore: unsafe { device.create_semaphore(&semaphore_info, None) }
                .unwrap(),
            draw_commands_fence: unsafe { device.create_fence(&fence_info, None) }.unwrap(),
        }
    }

    unsafe fn _cleanup(&self, device: &ash::Device) {
        device.destroy_semaphore(self.present_complete_semaphore, None);
        device.destroy_semaphore(self.rendering_complete_semaphore, None);
        device.destroy_fence(self.draw_commands_fence, None)
    }
}

struct Swapchain {
    swapchain_loader: SwapchainLoader,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
}

impl Swapchain {
    fn new(
        swapchain_loader: SwapchainLoader,
        info: vk::SwapchainCreateInfoKHR,
    ) -> anyhow::Result<Self> {
        let swapchain = unsafe { swapchain_loader.create_swapchain(&info, None) }.unwrap();
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }.unwrap();

        Ok(Self {
            images,
            swapchain,
            swapchain_loader,
        })
    }

    unsafe fn cleanup(&self) {
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
    }
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
struct CameraProperties {
    view_inverse: Mat4,
    proj_inverse: Mat4,
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

    let geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
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

    let scratch_buffer = ScratchBuffer::new(build_sizes.build_scratch_size, allocator)?;

    let blas = AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        "blas",
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        as_loader,
        device,
        allocator,
        command_buffer,
        queue,
        &scratch_buffer,
        geometry_info,
        offset,
    )?;

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
            device_address: instances.device_address(device),
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
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .geometries(geometries);

    let build_sizes = unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &geometry_info,
            &[num_instances],
        )
    };

    scratch_buffer.ensure_size_of(build_sizes.build_scratch_size, allocator)?;

    AccelerationStructure::new(
        build_sizes.acceleration_structure_size,
        "tlas",
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        as_loader,
        device,
        allocator,
        command_buffer,
        queue,
        &scratch_buffer,
        geometry_info,
        offset,
    )
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

fn vertex(a: f32, b: f32, c: f32) -> Vertex {
    Vertex {
        pos: Vec3::new(a, b, c) * 2.0 - Vec3::broadcast(1.0),
    }
}

fn vertices() -> [Vertex; 8] {
    [
        vertex(0.0, 0.0, 0.0),
        vertex(1.0, 0.0, 0.0),
        vertex(0.0, 1.0, 0.0),
        vertex(1.0, 1.0, 0.0),
        vertex(0.0, 0.0, 1.0),
        vertex(1.0, 0.0, 1.0),
        vertex(0.0, 1.0, 1.0),
        vertex(1.0, 1.0, 1.0),
    ]
}

fn indices() -> [u16; 36] {
    [
        0, 1, 2, 2, 1, 3, // bottom
        3, 1, 5, 3, 5, 7, // front
        0, 2, 4, 4, 2, 6, // back
        1, 0, 4, 1, 4, 5, // left
        2, 3, 6, 6, 3, 7, // right
        5, 4, 6, 5, 6, 7, // top
    ]
}
