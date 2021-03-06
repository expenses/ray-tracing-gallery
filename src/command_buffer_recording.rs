use crate::gpu_structs::unsafe_bytes_of;
use crate::util_structs::{
    AccelerationStructure, Allocator, Buffer, Device, Image, ScratchBuffer, ShaderBindingTable,
};
use ash::vk;
use shared_structs::PushConstantBufferAddresses;

pub struct ShaderBindingTables {
    pub raygen: ShaderBindingTable,
    pub hit: ShaderBindingTable,
    pub miss: ShaderBindingTable,
}

pub struct GlobalResources {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub general_ds: vk::DescriptorSet,
    pub shader_binding_tables: ShaderBindingTables,
    pub model_info_buffer: Buffer,
}

pub struct PerFrameResources {
    pub storage_image: Image,
    pub ray_tracing_ds: vk::DescriptorSet,
    pub ray_tracing_uniforms: Buffer,
    pub tlas: AccelerationStructure,
    pub scratch_buffer: ScratchBuffer,
    pub instances_buffer: Buffer,
    pub num_instances: u32,
}

impl PerFrameResources {
    pub fn resize(
        &mut self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        format: vk::Format,
        index: usize,
        device: &Device,
        allocator: &mut Allocator,
    ) -> anyhow::Result<()> {
        self.storage_image.cleanup(allocator)?;
        self.storage_image = Image::new_storage_image(
            extent.width,
            extent.height,
            &format!("storage image {}", index),
            format,
            command_buffer,
            allocator,
        )?;

        self.write_descriptor_sets(device);

        Ok(())
    }

    pub fn write_descriptor_sets(&self, device: &ash::Device) {
        let storage_image_info = &[self.storage_image.descriptor_image_info()];
        let uniform_buffer_info = &[self.ray_tracing_uniforms.descriptor_buffer_info()];

        let writes = &[
            *vk::WriteDescriptorSet::builder()
                .dst_set(self.ray_tracing_ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(storage_image_info),
            *vk::WriteDescriptorSet::builder()
                .dst_set(self.ray_tracing_ds)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(uniform_buffer_info),
        ];

        unsafe {
            device.update_descriptor_sets(writes, &[]);
        }
    }

    pub unsafe fn record(
        &mut self,
        command_buffer: vk::CommandBuffer,
        device: &Device,
        swapchain_image: vk::Image,
        extent: vk::Extent2D,
        global: &GlobalResources,
    ) -> anyhow::Result<()> {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            global.pipeline,
        );

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            global.pipeline_layout,
            0,
            &[global.general_ds, self.ray_tracing_ds],
            &[],
        );

        device.cmd_push_constants(
            command_buffer,
            global.pipeline_layout,
            vk::ShaderStageFlags::RAYGEN_KHR
                | vk::ShaderStageFlags::ANY_HIT_KHR
                | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            0,
            unsafe_bytes_of(&PushConstantBufferAddresses {
                model_info: global.model_info_buffer.device_address(device),
                uniforms: self.ray_tracing_uniforms.device_address(device),
                acceleration_structure: self.tlas.buffer.device_address(device),
            }),
        );

        device.rt_pipeline_loader.cmd_trace_rays(
            command_buffer,
            &global.shader_binding_tables.raygen.address_region,
            &global.shader_binding_tables.miss.address_region,
            &global.shader_binding_tables.hit.address_region,
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

        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            None,
            &[],
            &[
                vk_sync::ImageBarrier {
                    previous_accesses: &[vk_sync::AccessType::Nothing],
                    next_accesses: &[vk_sync::AccessType::TransferWrite],
                    image: swapchain_image,
                    range: subresource_range,
                    discard_contents: true,
                    ..Default::default()
                },
                vk_sync::ImageBarrier {
                    previous_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                    next_accesses: &[vk_sync::AccessType::TransferRead],
                    previous_layout: vk_sync::ImageLayout::General,
                    image: self.storage_image.image,
                    range: subresource_range,
                    discard_contents: false,
                    ..Default::default()
                },
            ],
        );

        device.cmd_copy_image(
            command_buffer,
            self.storage_image.image,
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

        vk_sync::cmd::pipeline_barrier(
            device,
            command_buffer,
            None,
            &[],
            &[
                vk_sync::ImageBarrier {
                    previous_accesses: &[vk_sync::AccessType::TransferWrite],
                    next_accesses: &[vk_sync::AccessType::Present],
                    image: swapchain_image,
                    range: subresource_range,
                    discard_contents: false,
                    ..Default::default()
                },
                vk_sync::ImageBarrier {
                    previous_accesses: &[vk_sync::AccessType::TransferRead],
                    next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
                    next_layout: vk_sync::ImageLayout::General,
                    image: self.storage_image.image,
                    range: subresource_range,
                    discard_contents: true,
                    ..Default::default()
                },
            ],
        );

        Ok(())
    }
}
