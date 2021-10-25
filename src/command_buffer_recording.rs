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
    pub tonemap_render_pass: vk::RenderPass,
    pub tonemap_pipeline: vk::Pipeline,
    pub tonemap_pipeline_layout: vk::PipelineLayout,
}

pub struct PerFrameResources {
    pub storage_image: Image,
    pub ray_tracing_ds: vk::DescriptorSet,
    pub ray_tracing_uniforms: Buffer,
    pub tlas: AccelerationStructure,
    pub scratch_buffer: ScratchBuffer,
    pub instances_buffer: Buffer,
    pub num_instances: u32,
    pub tonemap_ds: vk::DescriptorSet,
}

impl PerFrameResources {
    pub fn resize(
        &mut self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        format: vk::Format,
        index: usize,
        device: &Device,
        allocator: &Allocator,
        sampler: vk::Sampler,
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

        self.write_descriptor_sets(device, sampler);

        Ok(())
    }

    pub fn write_descriptor_sets(&self, device: &ash::Device, sampler: vk::Sampler) {
        let storage_image_info = &[*self.storage_image.descriptor_image_info()];
        let uniform_buffer_info = &[self.ray_tracing_uniforms.descriptor_buffer_info()];
        let storage_image_with_sampler_info =
            &[*self.storage_image.descriptor_image_info().sampler(sampler)];

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
            *vk::WriteDescriptorSet::builder()
                .dst_set(self.tonemap_ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(storage_image_with_sampler_info),
        ];

        unsafe {
            device.update_descriptor_sets(writes, &[]);
        }
    }

    pub unsafe fn record(
        &mut self,
        command_buffer: vk::CommandBuffer,
        device: &Device,
        tonemap_framebuffer: vk::Framebuffer,
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

        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        let viewport = *vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let tonemap_render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(global.tonemap_render_pass)
            .framebuffer(tonemap_framebuffer)
            .render_area(area)
            .clear_values(&[vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }]);

        device.cmd_begin_render_pass(
            command_buffer,
            &tonemap_render_pass_info,
            vk::SubpassContents::INLINE,
        );

        device.cmd_set_scissor(command_buffer, 0, &[area]);
        device.cmd_set_viewport(command_buffer, 0, &[viewport]);

        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            global.tonemap_pipeline,
        );

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            global.tonemap_pipeline_layout,
            0,
            &[self.tonemap_ds],
            &[],
        );

        device.cmd_push_constants(
            command_buffer,
            global.tonemap_pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&colstodian::tonemap::BakedLottesTonemapperParams::from(
                colstodian::tonemap::LottesTonemapperParams {
                    ..Default::default()
                },
            )),
        );

        device.cmd_draw(command_buffer, 3, 1, 0, 0);

        device.cmd_end_render_pass(command_buffer);

        Ok(())
    }
}
