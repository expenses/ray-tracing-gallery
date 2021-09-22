use crate::util_functions::{
    cmd_pipeline_image_memory_barrier_explicit, PipelineImageMemoryBarrierParams,
};
use crate::util_structs::{
    AccelerationStructure, Allocator, Buffer, Image, ScratchBuffer, ShaderBindingTable,
};
use ash::extensions::khr::{
    AccelerationStructure as AccelerationStructureLoader,
    RayTracingPipeline as RayTracingPipelineLoader,
};
use ash::vk;

pub struct GlobalResources {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub raygen_sbt: ShaderBindingTable,
    pub closest_hit_sbt: ShaderBindingTable,
    pub miss_sbt: ShaderBindingTable,
    pub general_ds: vk::DescriptorSet,

    pub as_loader: AccelerationStructureLoader,
    pub pipeline_loader: RayTracingPipelineLoader,
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
    pub unsafe fn record(
        &mut self,
        command_buffer: vk::CommandBuffer,
        swapchain_image: vk::Image,
        extent: vk::Extent2D,
        global: &GlobalResources,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> anyhow::Result<()> {
        self.tlas.update_tlas(
            &self.instances_buffer,
            self.num_instances,
            command_buffer,
            allocator,
            &global.as_loader,
            &mut self.scratch_buffer,
        )?;

        // wait for tlas update to finish
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::DependencyFlags::empty(),
            &[*vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR)],
            &[],
            &[],
        );

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

        global.pipeline_loader.cmd_trace_rays(
            command_buffer,
            &global.raygen_sbt.address_region,
            &global.miss_sbt.address_region,
            &global.closest_hit_sbt.address_region,
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

        cmd_pipeline_image_memory_barrier_explicit(&PipelineImageMemoryBarrierParams {
            device,
            buffer: command_buffer,
            // We just wrote the color attachment
            src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            // We need to do a transfer next.
            dst_stage: vk::PipelineStageFlags::TRANSFER,
            image_memory_barriers: &[
                // prepare swapchain image to be a destination
                *vk::ImageMemoryBarrier::builder()
                    .image(swapchain_image)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .subresource_range(*subresource_range)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE),
                // prepare storage image to be a source
                *vk::ImageMemoryBarrier::builder()
                    .image(self.storage_image.image)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .subresource_range(*subresource_range)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
            ],
        });

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

        cmd_pipeline_image_memory_barrier_explicit(&PipelineImageMemoryBarrierParams {
            device,
            buffer: command_buffer,
            // We just did a transfer
            src_stage: vk::PipelineStageFlags::TRANSFER,
            // Nothing happens after this.
            dst_stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            image_memory_barriers: &[
                // Reset swapchain image
                *vk::ImageMemoryBarrier::builder()
                    .image(swapchain_image)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .subresource_range(*subresource_range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE),
                // Reset storage image
                *vk::ImageMemoryBarrier::builder()
                    .image(self.storage_image.image)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .subresource_range(*subresource_range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ),
            ],
        });

        Ok(())
    }
}
