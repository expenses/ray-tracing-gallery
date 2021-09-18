use crate::util_functions::{load_shader_module, load_shader_module_as_stage_info};
use ash::vk;
use std::ffi::CStr;
use ultraviolet::{Mat4, Vec2};

pub const TILE_SIZE_X: u32 = 8;
pub const TILE_SIZE_Y: u32 = 4;

pub fn tiles_buffer_size_for_dimensions(width: u32, height: u32) -> vk::DeviceSize {
    (div_round_up(width, TILE_SIZE_X) * div_round_up(height, TILE_SIZE_Y)) as vk::DeviceSize
        * std::mem::size_of::<u32>() as u64
}

pub fn div_round_up(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

struct ShaderStageInfo {
    prepare: vk::PipelineShaderStageCreateInfo,
    pass_0: vk::PipelineShaderStageCreateInfo,
    pass_1: vk::PipelineShaderStageCreateInfo,
    pass_2: vk::PipelineShaderStageCreateInfo,
}

impl ShaderStageInfo {
    pub fn new(device: &ash::Device) -> anyhow::Result<Self> {
        let passes_module = load_shader_module(
            include_bytes!("shaders/shadow_denoiser_passes.hlsl.spv"),
            device,
        )?;

        Ok(Self {
            prepare: load_shader_module_as_stage_info(
                include_bytes!("shaders/shadow_denoiser_prepare.hlsl.spv"),
                vk::ShaderStageFlags::COMPUTE,
                device,
                None,
            )?,
            pass_0: *vk::PipelineShaderStageCreateInfo::builder()
                .module(passes_module)
                .stage(vk::ShaderStageFlags::COMPUTE)
                .name(CStr::from_bytes_with_nul(b"Pass0\0")?),
            pass_1: *vk::PipelineShaderStageCreateInfo::builder()
                .module(passes_module)
                .stage(vk::ShaderStageFlags::COMPUTE)
                .name(CStr::from_bytes_with_nul(b"Pass1\0")?),
            pass_2: *vk::PipelineShaderStageCreateInfo::builder()
                .module(passes_module)
                .stage(vk::ShaderStageFlags::COMPUTE)
                .name(CStr::from_bytes_with_nul(b"Pass2\0")?),
        })
    }
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
pub struct PreparePushConstants {
    pub buffer_dimensions: [u32; 2],
}

#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod, Debug)]
#[repr(C)]
pub struct PassPushConstants {
    pub projection_inverse: Mat4,
    pub buffer_diemensions: [i32; 2],
    pub inv_buffer_dimensions: Vec2,
    pub depth_similarity_sigma: f32,
}

fn push_constant_range<T>() -> vk::PushConstantRange {
    vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::COMPUTE,
        size: std::mem::size_of::<T>() as u32,
        offset: 0,
    }
}

pub struct Pipelines {
    pub prepare_pipeline_layout: vk::PipelineLayout,
    pub prepare_pipeline: vk::Pipeline,
    pub prepare_dsl: vk::DescriptorSetLayout,
    pub passes_pipeline_layout: vk::PipelineLayout,
    pub passes_dsl: vk::DescriptorSetLayout,
    pub pass_0_pipeline: vk::Pipeline,
    pub pass_1_pipeline: vk::Pipeline,
    pub pass_2_pipeline: vk::Pipeline,
}

impl Pipelines {
    pub fn new(
        device: &ash::Device,
        storage_image_dsl: vk::DescriptorSetLayout,
    ) -> anyhow::Result<Self> {
        let shader_stage_info = ShaderStageInfo::new(device)?;

        let storage_buffer = |binding| {
            *vk::DescriptorSetLayoutBinding::builder()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };

        let storage_image = |binding| {
            *vk::DescriptorSetLayoutBinding::builder()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };

        let prepare_dsl = unsafe {
            device.create_descriptor_set_layout(
                &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[storage_buffer(0)]),
                None,
            )
        }?;

        let prepare_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[storage_image_dsl, prepare_dsl])
                    .push_constant_ranges(&[push_constant_range::<PreparePushConstants>()]),
                None,
            )
        }?;

        let passes_dsl = unsafe {
            device.create_descriptor_set_layout(
                &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                    storage_image(0),
                    storage_buffer(1),
                    storage_image(2),
                    storage_image(3),
                    storage_image(4),
                ]),
                None,
            )
        }?;

        let passes_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[passes_dsl])
                    .push_constant_ranges(&[push_constant_range::<PassPushConstants>()]),
                None,
            )
        }?;

        let pipelines = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[
                    *vk::ComputePipelineCreateInfo::builder()
                        .layout(prepare_pipeline_layout)
                        .stage(shader_stage_info.prepare),
                    *vk::ComputePipelineCreateInfo::builder()
                        .layout(passes_pipeline_layout)
                        .stage(shader_stage_info.pass_0),
                    *vk::ComputePipelineCreateInfo::builder()
                        .layout(passes_pipeline_layout)
                        .stage(shader_stage_info.pass_1),
                    *vk::ComputePipelineCreateInfo::builder()
                        .layout(passes_pipeline_layout)
                        .stage(shader_stage_info.pass_2),
                ],
                None,
            )
        }
        .map_err(|(_, error)| error)?;

        Ok(Self {
            prepare_pipeline_layout,
            prepare_pipeline: pipelines[0],
            prepare_dsl,
            pass_0_pipeline: pipelines[1],
            pass_1_pipeline: pipelines[2],
            pass_2_pipeline: pipelines[3],
            passes_dsl,
            passes_pipeline_layout,
        })
    }
}
