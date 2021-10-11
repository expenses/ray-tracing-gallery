use crate::command_buffer_recording::PerFrameResources;
use crate::gpu_structs::{transpose_matrix_for_instance, AccelerationStructureInstance};
use crate::util_structs::{Allocator, Buffer, ImageManager, Model, ScratchBuffer};
use crate::HitShader;
use ash::vk;
use shared_structs::ModelInfo;
use ultraviolet::{Mat4, Vec3};

// A scene holds both the model buffers and blases, as well as information
// on how to update the instances buffer and tlas.
pub trait Scene {
    fn update(&mut self);

    fn write_resources(
        &self,
        resources: &mut PerFrameResources,
        command_buffer: vk::CommandBuffer,
        allocator: &mut Allocator,
    ) -> anyhow::Result<()>;

    fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()>;
}

pub struct DefaultScene {
    lain_rotation: f32,
    lain_base_transform: Mat4,
    lain_instance_offset: usize,
    plane_model: Model,
    tori_model: Model,
    lain_model: Model,
    fence_model: Model,
}

impl DefaultScene {
    pub fn new(
        command_buffer: vk::CommandBuffer,
        scratch_buffer: &mut ScratchBuffer,
        allocator: &mut Allocator,
        image_manager: &mut ImageManager,
        buffers_to_cleanup: &mut Vec<Buffer>,
    ) -> anyhow::Result<(Self, Vec<AccelerationStructureInstance>, Vec<ModelInfo>)> {
        let mut model_info = Vec::new();

        let scene = Self {
            plane_model: Model::load_gltf(
                include_bytes!("../resources/plane.glb"),
                "plane",
                0,
                allocator,
                image_manager,
                command_buffer,
                scratch_buffer,
                &mut model_info,
                buffers_to_cleanup,
            )?,
            tori_model: Model::load_gltf(
                include_bytes!("../resources/tori.glb"),
                "tori",
                1,
                allocator,
                image_manager,
                command_buffer,
                scratch_buffer,
                &mut model_info,
                buffers_to_cleanup,
            )?,
            lain_model: Model::load_gltf(
                include_bytes!("../resources/lain.glb"),
                "lain",
                1,
                allocator,
                image_manager,
                command_buffer,
                scratch_buffer,
                &mut model_info,
                buffers_to_cleanup,
            )?,
            fence_model: Model::load_gltf(
                include_bytes!("../resources/fence.glb"),
                "fence",
                0,
                allocator,
                image_manager,
                command_buffer,
                scratch_buffer,
                &mut model_info,
                buffers_to_cleanup,
            )?,
            lain_base_transform: Mat4::from_translation(Vec3::new(-2.0, 0.0, -1.0))
                * Mat4::from_scale(0.5),
            lain_rotation: 150.0_f32.to_radians(),
            lain_instance_offset: std::mem::size_of::<AccelerationStructureInstance>() * 2,
        };

        let mut instances = vec![
            AccelerationStructureInstance::new(
                Mat4::from_scale(10.0),
                &scene.plane_model,
                &allocator.device,
                HitShader::Textured,
                false,
            ),
            AccelerationStructureInstance::new(
                Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
                &scene.tori_model,
                &allocator.device,
                HitShader::Textured,
                false,
            ),
            AccelerationStructureInstance::new(
                scene.lain_base_transform * Mat4::from_rotation_y(scene.lain_rotation),
                &scene.lain_model,
                &allocator.device,
                HitShader::Textured,
                false,
            ),
            AccelerationStructureInstance::new(
                Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
                &scene.plane_model,
                &allocator.device,
                HitShader::Portal,
                true,
            ),
            // fence model??
        ];

        {
            use rand::Rng;

            let mut rng = rand::thread_rng();

            for _ in 0..100 {
                instances.push(AccelerationStructureInstance::new(
                    Mat4::from_translation(Vec3::new(
                        rng.gen_range(-10.0..10.0),
                        rng.gen_range(0.5..2.5),
                        rng.gen_range(-10.0..10.0),
                    )) * Mat4::from_rotation_y(rng.gen_range(0.0..100.0))
                        * Mat4::from_scale(rng.gen_range(0.01..0.1)),
                    &scene.tori_model,
                    &allocator.device,
                    if rng.gen() {
                        HitShader::Textured
                    } else {
                        HitShader::Mirror
                    },
                    false,
                ))
            }
        }

        Ok((scene, instances, model_info))
    }
}

impl Scene for DefaultScene {
    fn update(&mut self) {
        self.lain_rotation += 0.05;
    }

    fn write_resources(
        &self,
        resources: &mut PerFrameResources,
        command_buffer: vk::CommandBuffer,
        allocator: &mut Allocator,
    ) -> anyhow::Result<()> {
        let lain_instance_transform = transpose_matrix_for_instance(
            self.lain_base_transform * Mat4::from_rotation_y(self.lain_rotation),
        );

        resources.instances_buffer.write_mapped(
            // The transform is the first 48 bytes of the instance.
            bytemuck::bytes_of(&lain_instance_transform),
            self.lain_instance_offset,
        )?;

        resources.tlas.update_tlas(
            &resources.instances_buffer,
            resources.num_instances,
            command_buffer,
            allocator,
            &mut resources.scratch_buffer,
        )?;

        // wait for tlas update to finish
        unsafe {
            allocator.device.cmd_pipeline_barrier(
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
        }

        Ok(())
    }

    fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()> {
        self.plane_model.cleanup(allocator)?;
        self.tori_model.cleanup(allocator)?;
        self.lain_model.cleanup(allocator)?;
        self.fence_model.cleanup(allocator)?;
        Ok(())
    }
}

pub struct LoadedModelScene {
    model: Model,
}

impl LoadedModelScene {
    pub fn new(
        filename: &str,
        command_buffer: vk::CommandBuffer,
        scratch_buffer: &mut ScratchBuffer,
        allocator: &mut Allocator,
        image_manager: &mut ImageManager,
        buffers_to_cleanup: &mut Vec<Buffer>,
    ) -> anyhow::Result<(Self, Vec<AccelerationStructureInstance>, Vec<ModelInfo>)> {
        let mut model_info = Vec::new();

        let scene = Self {
            model: Model::load_gltf(
                &std::fs::read(filename)?[..],
                filename,
                0,
                allocator,
                image_manager,
                command_buffer,
                scratch_buffer,
                &mut model_info,
                buffers_to_cleanup,
            )?,
        };

        let instances = vec![AccelerationStructureInstance::new(
            Mat4::identity(),
            &scene.model,
            &allocator.device,
            HitShader::Textured,
            false,
        )];

        Ok((scene, instances, model_info))
    }
}

impl Scene for LoadedModelScene {
    fn update(&mut self) {}

    fn write_resources(
        &self,
        _resources: &mut PerFrameResources,
        _command_buffer: vk::CommandBuffer,
        _allocator: &mut Allocator,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()> {
        self.model.cleanup(allocator)
    }
}

pub enum EitherScene<A, B> {
    SceneA(A),
    SceneB(B),
}

impl<A: Scene, B: Scene> Scene for EitherScene<A, B> {
    fn update(&mut self) {
        match self {
            Self::SceneA(a) => a.update(),
            Self::SceneB(b) => b.update(),
        }
    }

    fn write_resources(
        &self,
        resources: &mut PerFrameResources,
        command_buffer: vk::CommandBuffer,
        allocator: &mut Allocator,
    ) -> anyhow::Result<()> {
        match self {
            Self::SceneA(a) => a.write_resources(resources, command_buffer, allocator),
            Self::SceneB(b) => b.write_resources(resources, command_buffer, allocator),
        }
    }

    fn cleanup(&self, allocator: &mut Allocator) -> anyhow::Result<()> {
        match self {
            Self::SceneA(a) => a.cleanup(allocator),
            Self::SceneB(b) => b.cleanup(allocator),
        }
    }
}
