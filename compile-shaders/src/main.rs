use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};

fn main() -> anyhow::Result<()> {
    let extensions = &[
        "SPV_KHR_ray_tracing",
        "SPV_EXT_descriptor_indexing",
        "SPV_KHR_shader_clock",
        "SPV_KHR_physical_storage_buffer",
    ];

    let capabilities = &[
        Capability::RayTracingKHR,
        Capability::Int8,
        Capability::Int64,
        Capability::RuntimeDescriptorArray,
        Capability::ShaderClockKHR,
        Capability::PhysicalStorageBufferAddresses,
    ];

    compile_shader("shaders/ray-tracing", extensions, capabilities)?;

    Ok(())
}

fn compile_shader(
    path: &str,
    extensions: &[&str],
    capabilities: &[Capability],
) -> anyhow::Result<()> {
    let mut builder = SpirvBuilder::new(path, "spirv-unknown-spv1.4")
        .print_metadata(MetadataPrintout::None)
        .multimodule(false);

    for extension in extensions {
        builder = builder.extension(*extension);
    }

    for capability in capabilities {
        builder = builder.capability(*capability);
    }

    let result = builder.build()?;

    std::fs::copy(result.module.unwrap_single(), &format!("{}.spv", path))?;

    Ok(())
}

fn compile_shader_debug(
    path: &str,
    extensions: &[&str],
    capabilities: &[Capability],
) -> anyhow::Result<()> {
    let mut builder = SpirvBuilder::new(path, "spirv-unknown-spv1.4")
        .print_metadata(MetadataPrintout::None)
        .multimodule(true);

    for extension in extensions {
        builder = builder.extension(*extension);
    }

    for capability in capabilities {
        builder = builder.capability(*capability);
    }

    let result = builder.build()?;

    for (name, path) in result.module.unwrap_multi() {
        std::fs::copy(path, &format!("shaders/{}_debug.spv", name))?;
    }

    Ok(())
}
