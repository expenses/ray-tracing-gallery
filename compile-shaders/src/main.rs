use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() {
    let result = SpirvBuilder::new(
        "shaders/ray-tracing",
        "spirv-unknown-spv1.3",
    ).print_metadata(MetadataPrintout::None)
    .extension("SPV_KHR_ray_tracing")
    .capability(spirv_builder::Capability::RayTracingKHR)
    .capability(spirv_builder::Capability::Int8)
    .multimodule(false)
    .build()
    .unwrap();

    std::fs::copy(result.module.unwrap_single(), "shaders/ray-tracing.spv").unwrap();
}
