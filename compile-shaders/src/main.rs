use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() {
    let result = SpirvBuilder::new("shaders/ray-tracing", "spirv-unknown-spv1.4")
        .print_metadata(MetadataPrintout::None)
        .extension("SPV_KHR_ray_tracing")
        .extension("SPV_EXT_descriptor_indexing")
        .capability(spirv_builder::Capability::RayTracingKHR)
        .capability(spirv_builder::Capability::Int8)
        .capability(spirv_builder::Capability::Int64)
        .capability(spirv_builder::Capability::RuntimeDescriptorArray)
        .multimodule(false)
        .build()
        .unwrap();

    std::fs::copy(result.module.unwrap_single(), "shaders/ray-tracing.spv").unwrap();
}
