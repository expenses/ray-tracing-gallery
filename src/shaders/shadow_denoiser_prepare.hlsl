[[vk::binding(2, 0)]] RWTexture2D<uint> input_ray_tracer_output;

[[vk::binding(0, 1)]] RWStructuredBuffer<uint> output_shadow_mask;

[[vk::push_constant]] struct PushConstants {
    uint2 buffer_dimensions;
} push_constants;

uint2 FFX_DNSR_Shadows_GetBufferDimensions() {
    return push_constants.buffer_dimensions;
} 

bool FFX_DNSR_Shadows_HitsLight(uint2 dispatch_id, uint2 group_thread_id, uint2 group_id) {
    return input_ray_tracer_output[dispatch_id];
}

void FFX_DNSR_Shadows_WriteMask(uint offset, uint value) {
    output_shadow_mask[offset] = value;
}

#include "../../external/FidelityFX-Denoiser/ffx-shadows-dnsr/ffx_denoiser_shadows_prepare.h"

#define TILE_SIZE_X 8
#define TILE_SIZE_Y 4

[numthreads(TILE_SIZE_X, TILE_SIZE_Y, 1)]
void main(uint2 group_thread_id: SV_GroupThreadID, uint2 group_id: SV_GroupID) {
    FFX_DNSR_Shadows_PrepareShadowMask(group_thread_id, group_id);
}