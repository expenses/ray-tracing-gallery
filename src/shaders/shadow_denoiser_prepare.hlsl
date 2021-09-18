[[vk::binding(2, 0)]] RWTexture2D<uint> input_ray_tracer_output;

[[vk::binding(0, 1)]] RWStructuredBuffer<uint> output_shadow_mask;

[[vk::push_constant]] struct PushConstants {
    uint2 buffer_dimensions;
} push_constants;

uint2 FFX_DNSR_Shadows_GetBufferDimensions() {
    return push_constants.buffer_dimensions;
} 

#define TILE_SIZE_X 8
#define TILE_SIZE_Y 4

uint LaneIdToBitShift(uint2 localID) {
    return localID.y * TILE_SIZE_X + localID.x;
}

bool WaveMaskToBool(uint mask, uint2 localID) {
    return bool((1u << LaneIdToBitShift(localID.xy)) & mask);
}

bool FFX_DNSR_Shadows_HitsLight(uint2 did, uint2 gtid, uint2 gid) {
    return !WaveMaskToBool(input_ray_tracer_output[gid], gtid);
}

void FFX_DNSR_Shadows_WriteMask(uint offset, uint value) {
    output_shadow_mask[offset] = value;
}

uint2 clamp(uint2 value, uint min_value, uint2 max_value) {
    return max(min(value, max_value), min_value);
}

#include "../../external/FidelityFX-Denoiser/ffx-shadows-dnsr/ffx_denoiser_shadows_prepare.h"

[numthreads(TILE_SIZE_X, TILE_SIZE_Y, 1)]
void main(uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID) {
    FFX_DNSR_Shadows_PrepareShadowMask(gtid, gid);
}