[[vk::binding(0, 0)]] RWTexture2D<float4> normals_and_depth_buffer;
[[vk::binding(1, 0)]] RWTexture2D<float4> previous_normals_and_depth_buffer;

[[vk::push_constant]] struct PushConstants {
    //float4x4 ProjectionInverse;
    int2     BufferDimensions;
    float2   InvBufferDimensions;
    //float    DepthSimilaritySigma;
    bool is_first_frame;
    float3 eye;
} push_constants;

int2 FFX_DNSR_Shadows_GetBufferDimensions() {
    return push_constants.BufferDimensions;
}

float2 FFX_DNSR_Shadows_GetInvBufferDimensions() {
    return push_constants.InvBufferDimensions;
}

float FFX_DNSR_Shadows_ReadDepth(int2 p) {
    return normals_and_depth_buffer.Load(p).a;
}

float FFX_DNSR_Shadows_ReadPreviousDepth(int2 p) {
    return previous_normals_and_depth_buffer.Load(p).a;
}

float3 FFX_DNSR_Shadows_ReadNormals(int2 p) {
    return normalize(normals_and_depth_buffer.Load(p).rgb * 2.0 - 1.0);
}

float3 FFX_DNSR_Shadows_GetEye() {
    return push_constants.eye;
}

bool FFX_DNSR_Shadows_IsShadowReciever(uint2 p) {
    float depth = FFX_DNSR_Shadows_ReadDepth(p);
    return (depth > 0.0f) && (depth < 1.0f);
}

// Unimplemented
float2 FFX_DNSR_Shadows_ReadVelocity(int2 p) {
    return float2(0.0);
}


bool FFX_DNSR_Shadows_IsFirstFrame() {
	push_constants.is_first_frame;
}

#include "../../external/FidelityFX-Denoiser/ffx-shadows-dnsr/ffx_denoiser_shadows_tileclassification.h"

[numthreads(8, 8, 1)]
void main(uint group_index : SV_GroupIndex, uint2 gid : SV_GroupID) {
    FFX_DNSR_Shadows_TileClassification(group_index, gid);
}
