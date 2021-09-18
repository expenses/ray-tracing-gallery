[[vk::binding(0, 0)]] RWTexture2D<float> t2d_DepthBuffer;
[[vk::binding(1, 0)]] RWTexture2D<float3> t2d_NormalBuffer;
[[vk::binding(2, 0)]] StructuredBuffer<uint> sb_tileMetaData;

[[vk::binding(3, 0)]] RWTexture2D<float2> rqt2d_input;

[[vk::binding(4, 0)]] RWTexture2D<float2> rwt2d_history;
[[vk::binding(5, 0)]] RWTexture2D<unorm float4> rwt2d_output;

[[vk::push_constant]] struct PushConstants {
    float4x4 ProjectionInverse;
    int2     BufferDimensions;
    float2   InvBufferDimensions;
    float    DepthSimilaritySigma;
} push_constants;

float2 FFX_DNSR_Shadows_GetInvBufferDimensions() {
    return push_constants.InvBufferDimensions;
}

int2 FFX_DNSR_Shadows_GetBufferDimensions() {
    return push_constants.BufferDimensions;
}

float4x4 FFX_DNSR_Shadows_GetProjectionInverse() {
    return push_constants.ProjectionInverse;
}

float FFX_DNSR_Shadows_GetDepthSimilaritySigma() {
    return push_constants.DepthSimilaritySigma;
}

float FFX_DNSR_Shadows_ReadDepth(int2 p) {
    return t2d_DepthBuffer.Load(p);
}

float16_t3 FFX_DNSR_Shadows_ReadNormals(int2 p) {
    return normalize(((float16_t3)t2d_NormalBuffer.Load(p)) * 2 - 1.f);
}

bool FFX_DNSR_Shadows_IsShadowReciever(uint2 p) {
    float depth = FFX_DNSR_Shadows_ReadDepth(p);
    return (depth > 0.0f) && (depth < 1.0f);
}

float16_t2 FFX_DNSR_Shadows_ReadInput(int2 p) {
    return (float16_t2)rqt2d_input.Load(p).xy;
}

uint FFX_DNSR_Shadows_ReadTileMetaData(uint p) {
    return sb_tileMetaData[p];
}

#include "../../external/FidelityFX-Denoiser/ffx-shadows-dnsr/ffx_denoiser_shadows_filter.h"

[numthreads(8, 8, 1)]
[shader("compute")]
void Pass0(uint2 gid : SV_GroupID, uint2 gtid : SV_GroupThreadID, uint2 did : SV_DispatchThreadID) {
    const uint PASS_INDEX = 0;
    const uint STEP_SIZE = 1;

    bool bWriteOutput = false;
    float2 const results = FFX_DNSR_Shadows_FilterSoftShadowsPass(gid, gtid, did, bWriteOutput, PASS_INDEX, STEP_SIZE);

    if (bWriteOutput) {
        rwt2d_history[did] = results;
    }
}

[numthreads(8, 8, 1)]
[shader("compute")]
void Pass1(uint2 gid : SV_GroupID, uint2 gtid : SV_GroupThreadID, uint2 did : SV_DispatchThreadID) {
    const uint PASS_INDEX = 1;
    const uint STEP_SIZE = 2;

    bool bWriteOutput = false;
    float2 const results = FFX_DNSR_Shadows_FilterSoftShadowsPass(gid, gtid, did, bWriteOutput, PASS_INDEX, STEP_SIZE);
    if (bWriteOutput) {
        rwt2d_history[did] = results;
    }
}

/*
float ShadowContrastRemapping(float x) {
    const float a = 10.f;
    const float b = -1.0f;
    const float c = 1 / pow(2, a);
    const float d = exp(-b);
    const float e = 1 / (1 / pow((1 + d), a) - c);
    const float m = 1 / pow((1 + pow(d, x)), a) - c;

    return m * e;
}
*/

[numthreads(8, 8, 1)]
[shader("compute")]
void Pass2(uint2 gid : SV_GroupID, uint2 gtid : SV_GroupThreadID, uint2 did : SV_DispatchThreadID): SV_TARGET {
    const uint PASS_INDEX = 2;
    const uint STEP_SIZE = 4;

    bool bWriteOutput = false;
    float2 const results = FFX_DNSR_Shadows_FilterSoftShadowsPass(gid, gtid, did, bWriteOutput, PASS_INDEX, STEP_SIZE);

    // Recover some of the contrast lost during denoising
    const float shadow_remap = max(1.2f - results.y, 1.0f);
    const float mean = pow(abs(results.x), shadow_remap);

    if (bWriteOutput) {
        rwt2d_output[did].x = mean;
    }
}
