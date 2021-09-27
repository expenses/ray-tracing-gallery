#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference: enable
#extension GL_EXT_scalar_block_layout : enable

#include "hit_shader_common.glsl"

void main() {
    ModelInformations infos = ModelInformations(push_constant_buffer_addresses.model_info);

    ModelInfo info = infos.buf[gl_InstanceCustomIndexEXT];

    Vertex interpolated = interpolate_triangle(load_triangle(info), compute_barycentric_coords());

    float alpha = texture(textures[nonuniformEXT(info.texture_index)], interpolated.uv).a;

    if (alpha < 0.5) {
        ignoreIntersectionEXT;
    }
}
