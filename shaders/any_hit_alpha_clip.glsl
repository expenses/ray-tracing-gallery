#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference: enable
#extension GL_EXT_scalar_block_layout : enable

#include "hit_shader_common.glsl"

void main() {
    ModelInfos infos = ModelInfos(push_constant_buffer_addresses.model_info);

    ModelInfo info = infos.buf[gl_InstanceCustomIndexEXT];

    GeometryInfos geo_infos = GeometryInfos(info.geometry_info_address);

    GeometryInfo geo_info = geo_infos.buf[gl_GeometryIndexEXT];

    uvec3 indices = read_indices(geo_info);

    vec2 interpolated_uv = interpolate(read_vec2_triangle(info.uv_buffer_address, indices), compute_barycentric_coords());

    float alpha = texture(textures[nonuniformEXT(geo_info.diffuse_texture_index)], interpolated_uv).a;

    if (alpha < 0.5) {
        ignoreIntersectionEXT;
    }
}
