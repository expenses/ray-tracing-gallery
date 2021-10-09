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

    vec3 interpolated_normal = interpolate(read_vec3_triangle(info.normal_buffer_address, indices), compute_barycentric_coords());

    vec3 rotated_normal = mat3(gl_WorldToObject3x4EXT) * interpolated_normal;
    vec3 normal = normalize(rotated_normal);

    primary_payload.new_ray_direction = reflect(gl_WorldRayDirectionEXT, normal);
    primary_payload.new_ray_origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
}
