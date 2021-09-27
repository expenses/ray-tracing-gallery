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

    vec3 rotated_normal = mat3(gl_WorldToObject3x4EXT) * interpolated.normal;
    vec3 normal = normalize(rotated_normal);

    primary_payload.new_ray_direction = reflect(gl_WorldRayDirectionEXT, normal);
    primary_payload.new_ray_origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
}
