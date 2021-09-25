#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference: enable
#extension GL_EXT_scalar_block_layout : enable

#include "closest_hit_common.glsl"

void main() {
    vec3 portal_relative_position = vec3(0.0, 5.0, 0.0);

    primary_payload.new_ray_direction = gl_WorldRayDirectionEXT;
    primary_payload.new_ray_origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT + portal_relative_position;
}
