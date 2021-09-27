#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference: enable
#extension GL_EXT_scalar_block_layout : enable

#include "hit_shader_common.glsl"

vec3 compute_vector_and_project_onto_tangent_plane(vec3 point, Vertex vert) {
    vec3 vector_to_point = point - vert.pos;

    float dot_product = min(0.0, dot(vector_to_point, vert.normal));

    return vector_to_point - (dot_product * vert.normal);
}

// Ray Tracing Gems II, Chapter 4.3
// https://link.springer.com/content/pdf/10.1007%2F978-1-4842-7185-8.pdf
//
// `interpolated_point` is the model-space intersection point
vec3 get_shadow_terminator_fix_shadow_origin(Triangle tri, vec3 interpolated_point, vec3 barycentric_coords) {
    // Get the 3 offset for the points
    vec3 offset_a = compute_vector_and_project_onto_tangent_plane(interpolated_point, tri.a);
    vec3 offset_b = compute_vector_and_project_onto_tangent_plane(interpolated_point, tri.b);
    vec3 offset_c = compute_vector_and_project_onto_tangent_plane(interpolated_point, tri.c);

    // Interpolate an offset
    vec3 interpolated_offset = interpolate(offset_a, offset_b, offset_c, barycentric_coords);

    // Add the offset to the point
    vec3 local_space_fixed_point = interpolated_point + interpolated_offset;

    // project into world space.
    return gl_ObjectToWorldEXT * vec4(local_space_fixed_point, 1.0);
}

void main() {
    ModelInformations infos = ModelInformations(push_constant_buffer_addresses.model_info);

    ModelInfo info = infos.buf[gl_InstanceCustomIndexEXT];

    Uniforms uniforms = Uniforms(push_constant_buffer_addresses.uniforms);

    Triangle triangle = load_triangle(info);

    vec3 barycentric_coords = compute_barycentric_coords();
    Vertex interpolated = interpolate_triangle(triangle, barycentric_coords);

    // Just in-case we do any non-uniform scaling, we use a normal matrix here.
    // This is defined as 'the transpose of the inverse of the upper-left 3x3 part of the model matrix'
    //
    // See: https://learnopengl.com/Lighting/Basic-Lighting
    vec3 rotated_normal = mat3(gl_WorldToObject3x4EXT) * interpolated.normal;
    vec3 normal = normalize(rotated_normal);

    vec3 shadow_origin = get_shadow_terminator_fix_shadow_origin(triangle, interpolated.pos, barycentric_coords);

    // Textures get blocky without the `nonuniformEXT` here. Thanks again to:
    // https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/blob/596b641a5687307ee9f58193472e8b620ce84189/ray_tracing__advance/shaders/raytrace.rchit#L125
    vec3 colour = texture(textures[nonuniformEXT(info.texture_index)], interpolated.uv).rgb;

    float lighting = max(dot(normal, uniforms.sun_dir), 0.0);

    // Shadow casting
	float t_min = 0.001;
	float t_max = 10000.0;
	shadow_payload.shadowed = uint8_t(1);
    // Trace shadow ray and offset indices to match shadow hit/miss shader group indices
	traceRayEXT(
        accelerationStructureEXT(push_constant_buffer_addresses.acceleration_structure),
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 1, 0, 1,
        shadow_origin, t_min, uniforms.sun_dir, t_max, 1
    );
	
    lighting *= float(uint8_t(1) - shadow_payload.shadowed);

    primary_payload.colour = colour * ((lighting * 0.6) + 0.4);
}
