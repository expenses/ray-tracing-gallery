#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference: enable
#extension GL_EXT_scalar_block_layout : enable

#include "hit_shader_common.glsl"

// For `get_shadow_terminator_fix_shadow_origin`.
vec3 compute_vector_and_project_onto_tangent_plane(vec3 point, vec3 vertex_pos, vec3 vertex_normal) {
    vec3 vector_to_point = point - vertex_pos;

    float dot_product = min(0.0, dot(vector_to_point, vertex_normal));

    return vector_to_point - (dot_product * vertex_normal);
}

// Ray Tracing Gems II, Chapter 4.3
// https://link.springer.com/content/pdf/10.1007%2F978-1-4842-7185-8.pdf
//
// `interpolated_point` is the model-space intersection point
vec3 get_shadow_terminator_fix_shadow_origin(Triangle tri, vec3 interpolated_point, vec3 barycentric_coords) {
    // Get the 3 offset for the points
    vec3 offset_a = compute_vector_and_project_onto_tangent_plane(interpolated_point, tri.positions.a, tri.normals.a);
    vec3 offset_b = compute_vector_and_project_onto_tangent_plane(interpolated_point, tri.positions.b, tri.normals.b);
    vec3 offset_c = compute_vector_and_project_onto_tangent_plane(interpolated_point, tri.positions.c, tri.normals.c);

    // Interpolate an offset
    vec3 interpolated_offset = interpolate(offset_a, offset_b, offset_c, barycentric_coords);

    // Add the offset to the point
    vec3 local_space_fixed_point = interpolated_point + interpolated_offset;

    // project into world space.
    return gl_ObjectToWorldEXT * vec4(local_space_fixed_point, 1.0);
}

// Maps 2 randomly generated numbers from 0 to 1 onto a circle with a radius of 1.
vec2 rng_to_circle(vec2 rng) {
    float radius = sqrt(rng.x);
    float angle = rng.y * 2.0 * PI;

    return radius * vec2(cos(angle), sin(angle));
}

vec3 sun_dir_equation(vec2 rng, vec3 uniform_sun_dir, float uniform_sun_radius) {
    vec2 point = rng_to_circle(rng) * uniform_sun_radius;

    vec3 tangent = normalize(cross(uniform_sun_dir, vec3(0.0, 1.0, 0.0)));
    vec3 bitangent = normalize(cross(tangent, uniform_sun_dir));

    return normalize(uniform_sun_dir + point.x * tangent + point.y * bitangent);
}

vec2 animated_blue_noise(vec2 blue_noise, uint frame_index) {
    // The fractional part of the golden ratio
    float golden_ratio_fract = 0.618033988749;
    return fract(blue_noise + float(frame_index % 32) * golden_ratio_fract);
}

float sun_factor(vec3 shadow_origin, vec3 sun_dir) {
    float t_min = 0.001;
    float t_max = 10000.0;
    shadow_payload.shadowed = uint8_t(1);
    // Trace shadow ray and offset indices to match shadow hit/miss shader group indices
    traceRayEXT(
        accelerationStructureEXT(push_constant_buffer_addresses.acceleration_structure),
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 1, 0, 1,
        shadow_origin, t_min, sun_dir, t_max, 1
    );
    return float(1 - shadow_payload.shadowed);
}

vec2 sample_blue_noise(uint blue_noise_texture_index, uint iteration) {
    uvec2 coord = gl_LaunchIDEXT.xy;
    uvec2 offset = uvec2(13, 41);
    vec2 texture_size = vec2(64.0);

    uvec2 first_offset = iteration * 2 * offset;
    uvec2 second_offset = (iteration * 2 + 1) * offset;

    return vec2(
        texture(textures[blue_noise_texture_index], (coord + first_offset) / texture_size).r,
        texture(textures[blue_noise_texture_index], (coord + second_offset) / texture_size).r
    );
}

void main() {
    ModelInfos infos = ModelInfos(push_constant_buffer_addresses.model_info);

    ModelInfo info = infos.buf[gl_InstanceCustomIndexEXT];

    GeometryInfos geo_infos = GeometryInfos(info.geometry_info_address);

    GeometryInfo geo_info = geo_infos.buf[gl_GeometryIndexEXT];

    Uniforms uniforms = Uniforms(push_constant_buffer_addresses.uniforms);

    vec2 blue_noise = animated_blue_noise(sample_blue_noise(uniforms.blue_noise_texture_index, 0), uniforms.frame_index);

    Triangle triangle = load_triangle(info, geo_info);

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
    vec3 colour = texture(textures[nonuniformEXT(geo_info.texture_index)], interpolated.uv).rgb;

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

    float ambient_lighting = 0.1;
    primary_payload.colour = colour * ((lighting * (1.0 - ambient_lighting)) + ambient_lighting);
}
