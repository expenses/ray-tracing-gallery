#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference: enable
#extension GL_EXT_scalar_block_layout : enable

#include "hit_shader_common.glsl"
#include "pbr.glsl"

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

struct MaterialData {
    vec3 colour;
    float metallic;
    float roughness;
};

MaterialData read_material_from_textures(GeometryInfo info, vec2 uv) {
    MaterialData data;

    // Textures get blocky without the `nonuniformEXT` here. Thanks again to:
    // https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/blob/596b641a5687307ee9f58193472e8b620ce84189/ray_tracing__advance/shaders/raytrace.rchit#L125
    data.colour = texture(textures[nonuniformEXT(info.diffuse_texture_index)], uv).rgb;

    vec2 metallic_roughness = texture(textures[nonuniformEXT(info.metallic_roughness_texture_index)], uv)
        // This swizzle is the wrong way around for a reason:
        // https://docs.rs/gltf/0.15.2/gltf/material/struct.PbrMetallicRoughness.html#method.metallic_roughness_texture
        .bg;

    data.metallic = metallic_roughness.x;
    data.roughness = metallic_roughness.y;

    return data;
}

// Just in-case we do any non-uniform scaling, we use a normal matrix here.
// This is defined as 'the transpose of the inverse of the upper-left 3x3 part of the model matrix'
//
// See: https://learnopengl.com/Lighting/Basic-Lighting
vec3 rotate_normal_into_world_space(vec3 local_space_normal) {
    vec3 rotated_normal = mat3(gl_WorldToObject3x4EXT) * local_space_normal;
    return normalize(rotated_normal);
}

// Maps 2 randomly generated numbers from 0 to 1 onto a circle with a radius of 1.
//
// See Ray Tracing Gems II, Chapter 24.7.2, Page 381.
vec2 rng_to_circle(vec2 rng) {
    float radius = sqrt(rng.x);
    float angle = rng.y * 2.0 * PI;

    return radius * vec2(cos(angle), sin(angle));
}

// Randomly pick a direction vector that points towards a directional light of a given radius.
//
// See Ray Tracing Gems II, Chapter 24.7.2, Page 381.
vec3 sample_directional_light(vec2 rng, vec3 center_direction, float light_radius) {
    vec2 point = rng_to_circle(rng) * light_radius;

    vec3 tangent = normalize(cross(center_direction, vec3(0.0, 1.0, 0.0)));
    vec3 bitangent = normalize(cross(tangent, center_direction));

    return normalize(center_direction + point.x * tangent + point.y * bitangent);
}

// Sample a random vec2 from a blue noise texture.
//
// See Ray Tracing Gems II, Chapter 24.7.2, Page 381 & 382.
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

// Animate blue noise over time using the golden ratio.
//
// See Ray Tracing Gems II, Chapter 24.7.2, Page 383 & 384.
vec2 animated_blue_noise(vec2 blue_noise, uint frame_index) {
    // The fractional part of the golden ratio
    float golden_ratio_fract = 0.618033988749;
    return fract(blue_noise + float(frame_index % 32) * golden_ratio_fract);
}

vec3 calculate_normal(Vertex interpolated, Triangle triangle, ModelInfo info, GeometryInfo geo_info) {
    // Just rotate the vertex normal into world space if there is no normal map or tangents.
    if (geo_info.normal_map_texture_index < 0 || info.has_tangents == 0) {
        return rotate_normal_into_world_space(interpolated.normal);
    }

    vec3 map_normal = texture(textures[nonuniformEXT(geo_info.normal_map_texture_index)], interpolated.uv).rgb;

    // Convert the map normal into a unit vector
    map_normal = map_normal * 2.0 - vec3(1.0);

    vec3 normal = normalize(interpolated.normal);

    vec4 tangent_and_handedness = read_and_interpolate_tangents(info, read_indices(geo_info), compute_barycentric_coords());
    vec3 tangent = normalize(tangent_and_handedness.xyz);
    float handedness = tangent_and_handedness.w;

    vec3 bitangent = cross(normal, tangent) * handedness;

    mat3 local_space_tbn_matrix = mat3(tangent, bitangent, normal);

    vec3 local_space_normal = local_space_tbn_matrix * normalize(map_normal);

    return rotate_normal_into_world_space(local_space_normal);
}

float cast_shadow_ray(vec3 origin, vec3 direction) {
    float t_min = 0.001;
    float t_max = 10000.0;
    shadow_payload.shadowed = uint8_t(1);
    // Trace shadow ray and offset indices to match shadow hit/miss shader group indices
    traceRayEXT(
        accelerationStructureEXT(push_constant_buffer_addresses.acceleration_structure),
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 1, 0, 1,
        origin, t_min, direction, t_max, 1
    );

    return float(uint8_t(1) - shadow_payload.shadowed);
}

void main() {
    ModelInfos infos = ModelInfos(push_constant_buffer_addresses.model_info);

    ModelInfo info = infos.buf[gl_InstanceCustomIndexEXT];

    GeometryInfos geo_infos = GeometryInfos(info.geometry_info_address);

    GeometryInfo geo_info = geo_infos.buf[gl_GeometryIndexEXT];

    Uniforms uniforms = Uniforms(push_constant_buffer_addresses.uniforms);

    Triangle triangle = load_triangle(info, geo_info);

    vec3 barycentric_coords = compute_barycentric_coords();
    Vertex interpolated = interpolate_triangle(triangle, barycentric_coords);

    // Cast 4 random shadow rays

    float shadow_lighting_sum = 0.0;
    vec3 shadow_origin = get_shadow_terminator_fix_shadow_origin(triangle, interpolated.pos, barycentric_coords);

    for (uint i = 0; i < 2; i++) {
        // Important! If we have this function take the uniforms struct as input, it SEGFAULTs
        // `device.create_shader_module`. Fantastic.
        vec2 blue_noise = animated_blue_noise(sample_blue_noise(uniforms.blue_noise_texture_index, i), uniforms.frame_index);
        vec3 sun_dir = sample_directional_light(blue_noise, uniforms.sun_dir, uniforms.sun_radius);
        shadow_lighting_sum += cast_shadow_ray(shadow_origin, sun_dir);
    }

    float sun_factor = shadow_lighting_sum / 2.0;

    MaterialData material_data = read_material_from_textures(geo_info, interpolated.uv);

    BrdfInputParams params;
    params.normal = calculate_normal(interpolated, triangle, info, geo_info);
    // Important!! This is the negative of the ray direction as it's the
    // direction of output light.
    params.object_to_view = -gl_WorldRayDirectionEXT;
    params.light = uniforms.sun_dir;
    params.base_colour = material_data.colour;
    params.metallic = material_data.metallic;
    params.perceptual_roughness = material_data.roughness;
    // Corresponds to 4% reflectance on non-metallic (dielectric) materials (0.16 * 0.5 * 0.5).
    params.perceptual_dielectric_reflectance = 0.5;
    params.light_intensity = vec3(1.0) * sun_factor;
    params.ggx_lut_texture_index = uniforms.ggx_lut_texture_index;

    // This is simple and not at all physically accurate but it works for now.
    vec3 ambient_light = vec3(0.1);
    vec3 ambient_lighting = ambient_light * material_data.colour;

    primary_payload.colour = brdf(params) + ambient_lighting;
}
