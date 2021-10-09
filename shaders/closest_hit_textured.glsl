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

void main() {
    ModelInfos infos = ModelInfos(push_constant_buffer_addresses.model_info);

    ModelInfo info = infos.buf[gl_InstanceCustomIndexEXT];

    GeometryInfos geo_infos = GeometryInfos(info.geometry_info_address);

    GeometryInfo geo_info = geo_infos.buf[gl_GeometryIndexEXT];

    Uniforms uniforms = Uniforms(push_constant_buffer_addresses.uniforms);

    Triangle triangle = load_triangle(info, geo_info);

    vec3 barycentric_coords = compute_barycentric_coords();
    Vertex interpolated = interpolate_triangle(triangle, barycentric_coords);

    // Shadow casting
    vec3 shadow_origin = get_shadow_terminator_fix_shadow_origin(triangle, interpolated.pos, barycentric_coords);
    // shadow origin without the fix:
    //shadow_origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Important! If we have this function take the uniforms struct as input, it SEGFAULTs
    // `device.create_shader_module`. Fantastic.
    float sun_factor = sun_factor(shadow_origin, uniforms.sun_dir);

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

    // float ambient_lighting = 0.1;
    primary_payload.colour = brdf(params);
}
