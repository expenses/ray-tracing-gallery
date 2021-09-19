#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_buffer_reference: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_scalar_block_layout : enable

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 1) rayPayloadEXT bool shadowed;

hitAttributeEXT vec2 attribs;

#include "common.glsl"

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 uv;
};

struct ModelInfo {
    uint64_t vertex_buffer_address;
    uint64_t index_buffer_address;
    uint texture_index;
};

// Buffer references are defined in a wierd way, I probably wouldn't have worked them out without:
// https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/blob/596b641a5687307ee9f58193472e8b620ce84189/ray_tracing__advance/shaders/raytrace.rchit#L37-L59

layout(buffer_reference, scalar) buffer Vertices {
    float inner[];
};

layout(buffer_reference, scalar) buffer Indices {
    uint16_t inner[];
};

layout(set = 0, binding = 0) buffer ModelInformations {
    ModelInfo model_info[];
};

layout(set = 0, binding = 1) uniform sampler2D textures[];

Vertex load_vertex(uint index, ModelInfo info) {
    Vertex vertex;
 
    Vertices vertices = Vertices(info.vertex_buffer_address);

    const uint VERTEX_SIZE = 8;

    uint offset = index * VERTEX_SIZE;

    vertex.pos = vec3(vertices.inner[offset], vertices.inner[offset + 1], vertices.inner[offset + 2]);
    vertex.normal = vec3(vertices.inner[offset + 3], vertices.inner[offset + 4], vertices.inner[offset + 5]);
    vertex.uv = vec2(vertices.inner[offset + 6], vertices.inner[offset + 7]);

    return vertex;
}

vec3 interpolate(vec3 a, vec3 b, vec3 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

vec2 interpolate(vec2 a, vec2 b, vec2 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

vec3 compute_vector_and_project_onto_tangent_plane(vec3 point, Vertex vert) {
    vec3 vector_to_point = point - vert.pos;

    float dot = min(0.0, dot(vector_to_point, vert.normal));

    return vector_to_point - (dot * vert.normal);
}

// Ray Tracing Gems II, Chapter 4.3
// https://link.springer.com/content/pdf/10.1007%2F978-1-4842-7185-8.pdf
vec3 get_shadow_terminator_fix_shadow_origin(Vertex a, Vertex b, Vertex c, vec3 barycentric_coords) {
    // Get the model-space intersection point
    vec3 point = interpolate(a.pos, b.pos, c.pos, barycentric_coords);

    // Get the 3 offset for the points
    vec3 offset_a = compute_vector_and_project_onto_tangent_plane(point, a);
    vec3 offset_b = compute_vector_and_project_onto_tangent_plane(point, b);
    vec3 offset_c = compute_vector_and_project_onto_tangent_plane(point, c);

    // Interpolate an offset
    vec3 interpolated_offset = interpolate(offset_a, offset_b, offset_c, barycentric_coords);

    // Add the offset to the point and project into world space.
    return vec3(gl_ObjectToWorldEXT * vec4(point + interpolated_offset, 1.0));
}

void main() {
    uint model_index = gl_InstanceCustomIndexEXT;
    ModelInfo info = model_info[model_index];

    Indices indices = Indices(info.index_buffer_address);

    uint index_offset = gl_PrimitiveID * 3;

    uvec3 index = uvec3(indices.inner[index_offset], indices.inner[index_offset + 1], indices.inner[index_offset + 2]);

    const vec3 barycentric_coords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    
    Vertex a = load_vertex(index.x, info);
    Vertex b = load_vertex(index.y, info);
    Vertex c = load_vertex(index.z, info);

    vec3 interpolated_normal = interpolate(a.normal, b.normal, c.normal, barycentric_coords);

    // Just in-case we do any non-uniform scaling, we use a normal matrix here.
    // This is defined as 'the transpose of the inverse of the upper-left 3x3 part of the model matrix'
    //
    // See: https://learnopengl.com/Lighting/Basic-Lighting
    vec3 rotated_normal = mat3(gl_WorldToObject3x4EXT) * interpolated_normal;
    vec3 normal = normalize(rotated_normal);

    vec2 uv = interpolate(a.uv, b.uv, c.uv, barycentric_coords);

    vec3 shadow_origin = get_shadow_terminator_fix_shadow_origin(a, b, c, barycentric_coords);

    // Textures get blocky without the `nonuniformEXT` here. Thanks again to:
    // https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/blob/596b641a5687307ee9f58193472e8b620ce84189/ray_tracing__advance/shaders/raytrace.rchit#L125
    vec3 colour = texture(textures[nonuniformEXT(info.texture_index)], uv).rgb;

    float lighting = max(dot(normal, uniforms.sun_dir), 0.0);

    // Shadow casting
	float tmin = 0.001;
	float tmax = 10000.0;
	shadowed = true;  
    // Trace shadow ray and offset indices to match shadow hit/miss shader group indices
	traceRayEXT(
        topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 1, 0, 1,
        shadow_origin, tmin, uniforms.sun_dir, tmax, 1
    );
	
    lighting *= float(!shadowed);

    hitValue = colour;

    hitValue *= (lighting * 0.6) + 0.4;
}
