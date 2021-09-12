#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_buffer_reference: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_scalar_block_layout : enable

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 1) rayPayloadEXT bool shadowed;
hitAttributeEXT vec2 attribs;

layout(set = 0, binding = 3) uniform Uniforms {
    vec3 sun_dir;
};

struct Vertex {
    vec3 pos;
    vec3 normal;
};

struct ModelInfo {
    uint64_t vertex_buffer_address;
    uint64_t index_buffer_address;
};

// Buffer references are defined in a wierd way, I probably wouldn't have worked them out without:
// https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/blob/596b641a5687307ee9f58193472e8b620ce84189/ray_tracing__advance/shaders/raytrace.rchit#L37-L59

layout(buffer_reference, scalar) buffer Vertices {
    float inner[];
};

layout(buffer_reference, scalar) buffer Indices {
    uint16_t inner[];
};

layout(set = 0, binding = 2) buffer ModelInformations {
    ModelInfo model_info[];
};

Vertex load_vertex(uint index, ModelInfo info) {
    Vertex vertex;
 
    Vertices vertices = Vertices(info.vertex_buffer_address);

    const uint VERTEX_SIZE = 6;

    uint offset = index * VERTEX_SIZE;

    vertex.pos = vec3(vertices.inner[offset], vertices.inner[offset + 1], vertices.inner[offset + 2]);
    vertex.normal = vec3(vertices.inner[offset + 3], vertices.inner[offset + 4], vertices.inner[offset + 5]);

    return vertex;
}

vec3 interpolate(vec3 a, vec3 b, vec3 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

void main()
{
    uint model_index = gl_InstanceCustomIndexEXT;
    ModelInfo info = model_info[model_index];

    Indices indices = Indices(info.index_buffer_address);

    uint index_offset = gl_PrimitiveID * 3;

    uvec3 index = uvec3(indices.inner[index_offset], indices.inner[index_offset + 1], indices.inner[index_offset + 2]);

    const vec3 barycentric_coords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    
    Vertex v0 = load_vertex(index.x, info);
    Vertex v1 = load_vertex(index.y, info);
    Vertex v2 = load_vertex(index.z, info);

    vec3 interpolated_normal = interpolate(v0.normal, v1.normal, v2.normal, barycentric_coords);

    vec3 normal = normalize(vec3(interpolated_normal * gl_WorldToObjectEXT));

    float lighting = max(dot(normal, sun_dir), 0.0);

    // Shadow casting
	float tmin = 0.001;
	float tmax = 10000.0;
	vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	shadowed = true;  
    // Trace shadow ray and offset indices to match shadow hit/miss shader group indices
	traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 1, 0, 1, origin, tmin, sun_dir, tmax, 1);
	
    lighting *= float(!shadowed);

    hitValue = (normal * 0.5) + 0.5;

    hitValue *= (lighting * 0.6) + 0.4;
}
