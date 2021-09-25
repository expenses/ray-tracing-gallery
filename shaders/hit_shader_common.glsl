
layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 2) uniform Uniforms {
    mat4 view_inverse;
    mat4 proj_inverse;
    vec3 sun_dir;
    float sun_radius;
} uniforms;

struct PrimaryRayPayload {
    vec3 colour;
    vec3 new_ray_origin;
    vec3 new_ray_direction;
};

struct ShadowRayPayload {
    uint8_t shadowed;
};

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 uv;
};

struct PackedVertex {
    vec4 first_half;
    vec4 second_half;
};

layout(location = 0) rayPayloadInEXT PrimaryRayPayload primary_payload;
layout(location = 1) rayPayloadEXT ShadowRayPayload shadow_payload;

hitAttributeEXT vec2 attribs;

// Buffer references are defined in a wierd way, I probably wouldn't have worked them out without:
// https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/blob/596b641a5687307ee9f58193472e8b620ce84189/ray_tracing__advance/shaders/raytrace.rchit#L37-L59

layout(buffer_reference, scalar) buffer Vertices {
    PackedVertex buf[];
};

layout(buffer_reference, scalar) buffer Indices {
    uint16_t buf[];
};

struct ModelInfo {
    uint64_t vertex_buffer_address;
    uint64_t index_buffer_address;
    uint texture_index;
};

layout(set = 0, binding = 0) buffer ModelInformations {
    ModelInfo model_info[];
};

layout(set = 0, binding = 1) uniform sampler2D textures[];

Vertex unpack_vertex(PackedVertex packed) {
    Vertex vertex;

    vertex.pos = packed.first_half.xyz;
    vertex.normal = vec3(packed.first_half.w, packed.second_half.xy);
    vertex.uv = packed.second_half.zw;

    return vertex;
}

vec3 interpolate(vec3 a, vec3 b, vec3 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

vec2 interpolate(vec2 a, vec2 b, vec2 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}
