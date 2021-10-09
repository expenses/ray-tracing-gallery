layout(buffer_reference, scalar) buffer Uniforms {
    mat4 view_inverse;
    mat4 proj_inverse;
    vec3 sun_dir;
    float sun_radius;
};

struct PrimaryRayPayload {
    vec3 colour;
    vec3 new_ray_origin;
    vec3 new_ray_direction;
};

struct ShadowRayPayload {
    uint8_t shadowed;
};

layout(location = 0) rayPayloadInEXT PrimaryRayPayload primary_payload;
layout(location = 1) rayPayloadEXT ShadowRayPayload shadow_payload;

hitAttributeEXT vec2 attribs;

// Buffer references are defined in a wierd way, I probably wouldn't have worked them out without:
// https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/blob/596b641a5687307ee9f58193472e8b620ce84189/ray_tracing__advance/shaders/raytrace.rchit#L37-L59

layout(buffer_reference, scalar) buffer FloatBuffer {
    vec3 buf[];
};

layout(buffer_reference, scalar) buffer Vec2Buffer {
    vec2 buf[];
};

layout(buffer_reference, scalar) buffer IndexBuffer {
    uvec3 buf[];
};

struct ModelInfo {
    uint64_t position_buffer_address;
    uint64_t normal_buffer_address;
    uint64_t uv_buffer_address;
    uint64_t geometry_info_address;
};

layout(buffer_reference, scalar) buffer ModelInfos {
    ModelInfo buf[];
};

struct GeometryInfo {
    uint64_t index_buffer_address;
    uint diffuse_texture_index;
    uint metallic_roughness_texture_index;
    int normal_map_texture_index;
};

layout(buffer_reference, scalar) buffer GeometryInfos {
    GeometryInfo buf[];
};

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(push_constant) uniform PushConstantBufferAddresses {
    uint64_t model_info;
    uint64_t uniforms;
    uint64_t acceleration_structure;
} push_constant_buffer_addresses;

vec3 compute_barycentric_coords() {
    return vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
}

vec3 interpolate(vec3 a, vec3 b, vec3 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

vec2 interpolate(vec2 a, vec2 b, vec2 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

uvec3 read_indices(GeometryInfo info) {
    IndexBuffer indices = IndexBuffer(info.index_buffer_address);

    return indices.buf[gl_PrimitiveID];
}

vec2 read_vec2(uint64_t buffer_address, uint index) {
    Vec2Buffer vec2s = Vec2Buffer(buffer_address);
    return vec2s.buf[index];
}

vec3 read_vec3(uint64_t buffer_address, uint index) {
    FloatBuffer floats = FloatBuffer(buffer_address);
    return floats.buf[index];
}

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 uv;
};

Vertex load_vertex(ModelInfo info, uint index) {
    Vertex vertex;

    vertex.pos = read_vec3(info.position_buffer_address, index);
    vertex.normal = read_vec3(info.normal_buffer_address, index);
    vertex.uv = read_vec2(info.uv_buffer_address, index);

    return vertex;
}

struct Triangle {
    Vertex a;
    Vertex b;
    Vertex c;
};

// Only use this if you need to interpolate the pos, normal and uv.
Triangle load_triangle(ModelInfo info, GeometryInfo geo_info) {
    uvec3 indices = read_indices(geo_info);

    Triangle triangle;

    triangle.a = load_vertex(info, indices.x);
    triangle.b = load_vertex(info, indices.y);
    triangle.c = load_vertex(info, indices.z);

    return triangle;
}

Vertex interpolate_triangle(Triangle triangle, vec3 barycentric_coords) {
    Vertex vertex;

    vertex.pos = interpolate(triangle.a.pos, triangle.b.pos, triangle.c.pos, barycentric_coords);
    vertex.normal = interpolate(triangle.a.normal, triangle.b.normal, triangle.c.normal, barycentric_coords);
    vertex.uv = interpolate(triangle.a.uv, triangle.b.uv, triangle.c.uv, barycentric_coords);

    return vertex;

}
