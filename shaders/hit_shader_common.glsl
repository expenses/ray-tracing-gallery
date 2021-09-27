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

layout(buffer_reference, scalar) buffer ModelInformations {
    ModelInfo buf[];
};

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(push_constant) uniform PushConstantBufferAddresses {
    uint64_t model_info;
    uint64_t uniforms;
    uint64_t acceleration_structure;
} push_constant_buffer_addresses;

Vertex unpack_vertex(PackedVertex packed) {
    Vertex vertex;

    vertex.pos = packed.first_half.xyz;
    vertex.normal = vec3(packed.first_half.w, packed.second_half.xy);
    vertex.uv = packed.second_half.zw;

    return vertex;
}

vec3 compute_barycentric_coords() {
    return vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
}

vec3 interpolate(vec3 a, vec3 b, vec3 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

vec2 interpolate(vec2 a, vec2 b, vec2 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

struct Triangle {
    Vertex a;
    Vertex b;
    Vertex c;
};

Triangle load_triangle(ModelInfo info) {
    uint index_offset = gl_PrimitiveID * 3;

    Indices indices = Indices(info.index_buffer_address);

    uvec3 index = uvec3(
        indices.buf[index_offset],
        indices.buf[index_offset + 1],
        indices.buf[index_offset + 2]
    );

    Vertices vertices = Vertices(info.vertex_buffer_address);

    Triangle triangle;

    triangle.a = unpack_vertex(vertices.buf[index.x]);
    triangle.b = unpack_vertex(vertices.buf[index.y]);
    triangle.c = unpack_vertex(vertices.buf[index.z]);

    return triangle;
}


Vertex interpolate_triangle(Triangle tri, vec3 barycentric_coords) {
    Vertex interpolated;

    interpolated.pos = interpolate(tri.a.pos, tri.b.pos, tri.c.pos, barycentric_coords);
    interpolated.normal = interpolate(tri.a.normal, tri.b.normal, tri.c.normal, barycentric_coords);
    interpolated.uv = interpolate(tri.a.uv, tri.b.uv, tri.c.uv, barycentric_coords);

    return interpolated;
}
