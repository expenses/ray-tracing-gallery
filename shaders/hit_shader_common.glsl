const float PI = 3.141592653589793;

layout(buffer_reference, scalar) buffer Uniforms {
    mat4 view_inverse;
    mat4 proj_inverse;
    vec3 sun_dir;
    float sun_radius;
    uint blue_noise_texture_index;
    uint ggx_lut_texture_index;
    uint frame_index;
    uint8_t show_heatmap;
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

layout(buffer_reference, scalar) buffer Vec3Buffer {
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

uvec3 read_indices(GeometryInfo info) {
    IndexBuffer indices = IndexBuffer(info.index_buffer_address);

    return indices.buf[gl_PrimitiveID];
}

struct Vec3Triangle {
    vec3 a;
    vec3 b;
    vec3 c;
};

vec3 interpolate(Vec3Triangle triangle, vec3 barycentric_coords) {
    return interpolate(triangle.a, triangle.b, triangle.c, barycentric_coords);
}

Vec3Triangle read_vec3_triangle(uint64_t buffer_address, uvec3 indices) {
    Vec3Triangle triangle;

    Vec3Buffer vec3s = Vec3Buffer(buffer_address);

    triangle.a = vec3s.buf[indices.x];
    triangle.b = vec3s.buf[indices.y];
    triangle.c = vec3s.buf[indices.z];
    return triangle;
}

struct Vec2Triangle {
    vec2 a;
    vec2 b;
    vec2 c;
};

vec2 interpolate(Vec2Triangle triangle, vec3 barycentric_coords) {
    return triangle.a * barycentric_coords.x + triangle.b * barycentric_coords.y + triangle.c * barycentric_coords.z;
}

Vec2Triangle read_vec2_triangle(uint64_t buffer_address, uvec3 indices) {
    Vec2Triangle triangle;

    Vec2Buffer vec2s = Vec2Buffer(buffer_address);

    triangle.a = vec2s.buf[indices.x];
    triangle.b = vec2s.buf[indices.y];
    triangle.c = vec2s.buf[indices.z];
    return triangle;
}

struct Vertex {
    vec3 pos;
    vec3 normal;
    vec2 uv;
};

struct Triangle {
    Vec3Triangle positions;
    Vec3Triangle normals;
    Vec2Triangle uvs;
};

// Only use this if you need to interpolate the pos, normal and uv.
Triangle load_triangle(ModelInfo info, GeometryInfo geo_info) {
    uvec3 indices = read_indices(geo_info);

    Triangle triangle;

    triangle.positions = read_vec3_triangle(info.position_buffer_address, indices);
    triangle.normals = read_vec3_triangle(info.normal_buffer_address, indices);
    triangle.uvs = read_vec2_triangle(info.uv_buffer_address, indices);

    return triangle;
}


Vertex interpolate_triangle(Triangle triangle, vec3 barycentric_coords) {
    Vertex vertex;

    vertex.pos = interpolate(triangle.positions, barycentric_coords);
    vertex.normal = interpolate(triangle.normals, barycentric_coords);
    vertex.uv = interpolate(triangle.uvs, barycentric_coords);

    return vertex;

}
