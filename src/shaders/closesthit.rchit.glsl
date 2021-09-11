#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec2 attribs;

struct Vertex {
    vec3 pos;
    vec3 normal;
};

layout(set = 0, binding = 2) buffer Vertices {
    float vertices[][2];
};

layout(set = 0, binding = 3) buffer Indices {
    uint16_t indices[][2];
};

Vertex load_vertex(uint index, uint model_index) {
    Vertex vertex;
 
    const uint VERTEX_SIZE = 6;

    uint offset = index * VERTEX_SIZE;

    vertex.pos = vec3(vertices[model_index][offset], vertices[model_index][offset + 1], vertices[model_index][offset + 2]);
    vertex.normal = vec3(vertices[model_index][offset + 3], vertices[model_index][offset + 4], vertices[model_index][offset + 5]);

    return vertex;
}

vec3 interpolate(vec3 a, vec3 b, vec3 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

void main()
{
    uint model_index = gl_InstanceCustomIndexEXT;

    uint index_offset = gl_PrimitiveID * 3;

    uvec3 index = uvec3(indices[model_index][index_offset], indices[model_index][index_offset + 1], indices[model_index][index_offset + 2]);

    const vec3 barycentric_coords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
        
    Vertex v0 = load_vertex(index.x, model_index);
    Vertex v1 = load_vertex(index.y, model_index);
    Vertex v2 = load_vertex(index.z, model_index);

    vec3 normal = normalize(interpolate(v0.normal, v1.normal, v2.normal, barycentric_coords));

    hitValue = (normal * 0.5) + 0.5;
}
