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
    float vertices[];
};

layout(set = 0, binding = 3) buffer Indices {
    uint16_t indices[];
};

Vertex load_vertex(uint index) {
    Vertex vertex;
 
    const uint VERTEX_SIZE = 6;

    uint offset = index * VERTEX_SIZE;

    vertex.pos = vec3(vertices[offset], vertices[offset + 1], vertices[offset + 2]);
    vertex.normal = vec3(vertices[offset + 3], vertices[offset + 4], vertices[offset + 5]);

    return vertex;
}

vec3 interpolate(vec3 a, vec3 b, vec3 c, vec3 barycentric_coords) {
    return a * barycentric_coords.x + b * barycentric_coords.y + c * barycentric_coords.z;
}

void main()
{
    uvec3 index = uvec3(indices[3 * gl_PrimitiveID], indices[3 * gl_PrimitiveID + 1], indices[3 * gl_PrimitiveID + 2]);

    const vec3 barycentric_coords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
        
    Vertex v0 = load_vertex(index.x);
    Vertex v1 = load_vertex(index.y);
    Vertex v2 = load_vertex(index.z);

    vec3 normal = normalize(interpolate(v0.normal, v1.normal, v2.normal, barycentric_coords));

    hitValue = (normal * 0.5) + 0.5;
}
