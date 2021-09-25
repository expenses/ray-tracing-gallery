#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference: enable
#extension GL_EXT_scalar_block_layout : enable

#include "hit_shader_common.glsl"

void main() {
    uint model_index = gl_InstanceCustomIndexEXT;
    ModelInfo info = model_info[model_index];

    uint index_offset = gl_PrimitiveID * 3;

    Indices indices = Indices(info.index_buffer_address);

    uvec3 index = uvec3(
        indices.buf[index_offset],
        indices.buf[index_offset + 1],
        indices.buf[index_offset + 2]
    );

    Vertices vertices = Vertices(info.vertex_buffer_address);

    Vertex a = unpack_vertex(vertices.buf[index.x]);
    Vertex b = unpack_vertex(vertices.buf[index.y]);
    Vertex c = unpack_vertex(vertices.buf[index.z]);

    const vec3 barycentric_coords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);

    vec2 uv = interpolate(a.uv, b.uv, c.uv, barycentric_coords);

    float alpha = texture(textures[nonuniformEXT(info.texture_index)], uv).a;

    if (alpha < 0.5) {
        ignoreIntersectionEXT;
    }
}
