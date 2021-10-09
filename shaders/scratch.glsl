
mat3 compute_tbn_matrix(vec3 normal, Triangle triangle) {
    vec3 edge_1 = triangle.b.pos - triangle.a.pos;
    vec3 edge_2 = triangle.c.pos - triangle.a.pos;

    vec2 delta_uv_1 = triangle.b.uv - triangle.a.uv;
    vec2 delta_uv_2 = triangle.c.uv - triangle.a.uv;

    float divisor = delta_uv_1.x * delta_uv_2.y - delta_uv_2.x * delta_uv_1.y;

    vec3 tangent = (delta_uv_2.y * edge_1 - delta_uv_1.y * edge_2) / divisor;

    tangent = normalize(tangent);
    //tangent = normalize(tangent - dot(tangent, normal) * normal);
    //tangent = normalize(tangent - dot(tangent, normal) * normal);
    //tangent = normalize(tangent - dot(tangent, normal) * normal);
    //tangent = normalize(tangent - dot(tangent, normal) * normal);
    //tangent = normalize(tangent - dot(tangent, normal) * normal);
    //tangent = normalize(tangent - dot(tangent, normal) * normal);

    vec3 bitangent = cross(normal, tangent);

    return mat3(tangent, bitangent, normal);
}

mat3 compute_cotangent_frame(vec3 normal, Triangle triangle) {
    // get edge vectors of the pixel triangle
    vec3 dp1 = triangle.b.pos - triangle.a.pos;
    vec3 dp2 = triangle.c.pos - triangle.a.pos;
    vec2 duv1 = triangle.b.uv - triangle.a.uv;
    vec2 duv2 = triangle.c.uv - triangle.a.uv;
    // solve the linear system
    vec3 dp2perp = cross(dp2, normal);
    vec3 dp1perp = cross(normal, dp1);
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
    // construct a scale-invariant frame
    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );

    vec3 tangent = T * invmax;

    tangent = normalize(tangent - dot(tangent, normal) * normal);
    tangent = normalize(tangent - dot(tangent, normal) * normal);
    tangent = normalize(tangent - dot(tangent, normal) * normal);


    vec3 bitangent = cross(normal, tangent);

    return mat3(tangent, bitangent, normal);
}
