#version 460
#extension GL_EXT_ray_tracing : enable

#include "common.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = 0, rgba8) uniform image2D image;
layout(set = 1, binding = 2, r8ui) uniform uimage2D shadow_image;
layout(set = 1, binding = 3, rgba8) uniform image2D normals_and_depth_image;

layout(location = 0) rayPayloadEXT struct Payload {
    vec3 hit_value;
    bool in_shadow;
    vec3 normals;
    float hit_t;
} payload;

void main()  {
	const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
	const vec2 inUV = pixelCenter/vec2(gl_LaunchSizeEXT.xy);
	vec2 d = inUV * 2.0 - 1.0;

	vec4 origin = uniforms.view_inverse * vec4(0,0,0,1);
	vec4 target = uniforms.proj_inverse * vec4(d.x, d.y, 1, 1) ;
	vec4 direction = uniforms.view_inverse * vec4(normalize(target.xyz), 0) ;

	float tmin = 0.001;
	float tmax = 10000.0;

    payload.hit_value = vec3(0.0);
    payload.in_shadow = false;
    payload.normals = vec3(0.0);
    payload.hit_t = tmax;

    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, tmin, direction.xyz, tmax, 0);

    ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);

    float depth = payload.hit_t / tmax;
    vec3 normals_in_range = (payload.normals + 1.0) * 0.5;

	imageStore(image, pixel, vec4(payload.hit_value, 1.0));
	imageStore(shadow_image, pixel, uvec4(uint(!payload.in_shadow), uvec3(0)));
	imageStore(normals_and_depth_image, pixel, vec4(normals_in_range, depth));
}
