#version 460
#extension GL_EXT_ray_tracing : enable

#include "common.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = 0, rgba8) uniform image2D image;
//layout(set = 2, binding = 0, r32ui) uniform image2D shadow_mask;

layout(location = 0) rayPayloadEXT struct Payload {
    vec3 hit_value;
    bool in_shadow;
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

    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, origin.xyz, tmin, direction.xyz, tmax, 0);

	imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(payload.hit_value, 0.0));
}
