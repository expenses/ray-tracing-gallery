layout(set = 1, binding = 2) uniform Uniforms {
	mat4 view_inverse;
	mat4 proj_inverse;
	vec3 sun_dir;
	float sun_radius;
} uniforms;

