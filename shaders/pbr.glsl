const float PI = 3.141592653589793;

// See: https://google.github.io/filament/Filament.md.html#notation
struct BaseParams {
	// v
	vec3 view;
	// n
	vec3 normal;
	// l
	vec3 light;
	// h
	vec3 halfway;
	// Also, annoying, sometimes called 'a'
	float roughness;
};

struct DotParams {
	float normal_dot_halfway;
	float normal_dot_view;
	float normal_dot_light;
	float light_dot_halfway;
	float roughness;
};

float clamp_dot(vec3 a, vec3 b) {
	return clamp(dot(a, b), 0.0, 1.0);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/standardmodelsummary
DotParams calculate_dot_params(BaseParams base) {
	DotParams params;
	params.normal_dot_view = clamp_dot(base.normal, base.view);
	params.normal_dot_halfway = clamp_dot(base.normal, base.halfway);
	params.normal_dot_light = clamp_dot(base.normal, base.light);
	params.light_dot_halfway = clamp_dot(base.light, base.halfway);
	params.roughness = base.roughness;
	return params;
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/normaldistributionfunction(speculard)
// The normal distribution function.

float D_GGX(DotParams params) {
	float NoH = params.normal_dot_halfway;
	float roughness = params.roughness;

    float a = NoH * roughness;
    float k = roughness / (1.0 - NoH * NoH + a * a);
    return k * k * (1.0 / PI);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
// Geometric shadowing function

float V_SmithGGXCorrelated(DotParams params) {
	float NoV = params.normal_dot_view;
	float NoL = params.normal_dot_light;
	float roughness = params.roughness;

    float a2 = roughness * roughness;
    float GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
}

// Fresnel

vec3 F_Schlick(float u, vec3 f0, float f90) {
    return f0 + (vec3(f90) - f0) * pow(1.0 - u, 5.0);
}

// The same bit only a single float
float F_Schlick(float u, float f0, float f90) {
    return f0 + (f90 - f0) * pow(1.0 - u, 5.0);
}


// Diffuse

float Fd_Lambert() {
    return 1.0 / PI;
}

// Disney diffuse (more fun!)

// Compute f90 according to burley.
float compute_f90(DotParams params) {
	float LoH = params.light_dot_halfway;
	float roughness = params.roughness;

	return 0.5 + 2.0 * roughness * LoH * LoH;
}

float Fd_Burley(DotParams params) {
	float NoV = params.normal_dot_view;
	float NoL = params.normal_dot_light;

    float f90 = compute_f90(params);
    float lightScatter = F_Schlick(NoL, 1.0, f90);
    float viewScatter = F_Schlick(NoV, 1.0, f90);
    return lightScatter * viewScatter * (1.0 / PI);
}

// https://google.github.io/filament/Filament.md.html#materialsystem/standardmodelsummary

struct BrdfInputParams {
	vec3 normal;
	vec3 view;
	vec3 light;
	float perceptual_roughness;
	vec3 base_colour;
	float metallic;
	float reflectance;
	vec3 light_intensity;
};

vec3 brdf(BrdfInputParams input_params) {
	BaseParams params;
	params.normal = input_params.normal;
	params.view = input_params.view;
	params.light = input_params.light;
	params.halfway = normalize(input_params.view + input_params.light);
	params.roughness = input_params.perceptual_roughness * input_params.perceptual_roughness;

	DotParams dot_params = calculate_dot_params(params);

	float D = D_GGX(dot_params);

	// from:
	// https://google.github.io/filament/Filament.md.html#materialsystem/parameterization/remapping
	vec3 f0 = 0.16 * input_params.reflectance * input_params.reflectance * (1.0 - input_params.metallic) + input_params.base_colour * input_params.metallic;
	float f90 = compute_f90(dot_params);

	vec3 F = F_Schlick(dot_params.light_dot_halfway, f0, f90);
	float V = V_SmithGGXCorrelated(dot_params);

    // specular BRDF factor
    // something here is wrong uthghgghhg
	vec3 Fr = (D * V) * F;

	vec3 Fd = input_params.base_colour * Fd_Burley(dot_params);

	vec3 colour = input_params.light_intensity * dot_params.normal_dot_light * (Fr + Fd);

	return colour;
}
