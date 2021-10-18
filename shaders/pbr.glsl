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
	uint ggx_lut_texture_index;
	float perceptual_roughness;
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
	// Some edges look black if we don't use this epsilon.
	params.normal_dot_view = clamp(dot(base.normal, base.view), 10.0e-10, 1.0);
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

// IBL (wip)

/*
// https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/ec9851c716b1ae5f2e639223dd5c2218a929a4da/source/Renderer/shaders/ibl.glsl#L19
vec3 getIBLRadianceGGX(DotParams params, BaseParams base_params, vec3 f0, float specularWeight) {
	float NdotV = params.normal_dot_view;
	float roughness = base_params.perceptual_roughness;
	vec3 v = base_params.view;
	vec3 n = base_params.normal;

    //float lod = roughness * float(u_MipCount - 1);
    vec3 reflection = normalize(reflect(-v, n));

    vec2 brdfSamplePoint = clamp(vec2(NdotV, 1.0 - roughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    vec2 f_ab = texture(textures[base_params.ggx_lut_texture_index], brdfSamplePoint).rg;
    vec4 specularSample = vec4(base_params.ambient_light, 1.0);//getSpecularSample(reflection, lod);

    vec3 specularLight = specularSample.rgb;

    // see https://bruop.github.io/ibl/#single_scattering_results at Single Scattering Results
    // Roughness dependent fresnel, from Fdez-Aguera
    vec3 Fr = max(vec3(1.0 - roughness), f0) - f0;
    vec3 k_S = f0 + Fr * pow(1.0 - NdotV, 5.0);
    vec3 FssEss = k_S * f_ab.x + f_ab.y;

    return specularWeight * specularLight * FssEss;
}

// https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/ec9851c716b1ae5f2e639223dd5c2218a929a4da/source/Renderer/shaders/ibl.glsl#L81
vec3 getIBLRadianceLambertian(DotParams params, BaseParams base_params, vec3 f0, float specularWeight, vec3 diffuse_colour) {
	float NdotV = params.normal_dot_view;
	float roughness = base_params.perceptual_roughness;

    vec2 brdfSamplePoint = clamp(vec2(NdotV, roughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
    vec2 f_ab = texture(textures[base_params.ggx_lut_texture_index], brdfSamplePoint).rg;

    vec3 irradiance = base_params.ambient_light;//getDiffuseLight(n);

    // see https://bruop.github.io/ibl/#single_scattering_results at Single Scattering Results
    // Roughness dependent fresnel, from Fdez-Aguera

    vec3 Fr = max(vec3(1.0 - roughness), f0) - f0;
    vec3 k_S = f0 + Fr * pow(1.0 - NdotV, 5.0);
    vec3 FssEss = specularWeight * k_S * f_ab.x + f_ab.y; // <--- GGX / specular light contribution (scale it down if the specularWeight is low)

    // Multiple scattering, from Fdez-Aguera
    float Ems = (1.0 - (f_ab.x + f_ab.y));
    vec3 F_avg = specularWeight * (f0 + (1.0 - f0) / 21.0);
    vec3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
    vec3 k_D = diffuse_colour * (1.0 - FssEss + FmsEms); // we use +FmsEms as indicated by the formula in the blog post (might be a typo in the implementation)

    return (FmsEms + k_D) * irradiance;
}
*/

// https://google.github.io/filament/Filament.md.html#materialsystem/standardmodelsummary

struct BrdfInputParams {
	vec3 normal;
	vec3 object_to_view;
	vec3 light;
	float perceptual_roughness;
	vec3 base_colour;
	float metallic;
	float perceptual_dielectric_reflectance;
	vec3 light_intensity;
	uint ggx_lut_texture_index;
};

vec3 brdf(BrdfInputParams input_params) {
	BaseParams params;
	params.normal = input_params.normal;
	params.view = input_params.object_to_view;
	params.light = input_params.light;
	params.halfway = normalize(input_params.object_to_view + input_params.light);
	params.roughness = input_params.perceptual_roughness * input_params.perceptual_roughness;
	params.ggx_lut_texture_index = input_params.ggx_lut_texture_index;
	params.perceptual_roughness = input_params.perceptual_roughness;

	DotParams dot_params = calculate_dot_params(params);

	float Distribution_function = D_GGX(dot_params);

	// from:
	// https://google.github.io/filament/Filament.md.html#materialsystem/parameterization/remapping
	float dielectric_f0 = 0.16 * input_params.perceptual_dielectric_reflectance * input_params.perceptual_dielectric_reflectance;
	vec3 metallic_f0 = input_params.base_colour;

	vec3 f0 = dielectric_f0 * (1.0 - input_params.metallic) + metallic_f0 * input_params.metallic;
	float f90 = compute_f90(dot_params);

	vec3 Fresnel = F_Schlick(dot_params.light_dot_halfway, f0, f90);
	float Geometric_shadowing = V_SmithGGXCorrelated(dot_params);

    // Specular BRDF factor.
	vec3 specular_brdf_factor = (Distribution_function * Geometric_shadowing) * Fresnel;

	// Diffuse BRDF factor.
	vec3 diffuse_brdf_factor = input_params.base_colour * Fd_Burley(dot_params);

	vec3 combined_factor = diffuse_brdf_factor + specular_brdf_factor;

	// For normal map debugging.
	//return input_params.normal * 0.5 + 0.5;

	return input_params.light_intensity * dot_params.normal_dot_light * combined_factor;
}
