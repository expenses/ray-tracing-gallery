use spirv_std::glam::Vec3;
use spirv_std::num_traits::Float;

struct BaseParams {
    view: Vec3,
    normal: Vec3,
    light: Vec3,
    halfway: Vec3,
    roughness: f32,
}

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.min(max).max(min)
}

fn clamp_dot(a: Vec3, b: Vec3) -> f32 {
    clamp(a.dot(b), 0.0, 1.0)
}

impl BaseParams {
    fn as_dot_params(&self) -> DotParams {
        DotParams {
            // It's important that we use an epsilon here,
            normal_dot_view: clamp(self.normal.dot(self.view), 1.0e-10, 1.0),
            normal_dot_halfway: clamp_dot(self.normal, self.halfway),
            normal_dot_light: clamp_dot(self.normal, self.light),
            light_dot_halfway: clamp_dot(self.light, self.halfway),
            roughness: self.roughness,
        }
    }
}

struct DotParams {
    normal_dot_halfway: f32,
    normal_dot_view: f32,
    normal_dot_light: f32,
    light_dot_halfway: f32,
    roughness: f32,
}

fn V_SmithGGXCorrelated(params: DotParams) -> f32 {
    let NoV = params.normal_dot_view;
    let NoL = params.normal_dot_light;
    let roughness = params.roughness;

    let a2 = roughness * roughness;
    let GGXV = NoL * (NoV * NoV * (1.0 - a2) + a2).sqrt();
    let GGXL = NoV * (NoL * NoL * (1.0 - a2) + a2).sqrt();
    0.5 / (GGXV + GGXL)
}

#[test]
fn test_ggx_division_by_zero() {
    let view = Vec3::new(1.0, 0.0, 0.0);
    let light = Vec3::new(0.0, 1.0, 0.0);

    let params = BaseParams {
        normal: Vec3::new(0.0, 1.0, 0.0),
        view,
        light,
        halfway: (view + light).normalize(),
        roughness: 0.0,
    }
    .as_dot_params();

    let result = V_SmithGGXCorrelated(params);

    assert!(result.is_finite(), "{}", result);
}
