use spirv_std::glam::Vec3;
use spirv_std::num_traits::Float;
use spirv_std::arch::{signed_min, signed_max};

pub fn heatmap_temperature(heat: f32) -> Vec3 {
    let colours = [
        Vec3::new(0.0 / 255.0, 2.0 / 255.0, 91.0 / 255.0),
        Vec3::new(0.0 / 255.0, 108.0 / 255.0, 251.0 / 255.0),
        Vec3::new(0.0 / 255.0, 221.0 / 255.0, 221.0 / 255.0),
        Vec3::new(51.0 / 255.0, 221.0 / 255.0, 0.0 / 255.0),
        Vec3::new(255.0 / 255.0, 252.0 / 255.0, 0.0 / 255.0),
        Vec3::new(255.0 / 255.0, 180.0 / 255.0, 0.0 / 255.0),
        Vec3::new(255.0 / 255.0, 104.0 / 255.0, 0.0 / 255.0),
        Vec3::new(226.0 / 255.0, 22.0 / 255.0, 0.0 / 255.0),
        Vec3::new(191.0 / 255.0, 0.0 / 255.0, 83.0 / 255.0),
        Vec3::new(145.0 / 255.0, 0.0 / 255.0, 65.0 / 255.0),
    ];

    let heat = saturate(heat) * 10.0;
    let heat_index_int = heat as i32;

    let cur = heat_index_int as usize;
    let prv = signed_max(heat_index_int - 1, 0) as usize;
    let nxt = signed_min(heat_index_int + 1, 9) as usize;

    let heat_floor = heat.floor();
    let heat_ceil = heat.ceil();

    let blur = 0.8;

    let wc = smoothstep(heat_floor - blur, heat_floor + blur, heat)
        * (1.0 - smoothstep(heat_ceil - blur, heat_ceil + blur, heat));
    let wp = 1.0 - smoothstep(heat_floor - blur, heat_floor + blur, heat);
    let wn = smoothstep(heat_ceil - blur, heat_ceil + blur, heat);

    let result = wc * colours[cur] + wp * colours[prv] + wn * colours[nxt];
    result.clamp(Vec3::splat(0.0), Vec3::splat(1.0))
}

fn saturate(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn smoothstep(edge_0: f32, edge_1: f32, x: f32) -> f32 {
    let t = saturate((x - edge_0) / (edge_1 - edge_0));
    t * t * (3.0 - 2.0 * t)
}
