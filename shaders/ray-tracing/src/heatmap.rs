use spirv_std::glam::Vec3;

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

    let heat_index = heat * 10.0;
    let heat_index_int = heat_index as i32;

    let cur = clamp_int(heat_index_int) as usize;
    let prv = clamp_int(heat_index_int - 1) as usize;
    let nxt = clamp_int(heat_index_int + 1) as usize;

    let blur = 0.8;

    let wc = smoothstep(cur as f32 - blur, cur as f32 + blur, heat_index)
        * (1.0 - smoothstep((cur + 1) as f32 - blur, (cur + 1) as f32 + blur, heat_index));
    let wp = 1.0 - smoothstep(cur as f32 - blur, cur as f32 + blur, heat_index);
    let wn = smoothstep((cur + 1) as f32 - blur, (cur + 1) as f32 + blur, heat_index);

    let r = wc * colours[cur] + wp * colours[prv] + wn * colours[nxt];
    Vec3::new(
        clamp(r.x, 0.0, 1.0),
        clamp(r.y, 0.0, 1.0),
        clamp(r.z, 0.0, 1.0),
    )
}

fn clamp_int(int: i32) -> i32 {
    int.max(0).min(9)
}

pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

fn smoothstep(edge_0: f32, edge_1: f32, x: f32) -> f32 {
    let t = clamp((x - edge_0) / (edge_1 - edge_0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
