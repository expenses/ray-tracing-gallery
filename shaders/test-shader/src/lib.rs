#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std
)]

// This needs to be here to provide `#[panic_handler]`.
extern crate spirv_std;

<<<<<<< HEAD
#[spirv(fragment)]
pub fn x() {
    let x = min_i32(1, 2);
    let y = min_u32(1, 2);
    let z = min_f32(1.0, 2.0);
}

#[inline(never)]
fn min_i32(a: i32, b: i32) -> i32 {
    a.min(b)
}

#[inline(never)]
fn min_u32(a: u32, b: u32) -> u32 {
    a.min(b)
}

#[inline(never)]
fn min_f32(a: f32, b: f32) -> f32 {
    a.min(b)
=======
use spirv_std::{
    glam::{Vec2, Vec4},
    image::SampledImage,
    Image,
};

#[spirv(compute(threads(64)))]
pub fn compute(
    #[spirv(descriptor_set = 1, binding = 1)] image: &SampledImage<Image!(2D, type=f32, sampled)>,
) {
    let colour: Vec4 = unsafe { image.sample(Vec2::new(0.0, 0.0)) };
>>>>>>> 15b6961 (Implement blue noise sampling!)
}
