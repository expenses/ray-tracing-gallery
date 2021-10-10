#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std
)]

// This needs to be here to provide `#[panic_handler]`.
extern crate spirv_std;

use spirv_std::{
    Image, image::SampledImage,
    glam::{Vec2, Vec4},
};

#[spirv(compute(threads(64)))]
pub fn compute(
    #[spirv(descriptor_set = 1, binding = 1)] image: &SampledImage<Image!(2D, type=f32, sampled)>,
) {
    let colour: Vec4 = unsafe {
        image.sample(Vec2::new(0.0, 0.0))
    };
}
