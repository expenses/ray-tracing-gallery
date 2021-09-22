#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std
)]

// This needs to be here to provide `#[panic_handler]`.
extern crate spirv_std;

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

#[spirv(miss)]
pub fn main(#[spirv(incoming_ray_payload)] shadowed: &mut bool) {
    *shadowed = false;
}

/*
#[spirv(miss)]
pub fn primary_miss()
*/
