:: Keep this target up-to-date with the rust-gpu one.
:: rustup +nightly-2021-08-27-x86_64-pc-windows-msvc component add rust-src rustc-dev llvm-tools-preview
cargo +nightly-2021-08-27-x86_64-pc-windows-msvc run -p compile-shaders --release

glslc --target-spv=spv1.4 -fshader-stage=rchit shaders/closest_hit_textured.glsl -o shaders/closest_hit_textured.spv
glslc --target-spv=spv1.4 -fshader-stage=rchit shaders/closest_hit_mirror.glsl -o shaders/closest_hit_mirror.spv
glslc --target-spv=spv1.4 -fshader-stage=rchit shaders/closest_hit_portal.glsl -o shaders/closest_hit_portal.spv
