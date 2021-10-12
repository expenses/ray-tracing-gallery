:: Keep this target up-to-date with the rust-gpu one.
:: rustup +nightly-2021-09-29-x86_64-pc-windows-msvc component add rust-src rustc-dev llvm-tools-preview
cargo +nightly-2021-09-29-x86_64-pc-windows-msvc run -p compile-shaders --release

glslc --target-spv=spv1.4 -fshader-stage=rchit shaders/closest_hit_textured.glsl -o shaders/closest_hit_textured.spv

spirv-opt shaders/closest_hit_textured.spv -O -o shaders/closest_hit_textured.spv
