:: Keep this target up-to-date with the rust-gpu one.
:: rustup +nightly-2021-08-27-x86_64-pc-windows-msvc component add rust-src rustc-dev llvm-tools-preview
cargo +nightly-2021-08-27-x86_64-pc-windows-msvc run -p compile-shaders --release

glslc --target-spv=spv1.4 -fshader-stage=rchit shaders/closesthit.glsl -o shaders/closesthit.spv
