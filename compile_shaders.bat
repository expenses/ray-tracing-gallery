glslc -g --target-spv=spv1.4 -fshader-stage=rchit src/shaders/closesthit.rchit.glsl -o src/shaders/closesthit.rchit.spv
glslc -g --target-spv=spv1.4 -fshader-stage=rmiss src/shaders/miss.rmiss.glsl -o src/shaders/miss.rmiss.spv
glslc -g --target-spv=spv1.4 -fshader-stage=rmiss src/shaders/shadow.rmiss.glsl -o src/shaders/shadow.rmiss.spv
glslc -g --target-spv=spv1.4 -fshader-stage=rgen src/shaders/raygen.rgen.glsl -o src/shaders/raygen.rgen.spv

dxc/dxc.exe src/shaders/shadow_denoiser_prepare.hlsl -spirv -T cs_6_0 -fspv-target-env=vulkan1.1 -Fo src/shaders/shadow_denoiser_prepare.hlsl.spv -Wno-ambig-lit-shift

dxc/dxc.exe src/shaders/shadow_denoiser_passes.hlsl -spirv -T lib_6_3 -fspv-target-env=vulkan1.1 -enable-16bit-types -Fo src/shaders/shadow_denoiser_passes.hlsl.spv -Wno-conversion -Wno-ambig-lit-shift

dxc/dxc.exe src/shaders/shadow_denoiser_classify.hlsl -spirv -T lib_6_3 -fspv-target-env='vulkan1.1'