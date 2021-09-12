glslc -g --target-spv=spv1.4 -fshader-stage=rchit src/shaders/closesthit.rchit.glsl -o src/shaders/closesthit.rchit.spv
glslc -g --target-spv=spv1.4 -fshader-stage=rmiss src/shaders/miss.rmiss.glsl -o src/shaders/miss.rmiss.spv
glslc -g --target-spv=spv1.4 -fshader-stage=rmiss src/shaders/shadow.rmiss.glsl -o src/shaders/shadow.rmiss.spv
glslc -g --target-spv=spv1.4 -fshader-stage=rgen src/shaders/raygen.rgen.glsl -o src/shaders/raygen.rgen.spv