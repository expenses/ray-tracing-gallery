glslc --target-spv=spv1.4 src/shaders/closesthit.rchit -o src/shaders/closesthit.rchit.spv
glslc --target-spv=spv1.4 src/shaders/miss.rmiss -o src/shaders/miss.rmiss.spv
glslc --target-spv=spv1.4 src/shaders/raygen.rgen -o src/shaders/raygen.rgen.spv