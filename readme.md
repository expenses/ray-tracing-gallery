# Ray Tracing Gallery

A work-in-progress

![](screenshots/soft-shadows-and-ambient-lighting.png)

## Features

* Top-Level Acceleration Structure updates to dynamically alter the scene.
* Basic lighting and shadows
* Double-buffering
* A controllable camera and directional light source
* Implements a fix for shadow terminator problems from [Ray Tracing Gems II], Chapter 4.
* Most shaders([\*](https://github.com/EmbarkStudios/rust-gpu/issues/754)) are written using [rust-gpu](https://github.com/EmbarkStudios/rust-gpu)!
* Blue noise soft shadows from [Ray Tracing Gems II], Chapter 24.

## Todo

* Ray traced ambient occlusion
* Shadow denoising

## Acknowledgements and sources

* The [NVIDIA Vulkan Ray Tracing Tutorial](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/)
* [Sascha Willems'](https://github.com/SaschaWillems) [Vulkan examples](https://github.com/SaschaWillems/Vulkan/)
* [evopen](https://github.com/evopen)/[silly-cat-engine](https://github.com/evopen/silly-cat-engine) for some help with understanding the `vk::AccelerationStructureInstanceKHR` struct and various other rust-specific details.
* [Lain model by woopoodle on Sketchfab](https://sketchfab.com/3d-models/lain-bf255be16da34df08d48abb5443a6706)
* Blue noise texture from https://momentsingraphics.de/BlueNoise.html.
* GGX LUT texture taken https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/master/assets/images/lut_ggx.png and flipped on the V axis for Vulkan.
* [Yet another blog explaining Vulkan synchronization](https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/) - helped me a lot to fully understand pipeline barriers.
* [Vulkan Guide - Double Buffering](https://vkguide.dev/docs/chapter-4/double_buffering/)
* [Ray Tracing Gems II], Chapter 16, Page 239. for clarifying how in-place Acceleration Structures work.

[Ray Tracing Gems II]: https://link.springer.com/content/pdf/10.1007%2F978-1-4842-7185-8.pdf
