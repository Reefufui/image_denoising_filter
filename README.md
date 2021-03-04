Vulkan validation layers can be installed from `https://vulkan.lunarg.com/sdk/home`

![](image.png =250x250)
![](result.bmp =250x250)

# Demo

The application launches a compute shader that applies denoicing alghorithm, by rendering output image into a storage bufferStaging.
The storage bufferStaging is then read from the GPU, and saved as `.bmp`.

## Building

The project uses CMake, and all dependencies are included,
If you then run the program, './build/vulkan_denoice'
a file named `result.png` should be created.
