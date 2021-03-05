cd build
make -j 5
cd ..
./build/vulkan_denoice
qview result.bmp &
qview result_cpu.bmp

