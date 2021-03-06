cd build
make -j 5
cd ..
./build/vulkan_denoice res/cornelbox.bmp
qview result.bmp &
qview cpu_result.bmp &
rm result.bmp
rm cpu_result.bmp
