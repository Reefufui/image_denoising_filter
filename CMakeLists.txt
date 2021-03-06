cmake_minimum_required (VERSION 3.8)
project (vulkan_denoice)

find_package(Vulkan)
find_package(OpenMP)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set(CMAKE_CXX_STANDARD 20)

include_directories(${Vulkan_INCLUDE_DIR} src/vendor)

set(ALL_LIBS OpenMP::OpenMP_CXX ${Vulkan_LIBRARY} )

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

add_executable(vulkan_denoice src/main.cpp src/vk_utils.h src/vk_utils.cpp src/Bitmap.h src/Bitmap.cpp)

set_target_properties(vulkan_denoice PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMPILE_FLAGS "-fopenmp -g"
    )

target_link_libraries(vulkan_denoice ${ALL_LIBS} )
