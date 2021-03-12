#!/bin/bash

if [ -d "build" ]; then
    cd build
    make -j 5
    cd ..
else
    mkdir build
    cd build
    cmake ..
    make -j 5
    cd ..
fi

./build/vulkan_denoice
