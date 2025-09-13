#!/usr/bin/env bash

# Remove existing build directory if it exists
rm -rf build

# Create a new build directory and navigate into it
mkdir -p build
cd build

# Run CMake to configure the project
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_OPENMP=ON \
    -DUSE_MPI=ON \
    -DUSE_CUDA=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_BENCHMARKS=ON \
    -DBUILD_EXAMPLES=ON \
    ..

# Compile the project using all available CPU cores
make -j$(nproc)
