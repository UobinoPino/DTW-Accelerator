cmake_minimum_required(VERSION 3.26)
project(DTW_ACCELERATOR LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -g")



# Find CUDA
find_package(CUDA REQUIRED)

# Define library
add_library(dtw_accelerator INTERFACE)
target_include_directories(dtw_accelerator INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define preprocessor flags based on available libraries
option(USE_OPENMP "Enable OpenMP support" OFF)
option(USE_MPI "Enable MPI support" OFF)
option(USE_CUDA "Enable CUDA support" OFF)

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_compile_definitions(dtw_accelerator INTERFACE USE_OPENMP)
    target_link_libraries(dtw_accelerator INTERFACE OpenMP::OpenMP_CXX)
endif()

if(USE_MPI)
    find_package(MPI REQUIRED)
    target_compile_definitions(dtw_accelerator INTERFACE USE_MPI)
    target_link_libraries(dtw_accelerator INTERFACE MPI::MPI_CXX)
endif()

if(USE_CUDA)
    target_compile_definitions(dtw_accelerator INTERFACE USE_CUDA)
    # Set CUDA architecture
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
endif()

# Example executable
add_executable(main include/dtw_accelerator/main.cpp)
target_link_libraries(main PRIVATE dtw_accelerator)