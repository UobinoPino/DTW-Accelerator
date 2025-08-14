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
    if(OpenMP_CXX_FOUND)
        target_compile_definitions(dtw_accelerator INTERFACE USE_OPENMP)
        target_link_libraries(dtw_accelerator INTERFACE OpenMP::OpenMP_CXX)

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP support enabled")
        message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")
        message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")

        # OpenMP-only executable
        add_executable(main_openmp include/dtw_accelerator/main.cpp)
        target_compile_definitions(main_openmp PRIVATE USE_OPENMP)
        target_link_libraries(main_openmp PRIVATE dtw_accelerator OpenMP::OpenMP_CXX)

        # OpenMP profiler test executable
        add_executable(test_openmp_profiler include/dtw_accelerator/parallel/test_openmp_profiler.cpp)
        target_compile_definitions(test_openmp_profiler PRIVATE USE_OPENMP)
        target_link_libraries(test_openmp_profiler PRIVATE dtw_accelerator OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found. Disabling OpenMP support.")
        set(USE_OPENMP OFF)
    endif()
endif()

if(USE_MPI)
    find_package(MPI REQUIRED)

    # Create MPI-enabled executable
    add_executable(main_mpi include/dtw_accelerator/main.cpp)
    target_compile_definitions(main_mpi PRIVATE USE_MPI)
    target_link_libraries(main_mpi PRIVATE dtw_accelerator MPI::MPI_CXX)
    target_include_directories(main_mpi PRIVATE ${MPI_CXX_INCLUDE_DIRS})

    # MPI test executable for debugging
    add_executable(test_mpi_dtw include/dtw_accelerator/parallel/test_mpi.cpp)
    target_compile_definitions(test_mpi_dtw PRIVATE USE_MPI)
    target_link_libraries(test_mpi_dtw PRIVATE dtw_accelerator MPI::MPI_CXX)

    # MPI profiler executable
    add_executable(test_mpi_profiler include/dtw_accelerator/parallel/test_mpi_profiler.cpp)
    target_compile_definitions(test_mpi_profiler PRIVATE USE_MPI)
    target_link_libraries(test_mpi_profiler PRIVATE dtw_accelerator MPI::MPI_CXX)
    target_include_directories(test_mpi_profiler PRIVATE ${MPI_CXX_INCLUDE_DIRS})

    message(STATUS "MPI support enabled")
    message(STATUS "MPI C++ Compiler: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI Include Path: ${MPI_CXX_INCLUDE_DIRS}")
    message(STATUS "MPI Libraries: ${MPI_CXX_LIBRARIES}")
endif()

if(USE_CUDA)
    target_compile_definitions(dtw_accelerator INTERFACE USE_CUDA)
    # Set CUDA architecture
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
endif()

# Example executable
add_executable(main include/dtw_accelerator/main.cpp)
target_link_libraries(main PRIVATE dtw_accelerator)

# Print build configuration
message(STATUS "====================================")
message(STATUS "DTW Accelerator Build Configuration:")
message(STATUS "------------------------------------")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Core DTW Profiler: ENABLED")
message(STATUS "OpenMP Support: ${USE_OPENMP}")
message(STATUS "MPI Support: ${USE_MPI}")
message(STATUS "CUDA Support: ${USE_CUDA}")
message(STATUS "====================================")