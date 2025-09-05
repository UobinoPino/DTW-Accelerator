cmake_minimum_required(VERSION 3.20)
project(DTWAccelerator VERSION 2.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops -fprefetch-loop-arrays")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fsanitize=address")

# Define library
add_library(dtw_accelerator INTERFACE)
target_include_directories(dtw_accelerator INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define preprocessor flags based on available libraries
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_MPI "Enable MPI support" ON)
option(USE_CUDA "Enable CUDA support" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        target_compile_definitions(dtw_accelerator INTERFACE USE_OPENMP)
        target_link_libraries(dtw_accelerator INTERFACE OpenMP::OpenMP_CXX)

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP support enabled")
        message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")
        message(STATUS "OpenMP flags: ${OpenMP_CXX_FLAGS}")

    else()
        message(WARNING "OpenMP not found. Disabling OpenMP support.")
        set(USE_OPENMP OFF)
    endif()
endif()

if(USE_MPI)
    find_package(MPI REQUIRED)

    if(MPI_CXX_FOUND)
        target_compile_definitions(dtw_accelerator INTERFACE USE_MPI)
        target_link_libraries(dtw_accelerator INTERFACE MPI::MPI_CXX)

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
        message(STATUS "MPI support enabled")
        message(STATUS "MPI version: ${MPI_VERSION}")
        message(STATUS "MPI C++ Compiler: ${MPI_CXX_COMPILER}")
        message(STATUS "MPI Include Path: ${MPI_CXX_INCLUDE_DIRS}")
        message(STATUS "MPI Libraries: ${MPI_CXX_LIBRARIES}")

    else()
        message(WARNING "MPI not found. Disabling MPI support.")
        set(USE_MPI OFF)
    endif()
    message(STATUS "MPI support enabled")
    message(STATUS "MPI C++ Compiler: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI Include Path: ${MPI_CXX_INCLUDE_DIRS}")
    message(STATUS "MPI Libraries: ${MPI_CXX_LIBRARIES}")
endif()

if(USE_CUDA)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    if(CUDAToolkit_FOUND)
        message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA Include Directory: ${CUDAToolkit_INCLUDE_DIRS}")
        message(STATUS "CUDA Libraries Directory: ${CUDAToolkit_LIBRARY_DIR}")

        if(CUDAToolkit_VERSION VERSION_LESS "12.0")
            message(FATAL_ERROR "CUDA 12.0+ required for C++20. Found: ${CUDAToolkit_VERSION}")
        endif()
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Set CUDA flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

        target_compile_definitions(dtw_accelerator INTERFACE USE_CUDA)

        # Create CUDA-specific library for kernels
        add_library(dtw_cuda_kernels STATIC
                include/dtw_accelerator/execution/parallel/cuda/kernels/dtw_core_kernels.cu
                include/dtw_accelerator/execution/parallel/cuda/kernels/path_kernels.cu
                include/dtw_accelerator/execution/parallel/cuda/kernels/matrix_kernels.cu
                include/dtw_accelerator/execution/parallel/cuda/execution/cuda_launcher.cu
        )

        target_include_directories(dtw_cuda_kernels PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

        set_target_properties(dtw_cuda_kernels PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_ARCHITECTURES "52;60;70;75;80;86"
                POSITION_INDEPENDENT_CODE ON
        )

        # Add CUDA-specific flags for C++20
        target_compile_options(dtw_cuda_kernels PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>
        )

        target_link_libraries(dtw_accelerator INTERFACE
                dtw_cuda_kernels
                CUDA::cudart
                CUDA::cuda_driver
        )


        message(STATUS "CUDA support enabled")
        message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        message(STATUS "CUDA Include Path: ${CUDAToolkit_INCLUDE_DIRS}")
        message(STATUS "CUDA Standard: C++20 (for CUDA files)")
        message(STATUS "C++ Standard: C++20 (for host files)")
    else()
        message(WARNING "CUDA Toolkit not found. Disabling CUDA support.")
        set(USE_CUDA OFF)
    endif()
endif()

# Examples
#if(BUILD_EXAMPLES)
#    add_executable(dtw_example include/examples/example_usage.cpp)
#    target_link_libraries(dtw_example PRIVATE dtw_accelerator)
#    if(USE_MPI)
#        add_executable(dtw_example_mpi include/examples/example_usage_mpi.cpp)
#        target_link_libraries(dtw_example_mpi PRIVATE dtw_accelerator)
#        target_compile_definitions(dtw_example_mpi PRIVATE USE_MPI)
#
#        add_executable(dtw_test_mpi include/examples/dtw_mpi_test.cpp)
#        target_link_libraries(dtw_test_mpi PRIVATE dtw_accelerator)
#        target_compile_definitions(dtw_test_mpi PRIVATE USE_MPI)
#
#    endif ()
#
#    add_executable(dtw_benchmark include/examples/benchmark.cpp)
#    target_link_libraries(dtw_benchmark PRIVATE dtw_accelerator)
#
#    add_executable(dtw_custom_strategy include/examples/custom_strategy.cpp)
#    target_link_libraries(dtw_custom_strategy PRIVATE dtw_accelerator)
#
#    add_executable(tests include/tests/test_concepts.cpp)
#    # target_link_libraries(tests PRIVATE dtw_accelerator)
#    target_link_libraries(tests PRIVATE dtw_accelerator GTest::GTest GTest::Main)
#
#
#    # Add CUDA examples if CUDA is enabled
#    if(USE_CUDA)
#        add_executable(dtw_cuda_example include/examples/cuda_example.cpp)
#        target_link_libraries(dtw_cuda_example PRIVATE dtw_accelerator)
#        target_compile_definitions(dtw_cuda_example PRIVATE USE_CUDA)
#    endif()
#endif()

# Build unit tests if enabled
if(BUILD_TESTS)
    add_subdirectory(include/tests)
    message(STATUS "Unit tests enabled")
else()
    message(STATUS "Unit tests disabled")
endif()

# Build benchmarks if enabled
if(BUILD_BENCHMARKS)
  #  add_subdirectory(include/tests/performance)
    message(STATUS "Performance benchmarks enabled")
else()
    message(STATUS "Performance benchmarks disabled")
endif()

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