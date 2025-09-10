cmake_minimum_required(VERSION 3.20)

# Performance test executables
add_executable(benchmark_strategies benchmark_strategies.cpp)
target_link_libraries(benchmark_strategies PRIVATE dtw_accelerator)

add_executable(benchmark_scaling benchmark_scaling.cpp)
target_link_libraries(benchmark_scaling PRIVATE dtw_accelerator)

add_executable(openmp_profile openmp_dtw_profile.cpp)
target_link_libraries(openmp_profile PRIVATE dtw_accelerator)

# Add CUDA-specific benchmarks if CUDA is enabled
if(USE_CUDA)
    add_executable(profile_cuda profile_cuda.cpp)
    target_link_libraries(profile_cuda PRIVATE dtw_accelerator)
    target_compile_definitions(profile_cuda PRIVATE USE_CUDA)

    add_executable(diagnose_cuda_issue diagnose_cuda_issue.cpp)
    target_link_libraries(diagnose_cuda_issue PRIVATE dtw_accelerator)
    target_compile_definitions(diagnose_cuda_issue PRIVATE USE_CUDA)
endif()

# Add test targets
add_custom_target(run_benchmarks
COMMAND benchmark_strategies
COMMAND benchmark_scaling
DEPENDS benchmark_strategies benchmark_scaling
COMMENT "Running all performance benchmarks"
)

# Individual benchmark targets
add_custom_target(benchmark_quick
COMMAND benchmark_strategies
DEPENDS benchmark_strategies
COMMENT "Running quick strategy comparison"
)

add_custom_target(benchmark_full
COMMAND benchmark_strategies
COMMAND benchmark_scaling
DEPENDS benchmark_strategies benchmark_scaling
COMMENT "Running full performance analysis"
)

if(USE_CUDA)
    add_custom_target(benchmark_cuda
            COMMAND profile_cuda
            COMMAND diagnose_cuda_issue
            DEPENDS benchmark_cuda_profile diagnose_cuda_issue
            COMMENT "Running CUDA-specific benchmarks and diagnostics"
    )
endif()

# Install benchmark binaries
install(TARGETS benchmark_strategies benchmark_scaling openmp_profile
RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/benchmarks)

# Install CUDA benchmark binaries if CUDA is enabled
if(USE_CUDA)
    install(TARGETS profile_cuda diagnose_cuda_issue
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/benchmarks)
endif()


# Comprehensive benchmark executable
add_executable(benchmark_comprehensive benchmark_comprehensive.cpp)
target_link_libraries(benchmark_comprehensive PRIVATE dtw_accelerator)

# Set compile definitions based on available backends
if(USE_OPENMP)
    target_compile_definitions(benchmark_comprehensive PRIVATE USE_OPENMP)
endif()

if(USE_MPI)
    target_compile_definitions(benchmark_comprehensive PRIVATE USE_MPI)
endif()

if(USE_CUDA)
    target_compile_definitions(benchmark_comprehensive PRIVATE USE_CUDA)
endif()

# Custom target to run benchmarks and generate plots
add_custom_target(run_comprehensive_benchmark
        COMMAND benchmark_comprehensive
        COMMAND ${CMAKE_COMMAND} -E echo "Generating plots..."
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/plot_results.py
        DEPENDS benchmark_comprehensive
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running comprehensive DTW benchmarks and generating plots"
)

# Target for MPI benchmarks (requires mpirun)
if(USE_MPI)
    add_custom_target(run_mpi_benchmark
            COMMAND mpirun -np 4 benchmark_comprehensive
            COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/plot_results.py
            DEPENDS benchmark_comprehensive
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Running MPI DTW benchmarks with 4 processes"
    )
endif()

# Install benchmark files
install(TARGETS benchmark_comprehensive
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/benchmarks
)

install(FILES plot_results.py
        DESTINATION ${CMAKE_INSTALL_BINDIR}/benchmarks
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

# Copy benchmark utilities header
install(FILES benchmark_utils.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dtw_accelerator/benchmarks)