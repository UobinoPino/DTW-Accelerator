cmake_minimum_required(VERSION 3.26)

# ==================================================
# SEPARATED BENCHMARKS FOR FAIR COMPARISON
# ==================================================

# Pure Sequential Baseline
add_executable(benchmark_sequential_pure benchmark_sequential_pure.cpp)
target_link_libraries(benchmark_sequential_pure PRIVATE dtw_accelerator)

# OpenMP-only benchmark
add_executable(benchmark_openmp_only benchmark_openmp_only.cpp)
target_link_libraries(benchmark_openmp_only PRIVATE dtw_accelerator)
if(USE_OPENMP)
    target_compile_definitions(benchmark_openmp_only PRIVATE USE_OPENMP)
endif()

# MPI-only benchmark
add_executable(benchmark_mpi_only benchmark_mpi_only.cpp)
target_link_libraries(benchmark_mpi_only PRIVATE dtw_accelerator)
if(USE_MPI)
    target_compile_definitions(benchmark_mpi_only PRIVATE USE_MPI)
endif()

add_executable(benchmark_cuda_only benchmark_cuda_only.cpp)
target_link_libraries(benchmark_cuda_only PRIVATE dtw_accelerator)
if(USE_CUDA)
    target_compile_definitions(benchmark_cuda_only PRIVATE USE_CUDA)
endif()



# ==================================================
# CUSTOM TARGETS FOR RUNNING BENCHMARKS
# ==================================================

# Fair benchmark targets
add_custom_target(run_fair_benchmarks
        COMMAND ${CMAKE_COMMAND} -E echo "Running fair separated benchmarks..."
        COMMAND benchmark_sequential_pure
        COMMAND benchmark_openmp_only
        COMMAND ${CMAKE_COMMAND} -E echo "For MPI, run: mpirun -np N benchmark_mpi_only"
        DEPENDS benchmark_sequential_pure benchmark_openmp_only benchmark_mpi_only
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running fair benchmarks (separated compilation)"
)

# Generate plots from fair benchmarks
add_custom_target(plot_fair_results
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/plot_results_updated.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating plots from fair benchmark results"
)


# Combined target to run everything
add_custom_target(run_all_benchmarks
        COMMAND ${CMAKE_COMMAND} -E echo "Running all benchmarks..."
        COMMAND benchmark_sequential_pure
        COMMAND benchmark_openmp_only
        COMMAND ${CMAKE_COMMAND} -E echo "For MPI: mpirun -np N benchmark_mpi_only"
        DEPENDS benchmark_sequential_pure benchmark_openmp_only benchmark_mpi_only
        COMMENT "Running all benchmarks (fair and legacy)"
)

# MPI-specific targets
if(USE_MPI)
    add_custom_target(run_mpi_fair_2
            COMMAND mpirun -np 2 benchmark_mpi_only
            DEPENDS benchmark_mpi_only
            COMMENT "Running fair MPI benchmark with 2 processes"
    )

    add_custom_target(run_mpi_fair_4
            COMMAND mpirun -np 4 benchmark_mpi_only
            DEPENDS benchmark_mpi_only
            COMMENT "Running fair MPI benchmark with 4 processes"
    )

    add_custom_target(run_mpi_fair_8
            COMMAND mpirun -np 8 benchmark_mpi_only
            DEPENDS benchmark_mpi_only
            COMMENT "Running fair MPI benchmark with 8 processes"
    )

    add_custom_target(run_mpi_fair_all
            COMMAND mpirun -np 1 benchmark_mpi_only
            COMMAND mpirun -np 2 benchmark_mpi_only
            COMMAND mpirun -np 4 benchmark_mpi_only
            COMMAND mpirun -np 8 benchmark_mpi_only
            DEPENDS benchmark_mpi_only
            COMMENT "Running fair MPI benchmarks with 1, 2, 4, 8 processes"
    )
endif()

# CUDA-specific targets
if(USE_CUDA)
    add_custom_target(run_cuda_benchmark
            COMMAND benchmark_cuda_only
            DEPENDS benchmark_cuda_only
            COMMENT "Running CUDA benchmark"
    )
endif()

# ==================================================
# INSTALLATION
# ==================================================

# Install benchmark binaries
install(TARGETS
        benchmark_sequential_pure
        benchmark_openmp_only
        benchmark_mpi_only
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/benchmarks
)

# Install Python plotting scripts
install(FILES
        plot_results.py
        plot_results_updated.py
        DESTINATION ${CMAKE_INSTALL_BINDIR}/benchmarks
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

# Install benchmark utilities header (if exists)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_utils.hpp)
    install(FILES benchmark_utils.hpp
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dtw_accelerator/benchmarks
    )
endif()

# ==================================================
# MESSAGES
# ==================================================

message(STATUS "Benchmark targets configured:")
message(STATUS "  - benchmark_sequential_pure (pure baseline)")
message(STATUS "  - benchmark_openmp_only (OpenMP only)")
message(STATUS "  - benchmark_mpi_only (MPI only)")
message(STATUS "")
message(STATUS "Use 'make run_fair_benchmarks' for accurate comparisons")
message(STATUS "Use 'make plot_fair_results' to generate plots")