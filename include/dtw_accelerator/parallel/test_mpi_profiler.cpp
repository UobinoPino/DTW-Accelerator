#include <vector>
#include <iostream>
#include <random>
#include <mpi.h>
#include "dtw_accelerator/parallel/mpi_dtw_profiler.hpp"

// Function to generate random time series data
std::vector<std::vector<double>> generate_random_series(int length, int dimensions, double min_val = 0.0, double max_val = 100.0) {
    std::vector<std::vector<double>> series(length, std::vector<double>(dimensions));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min_val, max_val);

    for (int i = 0; i < length; i++) {
        for (int j = 0; j < dimensions; j++) {
            series[i][j] = dist(gen);
        }
    }

    return series;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default parameters
    int sequence_length = 10000;
    int dimensions = 2;
    int block_size = 64;
    int fastdtw_radius = 3;
    int fastdtw_min_size = 100;

    // Parse command line arguments if provided
    if (argc > 1) sequence_length = std::atoi(argv[1]);
    if (argc > 2) dimensions = std::atoi(argv[2]);
    if (argc > 3) block_size = std::atoi(argv[3]);

    if (rank == 0) {
        std::cout << "=======================================" << std::endl;
        std::cout << "MPI DTW Performance Profiler" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::cout << "Sequence length: " << sequence_length << std::endl;
        std::cout << "Dimensions: " << dimensions << std::endl;
        std::cout << "MPI processes: " << size << std::endl;
        std::cout << "Block size: " << block_size << std::endl;
        std::cout << "=======================================" << std::endl;
    }

    // Generate random series (only on rank 0)
    std::vector<std::vector<double>> series_a, series_b;

    if (rank == 0) {
        series_a = generate_random_series(sequence_length, dimensions);
        series_b = generate_random_series(sequence_length, dimensions);
        std::cout << "Random time series generated." << std::endl;
    }

    // Broadcast dimensions to all processes to prepare space
    int dims[2] = {sequence_length, dimensions};
    MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        series_a.resize(dims[0], std::vector<double>(dims[1]));
        series_b.resize(dims[0], std::vector<double>(dims[1]));
    }

    // Broadcast series data
    for (int i = 0; i < sequence_length; i++) {
        MPI_Bcast(series_a[i].data(), dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(series_b[i].data(), dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Run the profiler tests
    dtw_accelerator::parallel::mpi::run_dtw_profiler_tests(series_a, series_b, true);
    MPI_Barrier(MPI_COMM_WORLD);




    // Run the profiler tests with constraints (Sakoe-Chiba R=3)
    if (rank == 0) {
        std::cout << "\n=======================================" << std::endl;
        std::cout << "TEST 2: DTW with Sakoe-Chiba Constraint (R=3)" << std::endl;
        std::cout << "=======================================" << std::endl;
    }
    dtw_accelerator::parallel::mpi::run_dtw_profiler_tests_with_constraints<dtw_accelerator::constraints::ConstraintType::SAKOE_CHIBA, 3>(
        series_a, series_b, true);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n=======================================" << std::endl;
        std::cout << "TEST 3: FastDTW (Radius=" << fastdtw_radius << ")" << std::endl;
        std::cout << "=======================================" << std::endl;
    }
    // Run the FastDTW profiler tests
    dtw_accelerator::parallel::mpi::run_fastdtw_profiler_tests(
            series_a, series_b, fastdtw_radius, fastdtw_min_size, true);

    MPI_Barrier(MPI_COMM_WORLD);

    // Test different FastDTW radii for comparison
    if (rank == 0) {
        std::cout << "\n=======================================" << std::endl;
        std::cout << "TEST 4: FastDTW Radius Comparison" << std::endl;
        std::cout << "=======================================" << std::endl;
    }

    std::vector<int> test_radii = {1, 3, 5, 10};
    for (int radius : test_radii) {
        if (rank == 0) {
            std::cout << "\n--- Testing FastDTW with radius=" << radius << " ---" << std::endl;
        }

        dtw_accelerator::parallel::mpi::run_fastdtw_profiler_tests(
                series_a, series_b, radius, fastdtw_min_size, true);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Final summary
    if (rank == 0) {
        std::cout << "\n=======================================" << std::endl;
        std::cout << "PROFILING COMPLETE" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::cout << "Profile files have been saved to disk." << std::endl;
        std::cout << "Look for *_profile_rank*.csv files in the current directory." << std::endl;
    }

    MPI_Finalize();
    return 0;
}