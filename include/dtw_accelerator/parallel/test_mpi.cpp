#include <iostream>
#include <iomanip>
#include <vector>
#include <mpi.h>
#include <chrono>

#include "dtw_accelerator/parallel/mpi_dtw.hpp"
#include "dtw_accelerator/core_dtw.hpp"

// Simple test function to verify correctness
void test_small_example() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "\n=== Testing Small Example ===" << std::endl;
        std::cout << "Processes: " << size << std::endl;
    }

    // Create small test sequences
    std::vector<std::vector<double>> A = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    std::vector<std::vector<double>> B = {{1.5, 2.5}, {3.5, 4.5}, {5.5, 6.5}};

    // Sequential reference
    double seq_cost = 0.0;
    if (rank == 0) {
        auto [cost, path] = dtw_accelerator::core::dtw_cpu(A, B);
        seq_cost = cost;
        std::cout << "Sequential DTW cost: " << seq_cost << std::endl;
        std::cout << "Path length: " << path.size() << std::endl;
    }

    MPI_Bcast(&seq_cost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI parallel version
    auto [mpi_cost, mpi_path] = dtw_accelerator::parallel::mpi::dtw_mpi(A, B);

    if (rank == 0) {
        std::cout << "MPI DTW cost: " << mpi_cost << std::endl;
        std::cout << "Path length: " << mpi_path.size() << std::endl;

        double error = std::abs(seq_cost - mpi_cost);
        std::cout << "Error: " << error << std::endl;

        if (error < 1e-6) {
            std::cout << "Test PASSED" << std::endl;
        } else {
            std::cout << "Test FAILED" << std::endl;
        }
    }
}

// Performance test for standard dtw and relative MPI implementation with varying sizes
void performance_test_standard() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "\n=== Performance Testing ===" << std::endl;
        std::cout << "Processes: " << size << std::endl;
        std::cout << std::setw(10) << "Size"
                  << std::setw(15) << "Sequential(ms)"
                  << std::setw(15) << "MPI(ms)"
                  << std::setw(15) << "Speedup"
                  << std::setw(15) << "Efficiency" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
    }

    std::vector<int> test_sizes = {100, 1000, 5000, 10000, 20000};

    for (int n : test_sizes) {
        // Generate random sequences
        std::vector<std::vector<double>> A(n, std::vector<double>(2));
        std::vector<std::vector<double>> B(n, std::vector<double>(2));

        if (rank == 0) {
            std::srand(42); // Fixed seed for reproducibility
            for (int i = 0; i < n; ++i) {
                A[i][0] = (double)rand() / RAND_MAX;
                A[i][1] = (double)rand() / RAND_MAX;
                B[i][0] = (double)rand() / RAND_MAX;
                B[i][1] = (double)rand() / RAND_MAX;
            }
        }

        // Broadcast sequences
        for (int i = 0; i < n; ++i) {
            MPI_Bcast(A[i].data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(B[i].data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Sequential timing (only on rank 0)
        double seq_time = 0.0;
        double seq_cost = 0.0;
        if (rank == 0) {
            auto start = std::chrono::high_resolution_clock::now();
            auto [cost, path] = dtw_accelerator::core::dtw_cpu(A, B);
            auto end = std::chrono::high_resolution_clock::now();
            seq_time = std::chrono::duration<double, std::milli>(end - start).count();
            seq_cost = cost;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // MPI parallel timing
        double mpi_start = MPI_Wtime();
        auto [mpi_cost, mpi_path] = dtw_accelerator::parallel::mpi::dtw_mpi(A, B);
        double mpi_end = MPI_Wtime();
        double mpi_time = (mpi_end - mpi_start) * 1000.0; // Convert to milliseconds

        if (rank == 0) {
            double speedup = seq_time / mpi_time;
            double efficiency = speedup / size * 100.0;

            std::cout << std::setw(10) << n
                      << std::setw(15) << std::fixed << std::setprecision(2) << seq_time
                      << std::setw(15) << std::fixed << std::setprecision(2) << mpi_time
                      << std::setw(15) << std::fixed << std::setprecision(2) << speedup
                      << std::setw(14) << std::fixed << std::setprecision(1) << efficiency << "%"
                      << std::endl;

            // Verify correctness
            double error = std::abs(seq_cost - mpi_cost);
            if (error > 1e-6) {
                std::cout << "  Warning: Results differ! Error = " << error << std::endl;
            }
        }

    }
}

// Performance test with varying sizes
void performance_test_constraints() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    using namespace dtw_accelerator::constraints;
    constexpr ConstraintType CT = ConstraintType::SAKOE_CHIBA;
    constexpr int R = 3; // Sakoe-Chiba radius

    if (rank == 0) {
        std::cout << "\n=== Performance Testing ===" << std::endl;
        std::cout << "Using constraint: " << ("Sakoe-Chiba") << std::endl;
        std::cout << "Processes: " << size << std::endl;
        std::cout << std::setw(10) << "Size "
                  << std::setw(15) << "Sequential (ms)"
                  << std::setw(15) << "MPI (ms)"
                  << std::setw(15) << "Speedup"
                  << std::setw(15) << "Efficiency" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
    }

    std::vector<int> test_sizes = {100, 1000, 5000, 10000, 20000};

    for (int n : test_sizes) {
        // Generate random sequences
        std::vector<std::vector<double>> A(n, std::vector<double>(2));
        std::vector<std::vector<double>> B(n, std::vector<double>(2));

        if (rank == 0) {
            std::srand(42); // Fixed seed for reproducibility
            for (int i = 0; i < n; ++i) {
                A[i][0] = (double)rand() / RAND_MAX;
                A[i][1] = (double)rand() / RAND_MAX;
                B[i][0] = (double)rand() / RAND_MAX;
                B[i][1] = (double)rand() / RAND_MAX;
            }
        }

        // Broadcast sequences
        for (int i = 0; i < n; ++i) {
            MPI_Bcast(A[i].data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(B[i].data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Sequential timing (only on rank 0)
        double seq_time = 0.0;
        double seq_cost = 0.0;
        if (rank == 0) {
            auto start = std::chrono::high_resolution_clock::now();
            auto [cost, path] = dtw_accelerator::core::dtw_with_constraint<CT, R>(A, B);
            auto end = std::chrono::high_resolution_clock::now();
            seq_time = std::chrono::duration<double, std::milli>(end - start).count();
            seq_cost = cost;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // MPI parallel timing
        double mpi_start = MPI_Wtime();
        auto [mpi_cost, mpi_path] = dtw_accelerator::parallel::mpi::dtw_mpi_with_constraint<CT, R>(A, B);
        double mpi_end = MPI_Wtime();
        double mpi_time = (mpi_end - mpi_start) * 1000.0; // Convert to milliseconds

        if (rank == 0) {
            double speedup = seq_time / mpi_time;
            double efficiency = speedup / size * 100.0;

            std::cout << std::setw(10) << n
                      << std::setw(15) << std::fixed << std::setprecision(2) << seq_time
                      << std::setw(15) << std::fixed << std::setprecision(2) << mpi_time
                      << std::setw(15) << std::fixed << std::setprecision(2) << speedup
                      << std::setw(14) << std::fixed << std::setprecision(1) << efficiency << "%"
                      << std::endl;

            // Verify correctness
            double error = std::abs(seq_cost - mpi_cost);
            if (error > 1e-6) {
                std::cout << "  Warning: Results differ! Error = " << error << std::endl;
            }
        }

    }
}

// Performance test for FastDTW
void performance_test_fastdtw() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "\n=== Performance Testing (FastDTW) ===" << std::endl;
        std::cout << "Processes: " << size << std::endl;
        std::cout << std::setw(10) << "Size"
                  << std::setw(10) << "Radius"
                  << std::setw(15) << "Sequential(ms)"
                  << std::setw(15) << "MPI(ms)"
                  << std::setw(15) << "Speedup"
                  << std::setw(15) << "Efficiency" << std::endl;
        std::cout << std::string(85, '-') << std::endl;
    }

    std::vector<int> test_sizes = {100, 1000, 5000, 10000, 20000};
    std::vector<int> radii = {1, 3, 5};

    for (int n : test_sizes) {
        for (int radius : radii) {
            // Generate random sequences
            std::vector<std::vector<double>> A(n, std::vector<double>(2));
            std::vector<std::vector<double>> B(n, std::vector<double>(2));

            if (rank == 0) {
                std::srand(42); // Fixed seed for reproducibility
                for (int i = 0; i < n; ++i) {
                    A[i][0] = (double)rand() / RAND_MAX;
                    A[i][1] = (double)rand() / RAND_MAX;
                    B[i][0] = (double)rand() / RAND_MAX;
                    B[i][1] = (double)rand() / RAND_MAX;
                }
            }

            // Broadcast sequences
            for (int i = 0; i < n; ++i) {
                MPI_Bcast(A[i].data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(B[i].data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }

            // Sequential FastDTW timing (only on rank 0)
            double seq_time = 0.0;
            double seq_cost = 0.0;
            if (rank == 0) {
                auto start = std::chrono::high_resolution_clock::now();
                auto [cost, path] = dtw_accelerator::fast::fastdtw_cpu(A, B, radius, 100);
                auto end = std::chrono::high_resolution_clock::now();
                seq_time = std::chrono::duration<double, std::milli>(end - start).count();
                seq_cost = cost;
            }

            MPI_Barrier(MPI_COMM_WORLD);

            // MPI FastDTW timing
            double mpi_start = MPI_Wtime();
            auto [mpi_cost, mpi_path] = dtw_accelerator::parallel::mpi::fastdtw_mpi(A, B, radius, 100);
            double mpi_end = MPI_Wtime();
            double mpi_time = (mpi_end - mpi_start) * 1000.0; // Convert to milliseconds

            if (rank == 0) {
                double speedup = seq_time / mpi_time;
                double efficiency = speedup / size * 100.0;

                std::cout << std::setw(10) << n
                          << std::setw(10) << radius
                          << std::setw(15) << std::fixed << std::setprecision(2) << seq_time
                          << std::setw(15) << std::fixed << std::setprecision(2) << mpi_time
                          << std::setw(15) << std::fixed << std::setprecision(2) << speedup
                          << std::setw(14) << std::fixed << std::setprecision(1) << efficiency << "%"
                          << std::endl;

                // Verify correctness
                double error = std::abs(seq_cost - mpi_cost);
                if (error > 1e-6) {
                    std::cout << "  Warning: Results differ! Error = " << error << std::endl;
                }
            }
        }
    }
}

// Accuracy comparison test
void accuracy_comparison_test() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "\n=== Accuracy Comparison Test ===" << std::endl;
        std::cout << "Comparing FastDTW approximation with standard DTW" << std::endl;
        std::cout << std::setw(10) << "Size"
                  << std::setw(10) << "Radius"
                  << std::setw(15) << "DTW Cost"
                  << std::setw(15) << "FastDTW Cost"
                  << std::setw(15) << "Error %"
                  << std::setw(15) << "Time Ratio" << std::endl;
        std::cout << std::string(85, '-') << std::endl;
    }

    std::vector<int> test_sizes = {100, 1000, 5000, 10000, 20000};
    std::vector<int> radii = {1, 3, 5};

    for (int n : test_sizes) {
        // Generate random sequences
        std::vector<std::vector<double>> A(n, std::vector<double>(2));
        std::vector<std::vector<double>> B(n, std::vector<double>(2));

        if (rank == 0) {
            std::srand(42); // Fixed seed for reproducibility
            for (int i = 0; i < n; ++i) {
                A[i][0] = (double)rand() / RAND_MAX;
                A[i][1] = (double)rand() / RAND_MAX;
                B[i][0] = (double)rand() / RAND_MAX;
                B[i][1] = (double)rand() / RAND_MAX;
            }
        }

        // Broadcast sequences
        for (int i = 0; i < n; ++i) {
            MPI_Bcast(A[i].data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(B[i].data(), 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Compute standard DTW for reference
        double dtw_cost = 0.0;
        double dtw_time = 0.0;
        if (rank == 0) {
            auto start = std::chrono::high_resolution_clock::now();
            auto [cost, path] = dtw_accelerator::core::dtw_cpu(A, B);
            auto end = std::chrono::high_resolution_clock::now();
            dtw_time = std::chrono::duration<double, std::milli>(end - start).count();
            dtw_cost = cost;
        }

        MPI_Bcast(&dtw_cost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&dtw_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        for (int radius : radii) {
            // Compute FastDTW
            double mpi_start = MPI_Wtime();
            auto [fastdtw_cost, fastdtw_path] = dtw_accelerator::parallel::mpi::fastdtw_mpi(A, B, radius, 100);
            double mpi_end = MPI_Wtime();
            double fastdtw_time = (mpi_end - mpi_start) * 1000.0;

            if (rank == 0) {
                double error_pct = std::abs(dtw_cost - fastdtw_cost) / dtw_cost * 100.0;
                double time_ratio = dtw_time / fastdtw_time;

                std::cout << std::setw(10) << n
                          << std::setw(10) << radius
                          << std::setw(15) << std::fixed << std::setprecision(2) << dtw_cost
                          << std::setw(15) << std::fixed << std::setprecision(2) << fastdtw_cost
                          << std::setw(14) << std::fixed << std::setprecision(2) << error_pct << "%"
                          << std::setw(15) << std::fixed << std::setprecision(2) << time_ratio << "x"
                          << std::endl;
            }
        }
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "DTW MPI Implementation Test Suite" << std::endl;
        std::cout << "Running with " << size << " processes" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Run tests
    test_small_example();
    performance_test_standard();
    performance_test_constraints();
    performance_test_fastdtw();
    accuracy_comparison_test();

    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "All tests completed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}


//TODO:
//- Implemented the MPI version of fast_dtw: fastdtw_mpi
//- Updated mpi_dtw_profiler.hpp and test_mpi_profiler.cpp to benchmark and debug also the function fast_dtw_mpi
//- Updated test_mpi.cpp to verify correctness of fast_dtw and accuracy comparison between dtw_cpu and fast_dtw
//- Integrated fast_dtw_mpi result in main.cpp for the general benchmark of all implementations