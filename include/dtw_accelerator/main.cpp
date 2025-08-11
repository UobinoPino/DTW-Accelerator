#include "dtw_accelerator.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>

#ifdef USE_MPI
#include <mpi.h>
#include "dtw_accelerator/parallel/mpi_dtw.hpp"
#endif

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

template<typename Func>
auto measure_time(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = func();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    return std::make_pair(result, duration);
}

int main(int argc, char** argv) {
    int rank = 0, size = 1;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    int sequence_length;
    int dimensions = 2;

    if (rank == 0) {
        std::cout << "==============================================\n";
        std::cout << "DTW Performance Comparison\n";
        std::cout << "==============================================\n";

        if (argc > 1) {
            sequence_length = std::atoi(argv[1]);
        } else {
            std::cout << "Enter the length of time series: ";
            std::cin >> sequence_length;
        }

        if (argc > 2) {
            dimensions = std::atoi(argv[2]);
        }

        std::cout << "Sequence length: " << sequence_length << std::endl;
        std::cout << "Dimensions: " << dimensions << std::endl;
#ifdef USE_MPI
        std::cout << "MPI Processes: " << size << std::endl;
#endif
        std::cout << "----------------------------------------------\n";
    }

#ifdef USE_MPI
    // Broadcast sequence_length and dimensions to all processes
    MPI_Bcast(&sequence_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    // Generate random series (only on rank 0 for consistency)
    std::vector<std::vector<double>> series_a, series_b;

    if (rank == 0) {
        series_a = generate_random_series(sequence_length, dimensions);
        series_b = generate_random_series(sequence_length, dimensions);
        std::cout << "Time series generated with length " << sequence_length << std::endl;
        std::cout << "----------------------------------------------\n";
    }

    // Results storage
    struct Result {
        std::string name;
        double cost;
        double time_ms;
        double speedup;
    };
    std::vector<Result> results;

    // CPU Sequential version (only on rank 0)
    if (rank == 0) {
        std::cout << "\n=== Sequential DTW (CPU) ===" << std::endl;
        auto [result_dtw, time_dtw] = measure_time([&]() {
            return dtw_accelerator::dtw_cpu(series_a, series_b);
        });
        auto& [cost_dtw, path_dtw] = result_dtw;
        std::cout << "Cost: " << cost_dtw << std::endl;
        std::cout << "Time: " << time_dtw << " ms" << std::endl;

        results.push_back({"Sequential CPU", cost_dtw, time_dtw, 1.0});

        // Constrained DTW tests
        std::cout << "\n=== DTW with Sakoe-Chiba Constraint ===" << std::endl;
        auto [result_wc_sc, time_wc_sc] = measure_time([&]() {
            return dtw_accelerator::core::dtw_with_constraint<
                    dtw_accelerator::constraints::ConstraintType::SAKOE_CHIBA, 3>(series_a, series_b);
        });
        auto& [cost_dtw_wc_sc, path_dtw_wc_sc] = result_wc_sc;
        std::cout << "Cost: " << cost_dtw_wc_sc << std::endl;
        std::cout << "Time: " << time_wc_sc << " ms" << std::endl;
        std::cout << "Speedup: " << time_dtw / time_wc_sc << "x" << std::endl;

        results.push_back({"Sakoe-Chiba R=3", cost_dtw_wc_sc, time_wc_sc, time_dtw / time_wc_sc});

        // FastDTW tests
        std::cout << "\n=== FastDTW CPU ===" << std::endl;
        auto [result_fastdtw, time_fastdtw] = measure_time([&]() {
            return dtw_accelerator::fastdtw_cpu(series_a, series_b);
        });
        auto& [cost_fastdtw, path_fastdtw] = result_fastdtw;
        std::cout << "Cost: " << cost_fastdtw << std::endl;
        std::cout << "Time: " << time_fastdtw << " ms" << std::endl;
        std::cout << "Speedup: " << time_dtw / time_fastdtw << "x" << std::endl;

        results.push_back({"FastDTW", cost_fastdtw, time_fastdtw, time_dtw / time_fastdtw});

        auto [result_fastdtw_r3, time_fastdtw_r3] = measure_time([&]() {
            return dtw_accelerator::fastdtw_cpu(series_a, series_b, 3);
        });
        auto& [cost_fastdtw_r3, path_fastdtw_r3] = result_fastdtw_r3;
        std::cout << "\nFastDTW with radius=3:" << std::endl;
        std::cout << "Cost: " << cost_fastdtw_r3 << std::endl;
        std::cout << "Time: " << time_fastdtw_r3 << " ms" << std::endl;
        std::cout << "Speedup: " << time_dtw / time_fastdtw_r3 << "x" << std::endl;

        results.push_back({"FastDTW R=3", cost_fastdtw_r3, time_fastdtw_r3, time_dtw / time_fastdtw_r3});
    }

#ifdef USE_MPI
    // MPI Parallel versions
    if (rank == 0) {
        std::cout << "\n=== MPI Parallel DTW ===" << std::endl;
        std::cout << "Running with " << size << " processes..." << std::endl;
    }

    // Ensure all processes have the data
    if (rank != 0) {
        series_a.resize(sequence_length, std::vector<double>(dimensions));
        series_b.resize(sequence_length, std::vector<double>(dimensions));
    }

    // Broadcast the series data to all processes
    for (int i = 0; i < sequence_length; ++i) {
        MPI_Bcast(series_a[i].data(), dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(series_b[i].data(), dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Test wavefront (standard) MPI implementation
    double start_time = MPI_Wtime();
    auto [result_mpi, path_mpi] = dtw_accelerator::parallel::mpi::dtw_mpi(series_a, series_b);
    double end_time = MPI_Wtime();
    double time_mpi = (end_time - start_time) * 1000.0; // Convert to ms

    if (rank == 0) {
        std::cout << "MPI Wavefront DTW:" << std::endl;
        std::cout << "Cost: " << result_mpi << std::endl;
        std::cout << "Time: " << time_mpi << " ms" << std::endl;

        // Calculate speedup
        double sequential_time = results.empty() ? time_mpi : results[0].time_ms;
        double speedup = sequential_time / time_mpi;
        std::cout << "Speedup vs Sequential: " << speedup << "x" << std::endl;
        std::cout << "Parallel Efficiency: " << (speedup / size) * 100 << "%" << std::endl;

        results.push_back({"MPI Wavefront", result_mpi, time_mpi, speedup});

    }

    // Test MPI with Sakoe-Chiba constraint (R=3)
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    auto [result_mpi_sc3, path_mpi_sc3] = dtw_accelerator::parallel::mpi::dtw_mpi_with_constraint<
            dtw_accelerator::constraints::ConstraintType::SAKOE_CHIBA, 3>(series_a, series_b);
    end_time = MPI_Wtime();
    double time_mpi_sc3 = (end_time - start_time) * 1000.0;

    if (rank == 0) {
        std::cout << "\nMPI Sakoe-Chiba (R=3):" << std::endl;
        std::cout << "Cost: " << result_mpi_sc3 << std::endl;
        std::cout << "Time: " << time_mpi_sc3 << " ms" << std::endl;
        double baseline_time = results.empty() ? time_mpi : results[0].time_ms;
        std::cout << "Speedup vs Sequential: " << baseline_time / time_mpi_sc3 << "x" << std::endl;
        std::cout << "Parallel Efficiency: " << (baseline_time / time_mpi_sc3 / size) * 100 << "%" << std::endl;
        results.push_back({"MPI Sakoe-Chiba R=3", result_mpi_sc3, time_mpi_sc3,
                           baseline_time / time_mpi_sc3});

    }




#endif

    // Summary table (only on rank 0)
    if (rank == 0 && !results.empty()) {
        std::cout << "\n==============================================\n";
        std::cout << "PERFORMANCE SUMMARY\n";
        std::cout << "==============================================\n";
        std::cout << std::left << std::setw(20) << "Method"
                  << std::right << std::setw(15) << "Cost"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(15) << "Speedup" << std::endl;
        std::cout << "----------------------------------------------\n";

        for (const auto& r : results) {
            std::cout << std::left << std::setw(20) << r.name
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << r.cost
                      << std::setw(15) << std::fixed << std::setprecision(2) << r.time_ms
                      << std::setw(15) << std::fixed << std::setprecision(2) << r.speedup << "x" << std::endl;
        }

        std::cout << "==============================================\n";

        // Find the best performing method
        auto best = std::min_element(results.begin(), results.end(),
                                     [](const Result& a, const Result& b) { return a.time_ms < b.time_ms; });
        if (best != results.end()) {
            std::cout << "\nBest performing method: " << best->name
                      << " (" << best->speedup << "x speedup)" << std::endl;
        }
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
