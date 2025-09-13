/**
 * @file benchmark_mpi_only.cpp
 * @brief MPI-specific benchmark for process scaling
 * @author UobinoPino
 * @date 2024
 *
 * This file benchmarks ONLY MPI implementations.
 * Should be compiled with MPI but WITHOUT OpenMP.
 * Run with: mpirun -np N ./benchmark_mpi_only
 */

#include "dtw_accelerator/dtw_accelerator.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include <string>
#include <map>

#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace dtw_accelerator;
using namespace std::chrono;

struct BenchmarkConfig {
    std::vector<int> problem_sizes;
    int dimensions = 3;
    int num_runs = 3;
    int block_size = 64;
    int fastdtw_radius = 2;
    int fastdtw_min_size = 100;
};

struct BenchmarkResult {
    std::string backend = "MPI";
    int num_processes;
    std::string constraint_type;
    int problem_size;
    double time_ms;
};

class MPIBenchmark {
private:
    BenchmarkConfig config;
    std::vector<BenchmarkResult> results;
    int mpi_rank = 0;
    int mpi_size = 1;

    DoubleTimeSeries generate_series(int size, int dim, unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(-10.0, 10.0);

        DoubleTimeSeries series(size, dim);
        for (int i = 0; i < size; ++i) {
            for (int d = 0; d < dim; ++d) {
                series[i][d] = dis(gen);
            }
        }
        return series;
    }

    template<typename Func>
    double measure_time(Func&& func, int num_runs = 3) {
        double total_time = 0.0;

        // Warm-up
        func();

#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        for (int run = 0; run < num_runs; ++run) {
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            auto start = high_resolution_clock::now();
            func();
            auto end = high_resolution_clock::now();

            duration<double, std::milli> elapsed = end - start;
            total_time += elapsed.count();
        }

        return total_time / num_runs;
    }

    void log_result(const std::string& constraint, int size, double time_ms) {
        if (mpi_rank == 0) {
            results.push_back({"MPI", mpi_size, constraint, size, time_ms});

            std::cout << std::left << std::setw(10) << "MPI"
                      << std::setw(10) << mpi_size
                      << std::setw(15) << constraint
                      << std::setw(10) << size
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << time_ms << " ms"
                      << std::endl;
        }
    }

public:
    MPIBenchmark() {
#ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

        // Initialize problem sizes
        for (int exp = 8; exp <= 15; ++exp) {
            config.problem_sizes.push_back(1 << exp);  // 2^exp
        }
    }

    void benchmark_no_constraints() {
#ifdef USE_MPI
        if (mpi_rank == 0) {
            std::cout << "\n======== MPI BENCHMARK (No Constraints) ========\n";
            std::cout << "MPI Processes: " << mpi_size << "\n";
            std::cout << std::left << std::setw(10) << "Backend"
                      << std::setw(10) << "Processes"
                      << std::setw(15) << "Constraint"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::endl;
            std::cout << std::string(57, '-') << std::endl;
        }

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 42);
            auto B = generate_series(size, config.dimensions, 43);

            MPI_Barrier(MPI_COMM_WORLD);

            double mpi_time = measure_time([&]() {
                auto result = dtw_mpi<distance::MetricType::EUCLIDEAN>(
                    A, B, config.block_size, 0, MPI_COMM_WORLD);
            }, config.num_runs);

            log_result("None", size, mpi_time);
        }
#else
        std::cout << "ERROR: MPI not enabled in compilation!\n";
#endif
    }

    void benchmark_sakoe_chiba() {
#ifdef USE_MPI
        if (mpi_rank == 0) {
            std::cout << "\n======== MPI BENCHMARK (Sakoe-Chiba) ========\n";
            std::cout << "MPI Processes: " << mpi_size << "\n";
            std::cout << std::left << std::setw(10) << "Backend"
                      << std::setw(10) << "Processes"
                      << std::setw(15) << "Constraint"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::endl;
            std::cout << std::string(57, '-') << std::endl;
        }

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 44);
            auto B = generate_series(size, config.dimensions, 45);

            execution::MPIStrategy mpi_strategy(config.block_size, 0);

            MPI_Barrier(MPI_COMM_WORLD);

            double mpi_time = measure_time([&]() {
                auto result = dtw_sakoe_chiba<distance::MetricType::EUCLIDEAN, 10>(
                    A, B, mpi_strategy);
            }, config.num_runs);

            log_result("Sakoe-Chiba", size, mpi_time);
        }
#endif
    }

    void benchmark_itakura() {
#ifdef USE_MPI
        if (mpi_rank == 0) {
            std::cout << "\n======== MPI BENCHMARK (Itakura) ========\n";
            std::cout << "MPI Processes: " << mpi_size << "\n";
            std::cout << std::left << std::setw(10) << "Backend"
                      << std::setw(10) << "Processes"
                      << std::setw(15) << "Constraint"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::endl;
            std::cout << std::string(57, '-') << std::endl;
        }

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 46);
            auto B = generate_series(size, config.dimensions, 47);

            execution::MPIStrategy mpi_strategy(config.block_size, 0);

            MPI_Barrier(MPI_COMM_WORLD);

            double mpi_time = measure_time([&]() {
                auto result = dtw_itakura<distance::MetricType::EUCLIDEAN, 2.0>(
                    A, B, mpi_strategy);
            }, config.num_runs);

            log_result("Itakura", size, mpi_time);
        }
#endif
    }

    void benchmark_fastdtw() {
#ifdef USE_MPI
        if (mpi_rank == 0) {
            std::cout << "\n======== MPI BENCHMARK (FastDTW) ========\n";
            std::cout << "MPI Processes: " << mpi_size << "\n";
            std::cout << std::left << std::setw(10) << "Backend"
                      << std::setw(10) << "Processes"
                      << std::setw(15) << "Constraint"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::endl;
            std::cout << std::string(57, '-') << std::endl;
        }

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 48);
            auto B = generate_series(size, config.dimensions, 49);

            MPI_Barrier(MPI_COMM_WORLD);

            double mpi_time = measure_time([&]() {
                auto result = fastdtw_mpi<distance::MetricType::EUCLIDEAN>(
                    A, B, config.fastdtw_radius, config.fastdtw_min_size,
                    config.block_size, 0, MPI_COMM_WORLD);
            }, config.num_runs);

            log_result("FastDTW", size, mpi_time);
        }
#endif
    }

    void save_results(const std::string& filename) {
        if (mpi_rank != 0) return;

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        // Write header
        file << "Backend,Processes,Constraint,Size,Time_ms\n";

        // Write data
        for (const auto& result : results) {
            file << result.backend << ","
                 << result.num_processes << ","
                 << result.constraint_type << ","
                 << result.problem_size << ","
                 << result.time_ms << "\n";
        }

        file.close();
        std::cout << "\nMPI results saved to " << filename << std::endl;
    }

    void run_all() {
#ifdef USE_MPI
        if (mpi_rank == 0) {
            std::cout << "========================================\n";
            std::cout << "       MPI SCALING BENCHMARK           \n";
            std::cout << "========================================\n";
            std::cout << "Problem sizes: " << config.problem_sizes.front()
                      << " to " << config.problem_sizes.back() << "\n";
            std::cout << "Dimensions: " << config.dimensions << "\n";
            std::cout << "MPI Processes: " << mpi_size << "\n";
            std::cout << "Runs per measurement: " << config.num_runs << "\n";
            std::cout << "========================================\n";
        }

        benchmark_no_constraints();
        benchmark_sakoe_chiba();
        benchmark_itakura();
        benchmark_fastdtw();

        // Save with process count in filename
        std::string filename = "dtw_benchmark_mpi_" + std::to_string(mpi_size) + ".csv";
        save_results(filename);
#else
        std::cerr << "ERROR: MPI support not compiled!\n";
        std::cerr << "Recompile with -DUSE_MPI=ON\n";
#endif
    }
};

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif

    MPIBenchmark benchmark;
    benchmark.run_all();

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
