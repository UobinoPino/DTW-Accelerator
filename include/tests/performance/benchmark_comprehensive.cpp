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

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace dtw_accelerator;
using namespace std::chrono;

// Benchmark configuration
struct BenchmarkConfig {
    std::vector<int> problem_sizes;  // 2^7 to 2^15
    int dimensions = 3;
    int num_runs = 2;  // Average over multiple runs
    int fastdtw_radius = 2;
    int fastdtw_min_size = 100;
    int sakoe_chiba_radius = 25;
    double itakura_slope = 2.0;
};

// Result structure
struct BenchmarkResult {
    std::string strategy_name;
    std::string constraint_type;
    int problem_size;
    double time_ms;
    double speedup;
};

// Benchmark class
class DTWBenchmarkSuite {
private:
    BenchmarkConfig config;
    std::vector<BenchmarkResult> results;
    std::map<int, double> baseline_times;  // Store sequential baseline times per size
    int mpi_rank = 0;
    int mpi_size = 1;

    // Generate random time series
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

    // Measure execution time
    template<typename Func>
    double measure_time(Func&& func, int num_runs = 3) {
        double total_time = 0.0;

        for (int run = 0; run < num_runs; ++run) {
            auto start = high_resolution_clock::now();
            func();
            auto end = high_resolution_clock::now();

            duration<double, std::milli> elapsed = end - start;
            total_time += elapsed.count();
        }

        return total_time / num_runs;
    }

    // Log result
    void log_result(const std::string& strategy, const std::string& constraint,
                    int size, double time_ms) {
        double speedup = 1.0;
        if (baseline_times.count(size) > 0) {
            speedup = baseline_times[size] / time_ms;
        }

        results.push_back({strategy, constraint, size, time_ms, speedup});

        if (mpi_rank == 0) {
            std::cout << std::left << std::setw(20) << strategy
                      << std::setw(15) << constraint
                      << std::setw(10) << size
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << time_ms << " ms"
                      << std::setw(10) << speedup << "x"
                      << std::endl;
        }
    }

public:
    DTWBenchmarkSuite() {
        // Initialize problem sizes: 2^7 to 2^14
        for (int exp = 7; exp <= 15; ++exp) {
            config.problem_sizes.push_back(1 << exp);  // 2^exp
        }

#ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    }

    // Benchmark NO CONSTRAINTS
    void benchmark_no_constraints() {
        if (mpi_rank == 0) {
            std::cout << "\n======== NO CONSTRAINTS BENCHMARK ========\n";
            std::cout << std::left << std::setw(20) << "Strategy"
                      << std::setw(15) << "Constraint"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::setw(10) << "Speedup"
                      << std::endl;
            std::cout << std::string(67, '-') << std::endl;
        }

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 42);
            auto B = generate_series(size, config.dimensions, 43);

            // Sequential (baseline)
            double seq_time = measure_time([&]() {
                auto result = dtw_sequential<MetricType::EUCLIDEAN>(A, B);
            }, config.num_runs);
            baseline_times[size] = seq_time;  // Avoid division by zero
            log_result("Sequential", "None", size, seq_time);  // Use "None" consistently

            // Blocked
            double blocked_time = measure_time([&]() {
                auto result = dtw_blocked<MetricType::EUCLIDEAN>(A, B, 64);
            }, config.num_runs);
            log_result("Blocked", "None", size, blocked_time);  // Use "None" consistently

#ifdef USE_OPENMP
            // OpenMP
            double omp_time = measure_time([&]() {
                auto result = dtw_openmp<MetricType::EUCLIDEAN>(A, B, 0, 64);
            }, config.num_runs);
            log_result("OpenMP", "None", size, omp_time);
#endif

#ifdef USE_MPI
            // MPI
            MPI_Barrier(MPI_COMM_WORLD);
            double mpi_time = measure_time([&]() {
                auto result = dtw_mpi<MetricType::EUCLIDEAN>(A, B, 64, 0);
            }, config.num_runs);

            // Only root reports
            if (mpi_rank == 0) {
                log_result("MPI", "None", size, mpi_time);
            }
#endif

#ifdef USE_CUDA
            // CUDA
            if (cuda::is_available()) {  // CUDA efficient for larger sizes
                double cuda_time = measure_time([&]() {
                    auto result = cuda::dtw_cuda<MetricType::EUCLIDEAN>(A, B, 64);
                }, config.num_runs);
                log_result("CUDA", "None", size, cuda_time);
            }
#endif
        }
    }

    // Benchmark SAKOE-CHIBA CONSTRAINT
    void benchmark_sakoe_chiba() {
        if (mpi_rank == 0) {
            std::cout << "\n======== SAKOE-CHIBA CONSTRAINT BENCHMARK ========\n";
            std::cout << std::left << std::setw(20) << "Strategy"
                      << std::setw(15) << "Constraint"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::setw(10) << "Speedup"
                      << std::endl;
            std::cout << std::string(67, '-') << std::endl;
        }

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 44);
            auto B = generate_series(size, config.dimensions, 45);

            execution::SequentialStrategy seq_strategy;
            execution::BlockedStrategy blocked_strategy(64);

            // Sequential with Sakoe-Chiba
            double seq_sc_time = measure_time([&]() {
                auto result = dtw_sakoe_chiba<MetricType::EUCLIDEAN, 10>(
                    A, B, seq_strategy);
            }, config.num_runs);
            log_result("Sequential", "Sakoe-Chiba", size, seq_sc_time);

            // Blocked with Sakoe-Chiba
            double blocked_sc_time = measure_time([&]() {
                auto result = dtw_sakoe_chiba<MetricType::EUCLIDEAN, 10>(
                    A, B, blocked_strategy);
            }, config.num_runs);
            log_result("Blocked", "Sakoe-Chiba", size, blocked_sc_time);

#ifdef USE_OPENMP
            // OpenMP with Sakoe-Chiba
            execution::OpenMPStrategy omp_strategy(0, 64);
            double omp_sc_time = measure_time([&]() {
                auto result = dtw_sakoe_chiba<MetricType::EUCLIDEAN, 10>(
                    A, B, omp_strategy);
            }, config.num_runs);
            log_result("OpenMP", "Sakoe-Chiba", size, omp_sc_time);
#endif

#ifdef USE_MPI
            // MPI with Sakoe-Chiba
            execution::MPIStrategy mpi_strategy(64, 0);
            MPI_Barrier(MPI_COMM_WORLD);
            double mpi_sc_time = measure_time([&]() {
                auto result = dtw_sakoe_chiba<MetricType::EUCLIDEAN, 10>(
                    A, B, mpi_strategy);
            }, config.num_runs);

            if (mpi_rank == 0) {
                log_result("MPI", "Sakoe-Chiba", size, mpi_sc_time);
            }
#endif
        }
    }

    // Benchmark ITAKURA CONSTRAINT
    void benchmark_itakura() {
        if (mpi_rank == 0) {
            std::cout << "\n======== ITAKURA CONSTRAINT BENCHMARK ========\n";
            std::cout << std::left << std::setw(20) << "Strategy"
                      << std::setw(15) << "Constraint"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::setw(10) << "Speedup"
                      << std::endl;
            std::cout << std::string(67, '-') << std::endl;
        }

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 46);
            auto B = generate_series(size, config.dimensions, 47);

            execution::SequentialStrategy seq_strategy;
            execution::BlockedStrategy blocked_strategy(64);

            // Sequential with Itakura
            double seq_it_time = measure_time([&]() {
                auto result = dtw_itakura<MetricType::EUCLIDEAN, 2.0>(
                    A, B, seq_strategy);
            }, config.num_runs);
            log_result("Sequential", "Itakura", size, seq_it_time);

            // Blocked with Itakura
            double blocked_it_time = measure_time([&]() {
                auto result = dtw_itakura<MetricType::EUCLIDEAN, 2.0>(
                    A, B, blocked_strategy);
            }, config.num_runs);
            log_result("Blocked", "Itakura", size, blocked_it_time);

#ifdef USE_OPENMP
            // OpenMP with Itakura
            execution::OpenMPStrategy omp_strategy(0, 64);
            double omp_it_time = measure_time([&]() {
                auto result = dtw_itakura<MetricType::EUCLIDEAN, 2.0>(
                    A, B, omp_strategy);
            }, config.num_runs);
            log_result("OpenMP", "Itakura", size, omp_it_time);
#endif

#ifdef USE_MPI
            // MPI with Itakura
            execution::MPIStrategy mpi_strategy(64, 0);
            MPI_Barrier(MPI_COMM_WORLD);
            double mpi_it_time = measure_time([&]() {
                auto result = dtw_itakura<MetricType::EUCLIDEAN, 2.0>(
                    A, B, mpi_strategy);
            }, config.num_runs);

            if (mpi_rank == 0) {
                log_result("MPI", "Itakura", size, mpi_it_time);
            }
#endif
        }
    }

    // Benchmark FASTDTW
    void benchmark_fastdtw() {
        if (mpi_rank == 0) {
            std::cout << "\n======== FASTDTW BENCHMARK ========\n";
            std::cout << std::left << std::setw(20) << "Strategy"
                      << std::setw(15) << "Constraint"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::setw(10) << "Speedup"
                      << std::endl;
            std::cout << std::string(67, '-') << std::endl;
        }

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 48);
            auto B = generate_series(size, config.dimensions, 49);

            // Sequential FastDTW
            double seq_fast_time = measure_time([&]() {
                auto result = fastdtw_sequential<MetricType::EUCLIDEAN>(
                    A, B, config.fastdtw_radius, config.fastdtw_min_size);
            }, config.num_runs);
            log_result("Sequential", "FastDTW", size, seq_fast_time);

            // Blocked FastDTW
            double blocked_fast_time = measure_time([&]() {
                auto result = fastdtw_blocked<MetricType::EUCLIDEAN>(
                    A, B, config.fastdtw_radius, config.fastdtw_min_size, 64);
            }, config.num_runs);
            log_result("Blocked", "FastDTW", size, blocked_fast_time);

#ifdef USE_OPENMP
            // OpenMP FastDTW
            double omp_fast_time = measure_time([&]() {
                auto result = fastdtw_openmp<MetricType::EUCLIDEAN>(
                    A, B, config.fastdtw_radius, config.fastdtw_min_size, 0, 64);
            }, config.num_runs);
            log_result("OpenMP", "FastDTW", size, omp_fast_time);
#endif

#ifdef USE_MPI
            // MPI FastDTW
            MPI_Barrier(MPI_COMM_WORLD);
            double mpi_fast_time = measure_time([&]() {
                auto result = fastdtw_mpi<MetricType::EUCLIDEAN>(
                    A, B, config.fastdtw_radius, config.fastdtw_min_size, 64, 0);
            }, config.num_runs);

            if (mpi_rank == 0) {
                log_result("MPI", "FastDTW", size, mpi_fast_time);
            }
#endif
        }
    }

    // Save results to CSV
    void save_results(const std::string& filename) {
        if (mpi_rank != 0) return;

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        // Write header
        file << "Strategy,Constraint,Size,Time_ms,Speedup\n";

        // Write data
        for (const auto& result : results) {
            file << result.strategy_name << ","
                 << result.constraint_type << ","
                 << result.problem_size << ","
                 << result.time_ms << ","
                 << result.speedup << "\n";
        }

        file.close();
        std::cout << "\nResults saved to " << filename << std::endl;
    }

    // Run all benchmarks
    void run_all() {
        if (mpi_rank == 0) {
            std::cout << "========================================\n";
            std::cout << "    DTW COMPREHENSIVE BENCHMARK SUITE   \n";
            std::cout << "========================================\n";
            std::cout << "Problem sizes: 2^7 to 2^14\n";
            std::cout << "Dimensions: " << config.dimensions << "\n";
            std::cout << "Runs per measurement: " << config.num_runs << "\n";
#ifdef USE_MPI
            std::cout << "MPI Processes: " << mpi_size << "\n";
#endif
#ifdef USE_OPENMP
            std::cout << "OpenMP Threads: " << omp_get_max_threads() << "\n";
#endif
#ifdef USE_CUDA
            if (cuda::is_available()) {
                std::cout << "CUDA: " << cuda::device_info() << "\n";
            }
#endif
            std::cout << "========================================\n";
        }

        benchmark_no_constraints();
        benchmark_sakoe_chiba();
        benchmark_itakura();
        benchmark_fastdtw();

        save_results("dtw_benchmark_results.csv");
    }
};

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif

    DTWBenchmarkSuite benchmark;
    benchmark.run_all();

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}