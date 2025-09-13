/**
 * @file benchmark_openmp_only.cpp
 * @brief OpenMP-specific benchmark for thread scaling
 * @author UobinoPino
 * @date 2024
 *
 * This file benchmarks ONLY OpenMP implementations.
 * Should be compiled with OpenMP flags but WITHOUT MPI.
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

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace dtw_accelerator;
using namespace std::chrono;

struct BenchmarkConfig {
    std::vector<int> problem_sizes;
    std::vector<int> thread_counts = {1, 2, 4, 8};
    int dimensions = 3;
    int num_runs = 3;
    int block_size = 512;
    int fastdtw_radius = 2;
    int fastdtw_min_size = 100;
};

struct BenchmarkResult {
    std::string backend = "OpenMP";
    int num_threads;
    std::string constraint_type;
    int problem_size;
    double time_ms;
};

class OpenMPBenchmark {
private:
    BenchmarkConfig config;
    std::vector<BenchmarkResult> results;

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

        for (int run = 0; run < num_runs; ++run) {
            auto start = high_resolution_clock::now();
            func();
            auto end = high_resolution_clock::now();

            duration<double, std::milli> elapsed = end - start;
            total_time += elapsed.count();
        }

        return total_time / num_runs;
    }

    void log_result(int threads, const std::string& constraint,
                    int size, double time_ms) {
        results.push_back({"OpenMP", threads, constraint, size, time_ms});

        std::cout << std::left << std::setw(10) << "OpenMP"
                  << std::setw(10) << threads
                  << std::setw(15) << constraint
                  << std::setw(10) << size
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << time_ms << " ms"
                  << std::endl;
    }

public:
    OpenMPBenchmark() {
        // Initialize problem sizes
        for (int exp = 8; exp <= 15; ++exp) {
            config.problem_sizes.push_back(1 << exp);  // 2^exp
        }
    }

    void benchmark_no_constraints() {
#ifdef USE_OPENMP
        std::cout << "\n======== OPENMP SCALING (No Constraints) ========\n";
        std::cout << std::left << std::setw(10) << "Backend"
                  << std::setw(10) << "Threads"
                  << std::setw(15) << "Constraint"
                  << std::setw(10) << "Size"
                  << std::setw(12) << "Time"
                  << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int threads : config.thread_counts) {
            for (int size : config.problem_sizes) {
                auto A = generate_series(size, config.dimensions, 42);
                auto B = generate_series(size, config.dimensions, 43);

                double omp_time = measure_time([&]() {
                    auto result = dtw_openmp<distance::MetricType::EUCLIDEAN>(
                        A, B, threads, config.block_size);
                }, config.num_runs);

                log_result(threads, "None", size, omp_time);
            }
        }
#else
        std::cout << "ERROR: OpenMP not enabled in compilation!\n";
        std::cout << "Compile with -fopenmp flag\n";
#endif
    }

    void benchmark_sakoe_chiba() {
#ifdef USE_OPENMP
        std::cout << "\n======== OPENMP SCALING (Sakoe-Chiba) ========\n";
        std::cout << std::left << std::setw(10) << "Backend"
                  << std::setw(10) << "Threads"
                  << std::setw(15) << "Constraint"
                  << std::setw(10) << "Size"
                  << std::setw(12) << "Time"
                  << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int threads : config.thread_counts) {
            for (int size : config.problem_sizes) {
                auto A = generate_series(size, config.dimensions, 44);
                auto B = generate_series(size, config.dimensions, 45);

                execution::OpenMPStrategy omp_strategy(threads, config.block_size);

                double omp_time = measure_time([&]() {
                    auto result = dtw_sakoe_chiba<distance::MetricType::EUCLIDEAN, 10>(
                        A, B, omp_strategy);
                }, config.num_runs);

                log_result(threads, "Sakoe-Chiba", size, omp_time);
            }
        }
#endif
    }

    void benchmark_itakura() {
#ifdef USE_OPENMP
        std::cout << "\n======== OPENMP SCALING (Itakura) ========\n";
        std::cout << std::left << std::setw(10) << "Backend"
                  << std::setw(10) << "Threads"
                  << std::setw(15) << "Constraint"
                  << std::setw(10) << "Size"
                  << std::setw(12) << "Time"
                  << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int threads : config.thread_counts) {
            for (int size : config.problem_sizes) {
                auto A = generate_series(size, config.dimensions, 46);
                auto B = generate_series(size, config.dimensions, 47);

                execution::OpenMPStrategy omp_strategy(threads, config.block_size);

                double omp_time = measure_time([&]() {
                    auto result = dtw_itakura<distance::MetricType::EUCLIDEAN, 2.0>(
                        A, B, omp_strategy);
                }, config.num_runs);

                log_result(threads, "Itakura", size, omp_time);
            }
        }
#endif
    }

    void benchmark_fastdtw() {
#ifdef USE_OPENMP
        std::cout << "\n======== OPENMP SCALING (FastDTW) ========\n";
        std::cout << std::left << std::setw(10) << "Backend"
                  << std::setw(10) << "Threads"
                  << std::setw(15) << "Constraint"
                  << std::setw(10) << "Size"
                  << std::setw(12) << "Time"
                  << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int threads : config.thread_counts) {
            for (int size : config.problem_sizes) {
                auto A = generate_series(size, config.dimensions, 48);
                auto B = generate_series(size, config.dimensions, 49);

                double omp_time = measure_time([&]() {
                    auto result = fastdtw_openmp<distance::MetricType::EUCLIDEAN>(
                        A, B, config.fastdtw_radius, config.fastdtw_min_size,
                        threads, config.block_size);
                }, config.num_runs);

                log_result(threads, "FastDTW", size, omp_time);
            }
        }
#endif
    }

    void save_results(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        // Write header
        file << "Backend,Threads,Constraint,Size,Time_ms\n";

        // Write data
        for (const auto& result : results) {
            file << result.backend << ","
                 << result.num_threads << ","
                 << result.constraint_type << ","
                 << result.problem_size << ","
                 << result.time_ms << "\n";
        }

        file.close();
        std::cout << "\nOpenMP results saved to " << filename << std::endl;
    }

    void run_all() {
#ifdef USE_OPENMP
        std::cout << "========================================\n";
        std::cout << "      OPENMP SCALING BENCHMARK         \n";
        std::cout << "========================================\n";
        std::cout << "Problem sizes: " << config.problem_sizes.front()
                  << " to " << config.problem_sizes.back() << "\n";
        std::cout << "Dimensions: " << config.dimensions << "\n";
        std::cout << "Thread counts: ";
        for (int t : config.thread_counts) std::cout << t << " ";
        std::cout << "\n";
        std::cout << "Max available threads: " << omp_get_max_threads() << "\n";
        std::cout << "Runs per measurement: " << config.num_runs << "\n";
        std::cout << "========================================\n";

        benchmark_no_constraints();
        benchmark_sakoe_chiba();
        benchmark_itakura();
        benchmark_fastdtw();

        save_results("dtw_benchmark_openmp.csv");
#else
        std::cerr << "ERROR: OpenMP support not compiled!\n";
        std::cerr << "Recompile with -DUSE_OPENMP=ON and -fopenmp\n";
#endif
    }
};

int main(int argc, char** argv) {
    OpenMPBenchmark benchmark;
    benchmark.run_all();
    return 0;
}