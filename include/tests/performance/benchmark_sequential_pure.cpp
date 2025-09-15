/**
 * @file benchmark_sequential_pure.cpp
 * @brief Pure sequential benchmark without any parallel library contamination
 * @author UobinoPino
 * @date 2024
 *
 * This file benchmarks ONLY the sequential implementations without
 * any OpenMP or MPI libraries linked, providing a true baseline.
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

// IMPORTANT: This file should be compiled WITHOUT -fopenmp or MPI flags
// to get true sequential baseline performance

using namespace dtw_accelerator;
using namespace std::chrono;

// Benchmark configuration
struct BenchmarkConfig {
    std::vector<int> problem_sizes;
    int dimensions = 3;
    int num_runs = 2;
    int fastdtw_radius = 2;
    int fastdtw_min_size = 100;
};

// Result structure
struct BenchmarkResult {
    std::string strategy_name;
    std::string constraint_type;
    int problem_size;
    double time_ms;
};

class SequentialBenchmark {
private:
    BenchmarkConfig config;
    std::vector<BenchmarkResult> results;

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

        // Warm-up run
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

    void log_result(const std::string& strategy, const std::string& constraint,
                    int size, double time_ms) {
        results.push_back({strategy, constraint, size, time_ms});

        std::cout << std::left << std::setw(20) << strategy
                  << std::setw(15) << constraint
                  << std::setw(10) << size
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << time_ms << " ms"
                  << std::endl;
    }

public:
    SequentialBenchmark() {
        // Initialize problem sizes 1555555
        for (int exp = 8; exp <= 14; ++exp) {
            config.problem_sizes.push_back(1 << exp);  // 2^exp
        }
    }

    void benchmark_no_constraints() {
        std::cout << "\n======== SEQUENTIAL BASELINE (No Constraints) ========\n";
        std::cout << std::left << std::setw(20) << "Strategy"
                  << std::setw(15) << "Constraint"
                  << std::setw(10) << "Size"
                  << std::setw(12) << "Time"
                  << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 42);
            auto B = generate_series(size, config.dimensions, 43);

            // Sequential
            double seq_time = measure_time([&]() {
                auto result = dtw_sequential<distance::MetricType::EUCLIDEAN>(A, B);
            }, config.num_runs);
            log_result("Sequential", "None", size, seq_time);

            // Blocked
            double blocked_time = measure_time([&]() {
                auto result = dtw_blocked<distance::MetricType::EUCLIDEAN>(A, B, 512);
            }, config.num_runs);
            log_result("Blocked", "None", size, blocked_time);
        }
    }

    void benchmark_sakoe_chiba() {
        std::cout << "\n======== SEQUENTIAL BASELINE (Sakoe-Chiba) ========\n";
        std::cout << std::left << std::setw(20) << "Strategy"
                  << std::setw(15) << "Constraint"
                  << std::setw(10) << "Size"
                  << std::setw(12) << "Time"
                  << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 44);
            auto B = generate_series(size, config.dimensions, 45);

            execution::SequentialStrategy seq_strategy;
            execution::BlockedStrategy blocked_strategy(64);

            // Sequential with Sakoe-Chiba
            double seq_time = measure_time([&]() {
                auto result = dtw_sakoe_chiba<distance::MetricType::EUCLIDEAN, 10>(
                    A, B, seq_strategy);
            }, config.num_runs);
            log_result("Sequential", "Sakoe-Chiba", size, seq_time);

            // Blocked with Sakoe-Chiba
            double blocked_time = measure_time([&]() {
                auto result = dtw_sakoe_chiba<distance::MetricType::EUCLIDEAN, 10>(
                    A, B, blocked_strategy);
            }, config.num_runs);
            log_result("Blocked", "Sakoe-Chiba", size, blocked_time);
        }
    }

    void benchmark_itakura() {
        std::cout << "\n======== SEQUENTIAL BASELINE (Itakura) ========\n";
        std::cout << std::left << std::setw(20) << "Strategy"
                  << std::setw(15) << "Constraint"
                  << std::setw(10) << "Size"
                  << std::setw(12) << "Time"
                  << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 46);
            auto B = generate_series(size, config.dimensions, 47);

            execution::SequentialStrategy seq_strategy;
            execution::BlockedStrategy blocked_strategy(64);

            // Sequential with Itakura
            double seq_time = measure_time([&]() {
                auto result = dtw_itakura<distance::MetricType::EUCLIDEAN, 2.0>(
                    A, B, seq_strategy);
            }, config.num_runs);
            log_result("Sequential", "Itakura", size, seq_time);

            // Blocked with Itakura
            double blocked_time = measure_time([&]() {
                auto result = dtw_itakura<distance::MetricType::EUCLIDEAN, 2.0>(
                    A, B, blocked_strategy);
            }, config.num_runs);
            log_result("Blocked", "Itakura", size, blocked_time);
        }
    }

    void benchmark_fastdtw() {
        std::cout << "\n======== SEQUENTIAL BASELINE (FastDTW) ========\n";
        std::cout << std::left << std::setw(20) << "Strategy"
                  << std::setw(15) << "Constraint"
                  << std::setw(10) << "Size"
                  << std::setw(12) << "Time"
                  << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 48);
            auto B = generate_series(size, config.dimensions, 49);

            // Sequential FastDTW
            double seq_time = measure_time([&]() {
                auto result = fastdtw_sequential<distance::MetricType::EUCLIDEAN>(
                        A, B, config.fastdtw_radius, config.fastdtw_min_size);
            }, config.num_runs);
            log_result("Sequential", "FastDTW", size, seq_time);

            // Blocked FastDTW
            double blocked_time = measure_time([&]() {
                auto result = fastdtw_blocked<distance::MetricType::EUCLIDEAN>(
                        A, B, config.fastdtw_radius, config.fastdtw_min_size, 64);
            }, config.num_runs);
            log_result("Blocked", "FastDTW", size, blocked_time);
        }
    }

    void save_results(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        // Write header
        file << "Strategy,Constraint,Size,Time_ms\n";

        // Write data
        for (const auto& result : results) {
            file << result.strategy_name << ","
                 << result.constraint_type << ","
                 << result.problem_size << ","
                 << result.time_ms << "\n";
        }

        file.close();
        std::cout << "\nBaseline results saved to " << filename << std::endl;
    }

    void run_all() {
        std::cout << "========================================\n";
        std::cout << "    PURE SEQUENTIAL BASELINE BENCHMARK \n";
        std::cout << "========================================\n";
        std::cout << "Problem sizes: " << config.problem_sizes.front()
                  << " to " << config.problem_sizes.back() << "\n";
        std::cout << "Dimensions: " << config.dimensions << "\n";
        std::cout << "Runs per measurement: " << config.num_runs << "\n";
        std::cout << "========================================\n";

        benchmark_no_constraints();
        benchmark_sakoe_chiba();
        benchmark_itakura();
        benchmark_fastdtw();

        save_results("dtw_baseline_sequential.csv");
    }
};

int main(int argc, char** argv) {
    SequentialBenchmark benchmark;
    benchmark.run_all();
    return 0;
}