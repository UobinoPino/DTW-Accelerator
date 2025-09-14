/**
 * @file benchmark_cuda_only.cpp
 * @brief CUDA-specific benchmark for GPU acceleration
 * @author UobinoPino
 * @date 2024
 *
 * This file benchmarks ONLY CUDA implementations.
 * Should be compiled with CUDA support but WITHOUT OpenMP/MPI.
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

#ifdef USE_CUDA
#include "dtw_accelerator/execution/parallel/cuda/cuda_dtw.hpp"
#endif

using namespace dtw_accelerator;
using namespace std::chrono;

struct BenchmarkConfig {
    std::vector<int> problem_sizes;
    std::vector<int> tile_sizes = {32, 64, 128, 256, 512, 1024, 2048};  // CUDA tile sizes
    int dimensions = 3;
    int num_runs = 3;
    int fastdtw_radius = 2;
    int fastdtw_min_size = 100;
};

struct BenchmarkResult {
    std::string backend = "CUDA";
    int tile_size;
    std::string constraint_type;
    int problem_size;
    double time_ms;
};

class CUDABenchmark {
private:
    BenchmarkConfig config;
    std::vector<BenchmarkResult> results;
    bool cuda_available = false;

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

        // Warm-up run (important for GPU)
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

    void log_result(int tile_size, const std::string& constraint,
                    int size, double time_ms) {
        results.push_back({"CUDA", tile_size, constraint, size, time_ms});

        std::cout << std::left << std::setw(10) << "CUDA"
                  << std::setw(10) << tile_size
                  << std::setw(15) << constraint
                  << std::setw(10) << size
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << time_ms << " ms"
                  << std::endl;
    }

public:
    CUDABenchmark() {
        // Initialize problem sizes - focus on larger sizes where GPU excels
        for (int exp = 8; exp <= 14; ++exp) {
            config.problem_sizes.push_back(1 << exp);  // 2^exp
        }

#ifdef USE_CUDA
        // Check CUDA availability and get device info
        cuda_available = cuda::is_available();

#endif
    }

    void benchmark_no_constraints() {
#ifdef USE_CUDA
        if (!cuda_available) {
            std::cout << "CUDA is not available on this system!\n";
            return;
        }

        std::cout << "\n======== CUDA BENCHMARK (No Constraints) ========\n";
        std::cout << std::left << std::setw(10) << "Backend"
                  << std::setw(10) << "TileSize"
                  << std::setw(15) << "Constraint"
                  << std::setw(10) << "Size"
                  << std::setw(12) << "Time"
                  << std::endl;
        std::cout << std::string(57, '-') << std::endl;

        for (int tile_size : config.tile_sizes) {
            for (int size : config.problem_sizes) {
                auto A = generate_series(size, config.dimensions, 42);
                auto B = generate_series(size, config.dimensions, 43);

                double cuda_time = measure_time([&]() {
                    auto result = cuda::dtw_cuda<distance::MetricType::EUCLIDEAN>(
                        A, B, tile_size);
                }, config.num_runs);

                log_result(tile_size, "None", size, cuda_time);
            }
        }
#else
        std::cout << "ERROR: CUDA not enabled in compilation!\n";
        std::cout << "Compile with -DUSE_CUDA=ON and CUDA toolkit installed\n";
#endif
    }

    void benchmark_memory_transfer() {
#ifdef USE_CUDA
        if (!cuda_available) {
            return;
        }

        std::cout << "\n======== CUDA MEMORY TRANSFER ANALYSIS ========\n";
        std::cout << std::left << std::setw(10) << "Size"
                  << std::setw(15) << "H2D Time (ms)"
                  << std::setw(15) << "Compute (ms)"
                  << std::setw(15) << "D2H Time (ms)"
                  << std::setw(15) << "Total (ms)"
                  << std::setw(15) << "Compute %"
                  << std::endl;
        std::cout << std::string(85, '-') << std::endl;

        // Test memory transfer overhead for different sizes
        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 50);
            auto B = generate_series(size, config.dimensions, 51);

            // Measure with detailed timing (if supported by CUDA implementation)
            double total_time = measure_time([&]() {
                auto result = cuda::dtw_cuda<distance::MetricType::EUCLIDEAN>(
                    A, B, 64);
            }, config.num_runs);

            // For now, just show total time
            // In a real implementation, you'd measure H2D, compute, and D2H separately
            std::cout << std::left << std::setw(10) << size
                      << std::setw(15) << "N/A"
                      << std::setw(15) << "N/A"
                      << std::setw(15) << "N/A"
                      << std::fixed << std::setprecision(2)
                      << std::setw(15) << total_time
                      << std::setw(15) << "N/A"
                      << std::endl;
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
        file << "Backend,TileSize,Constraint,Size,Time_ms,Device\n";

        // Write data
        for (const auto& result : results) {
            file << result.backend << ","
                 << result.tile_size << ","
                 << result.constraint_type << ","
                 << result.problem_size << ","
                 << result.time_ms << "\n";
        }

        file.close();
        std::cout << "\nCUDA results saved to " << filename << std::endl;
    }

    void run_all() {
#ifdef USE_CUDA
        std::cout << "========================================\n";
        std::cout << "       CUDA BENCHMARK SUITE           \n";
        std::cout << "========================================\n";

        if (!cuda_available) {
            std::cout << "ERROR: CUDA is not available on this system!\n";
            std::cout << "Possible reasons:\n";
            std::cout << "  - No NVIDIA GPU detected\n";
            std::cout << "  - CUDA drivers not installed\n";
            std::cout << "  - GPU compute capability mismatch\n";
            return;
        }

        std::cout << "Problem sizes: " << config.problem_sizes.front()
                  << " to " << config.problem_sizes.back() << "\n";
        std::cout << "Dimensions: " << config.dimensions << "\n";
        std::cout << "Tile sizes: ";
        for (int t : config.tile_sizes) std::cout << t << " ";
        std::cout << "\n";
        std::cout << "Runs per measurement: " << config.num_runs << "\n";
        std::cout << "========================================\n";

        benchmark_no_constraints();
        benchmark_memory_transfer();

        save_results("dtw_benchmark_cuda.csv");
#else
        std::cerr << "ERROR: CUDA support not compiled!\n";
        std::cerr << "Recompile with -DUSE_CUDA=ON and ensure CUDA toolkit is installed\n";
#endif
    }
};

int main(int argc, char** argv) {
    CUDABenchmark benchmark;
    benchmark.run_all();
    return 0;
}