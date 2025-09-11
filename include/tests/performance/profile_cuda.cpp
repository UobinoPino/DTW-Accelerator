#include "dtw_accelerator/dtw_accelerator.hpp"
#include "benchmark_utils.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace dtw_accelerator;
using namespace dtw_benchmark;

class StrategyBenchmark {
private:
    DataGenerator generator;
    BenchmarkReporter reporter;

public:
    StrategyBenchmark(const std::string& name) : generator(42), reporter(name) {}

    void run_strategy_comparison(size_t n, size_t m, size_t dimensions) {
        std::cout << "\n--- Benchmarking DTW Strategies ---\n";
        std::cout << "Sequence A: " << n << " points, " << dimensions << " dimensions\n";
        std::cout << "Sequence B: " << m << " points, " << dimensions << " dimensions\n";

        // Calculate memory using Matrix class
        size_t matrix_memory = MemoryEstimator::estimate_dtw_matrix_memory(n, m);
        std::cout << "Matrix size: " << matrix_memory/(1024.0*1024.0) << " MB (contiguous memory)\n";

        // Calculate TimeSeries memory
        size_t ts_memory = MemoryEstimator::estimate_time_series_memory(n, dimensions) +
                           MemoryEstimator::estimate_time_series_memory(m, dimensions);
        std::cout << "TimeSeries size: " << ts_memory/(1024.0*1024.0) << " MB (contiguous memory)\n\n";

        // Generate test data using TimeSeries
        auto A = generator.generate_random_series(n, dimensions);
        auto B = generator.generate_random_series(m, dimensions);


#ifdef USE_CUDA
        // CUDA Strategy
        if (cuda::is_available()) {
            std::cout << "Testing CUDA Strategy..." << std::flush;
            auto result = run_benchmark("CUDA", [&]() {
                return cuda::dtw_cuda<MetricType::EUCLIDEAN>(A, B);
            }, n, m, dimensions);
            reporter.add_result(result);
            std::cout << " Done (" << result.time_ms << " ms)\n";
        }
#endif

    }


    void print_results() {
        reporter.print_summary();
    }

    void export_results(const std::string& filename) {
        reporter.export_csv(filename);
    }
};

// Main benchmark executable
int main(int argc, char** argv) {
    std::cout << "==============================================\n";
    std::cout << "    DTW Accelerator Performance Benchmark    \n";
    std::cout << "      (Using TimeSeries & Matrix Classes)    \n";
    std::cout << "==============================================\n";


    // Print system info
    std::cout << "\nSystem Configuration:\n";
    std::cout << "- TimeSeries: Contiguous memory, row-major layout\n";
    std::cout << "- Matrix: Contiguous memory implementation\n";

#ifdef USE_CUDA
    if (cuda::is_available()) {
        std::cout << "- CUDA: " << dtw_accelerator::parallel::cuda::get_cuda_device_info() << "\n";
    }
#endif
    std::cout << "\n";

    // Test different problem sizes
    struct TestCase {
        size_t n, m, dim;
        std::string description;
    };

    std::vector<TestCase> test_cases = {
          //  {1000, 1000, 3, "Large (1000x1000x3)"},
            //  {5000, 5000, 3, "XXLarge (5000x5000x3)"},
            // {10000, 10000, 3, "XXXLarge (10000x10000x3)"},
            // {20000, 20000, 3, "Huge (20000x20000x3)"},
          //  {16384, 16384, 3, "Huge x (25000x25000x3)"}
           // {30000, 30000, 3, "Huge xx (30000x30000x3)"}
            {32768, 32768, 3, "Huge xxx (32768x32768x3)"}

    };

    for (const auto& tc : test_cases) {
        std::cout << "\n\n############################################\n";
        std::cout << "Test Case: " << tc.description << "\n";
        std::cout << "############################################\n";

        StrategyBenchmark benchmark(tc.description);

        // Run memory comparison first
        // Run strategy comparison
        benchmark.run_strategy_comparison(tc.n, tc.m, tc.dim);

        benchmark.print_results();

        // Export results to CSV
        std::string filename = "benchmark_" + std::to_string(tc.n) + "x" +
                               std::to_string(tc.m) + "x" + std::to_string(tc.dim) + ".csv";
        benchmark.export_results(filename);
    }

    // Final memory efficiency report
    std::cout << "\n\n============================================\n";
    std::cout << "        Memory Efficiency Summary           \n";
    std::cout << "============================================\n";
    std::cout << "TimeSeries class benefits:\n";
    std::cout << "- Contiguous memory layout (single allocation)\n";
    std::cout << "- Better cache locality (row-major access)\n";
    std::cout << "- Direct pointer access for MPI/CUDA\n";
    std::cout << "- Reduced memory overhead vs nested vectors\n";
    std::cout << "- Zero-copy conversion to GPU buffers\n";
    std::cout << "\nMatrix class benefits:\n";
    std::cout << "- Contiguous memory for cost matrix\n";
    std::cout << "- Efficient row-wise access\n";
    std::cout << "- Reduced memory fragmentation\n";


    return 0;
}
