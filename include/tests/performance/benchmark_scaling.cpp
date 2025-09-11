//////#include "dtw_accelerator/dtw_accelerator.hpp"
//////#include "benchmark_utils.hpp"
//////#include <iostream>
//////#include <vector>
//////#include <cmath>
//////
//////#ifdef USE_OPENMP
//////#include <omp.h>
//////#endif
//////
//////using namespace dtw_accelerator;
//////using namespace dtw_benchmark;
//////
//////class ScalingBenchmark {
//////private:
//////    DataGenerator generator;
//////
//////public:
//////    ScalingBenchmark() : generator(42) {}
//////
//////    void test_weak_scaling() {
//////        std::cout << "\n=== Weak Scaling Test ===\n";
//////        std::cout << "(Problem size per thread remains constant)\n\n";
//////
//////#ifdef USE_OPENMP
//////        int max_threads = omp_get_max_threads();
//////        size_t base_size = 1000;
//////
//////        std::cout << std::left << std::setw(10) << "Threads"
//////                  << std::setw(15) << "Problem Size"
//////                  << std::setw(12) << "Time (ms)"
//////                  << std::setw(12) << "Efficiency\n";
//////        std::cout << std::string(50, '-') << "\n";
//////
//////        double baseline_time = 0.0;
//////
//////        for (int threads = 1; threads <= max_threads; threads *= 2) {
//////            size_t problem_size = base_size * threads;
//////            auto A = generator.generate_random_series(problem_size, 3);
//////            auto B = generator.generate_random_series(problem_size, 3);
//////
//////            Timer timer;
//////            timer.start();
//////            auto result = dtw_openmp<MetricType::EUCLIDEAN>(A, B, threads, 64);
//////            double time = timer.elapsed_ms();
//////
//////            if (threads == 1) baseline_time = time;
//////            double efficiency = (baseline_time / time) * 100.0;
//////
//////            std::cout << std::left << std::setw(10) << threads
//////                      << std::setw(15) << problem_size
//////                      << std::fixed << std::setprecision(2)
//////                      << std::setw(12) << time
//////                      << std::setw(11) << efficiency << "%\n";
//////        }
//////#else
//////        std::cout << "OpenMP not available - skipping weak scaling test\n";
//////#endif
//////    }
//////
//////    void test_strong_scaling() {
//////        std::cout << "\n=== Strong Scaling Test ===\n";
//////        std::cout << "(Fixed problem size, varying thread count)\n\n";
//////
//////#ifdef USE_OPENMP
//////        int max_threads = omp_get_max_threads();
//////        size_t problem_size = 10000;
//////
//////        auto A = generator.generate_random_series(problem_size, 3);
//////        auto B = generator.generate_random_series(problem_size, 3);
//////
//////        std::cout << std::left << std::setw(10) << "Threads"
//////                  << std::setw(12) << "Time (ms)"
//////                  << std::setw(12) << "Speedup"
//////                  << std::setw(12) << "Efficiency\n";
//////        std::cout << std::string(45, '-') << "\n";
//////
//////        double baseline_time = 0.0;
//////
//////        for (int threads = 1; threads <= max_threads; threads *= 2) {
//////            Timer timer;
//////            timer.start();
//////            auto result = dtw_openmp<MetricType::EUCLIDEAN>(A, B, threads, 64);
//////            double time = timer.elapsed_ms();
//////
//////            if (threads == 1) baseline_time = time;
//////            double speedup = baseline_time / time;
//////            double efficiency = (speedup / threads) * 100.0;
//////
//////            std::cout << std::left << std::setw(10) << threads
//////                      << std::fixed << std::setprecision(2)
//////                      << std::setw(12) << time
//////                      << std::setw(11) << speedup << "x"
//////                      << std::setw(11) << efficiency << "%\n";
//////        }
//////#else
//////        std::cout << "OpenMP not available - skipping strong scaling test\n";
//////#endif
//////    }
//////
//////
//////    void test_block_size_scaling() {
//////        std::cout << "\n=== Block Size Scaling Test ===\n";
//////        std::cout << "(Finding optimal block size for cache)\n\n";
//////
//////        size_t problem_size = 10000;
//////        auto A = generator.generate_random_series(problem_size, 3);
//////        auto B = generator.generate_random_series(problem_size, 3);
//////
//////        std::vector<int> block_sizes = {16, 32, 64, 128, 256, 512};
//////
//////        std::cout << std::left << std::setw(12) << "Block Size"
//////                  << std::setw(12) << "Time (ms)"
//////                  << std::setw(15) << "Speedup\n";
//////        std::cout << std::string(60, '-') << "\n";
//////
//////        // Get baseline with no blocking
//////        Timer timer;
//////        timer.start();
//////        auto baseline_result = dtw_sequential<MetricType::EUCLIDEAN>(A, B);
//////        double baseline_time = timer.elapsed_ms();
//////
//////        for (int block_size : block_sizes) {
//////            timer.start();
//////            auto result = dtw_blocked<MetricType::EUCLIDEAN>(A, B, block_size);
//////            double time = timer.elapsed_ms();
//////
//////            double speedup = baseline_time / time;
//////
//////            std::cout << std::left << std::setw(12) << block_size
//////                      << std::fixed << std::setprecision(2)
//////                      << std::setw(12) << time
//////                      << std::setw(14) << speedup << "\n";
//////        }
//////    }
//////};
//////
//////int main() {
//////    std::cout << "==============================================\n";
//////    std::cout << "      DTW Accelerator Scaling Analysis       \n";
//////    std::cout << "==============================================\n";
//////
//////    ScalingBenchmark benchmark;
//////
//////    benchmark.test_block_size_scaling();
//////    benchmark.test_strong_scaling();
//////    benchmark.test_weak_scaling();
//////
//////    return 0;
//////}
////
////#include "dtw_accelerator/dtw_accelerator.hpp"
////#include "benchmark_utils.hpp"
////#include <iostream>
////#include <vector>
////#include <cmath>
////#include <fstream>
////#include <sstream>
////#include <iomanip>
////#include <map>
////#include <algorithm>
////#include <string>
////
////#ifdef USE_OPENMP
////#include <omp.h>
////#endif
////
////#ifdef USE_MPI
////#include <mpi.h>
////#endif
////
////#ifdef USE_CUDA
////#include "dtw_accelerator/execution/parallel/cuda/cuda_dtw.hpp"
////#endif
////
////using namespace dtw_accelerator;
////using namespace dtw_benchmark;
////
////// Structure to hold benchmark results
////struct BenchmarkRecord {
////    std::string backend;
////    std::string test_type;
////    int threads_or_ranks;
////    size_t problem_size;
////    double time_ms;
////    double speedup;
////    double efficiency;
////};
////
////class ScalingBenchmark {
////private:
////    DataGenerator generator;
////    std::vector<BenchmarkRecord> all_results;
////    bool is_mpi_root;
////    int mpi_rank;
////    int mpi_size;
////
////public:
////    ScalingBenchmark() : generator(42), is_mpi_root(true), mpi_rank(0), mpi_size(1) {
////#ifdef USE_MPI
////        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
////        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
////        is_mpi_root = (mpi_rank == 0);
////#endif
////    }
////
////    // OpenMP Weak Scaling Test
////    void test_openmp_weak_scaling() {
////#ifdef USE_OPENMP
////        if (!is_mpi_root) return;
////
////        std::cout << "\n=== OpenMP Weak Scaling Test ===\n";
////        std::cout << "(Problem size per thread remains constant)\n\n";
////
////        int max_threads = omp_get_max_threads();
////        size_t base_size = 1000;
////
////        std::cout << std::left << std::setw(10) << "Threads"
////                  << std::setw(15) << "Problem Size"
////                  << std::setw(12) << "Time (ms)"
////                  << std::setw(12) << "Efficiency\n";
////        std::cout << std::string(50, '-') << "\n";
////
////        double baseline_time = 0.0;
////
////        for (int threads = 1; threads <= max_threads; threads *= 2) {
////            size_t problem_size = base_size * threads;
////            auto A = generator.generate_random_series(problem_size, 3);
////            auto B = generator.generate_random_series(problem_size, 3);
////
////            Timer timer;
////            timer.start();
////            auto result = dtw_openmp<MetricType::EUCLIDEAN>(A, B, threads, 64);
////            double time = timer.elapsed_ms();
////
////            if (threads == 1) baseline_time = time;
////            double efficiency = (baseline_time / time) * 100.0;
////
////            std::cout << std::left << std::setw(10) << threads
////                      << std::setw(15) << problem_size
////                      << std::fixed << std::setprecision(2)
////                      << std::setw(12) << time
////                      << std::setw(11) << efficiency << "%\n";
////
////            // Store result
////            all_results.push_back({
////                "OpenMP", "WeakScaling", threads, problem_size,
////                time, baseline_time/time, efficiency/100.0
////            });
////        }
////#else
////        if (is_mpi_root) {
////            std::cout << "OpenMP not available - skipping OpenMP weak scaling test\n";
////        }
////#endif
////    }
////
////    // OpenMP Strong Scaling Test
////    void test_openmp_strong_scaling() {
////#ifdef USE_OPENMP
////        if (!is_mpi_root) return;
////
////        std::cout << "\n=== OpenMP Strong Scaling Test ===\n";
////        std::cout << "(Fixed problem size, varying thread count)\n\n";
////
////        int max_threads = omp_get_max_threads();
////        size_t problem_size = 5000;
////
////        auto A = generator.generate_random_series(problem_size, 3);
////        auto B = generator.generate_random_series(problem_size, 3);
////
////        std::cout << std::left << std::setw(10) << "Threads"
////                  << std::setw(12) << "Time (ms)"
////                  << std::setw(12) << "Speedup"
////                  << std::setw(12) << "Efficiency\n";
////        std::cout << std::string(45, '-') << "\n";
////
////        double baseline_time = 0.0;
////
////        for (int threads = 1; threads <= max_threads; threads *= 2) {
////            Timer timer;
////            timer.start();
////            auto result = dtw_openmp<MetricType::EUCLIDEAN>(A, B, threads, 64);
////            double time = timer.elapsed_ms();
////
////            if (threads == 1) baseline_time = time;
////            double speedup = baseline_time / time;
////            double efficiency = (speedup / threads) * 100.0;
////
////            std::cout << std::left << std::setw(10) << threads
////                      << std::fixed << std::setprecision(2)
////                      << std::setw(12) << time
////                      << std::setw(11) << speedup << "x"
////                      << std::setw(11) << efficiency << "%\n";
////
////            // Store result
////            all_results.push_back({
////                "OpenMP", "StrongScaling", threads, problem_size,
////                time, speedup, efficiency/100.0
////            });
////        }
////#else
////        if (is_mpi_root) {
////            std::cout << "OpenMP not available - skipping OpenMP strong scaling test\n";
////        }
////#endif
////    }
////
////    // MPI Weak Scaling Test
////    void test_mpi_weak_scaling() {
////#ifdef USE_MPI
////        std::cout << "\n=== MPI Weak Scaling Test ===\n";
////        std::cout << "(Problem size per process remains constant)\n\n";
////
////        size_t base_size = 1000;
////        size_t problem_size = base_size * mpi_size;
////
////        auto A = generator.generate_random_series(problem_size, 3);
////        auto B = generator.generate_random_series(problem_size, 3);
////
////        MPI_Barrier(MPI_COMM_WORLD);
////
////        Timer timer;
////        timer.start();
////        auto result = dtw_mpi<MetricType::EUCLIDEAN>(A, B, 64, 0, MPI_COMM_WORLD);
////        double time = timer.elapsed_ms();
////
////        // Gather timing results to root
////        double* all_times = nullptr;
////        if (is_mpi_root) {
////            all_times = new double[mpi_size];
////        }
////        MPI_Gather(&time, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
////
////        if (is_mpi_root) {
////            // Compute average time
////            double avg_time = 0.0;
////            for (int i = 0; i < mpi_size; i++) {
////                avg_time += all_times[i];
////            }
////            avg_time /= mpi_size;
////
////            // Assuming we have baseline from single process run
////            double baseline_time = avg_time; // This would ideally come from a 1-process run
////            double efficiency = (baseline_time / avg_time) * 100.0;
////
////            std::cout << std::left << std::setw(10) << "Processes"
////                      << std::setw(15) << "Problem Size"
////                      << std::setw(12) << "Time (ms)"
////                      << std::setw(12) << "Efficiency\n";
////            std::cout << std::string(50, '-') << "\n";
////
////            std::cout << std::left << std::setw(10) << mpi_size
////                      << std::setw(15) << problem_size
////                      << std::fixed << std::setprecision(2)
////                      << std::setw(12) << avg_time
////                      << std::setw(11) << efficiency << "%\n";
////
////            // Store result
////            all_results.push_back({
////                "MPI", "WeakScaling", mpi_size, problem_size,
////                avg_time, 1.0, efficiency/100.0
////            });
////
////            delete[] all_times;
////        }
////#else
////        if (is_mpi_root) {
////            std::cout << "MPI not available - skipping MPI weak scaling test\n";
////        }
////#endif
////    }
////
////    // MPI Strong Scaling Test
////    void test_mpi_strong_scaling() {
////#ifdef USE_MPI
////        std::cout << "\n=== MPI Strong Scaling Test ===\n";
////        std::cout << "(Fixed problem size, varying process count)\n\n";
////
////        size_t problem_size = 5000;
////
////        auto A = generator.generate_random_series(problem_size, 3);
////        auto B = generator.generate_random_series(problem_size, 3);
////
////        MPI_Barrier(MPI_COMM_WORLD);
////
////        Timer timer;
////        timer.start();
////        auto result = dtw_mpi<MetricType::EUCLIDEAN>(A, B, 64, 0, MPI_COMM_WORLD);
////        double time = timer.elapsed_ms();
////
////        // Gather timing results to root
////        double* all_times = nullptr;
////        if (is_mpi_root) {
////            all_times = new double[mpi_size];
////        }
////        MPI_Gather(&time, 1, MPI_DOUBLE, all_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
////
////        if (is_mpi_root) {
////            // Compute average time
////            double avg_time = 0.0;
////            for (int i = 0; i < mpi_size; i++) {
////                avg_time += all_times[i];
////            }
////            avg_time /= mpi_size;
////
////            // For strong scaling, we'd need baseline from 1-process run
////            // Here we'll use the current time as reference
////            static double baseline_time = -1.0;
////            if (baseline_time < 0) baseline_time = avg_time * mpi_size; // Estimate
////
////            double speedup = baseline_time / avg_time;
////            double efficiency = (speedup / mpi_size) * 100.0;
////
////            std::cout << std::left << std::setw(10) << "Processes"
////                      << std::setw(12) << "Time (ms)"
////                      << std::setw(12) << "Speedup"
////                      << std::setw(12) << "Efficiency\n";
////            std::cout << std::string(45, '-') << "\n";
////
////            std::cout << std::left << std::setw(10) << mpi_size
////                      << std::fixed << std::setprecision(2)
////                      << std::setw(12) << avg_time
////                      << std::setw(11) << speedup << "x"
////                      << std::setw(11) << efficiency << "%\n";
////
////            // Store result
////            all_results.push_back({
////                "MPI", "StrongScaling", mpi_size, problem_size,
////                avg_time, speedup, efficiency/100.0
////            });
////
////            delete[] all_times;
////        }
////#else
////        if (is_mpi_root) {
////            std::cout << "MPI not available - skipping MPI strong scaling test\n";
////        }
////#endif
////    }
////
////    // CUDA Scaling Tests
////    void test_cuda_scaling() {
////#ifdef USE_CUDA
////        if (!is_mpi_root) return;
////
////        if (!cuda::is_available()) {
////            std::cout << "CUDA device not available - skipping CUDA tests\n";
////            return;
////        }
////
////        std::cout << "\n=== CUDA Scaling Test ===\n";
////        std::cout << "Device: " << cuda::device_info() << "\n\n";
////
////        std::vector<size_t> problem_sizes = {500, 1000, 2000, 4000, 8000, 16000};
////
////        std::cout << std::left << std::setw(15) << "Problem Size"
////                  << std::setw(12) << "Time (ms)"
////                  << std::setw(15) << "vs Sequential"
////                  << std::setw(15) << "vs OpenMP\n";
////        std::cout << std::string(60, '-') << "\n";
////
////        for (size_t size : problem_sizes) {
////            auto A = generator.generate_random_series(size, 3);
////            auto B = generator.generate_random_series(size, 3);
////
////            // CUDA timing
////            Timer cuda_timer;
////            cuda_timer.start();
////            auto cuda_result = cuda::dtw_cuda<MetricType::EUCLIDEAN>(A, B, 64);
////            double cuda_time = cuda_timer.elapsed_ms();
////
////            // Sequential baseline
////            Timer seq_timer;
////            seq_timer.start();
////            auto seq_result = dtw_sequential<MetricType::EUCLIDEAN>(A, B);
////            double seq_time = seq_timer.elapsed_ms();
////
////            double speedup_vs_seq = seq_time / cuda_time;
////
////            std::cout << std::left << std::setw(15) << size
////                      << std::fixed << std::setprecision(2)
////                      << std::setw(12) << cuda_time
////                      << std::setw(14) << speedup_vs_seq << "x";
////
////#ifdef USE_OPENMP
////            // OpenMP comparison
////            Timer omp_timer;
////            omp_timer.start();
////            auto omp_result = dtw_openmp<MetricType::EUCLIDEAN>(A, B, 4, 64);
////            double omp_time = omp_timer.elapsed_ms();
////
////            double speedup_vs_omp = omp_time / cuda_time;
////            std::cout << std::setw(14) << speedup_vs_omp << "x";
////#else
////            std::cout << std::setw(15) << "N/A";
////#endif
////            std::cout << "\n";
////
////            // Store result
////            all_results.push_back({
////                "CUDA", "Scaling", 1, size,
////                cuda_time, speedup_vs_seq, speedup_vs_seq
////            });
////        }
////#else
////        if (is_mpi_root) {
////            std::cout << "CUDA not available - skipping CUDA scaling test\n";
////        }
////#endif
////    }
////
////    // Block size scaling (original test)
////    void test_block_size_scaling() {
////        if (!is_mpi_root) return;
////
////        std::cout << "\n=== Block Size Scaling Test ===\n";
////        std::cout << "(Finding optimal block size for cache)\n\n";
////
////        size_t problem_size = 5000;
////        auto A = generator.generate_random_series(problem_size, 3);
////        auto B = generator.generate_random_series(problem_size, 3);
////
////        std::vector<int> block_sizes = {16, 32, 64, 128, 256, 512};
////
////        std::cout << std::left << std::setw(12) << "Block Size"
////                  << std::setw(12) << "Time (ms)"
////                  << std::setw(15) << "Speedup\n";
////        std::cout << std::string(40, '-') << "\n";
////
////        // Get baseline with no blocking
////        Timer timer;
////        timer.start();
////        auto baseline_result = dtw_sequential<MetricType::EUCLIDEAN>(A, B);
////        double baseline_time = timer.elapsed_ms();
////
////        for (int block_size : block_sizes) {
////            timer.start();
////            auto result = dtw_blocked<MetricType::EUCLIDEAN>(A, B, block_size);
////            double time = timer.elapsed_ms();
////
////            double speedup = baseline_time / time;
////
////            std::cout << std::left << std::setw(12) << block_size
////                      << std::fixed << std::setprecision(2)
////                      << std::setw(12) << time
////                      << std::setw(14) << speedup << "x\n";
////
////            // Store result
////            all_results.push_back({
////                                          "Blocked", "BlockSize", block_size, problem_size,
////                                          time, speedup, speedup
////                                  });
////        }
////    }
////
////    // Export results to CSV
////    void export_to_csv(const std::string& filename) {
////        if (!is_mpi_root) return;
////
////        std::ofstream file(filename);
////        if (!file.is_open()) {
////            std::cerr << "Failed to open " << filename << " for writing\n";
////            return;
////        }
////
////        // Write header
////        file << "Backend,TestType,ThreadsOrRanks,ProblemSize,Time_ms,Speedup,Efficiency\n";
////
////        // Write data
////        for (const auto& result : all_results) {
////            file << result.backend << ","
////                 << result.test_type << ","
////                 << result.threads_or_ranks << ","
////                 << result.problem_size << ","
////                 << result.time_ms << ","
////                 << result.speedup << ","
////                 << result.efficiency << "\n";
////        }
////
////        file.close();
////        std::cout << "\nResults exported to " << filename << "\n";
////    }
////
////    // Generate HTML report with plots
////    void generate_html_report(const std::string& filename) {
////        if (!is_mpi_root) return;
////
////        std::ofstream html(filename);
////        if (!html.is_open()) {
////            std::cerr << "Failed to open " << filename << " for writing\n";
////            return;
////        }
////
////        html << R"(<!DOCTYPE html>
////<html>
////<head>
////    <title>DTW Accelerator Scaling Analysis</title>
////    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
////    <style>
////        body { font-family: Arial, sans-serif; margin: 20px; }
////        .chart-container { width: 45%; display: inline-block; margin: 20px; }
////        h1 { color: #333; }
////        h2 { color: #666; }
////    </style>
////</head>
////<body>
////    <h1>DTW Accelerator Scaling Analysis Results</h1>
////    <div id="charts"></div>
////
////    <script>
////        const data = [)";
////
////        // Export data as JavaScript array
////        bool first = true;
////        for (const auto& result : all_results) {
////            if (!first) html << ",";
////            html << "\n            {backend: '" << result.backend
////                 << "', test: '" << result.test_type
////                 << "', threads: " << result.threads_or_ranks
////                 << ", size: " << result.problem_size
////                 << ", time: " << result.time_ms
////                 << ", speedup: " << result.speedup
////                 << ", efficiency: " << result.efficiency << "}";
////            first = false;
////        }
////
////        html << R"(
////        ];
////
////        // Group data by test type
////        const grouped = {};
////        data.forEach(d => {
////            const key = d.backend + '_' + d.test;
////            if (!grouped[key]) grouped[key] = [];
////            grouped[key].push(d);
////        });
////
////        // Create charts
////        const chartsDiv = document.getElementById('charts');
////
////        Object.keys(grouped).forEach(key => {
////            const testData = grouped[key];
////            const backend = testData[0].backend;
////            const testType = testData[0].test;
////
////            // Create container
////            const container = document.createElement('div');
////            container.className = 'chart-container';
////
////            // Create canvas
////            const canvas = document.createElement('canvas');
////            container.appendChild(canvas);
////            chartsDiv.appendChild(container);
////
////            // Determine chart type and data
////            let chartConfig;
////
////            if (testType.includes('Scaling')) {
////                if (testType.includes('Strong')) {
////                    // Strong scaling - threads vs speedup
////                    chartConfig = {
////                        type: 'line',
////                        data: {
////                            labels: testData.map(d => d.threads),
////                            datasets: [{
////                                label: backend + ' Speedup',
////                                data: testData.map(d => d.speedup),
////                                borderColor: 'rgb(75, 192, 192)',
////                                tension: 0.1
////                            }, {
////                                label: 'Ideal',
////                                data: testData.map(d => d.threads),
////                                borderColor: 'rgb(255, 99, 132)',
////                                borderDash: [5, 5],
////                                tension: 0
////                            }]
////                        },
////                        options: {
////                            responsive: true,
////                            plugins: {
////                                title: {
////                                    display: true,
////                                    text: backend + ' Strong Scaling'
////                                }
////                            },
////                            scales: {
////                                x: { title: { display: true, text: 'Number of Threads/Processes' }},
////                                y: { title: { display: true, text: 'Speedup' }}
////                            }
////                        }
////                    };
////                } else if (testType.includes('Weak')) {
////                    // Weak scaling - problem size vs efficiency
////                    chartConfig = {
////                        type: 'line',
////                        data: {
////                            labels: testData.map(d => d.size),
////                            datasets: [{
////                                label: backend + ' Efficiency',
////                                data: testData.map(d => d.efficiency * 100),
////                                borderColor: 'rgb(54, 162, 235)',
////                                tension: 0.1
////                            }]
////                        },
////                        options: {
////                            responsive: true,
////                            plugins: {
////                                title: {
////                                    display: true,
////                                    text: backend + ' Weak Scaling'
////                                }
////                            },
////                            scales: {
////                                x: { title: { display: true, text: 'Problem Size' }},
////                                y: { title: { display: true, text: 'Efficiency (%)' }}
////                            }
////                        }
////                    };
////                } else {
////                    // General scaling - size vs time
////                    chartConfig = {
////                        type: 'line',
////                        data: {
////                            labels: testData.map(d => d.size),
////                            datasets: [{
////                                label: backend + ' Time',
////                                data: testData.map(d => d.time),
////                                borderColor: 'rgb(255, 159, 64)',
////                                tension: 0.1
////                            }]
////                        },
////                        options: {
////                            responsive: true,
////                            plugins: {
////                                title: {
////                                    display: true,
////                                    text: backend + ' Performance Scaling'
////                                }
////                            },
////                            scales: {
////                                x: { title: { display: true, text: 'Problem Size' }},
////                                y: { title: { display: true, text: 'Time (ms)' }}
////                            }
////                        }
////                    };
////                }
////            } else if (testType === 'BlockSize') {
////                // Block size optimization
////                chartConfig = {
////                    type: 'bar',
////                    data: {
////                        labels: testData.map(d => d.threads),
////                        datasets: [{
////                            label: 'Speedup',
////                            data: testData.map(d => d.speedup),
////                            backgroundColor: 'rgba(153, 102, 255, 0.5)'
////                        }]
////                    },
////                    options: {
////                        responsive: true,
////                        plugins: {
////                            title: {
////                                display: true,
////                                text: 'Block Size Optimization'
////                            }
////                        },
////                        scales: {
////                            x: { title: { display: true, text: 'Block Size' }},
////                            y: { title: { display: true, text: 'Speedup' }}
////                        }
////                    }
////                };
////            }
////
////            new Chart(canvas, chartConfig);
////        });
////    </script>
////</body>
////</html>)";
////
////        html.close();
////        std::cout << "HTML report generated: " << filename << "\n";
////    }
////
////    // Comparison across backends
////    void test_backend_comparison() {
////        if (!is_mpi_root) return;
////
////        std::cout << "\n=== Backend Comparison ===\n";
////        std::cout << "Comparing all available backends on same problem sizes\n\n";
////
////        std::vector<size_t> test_sizes = {500, 1000, 2000, 4000, 8000, 16000, 24000};
////
////        std::cout << std::left << std::setw(12) << "Size"
////                  << std::setw(15) << "Sequential"
////                  << std::setw(15) << "Blocked";
////#ifdef USE_OPENMP
////        std::cout << std::setw(15) << "OpenMP";
////#endif
////#ifdef USE_CUDA
////        std::cout << std::setw(15) << "CUDA";
////#endif
////        std::cout << "\n" << std::string(60, '-') << "\n";
////
////        for (size_t size : test_sizes) {
////            auto A = generator.generate_random_series(size, 3);
////            auto B = generator.generate_random_series(size, 3);
////
////            std::cout << std::left << std::setw(12) << size;
////
////            // Sequential
////            Timer timer;
////            timer.start();
////            auto seq_result = dtw_sequential<MetricType::EUCLIDEAN>(A, B);
////            double seq_time = timer.elapsed_ms();
////            std::cout << std::fixed << std::setprecision(2)
////                      << std::setw(15) << seq_time;
////
////            // Blocked
////            timer.start();
////            auto blocked_result = dtw_blocked<MetricType::EUCLIDEAN>(A, B, 64);
////            double blocked_time = timer.elapsed_ms();
////            std::cout << std::setw(15) << blocked_time;
////
////#ifdef USE_OPENMP
////            // OpenMP
////            timer.start();
////            auto omp_result = dtw_openmp<MetricType::EUCLIDEAN>(A, B, 0, 64);
////            double omp_time = timer.elapsed_ms();
////            std::cout << std::setw(15) << omp_time;
////#endif
////
////#ifdef USE_CUDA
////            // CUDA
////            if (cuda::is_available()) {
////                timer.start();
////                auto cuda_result = cuda::dtw_cuda<MetricType::EUCLIDEAN>(A, B, 64);
////                double cuda_time = timer.elapsed_ms();
////                std::cout << std::setw(15) << cuda_time;
////            } else {
////                std::cout << std::setw(15) << "N/A";
////            }
////#endif
////            std::cout << "\n";
////        }
////    }
////};
////
////int main(int argc, char** argv) {
////    // Initialize MPI if available
////#ifdef USE_MPI
////    MPI_Init(&argc, &argv);
////#endif
////
////    std::cout << "==============================================\n";
////    std::cout << "      DTW Accelerator Scaling Analysis       \n";
////    std::cout << "==============================================\n";
////
////    ScalingBenchmark benchmark;
////
////    // Run all tests
////    benchmark.test_block_size_scaling();
////    benchmark.test_backend_comparison();
////
////    // OpenMP tests
////    benchmark.test_openmp_strong_scaling();
////    benchmark.test_openmp_weak_scaling();
////
////    // MPI tests (will only run with MPI execution)
////    benchmark.test_mpi_strong_scaling();
////    benchmark.test_mpi_weak_scaling();
////
////    // CUDA tests
////    benchmark.test_cuda_scaling();
////
////    // Export results
////    benchmark.export_to_csv("scaling_results.csv");
////    benchmark.generate_html_report("scaling_report.html");
////
////    std::cout << "\n==============================================\n";
////    std::cout << "           Analysis Complete                  \n";
////    std::cout << "==============================================\n";
////    std::cout << "Results saved to:\n";
////    std::cout << "  - scaling_results.csv (raw data)\n";
////    std::cout << "  - scaling_report.html (interactive plots)\n";
////
////    // Finalize MPI if available
////#ifdef USE_MPI
////    MPI_Finalize();
////#endif
////
////    return 0;
////}
//
//#include "dtw_accelerator/dtw_accelerator.hpp"
//#include "benchmark_utils.hpp"
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <fstream>
//#include <sstream>
//#include <iomanip>
//#include <map>
//#include <algorithm>
//#include <string>
//
//#ifdef USE_OPENMP
//#include <omp.h>
//#endif
//
//#ifdef USE_MPI
//#include <mpi.h>
//#endif
//
//#ifdef USE_CUDA
//#include "dtw_accelerator/execution/parallel/cuda/cuda_dtw.hpp"
//#endif
//
//using namespace dtw_accelerator;
//using namespace dtw_benchmark;
//
//// Structure to hold benchmark results
//struct BenchmarkRecord {
//    std::string backend;
//    std::string test_type;
//    int parallelism_level;  // threads/processes/block_size
//    size_t problem_size;
//    double time_ms;
//    double speedup;
//    double efficiency;
//};
//
//class ScalingBenchmark {
//private:
//    DataGenerator generator;
//    std::vector<BenchmarkRecord> results;
//    int mpi_rank;
//    int mpi_size;
//    bool is_root;
//
//public:
//    ScalingBenchmark() : generator(42), mpi_rank(0), mpi_size(1), is_root(true) {
//#ifdef USE_MPI
//        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
//        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
//        is_root = (mpi_rank == 0);
//#endif
//    }
//
//    // OpenMP Strong Scaling Test
//    void test_openmp_strong_scaling() {
//#ifdef USE_OPENMP
//        if (!is_root) return;
//
//        std::cout << "\n=== OpenMP Strong Scaling Test ===\n";
//        std::cout << "(Fixed problem size, varying thread count)\n\n";
//
//        std::vector<size_t> problem_sizes = {1000, 2000, 4000, 8000};
//        int max_threads = omp_get_max_threads();
//
//        for (size_t problem_size : problem_sizes) {
//            auto A = generator.generate_random_series(problem_size, 3);
//            auto B = generator.generate_random_series(problem_size, 3);
//
//            // Get baseline (sequential)
//            Timer timer;
//            timer.start();
//            auto seq_result = dtw_sequential<MetricType::EUCLIDEAN>(A, B);
//            double baseline_time = timer.elapsed_ms();
//
//            results.push_back({
//                "OpenMP", "StrongScaling", 1, problem_size,
//                baseline_time, 1.0, 1.0
//            });
//
//            // Test with different thread counts
//            for (int threads = 2; threads <= max_threads; threads *= 2) {
//                timer.start();
//                auto result = dtw_openmp<MetricType::EUCLIDEAN>(A, B, threads, 64);
//                double time = timer.elapsed_ms();
//
//                double speedup = baseline_time / time;
//                double efficiency = speedup / threads;
//
//                results.push_back({
//                    "OpenMP", "StrongScaling", threads, problem_size,
//                    time, speedup, efficiency
//                });
//
//                std::cout << "Size: " << std::setw(6) << problem_size
//                          << " | Threads: " << std::setw(2) << threads
//                          << " | Time: " << std::fixed << std::setprecision(2)
//                          << std::setw(8) << time << " ms"
//                          << " | Speedup: " << std::setw(6) << speedup << "x"
//                          << " | Efficiency: " << std::setw(6)
//                          << (efficiency * 100) << "%\n";
//            }
//        }
//#else
//        if (is_root) {
//            std::cout << "OpenMP not available - skipping OpenMP strong scaling test\n";
//        }
//#endif
//    }
//
//    // OpenMP Weak Scaling Test
//    void test_openmp_weak_scaling() {
//#ifdef USE_OPENMP
//        if (!is_root) return;
//
//        std::cout << "\n=== OpenMP Weak Scaling Test ===\n";
//        std::cout << "(Problem size per thread remains constant)\n\n";
//
//        int max_threads = omp_get_max_threads();
//        size_t base_size = 1000;
//
//        // Get baseline with 1 thread
//        auto A1 = generator.generate_random_series(base_size, 3);
//        auto B1 = generator.generate_random_series(base_size, 3);
//
//        Timer timer;
//        timer.start();
//        auto result1 = dtw_openmp<MetricType::EUCLIDEAN>(A1, B1, 1, 64);
//        double baseline_time = timer.elapsed_ms();
//
//        results.push_back({
//            "OpenMP", "WeakScaling", 1, base_size,
//            baseline_time, 1.0, 1.0
//        });
//
//        std::cout << "Threads: 1 | Size: " << base_size
//                  << " | Time: " << baseline_time << " ms | Efficiency: 100%\n";
//
//        // Test with increasing threads and proportionally increasing problem size
//        for (int threads = 2; threads <= max_threads; threads *= 2) {
//            size_t problem_size = base_size * threads;
//            auto A = generator.generate_random_series(problem_size, 3);
//            auto B = generator.generate_random_series(problem_size, 3);
//
//            timer.start();
//            auto result = dtw_openmp<MetricType::EUCLIDEAN>(A, B, threads, 64);
//            double time = timer.elapsed_ms();
//
//            double efficiency = baseline_time / time;
//
//            results.push_back({
//                "OpenMP", "WeakScaling", threads, problem_size,
//                time, 1.0, efficiency
//            });
//
//            std::cout << "Threads: " << threads << " | Size: " << problem_size
//                      << " | Time: " << std::fixed << std::setprecision(2) << time
//                      << " ms | Efficiency: " << (efficiency * 100) << "%\n";
//        }
//#else
//        if (is_root) {
//            std::cout << "OpenMP not available - skipping OpenMP weak scaling test\n";
//        }
//#endif
//    }
//
//    // MPI Strong Scaling Test - runs with current MPI configuration
//    void test_mpi_strong_scaling() {
//#ifdef USE_MPI
//        std::vector<size_t> problem_sizes = {1000, 2000, 4000, 8000};
//
//        for (size_t problem_size : problem_sizes) {
//            auto A = generator.generate_random_series(problem_size, 3);
//            auto B = generator.generate_random_series(problem_size, 3);
//
//            MPI_Barrier(MPI_COMM_WORLD);
//
//            Timer timer;
//            timer.start();
//            auto result = dtw_mpi<MetricType::EUCLIDEAN>(A, B, 64, 0, MPI_COMM_WORLD);
//            double local_time = timer.elapsed_ms();
//
//            // Gather max time from all processes (worst case)
//            double max_time;
//            MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//
//            if (is_root) {
//                // We'll compute speedup later when combining results from multiple runs
//                results.push_back({
//                    "MPI", "StrongScaling", mpi_size, problem_size,
//                    max_time, 0.0, 0.0  // Speedup/efficiency computed later
//                });
//
//                std::cout << "MPI Strong Scaling - Processes: " << mpi_size
//                          << " | Size: " << problem_size
//                          << " | Time: " << max_time << " ms\n";
//            }
//        }
//#else
//        if (is_root) {
//            std::cout << "MPI not available - skipping MPI strong scaling test\n";
//        }
//#endif
//    }
//
//    // MPI Weak Scaling Test - runs with current MPI configuration
//    void test_mpi_weak_scaling() {
//#ifdef USE_MPI
//        size_t base_size = 1000;
//        size_t problem_size = base_size * mpi_size;
//
//        auto A = generator.generate_random_series(problem_size, 3);
//        auto B = generator.generate_random_series(problem_size, 3);
//
//        MPI_Barrier(MPI_COMM_WORLD);
//
//        Timer timer;
//        timer.start();
//        auto result = dtw_mpi<MetricType::EUCLIDEAN>(A, B, 64, 0, MPI_COMM_WORLD);
//        double local_time = timer.elapsed_ms();
//
//        // Gather max time from all processes
//        double max_time;
//        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//
//        if (is_root) {
//            results.push_back({
//                "MPI", "WeakScaling", mpi_size, problem_size,
//                max_time, 0.0, 0.0  // Efficiency computed later
//            });
//
//            std::cout << "MPI Weak Scaling - Processes: " << mpi_size
//                      << " | Size: " << problem_size
//                      << " | Time: " << max_time << " ms\n";
//        }
//#else
//        if (is_root) {
//            std::cout << "MPI not available - skipping MPI weak scaling test\n";
//        }
//#endif
//    }
//
//    // CUDA Scaling Test
//    void test_cuda_scaling() {
//#ifdef USE_CUDA
//        if (!is_root) return;
//
//        if (!cuda::is_available()) {
//            std::cout << "CUDA device not available - skipping CUDA tests\n";
//            return;
//        }
//
//        std::cout << "\n=== CUDA Scaling Test ===\n";
//        std::vector<size_t> problem_sizes = {500, 1000, 2000, 4000, 8000, 16000};
//
//        for (size_t size : problem_sizes) {
//            auto A = generator.generate_random_series(size, 3);
//            auto B = generator.generate_random_series(size, 3);
//
//            // Get sequential baseline
//            Timer timer;
//            timer.start();
//            auto seq_result = dtw_sequential<MetricType::EUCLIDEAN>(A, B);
//            double seq_time = timer.elapsed_ms();
//
//            // CUDA timing
//            timer.start();
//            auto cuda_result = cuda::dtw_cuda<MetricType::EUCLIDEAN>(A, B, 64);
//            double cuda_time = timer.elapsed_ms();
//
//            double speedup = seq_time / cuda_time;
//
//            results.push_back({
//                "CUDA", "Scaling", 1, size,
//                cuda_time, speedup, speedup
//            });
//
//            std::cout << "Size: " << std::setw(6) << size
//                      << " | CUDA Time: " << std::fixed << std::setprecision(2)
//                      << std::setw(8) << cuda_time << " ms"
//                      << " | Speedup: " << std::setw(6) << speedup << "x\n";
//        }
//#else
//        if (is_root) {
//            std::cout << "CUDA not available - skipping CUDA scaling test\n";
//        }
//#endif
//    }
//
//    // Block Size Optimization Test
//    void test_block_size_scaling() {
//        if (!is_root) return;
//
//        std::cout << "\n=== Block Size Optimization Test ===\n";
//
//        std::vector<size_t> problem_sizes = {1000, 2000, 4000};
//        std::vector<int> block_sizes = {16, 32, 64, 128, 256};
//
//        for (size_t problem_size : problem_sizes) {
//            auto A = generator.generate_random_series(problem_size, 3);
//            auto B = generator.generate_random_series(problem_size, 3);
//
//            // Get baseline
//            Timer timer;
//            timer.start();
//            auto baseline_result = dtw_sequential<MetricType::EUCLIDEAN>(A, B);
//            double baseline_time = timer.elapsed_ms();
//
//            for (int block_size : block_sizes) {
//                timer.start();
//                auto result = dtw_blocked<MetricType::EUCLIDEAN>(A, B, block_size);
//                double time = timer.elapsed_ms();
//
//                double speedup = baseline_time / time;
//
//                results.push_back({
//                                          "Blocked", "BlockSize", block_size, problem_size,
//                                          time, speedup, speedup
//                                  });
//
//                std::cout << "Size: " << problem_size
//                          << " | Block: " << block_size
//                          << " | Time: " << time << " ms"
//                          << " | Speedup: " << speedup << "x\n";
//            }
//        }
//    }
//
//    // Save results to CSV files optimized for plotting
//    void save_results(const std::string& prefix = "") {
//        if (!is_root) return;
//
//        // Create separate files for each backend and test type
//        std::map<std::string, std::ofstream> files;
//
//        for (const auto& result : results) {
//            std::string filename = prefix + result.backend + "_" + result.test_type + ".csv";
//
//            if (files.find(filename) == files.end()) {
//                files[filename].open(filename);
//                if (result.test_type == "StrongScaling") {
//                    files[filename] << "# Strong Scaling: Fixed problem size, varying parallelism\n";
//                    files[filename] << "# Columns: ProblemSize Parallelism Time_ms Speedup Efficiency\n";
//                } else if (result.test_type == "WeakScaling") {
//                    files[filename] << "# Weak Scaling: Problem size scales with parallelism\n";
//                    files[filename] << "# Columns: ProblemSize Parallelism Time_ms Speedup Efficiency\n";
//                } else {
//                    files[filename] << "# Columns: ProblemSize Parallelism Time_ms Speedup Efficiency\n";
//                }
//            }
//
//            files[filename] << result.problem_size << " "
//                            << result.parallelism_level << " "
//                            << result.time_ms << " "
//                            << result.speedup << " "
//                            << result.efficiency << "\n";
//        }
//
//        for (auto& [filename, file] : files) {
//            file.close();
//            std::cout << "Saved: " << filename << "\n";
//        }
//
//        // Also save a combined file for MPI results (for post-processing)
//#ifdef USE_MPI
//        std::string mpi_file = prefix + "mpi_raw.csv";
//        std::ofstream mpi_out(mpi_file, std::ios::app);
//        for (const auto& result : results) {
//            if (result.backend == "MPI") {
//                mpi_out << result.test_type << " "
//                        << result.parallelism_level << " "
//                        << result.problem_size << " "
//                        << result.time_ms << "\n";
//            }
//        }
//        mpi_out.close();
//#endif
//    }
//};
//
//int main(int argc, char** argv) {
//#ifdef USE_MPI
//    MPI_Init(&argc, &argv);
//#endif
//
//    ScalingBenchmark benchmark;
//
//    std::cout << "==============================================\n";
//    std::cout << "      DTW Accelerator Scaling Analysis       \n";
//    std::cout << "==============================================\n";
//
//    // Run tests based on available backends
//    benchmark.test_block_size_scaling();
//
//#ifdef USE_OPENMP
//    benchmark.test_openmp_strong_scaling();
//    benchmark.test_openmp_weak_scaling();
//#endif
//
//#ifdef USE_MPI
//    benchmark.test_mpi_strong_scaling();
//    benchmark.test_mpi_weak_scaling();
//#endif
//
//#ifdef USE_CUDA
//    benchmark.test_cuda_scaling();
//#endif
//
//    // Save results
//    benchmark.save_results();
//
//#ifdef USE_MPI
//    MPI_Finalize();
//#endif
//
//    return 0;
//}

/**
 * @file benchmark_scaling.cpp
 * @brief Scaling benchmarks for OpenMP and MPI implementations
 * @author DTW-Accelerator
 * @date 2024
 *
 * This file benchmarks the scaling performance of OpenMP (1-8 threads)
 * and MPI (1-8 processes) implementations across different problem sizes.
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

#ifdef USE_OPENMP
#include <omp.h>
#endif

using namespace dtw_accelerator;
using namespace std::chrono;

// Benchmark configuration
struct ScalingConfig {
    std::vector<int> problem_sizes;  // 2^7 to 2^14
    std::vector<int> thread_counts = {1, 2, 4, 8};  // OpenMP threads
    std::vector<int> process_counts = {1, 2, 4, 8}; // MPI processes
    int dimensions = 10;
    int num_runs = 3;
};

// Result structure
struct ScalingResult {
    std::string backend;      // "OpenMP" or "MPI"
    int num_workers;          // threads or processes
    int problem_size;
    double time_ms;
    double speedup;
    double efficiency;
};

// Benchmark class
class DTWScalingBenchmark {
private:
    ScalingConfig config;
    std::vector<ScalingResult> results;
    std::map<int, double> baseline_times;  // Sequential baseline per size
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
    void log_result(const std::string& backend, int workers,
                    int size, double time_ms) {
        double speedup = 1.0;
        double efficiency = 1.0;

        if (baseline_times.count(size) > 0) {
            speedup = baseline_times[size] / time_ms;
            efficiency = speedup / workers;
        }

        results.push_back({backend, workers, size, time_ms, speedup, efficiency});

        if (mpi_rank == 0) {
            std::cout << std::left << std::setw(10) << backend
                      << std::setw(10) << workers
                      << std::setw(10) << size
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << time_ms << " ms"
                      << std::setw(10) << speedup << "x"
                      << std::setw(10) << (efficiency * 100) << "%"
                      << std::endl;
        }
    }

public:
    DTWScalingBenchmark() {
        // Initialize problem sizes: 2^7 to 2^14
        for (int exp = 7; exp <= 14; ++exp) {
            config.problem_sizes.push_back(1 << exp);
        }

#ifdef USE_MPI
        int is_initialized = 0;
        MPI_Initialized(&is_initialized);
        if (is_initialized) {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        }
#endif
    }

    // Benchmark sequential baseline
    void benchmark_sequential() {
        if (mpi_rank == 0) {
            std::cout << "\n======== SEQUENTIAL BASELINE ========\n";
            std::cout << std::left << std::setw(10) << "Backend"
                      << std::setw(10) << "Workers"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::setw(10) << "Speedup"
                      << std::setw(10) << "Efficiency"
                      << std::endl;
            std::cout << std::string(72, '-') << std::endl;
        }

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 42);
            auto B = generate_series(size, config.dimensions, 43);

            double seq_time = measure_time([&]() {
                auto result = dtw_sequential<distance::MetricType::EUCLIDEAN>(A, B);
            }, config.num_runs);

            baseline_times[size] = seq_time;
            log_result("Sequential", 1, size, seq_time);
        }
    }

#ifdef USE_OPENMP
    // Benchmark OpenMP scaling
    void benchmark_openmp_scaling() {
        if (mpi_rank == 0) {
            std::cout << "\n======== OPENMP SCALING ========\n";
            std::cout << std::left << std::setw(10) << "Backend"
                      << std::setw(10) << "Threads"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::setw(10) << "Speedup"
                      << std::setw(10) << "Efficiency"
                      << std::endl;
            std::cout << std::string(72, '-') << std::endl;
        }

        for (int threads : config.thread_counts) {
            for (int size : config.problem_sizes) {
                auto A = generate_series(size, config.dimensions, 42);
                auto B = generate_series(size, config.dimensions, 43);

                double omp_time = measure_time([&]() {
                    auto result = dtw_openmp<distance::MetricType::EUCLIDEAN>(
                        A, B, threads, 64);
                }, config.num_runs);

                log_result("OpenMP", threads, size, omp_time);
            }
        }
    }
#endif

#ifdef USE_MPI
    // Benchmark MPI scaling (must be run with mpirun -np X)
    void benchmark_mpi_scaling() {
        // Only run if we're in the correct MPI configuration
        bool is_target_size = false;
        for (int procs : config.process_counts) {
            if (mpi_size == procs) {
                is_target_size = true;
                break;
            }
        }

        if (!is_target_size) {
            if (mpi_rank == 0) {
                std::cout << "\n[INFO] Skipping MPI benchmark for " << mpi_size
                          << " processes (not in target list)\n";
            }
            return;
        }

        if (mpi_rank == 0) {
            std::cout << "\n======== MPI SCALING (" << mpi_size << " processes) ========\n";
            std::cout << std::left << std::setw(10) << "Backend"
                      << std::setw(10) << "Processes"
                      << std::setw(10) << "Size"
                      << std::setw(12) << "Time"
                      << std::setw(10) << "Speedup"
                      << std::setw(10) << "Efficiency"
                      << std::endl;
            std::cout << std::string(72, '-') << std::endl;
        }

        for (int size : config.problem_sizes) {
            auto A = generate_series(size, config.dimensions, 42);
            auto B = generate_series(size, config.dimensions, 43);

            MPI_Barrier(MPI_COMM_WORLD);

            double mpi_time = measure_time([&]() {
                auto result = dtw_mpi<distance::MetricType::EUCLIDEAN>(
                    A, B, 64, 0, MPI_COMM_WORLD);
            }, config.num_runs);

            // Only root reports
            if (mpi_rank == 0) {
                log_result("MPI", mpi_size, size, mpi_time);
            }
        }
    }
#endif

    // Save results to CSV
    void save_results(const std::string& filename) {
        if (mpi_rank != 0) return;

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << filename << std::endl;
            return;
        }

        // Write header
        file << "Backend,Workers,Size,Time_ms,Speedup,Efficiency\n";

        // Write data
        for (const auto& result : results) {
            file << result.backend << ","
                 << result.num_workers << ","
                 << result.problem_size << ","
                 << result.time_ms << ","
                 << result.speedup << ","
                 << result.efficiency << "\n";
        }

        file.close();
        std::cout << "\nResults saved to " << filename << std::endl;
    }

    // Run all benchmarks
    void run_all() {
        if (mpi_rank == 0) {
            std::cout << "========================================\n";
            std::cout << "    DTW SCALING BENCHMARK SUITE        \n";
            std::cout << "========================================\n";
            std::cout << "Problem sizes: 2^7 to 2^14\n";
            std::cout << "Dimensions: " << config.dimensions << "\n";
            std::cout << "Runs per measurement: " << config.num_runs << "\n";
#ifdef USE_MPI
            std::cout << "MPI Processes: " << mpi_size << "\n";
#endif
#ifdef USE_OPENMP
            std::cout << "Max OpenMP Threads: " << omp_get_max_threads() << "\n";
#endif
            std::cout << "========================================\n";
        }

        // Always run sequential baseline
        if (mpi_size == 1) {
            benchmark_sequential();

#ifdef USE_OPENMP
            benchmark_openmp_scaling();
#endif
        }

#ifdef USE_MPI
        // MPI scaling benchmarks
        benchmark_mpi_scaling();
#endif

        // Save results (only from rank 0)
        if (mpi_rank == 0) {
            std::string filename = "dtw_scaling_results";
            if (mpi_size > 1) {
                filename += "_mpi" + std::to_string(mpi_size);
            }
            filename += ".csv";
            save_results(filename);
        }
    }
};

int main(int argc, char** argv) {
#ifdef USE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif

    DTWScalingBenchmark benchmark;
    benchmark.run_all();

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}