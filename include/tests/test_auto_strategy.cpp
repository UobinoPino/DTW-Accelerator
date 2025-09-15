/**
 * @file test_auto_strategy.cpp
 * @brief Comprehensive test for AutoStrategy to ensure it works in all scenarios
 * @author UobinoPino
 * @date 2024
 *
 * This file tests the AutoStrategy class with different problem sizes to verify:
 * 1. It doesn't crash
 * 2. It correctly selects different strategies based on problem size
 * 3. It produces valid results
 * 4. It handles all available backends gracefully
 */

#include "dtw_accelerator/dtw_accelerator.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <vector>
#include <string>
#include <set>

using namespace dtw_accelerator;
using namespace dtw_accelerator::execution;
using namespace std::chrono;

// Color codes for terminal output
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define RED "\033[31m"
#define BLUE "\033[34m"
#define RESET "\033[0m"

/**
 * @brief Test result structure
 */
struct TestResult {
    std::string test_name;
    int size_a;
    int size_b;
    int dimensions;
    std::string selected_strategy;
    double execution_time_ms;
    bool passed;
    std::string error_message;
};

/**
 * @brief Generate random time series for testing
 */
DoubleTimeSeries generate_random_series(int length, int dim, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-10.0, 10.0);

    DoubleTimeSeries series(length, dim);
    for (int i = 0; i < length; ++i) {
        for (int d = 0; d < dim; ++d) {
            series[i][d] = dis(gen);
        }
    }
    return series;
}

/**
 * @brief Test AutoStrategy with specific problem size
 */
TestResult test_auto_strategy(const std::string& test_name,
                              int size_a, int size_b, int dim) {
    TestResult result;
    result.test_name = test_name;
    result.size_a = size_a;
    result.size_b = size_b;
    result.dimensions = dim;
    result.passed = false;

    try {
        // Generate test data
        auto series_a = generate_random_series(size_a, dim, 42);
        auto series_b = generate_random_series(size_b, dim, 43);

        // Create AutoStrategy with problem dimensions
        AutoStrategy strategy(size_a, size_b);

        // Measure execution time
        auto start = high_resolution_clock::now();

        // Run DTW with AutoStrategy
        auto dtw_result = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::NONE>(
                series_a, series_b, strategy
        );

        auto end = high_resolution_clock::now();
        duration<double, std::milli> elapsed = end - start;

        // Record results
        result.selected_strategy = std::string(strategy.name());
        result.execution_time_ms = elapsed.count();
        result.passed = true;

        // Validate result
        if (dtw_result.first < 0) {
            result.passed = false;
        }
        if (dtw_result.second.empty() && size_a > 0 && size_b > 0) {
            result.passed = false;
            result.error_message = "Empty warping path for non-empty series";
        }

    } catch (const std::exception& e) {
        result.passed = false;
        result.error_message = std::string("Exception: ") + e.what();
    } catch (...) {
        result.passed = false;
        result.error_message = "Unknown exception";
    }

    return result;
}

/**
 * @brief Test AutoStrategy with constraints
 */
TestResult test_auto_strategy_with_constraints(const std::string& test_name,
                                               int size_a, int size_b, int dim) {
    TestResult result;
    result.test_name = test_name;
    result.size_a = size_a;
    result.size_b = size_b;
    result.dimensions = dim;
    result.passed = false;

    try {
        auto series_a = generate_random_series(size_a, dim, 44);
        auto series_b = generate_random_series(size_b, dim, 45);

        AutoStrategy strategy(size_a, size_b);

        auto start = high_resolution_clock::now();

        // Test with Sakoe-Chiba constraint
        auto dtw_result = dtw<MetricType::EUCLIDEAN,
                constraints::ConstraintType::SAKOE_CHIBA, 5>(
                series_a, series_b, strategy
        );

        auto end = high_resolution_clock::now();
        duration<double, std::milli> elapsed = end - start;

        result.selected_strategy = std::string(strategy.name());
        result.execution_time_ms = elapsed.count();
        result.passed = true;

        if (dtw_result.first < 0) {
            result.passed = false;
        }

    } catch (const std::exception& e) {
        result.passed = false;
        result.error_message = std::string("Exception: ") + e.what();
    } catch (...) {
        result.passed = false;
        result.error_message = "Unknown exception";
    }

    return result;
}

/**
 * @brief Test FastDTW with AutoStrategy
 */
TestResult test_fastdtw_auto(const std::string& test_name,
                             int size_a, int size_b, int dim) {
    TestResult result;
    result.test_name = test_name;
    result.size_a = size_a;
    result.size_b = size_b;
    result.dimensions = dim;
    result.passed = false;

    try {
        auto series_a = generate_random_series(size_a, dim, 46);
        auto series_b = generate_random_series(size_b, dim, 47);

        AutoStrategy strategy(size_a, size_b);

        auto start = high_resolution_clock::now();

        // Test FastDTW with AutoStrategy
        auto dtw_result = fastdtw<MetricType::EUCLIDEAN>(
                series_a, series_b, 2, 50, strategy
        );

        auto end = high_resolution_clock::now();
        duration<double, std::milli> elapsed = end - start;

        result.selected_strategy = std::string(strategy.name());
        result.execution_time_ms = elapsed.count();
        result.passed = true;

        if (dtw_result.first < 0) {
            result.passed = false;
        }

    } catch (const std::exception& e) {
        result.passed = false;
        result.error_message = std::string("Exception: ") + e.what();
    } catch (...) {
        result.passed = false;
        result.error_message = "Unknown exception";
    }

    return result;
}

/**
 * @brief Print test result
 */
void print_result(const TestResult& result) {
    std::cout << std::left << std::setw(35) << result.test_name;
    std::cout << std::setw(12) << (std::to_string(result.size_a) + "x" + std::to_string(result.size_b));
    std::cout << std::setw(20) << result.selected_strategy;

    if (result.passed) {
        std::cout << GREEN << std::setw(8) << "PASSED" << RESET;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(10) << result.execution_time_ms << " ms";
    } else {
        std::cout << RED << std::setw(8) << "FAILED" << RESET;
        std::cout << " Error: " << result.error_message;
    }
    std::cout << std::endl;
}

/**
 * @brief Check available backends
 */
void check_available_backends() {
    std::cout << "\n" << BLUE << "=== Available Backends ===" << RESET << "\n";

    std::cout << "Sequential: " << GREEN << "Available" << RESET << "\n";
    std::cout << "Blocked:    " << GREEN << "Available" << RESET << "\n";

#ifdef USE_OPENMP
    std::cout << "OpenMP:     " << GREEN << "Available";
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << " (" << omp_get_num_threads() << " threads)";
    }
    std::cout << RESET << "\n";
#else
    std::cout << "OpenMP:     " << YELLOW << "Not Available" << RESET << "\n";
#endif

#ifdef USE_MPI
    int flag = 0;
    MPI_Initialized(&flag);
    if (flag) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        std::cout << "MPI:        " << GREEN << "Available (" << size << " processes)" << RESET << "\n";
    } else {
        std::cout << "MPI:        " << YELLOW << "Available but not initialized" << RESET << "\n";
    }
#else
    std::cout << "MPI:        " << YELLOW << "Not Available" << RESET << "\n";
#endif

#ifdef USE_CUDA
    if (cuda::is_available()) {
        std::cout << "CUDA:       " << GREEN << "Available" << RESET;
        std::cout << " (" << cuda::device_info() << ")\n";
    } else {
        std::cout << "CUDA:       " << YELLOW << "Compiled but no device found" << RESET << "\n";
    }
#else
    std::cout << "CUDA:       " << YELLOW << "Not Available" << RESET << "\n";
#endif

    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Initialize MPI if available
#ifdef USE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif

    std::cout << "\n";
    std::cout << BLUE << "========================================" << RESET << "\n";
    std::cout << BLUE << "    AutoStrategy Comprehensive Test    " << RESET << "\n";
    std::cout << BLUE << "========================================" << RESET << "\n";

    // Check available backends
    check_available_backends();

    // Test suite
    std::vector<TestResult> results;

    std::cout << BLUE << "=== Running Tests ===" << RESET << "\n\n";
    std::cout << std::left << std::setw(35) << "Test Name";
    std::cout << std::setw(12) << "Size";
    std::cout << std::setw(20) << "Selected Strategy";
    std::cout << std::setw(8) << "Status";
    std::cout << std::setw(10) << "Time" << "\n";
    std::cout << std::string(95, '-') << "\n";

    // Test 1: Very small problem (should select Sequential)
    results.push_back(test_auto_strategy("Small Problem (10x10)", 10, 10, 2));
    print_result(results.back());

    // Test 2: Small problem (should select Sequential)
    results.push_back(test_auto_strategy("Small Problem (30x30)", 30, 30, 3));
    print_result(results.back());

    // Test 3: Medium problem (should select Blocked)
    results.push_back(test_auto_strategy("Medium Problem (75x75)", 75, 75, 3));
    print_result(results.back());

    // Test 4: Medium-Large problem (should select OpenMP if available)
    results.push_back(test_auto_strategy("Medium-Large Problem (150x150)", 150, 150, 3));
    print_result(results.back());

    // Test 5: Large problem (should select OpenMP or better)
    results.push_back(test_auto_strategy("Large Problem (500x500)", 500, 500, 3));
    print_result(results.back());

    // Test 6: Very large problem (should select CUDA if available)
    results.push_back(test_auto_strategy("Very Large Problem (1500x1500)", 1500, 1500, 2));
    print_result(results.back());

    // Test 7: Asymmetric sizes
    results.push_back(test_auto_strategy("Asymmetric (100x500)", 100, 500, 3));
    print_result(results.back());

    // Test 8: With constraints - small
    results.push_back(test_auto_strategy_with_constraints("Constrained Small (50x50)", 50, 50, 3));
    print_result(results.back());

    // Test 9: With constraints - large
    results.push_back(test_auto_strategy_with_constraints("Constrained Large (300x300)", 300, 300, 2));
    print_result(results.back());

    // Test 10: FastDTW with auto strategy
    results.push_back(test_fastdtw_auto("FastDTW Medium (200x200)", 200, 200, 3));
    print_result(results.back());

    // Test 11: FastDTW large
    results.push_back(test_fastdtw_auto("FastDTW Large (800x800)", 800, 800, 2));
    print_result(results.back());

    // Test 12: Edge case - single element
    results.push_back(test_auto_strategy("Edge Case (1x1)", 1, 1, 1));
    print_result(results.back());

    // Test 13: Edge case - single dimension
    results.push_back(test_auto_strategy("Single Dimension (100x100)", 100, 100, 1));
    print_result(results.back());

    // Test 14: High dimensional
    results.push_back(test_auto_strategy("High Dimensional (100x100x10)", 100, 100, 10));
    print_result(results.back());

    // Summary
    std::cout << "\n" << BLUE << "=== Test Summary ===" << RESET << "\n";

    int passed = 0;
    int failed = 0;

    // Group results by selected strategy
    std::map<std::string, int> strategy_counts;

    for (const auto& result : results) {
        if (result.passed) {
            passed++;
            strategy_counts[result.selected_strategy]++;
        } else {
            failed++;
        }
    }

    std::cout << "Total Tests: " << results.size() << "\n";
    std::cout << GREEN << "Passed: " << passed << RESET << "\n";
    if (failed > 0) {
        std::cout << RED << "Failed: " << failed << RESET << "\n";
    }

    std::cout << "\n" << BLUE << "Strategy Selection Distribution:" << RESET << "\n";
    for (const auto& [strategy, count] : strategy_counts) {
        std::cout << "  " << std::setw(20) << strategy << ": " << count << " times\n";
    }

    // Verify strategy selection logic
    std::cout << "\n" << BLUE << "=== Strategy Selection Verification ===" << RESET << "\n";

    bool selection_correct = true;

    // Check if different strategies were selected for different problem sizes
    std::set<std::string> unique_strategies;
    for (const auto& result : results) {
        if (result.passed) {
            unique_strategies.insert(result.selected_strategy);
        }
    }

    if (unique_strategies.size() == 1) {
        std::cout << YELLOW << "Warning: Only one strategy was selected across all tests!" << RESET << "\n";
        std::cout << "This might indicate the AutoStrategy is not working correctly.\n";
        selection_correct = false;
    } else {
        std::cout << GREEN << "✓ Multiple strategies selected based on problem size" << RESET << "\n";
        std::cout << "  Unique strategies used: ";
        for (const auto& s : unique_strategies) {
            std::cout << s << " ";
        }
        std::cout << "\n";
    }

    // Verify size-based selection
    std::cout << "\n" << BLUE << "Size-based Selection Check:" << RESET << "\n";

    for (const auto& result : results) {
        if (!result.passed) continue;

        bool expected_correct = true;
        std::string expected;

        int max_size = std::max(result.size_a, result.size_b);

        if (max_size < 50) {
            expected = "Sequential";
            if (result.selected_strategy.find("Sequential") == std::string::npos) {
                expected_correct = false;
            }
        } else if (max_size < 100) {
            expected = "Blocked or Sequential";
            if (result.selected_strategy.find("Blocked") == std::string::npos &&
                result.selected_strategy.find("Sequential") == std::string::npos) {
                expected_correct = false;
            }
        }

        if (!expected_correct) {
            std::cout << YELLOW << "  Size " << max_size << ": Expected " << expected
                      << ", got " << result.selected_strategy << RESET << "\n";
        }
    }

    // Final verdict
    std::cout << "\n" << BLUE << "=== Final Verdict ===" << RESET << "\n";

    if (failed == 0 && selection_correct) {
        std::cout << GREEN << "✓ ALL TESTS PASSED - AutoStrategy is working correctly!" << RESET << "\n";
    } else if (failed == 0) {
        std::cout << YELLOW << "⚠ Tests passed but strategy selection may need review" << RESET << "\n";
    } else {
        std::cout << RED << "✗ Some tests failed - AutoStrategy needs debugging" << RESET << "\n";
        std::cout << "Failed tests:\n";
        for (const auto& result : results) {
            if (!result.passed) {
                std::cout << "  - " << result.test_name << ": " << result.error_message << "\n";
            }
        }
    }

    std::cout << "\n";

    // Cleanup MPI if initialized
#ifdef USE_MPI
    int flag = 0;
    MPI_Initialized(&flag);
    if (flag) {
        MPI_Finalize();
    }
#endif

    return (failed == 0) ? 0 : 1;
}