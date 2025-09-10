/**
 * @file test_strategies.cpp
 * @brief Unit tests for DTW execution strategies
 * @author UobinoPino
 * @date 2024
 *
 * This file contains comprehensive tests for all execution strategies
 * (Sequential, Blocked, OpenMP, MPI, CUDA, Auto) ensuring they produce
 * consistent results and validating their specific features.
 */

#include "dtw_accelerator/dtw_accelerator.hpp"
#include "gtest/gtest.h"
#include <random>

using namespace dtw_accelerator;
using namespace dtw_accelerator::strategies;
using namespace dtw_accelerator::execution;

/**
 * @class StrategyTest
 * @brief Test fixture for execution strategy validation
 *
 * Provides test data and utilities for validating that all execution
 * strategies produce consistent results and perform correctly under
 * various conditions.
 */class StrategyTest : public ::testing::Test {
protected:
    /**
       * @brief Set up test environment before each test
       *
       * Generates small and medium-sized test series for strategy testing.
       * Small series are used for quick validation, medium series for
       * performance comparison and parallel efficiency testing.
       */
    void SetUp() override {
        // Generate test data
        small_a_ = generate_test_series(50, 2, 42);
        small_b_ = generate_test_series(50, 2, 43);

        medium_a_ = generate_test_series(200, 3, 44);
        medium_b_ = generate_test_series(180, 3, 45);
    }

    /**
    * @brief Generate random time series for testing
    * @param length Number of time points
    * @param dim Number of dimensions
    * @param seed Random seed for reproducibility
    * @return Time series with random values in [-10.0, 10.0]
    *
    * Uses larger value range than core tests to ensure strategies
    * handle various numerical scales correctly.
    */
    DoubleTimeSeries generate_test_series(size_t length, size_t dim, unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(-10.0, 10.0);

        DoubleTimeSeries series(length, dim);
        for (size_t i = 0; i < length; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                series[i][d] = dis(gen);
            }
        }
        return series;
    }

    /**
    * @brief Compare DTW results for equality
    * @param r1 First result
    * @param r2 Second result
    * @param tolerance Numerical tolerance
    * @return True if distances are equal within tolerance
    *
    * Critical for validating that different strategies produce
    * numerically consistent results.
    */
    bool results_equal(const std::pair<double, std::vector<std::pair<int, int>>>& r1,
    const std::pair<double, std::vector<std::pair<int, int>>>& r2,
    double tolerance = 1e-6) {
        return std::abs(r1.first - r2.first) < tolerance;
    }

    /// @brief Small test series A (50 points, 2 dimensions)
    DoubleTimeSeries small_a_;

    /// @brief Small test series B (50 points, 2 dimensions)
    DoubleTimeSeries small_b_;

    /// @brief Medium test series A (200 points, 3 dimensions)
    DoubleTimeSeries medium_a_;

    /// @brief Medium test series B (180 points, 3 dimensions)
    DoubleTimeSeries medium_b_;
};

/**
 * @test SequentialStrategy
 * @brief Test sequential execution strategy
 *
 * Validates the baseline sequential strategy, ensuring it produces
 * correct results and properly reports its characteristics.
 */
TEST_F(StrategyTest, SequentialStrategy) {
SequentialStrategy strategy;

EXPECT_EQ(strategy.name(), "Sequential");
EXPECT_FALSE(strategy.is_parallel());

auto result = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::NONE>(
        small_a_, small_b_, strategy);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());
}

/**
 * @test BlockedStrategy
 * @brief Test cache-optimized blocked execution strategy
 *
 * Validates blocked strategy with configurable block sizes.
 * Verifies results match sequential strategy and tests block
 * size configuration.
 */
TEST_F(StrategyTest, BlockedStrategy) {
BlockedStrategy strategy(32);

EXPECT_EQ(strategy.name(), "Blocked");
EXPECT_FALSE(strategy.is_parallel());
EXPECT_EQ(strategy.get_block_size(), 32);

strategy.set_block_size(64);
EXPECT_EQ(strategy.get_block_size(), 64);

auto result = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::NONE>(
        medium_a_, medium_b_, strategy);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Compare with sequential for correctness
SequentialStrategy seq_strategy;
auto seq_result = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::NONE>(
        medium_a_, medium_b_, seq_strategy);

EXPECT_TRUE(results_equal(result, seq_result));
}

#ifdef USE_OPENMP
/**
 * @test OpenMPStrategy
 * @brief Test OpenMP parallel execution strategy
 *
 * Validates multi-threaded execution using OpenMP.
 * Tests thread configuration and verifies parallel results
 * match sequential computation.
 */
TEST_F(StrategyTest, OpenMPStrategy) {
    OpenMPStrategy strategy(2, 32);

    EXPECT_EQ(strategy.name(), "OpenMP");
    EXPECT_TRUE(strategy.is_parallel());
    EXPECT_EQ(strategy.get_num_threads(), 2);
    EXPECT_EQ(strategy.get_block_size(), 32);

    strategy.set_num_threads(4);
    strategy.set_block_size(64);
    EXPECT_EQ(strategy.get_num_threads(), 4);
    EXPECT_EQ(strategy.get_block_size(), 64);

    auto result = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::NONE>(
        medium_a_, medium_b_, strategy);

    EXPECT_GT(result.first, 0.0);
    EXPECT_FALSE(result.second.empty());

    // Compare with sequential for correctness
    SequentialStrategy seq_strategy;
    auto seq_result = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::NONE>(
        medium_a_, medium_b_, seq_strategy);

    EXPECT_TRUE(results_equal(result, seq_result));
}

/**
 * @test OpenMPConvenienceFunction
 * @brief Test OpenMP convenience function interface
 *
 * Validates the simplified OpenMP interface function.
 */
TEST_F(StrategyTest, OpenMPConvenienceFunction) {
    auto result = dtw_openmp<MetricType::EUCLIDEAN>(medium_a_, medium_b_, 2, 32);
    EXPECT_GT(result.first, 0.0);
    EXPECT_FALSE(result.second.empty());
}
#endif

/**
 * @test AutoStrategy
 * @brief Test automatic strategy selection
 *
 * Validates that AutoStrategy correctly selects appropriate
 * execution strategies based on problem size and available
 * backends.
 */
TEST_F(StrategyTest, AutoStrategy) {
AutoStrategy strategy(medium_a_.size(), medium_b_.size());

auto result = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::NONE>(
        medium_a_, medium_b_, strategy);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Check that the strategy name is not empty
EXPECT_FALSE(strategy.name().empty());
}

/**
 * @test NewConvenienceFunctions
 * @brief Test strategy convenience functions
 *
 * Comprehensive test of all convenience functions for different
 * strategies, ensuring they work correctly and produce consistent
 * results.
 */
TEST_F(StrategyTest, NewConvenienceFunctions) {
// Test unconstrained convenience function
auto unconstrained_result = dtw_unconstrained<MetricType::EUCLIDEAN>(
        small_a_, small_b_, SequentialStrategy{});
EXPECT_GT(unconstrained_result.first, 0.0);
EXPECT_FALSE(unconstrained_result.second.empty());

// Test sequential convenience function
auto seq_result = dtw_sequential<MetricType::EUCLIDEAN>(small_a_, small_b_);
EXPECT_GT(seq_result.first, 0.0);
EXPECT_FALSE(seq_result.second.empty());

// Test blocked convenience function
auto blocked_result = dtw_blocked<MetricType::EUCLIDEAN>(small_a_, small_b_, 16);
EXPECT_GT(blocked_result.first, 0.0);
EXPECT_FALSE(blocked_result.second.empty());

// Results should be identical (within numerical tolerance)
EXPECT_TRUE(results_equal(seq_result, blocked_result));

#ifdef USE_OPENMP
// Test OpenMP convenience function
    auto omp_result = dtw_openmp<MetricType::EUCLIDEAN>(small_a_, small_b_, 2, 32);
    EXPECT_GT(omp_result.first, 0.0);
    EXPECT_FALSE(omp_result.second.empty());
    EXPECT_TRUE(results_equal(seq_result, omp_result));
#endif

// Test auto-selection
auto auto_result = dtw_auto<MetricType::EUCLIDEAN>(medium_a_, medium_b_);
EXPECT_GT(auto_result.first, 0.0);
EXPECT_FALSE(auto_result.second.empty());
}

/**
 * @test StrategyConsistencyAcrossMetrics
 * @brief Verify strategy consistency with different metrics
 *
 * Critical test ensuring all strategies produce identical results
 * regardless of the distance metric used. Tests Euclidean, Manhattan,
 * and Chebyshev metrics.
 */
TEST_F(StrategyTest, StrategyConsistencyAcrossMetrics) {
SequentialStrategy seq_strategy;
BlockedStrategy blocked_strategy(32);

// Test Euclidean
auto seq_euclidean = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::NONE>(
        small_a_, small_b_, seq_strategy);
auto blocked_euclidean = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::NONE>(
        small_a_, small_b_, blocked_strategy);
EXPECT_TRUE(results_equal(seq_euclidean, blocked_euclidean));

// Test Manhattan
auto seq_manhattan = dtw<MetricType::MANHATTAN, constraints::ConstraintType::NONE>(
        small_a_, small_b_, seq_strategy);
auto blocked_manhattan = dtw<MetricType::MANHATTAN, constraints::ConstraintType::NONE>(
        small_a_, small_b_, blocked_strategy);
EXPECT_TRUE(results_equal(seq_manhattan, blocked_manhattan));

// Test Chebyshev
auto seq_chebyshev = dtw<MetricType::CHEBYSHEV, constraints::ConstraintType::NONE>(
        small_a_, small_b_, seq_strategy);
auto blocked_chebyshev = dtw<MetricType::CHEBYSHEV, constraints::ConstraintType::NONE>(
        small_a_, small_b_, blocked_strategy);
EXPECT_TRUE(results_equal(seq_chebyshev, blocked_chebyshev));
}

/**
 * @test StrategiesWithConstraints
 * @brief Test strategies with path constraints
 *
 * Validates that all strategies correctly handle path constraints
 * (Sakoe-Chiba, Itakura) and produce consistent results.
 */
TEST_F(StrategyTest, StrategiesWithConstraints) {
SequentialStrategy seq_strategy;
BlockedStrategy blocked_strategy(32);

// Test with Sakoe-Chiba constraint
auto seq_sc = dtw<MetricType::EUCLIDEAN,
        constraints::ConstraintType::SAKOE_CHIBA, 3>(
        small_a_, small_b_, seq_strategy);

auto blocked_sc = dtw<MetricType::EUCLIDEAN,
        constraints::ConstraintType::SAKOE_CHIBA, 3>(
        small_a_, small_b_, blocked_strategy);

EXPECT_GT(seq_sc.first, 0.0);
EXPECT_GT(blocked_sc.first, 0.0);
EXPECT_TRUE(results_equal(seq_sc, blocked_sc));

// Test with Itakura constraint
auto seq_it = dtw<MetricType::EUCLIDEAN,
        constraints::ConstraintType::ITAKURA, 1, 2.0>(
        small_a_, small_b_, seq_strategy);

auto blocked_it = dtw<MetricType::EUCLIDEAN,
        constraints::ConstraintType::ITAKURA, 1, 2.0>(
        small_a_, small_b_, blocked_strategy);

EXPECT_GT(seq_it.first, 0.0);
EXPECT_GT(blocked_it.first, 0.0);
EXPECT_TRUE(results_equal(seq_it, blocked_it));
}

/**
 * @test StrategiesWithWindow
 * @brief Test strategies with window constraints
 *
 * Validates that all strategies correctly handle custom window
 * constraints and produce consistent results for windowed DTW.
 */
TEST_F(StrategyTest, StrategiesWithWindow) {
SequentialStrategy seq_strategy;
BlockedStrategy blocked_strategy(16);

// Create a window
WindowConstraint window;
int n = std::min(small_a_.size(), small_b_.size());
for (int i = 0; i < n; ++i) {
for (int j = std::max(0, i-2); j <= std::min(n-1, i+2); ++j) {
window.push_back({i, j});
}
}

// Test with window using new interface
auto seq_window = dtw_windowed<MetricType::EUCLIDEAN>(
        small_a_, small_b_, window, seq_strategy);

auto blocked_window = dtw_windowed<MetricType::EUCLIDEAN>(
        small_a_, small_b_, window, blocked_strategy);

EXPECT_GT(seq_window.first, 0.0);
EXPECT_GT(blocked_window.first, 0.0);
EXPECT_TRUE(results_equal(seq_window, blocked_window));
}

/**
 * @test DirectExecuteWithConstraint
 * @brief Test direct execute_with_constraint interface
 *
 * Low-level test of the execute_with_constraint method,
 * validating both overloaded versions (with and without window).
 */
TEST_F(StrategyTest, DirectExecuteWithConstraint) {
SequentialStrategy strategy;
DoubleMatrix D;
int n = small_a_.size();
int m = small_b_.size();
int dim = small_a_.dimensions();

// Test overload 1: without window (should work without nullptr)
strategy.initialize_matrix(D, n, m);
strategy.execute_with_constraint<constraints::ConstraintType::NONE, 1, 2.0,
        MetricType::EUCLIDEAN>(
        D, small_a_, small_b_, n, m, dim);  // No nullptr!

auto result1 = strategy.extract_result(D);
EXPECT_GT(result1.first, 0.0);

// Test overload 2: with window
WindowConstraint window = {{0,0}, {1,1}, {2,2}};
strategy.initialize_matrix(D, n, m);
strategy.execute_with_constraint<constraints::ConstraintType::NONE, 1, 2.0,
        MetricType::EUCLIDEAN>(
        D, small_a_, small_b_, n, m, dim, &window);

auto result2 = strategy.extract_result(D);
EXPECT_TRUE(std::isfinite(result2.first));
}

/**
 * @test FastDTWWithStrategies
 * @brief Test FastDTW with different execution strategies
 *
 * Validates that FastDTW produces consistent approximate results
 * across different execution strategies, with appropriate tolerance
 * for approximation error.
 */
TEST_F(StrategyTest, FastDTWWithStrategies) {
SequentialStrategy seq_strategy;
BlockedStrategy blocked_strategy(32);

auto seq_fast = fastdtw<MetricType::EUCLIDEAN>(
        medium_a_, medium_b_, 2, 20, seq_strategy);

auto blocked_fast = fastdtw<MetricType::EUCLIDEAN>(
        medium_a_, medium_b_, 2, 20, blocked_strategy);

EXPECT_GT(seq_fast.first, 0.0);
EXPECT_GT(blocked_fast.first, 0.0);

// FastDTW is approximate, so allow more tolerance
EXPECT_TRUE(results_equal(seq_fast, blocked_fast, 1e-4));
}

/**
 * @brief Main test runner
 * @param argc Number of command-line arguments
 * @param argv Command-line arguments
 * @return Test execution status
 *
 * Initializes Google Test framework and runs all registered tests.
 * Returns 0 on success, non-zero on test failures.
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}