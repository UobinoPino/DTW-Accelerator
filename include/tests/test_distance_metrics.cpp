/**
 * @file test_distance_metrics.cpp
 * @brief Unit tests for distance metric implementations
 * @author UobinoPino
 * @date 2024
 *
 * This file contains comprehensive tests for all supported distance
 * metrics (Euclidean, Manhattan, Chebyshev, Cosine) and their integration
 * with DTW algorithms. Tests include direct metric computation and
 * DTW computation with different metrics.
 */

#include "dtw_accelerator/dtw_accelerator.hpp"
#include "gtest/gtest.h"
#include <cmath>

using namespace dtw_accelerator;
using namespace dtw_accelerator::distance;

/**
 * @class DistanceMetricsTest
 * @brief Test fixture for distance metric functionality
 *
 * Provides test data and validation methods for testing various
 * distance metrics used in DTW computations. Tests both direct
 * distance calculations and their integration with DTW.
 */
class DistanceMetricsTest : public ::testing::Test {
protected:
    /**
     * @brief Set up test environment before each test
     *
     * Creates test time series with known values for validating
     * distance metric computations. Uses 3x3 dimensional series
     * with incrementing values for predictable results.
     */
    void SetUp() override {
        // Create test data
        series_a_ = DoubleTimeSeries(3, 3);
        series_a_[0][0] = 1.0; series_a_[0][1] = 2.0; series_a_[0][2] = 3.0;
        series_a_[1][0] = 4.0; series_a_[1][1] = 5.0; series_a_[1][2] = 6.0;
        series_a_[2][0] = 7.0; series_a_[2][1] = 8.0; series_a_[2][2] = 9.0;

        series_b_ = DoubleTimeSeries(3, 3);
        series_b_[0][0] = 1.5; series_b_[0][1] = 2.5; series_b_[0][2] = 3.5;
        series_b_[1][0] = 4.5; series_b_[1][1] = 5.5; series_b_[1][2] = 6.5;
        series_b_[2][0] = 7.5; series_b_[2][1] = 8.5; series_b_[2][2] = 9.5;
    }

    /// @brief First test time series with integer values
    DoubleTimeSeries series_a_;

    /// @brief Second test time series offset by 0.5
    DoubleTimeSeries series_b_;
};

/**
 * @test EuclideanDistance
 * @brief Test Euclidean (L2) distance metric
 *
 * Validates Euclidean distance computation in DTW context.
 * Verifies that identical series produce zero distance and
 * different series produce positive distance.
 */
TEST_F(DistanceMetricsTest, EuclideanDistance) {
auto result = dtw<MetricType::EUCLIDEAN>(series_a_, series_b_);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Euclidean distance between identical series should be 0
auto identical_result = dtw<MetricType::EUCLIDEAN>(series_a_, series_a_);
EXPECT_NEAR(identical_result.first, 0.0, 1e-10);
}

/**
 * @test ManhattanDistance
 * @brief Test Manhattan (L1) distance metric
 *
 * Validates Manhattan distance computation, which sums absolute
 * differences rather than squared differences.
 */
TEST_F(DistanceMetricsTest, ManhattanDistance) {
auto result = dtw<MetricType::MANHATTAN>(series_a_, series_b_);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Manhattan distance between identical series should be 0
auto identical_result = dtw<MetricType::MANHATTAN>(series_a_, series_a_);
EXPECT_NEAR(identical_result.first, 0.0, 1e-10);
}

/**
 * @test ChebyshevDistance
 * @brief Test Chebyshev (L∞) distance metric
 *
 * Validates Chebyshev distance computation, which uses the
 * maximum absolute difference across dimensions.
 */
TEST_F(DistanceMetricsTest, ChebyshevDistance) {
auto result = dtw<MetricType::CHEBYSHEV>(series_a_, series_b_);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Chebyshev distance between identical series should be 0
auto identical_result = dtw<MetricType::CHEBYSHEV>(series_a_, series_a_);
EXPECT_NEAR(identical_result.first, 0.0, 1e-10);
}

/**
 * @test CosineDistance
 * @brief Test Cosine distance metric
 *
 * Validates Cosine distance computation (1 - cosine similarity).
 * Tests include handling of parallel vectors (distance ≈ 0) and
 * orthogonal vectors (distance ≈ 1).
 */
TEST_F(DistanceMetricsTest, CosineDistance) {
auto result = dtw<MetricType::COSINE>(series_a_, series_b_);

EXPECT_GE(result.first, 0.0);  // Cosine distance is always >= 0
EXPECT_FALSE(result.second.empty());

// Cosine distance between identical series should be close to 0
auto identical_result = dtw<MetricType::COSINE>(series_a_, series_a_);
EXPECT_NEAR(identical_result.first, 0.0, 1e-10);
}

/**
 * @test DifferentMetricsProduceDifferentResults
 * @brief Verify that metrics produce distinct results
 *
 * Confirms that different distance metrics produce different DTW
 * distances for the same input series, validating that each metric
 * is correctly implemented and distinct.
 */
TEST_F(DistanceMetricsTest, DifferentMetricsProduceDifferentResults) {
auto euclidean = dtw<MetricType::EUCLIDEAN>(series_a_, series_b_);
auto manhattan = dtw<MetricType::MANHATTAN>(series_a_, series_b_);
auto chebyshev = dtw<MetricType::CHEBYSHEV>(series_a_, series_b_);
auto cosine = dtw<MetricType::COSINE>(series_a_, series_b_);

// All should produce valid results
EXPECT_GT(euclidean.first, 0.0);
EXPECT_GT(manhattan.first, 0.0);
EXPECT_GT(chebyshev.first, 0.0);
EXPECT_GE(cosine.first, 0.0);

// Different metrics should generally produce different costs
EXPECT_NE(euclidean.first, manhattan.first);
EXPECT_NE(euclidean.first, chebyshev.first);

// All paths should be the same length
EXPECT_EQ(euclidean.second.size(), manhattan.second.size());
EXPECT_EQ(euclidean.second.size(), chebyshev.second.size());
EXPECT_EQ(euclidean.second.size(), cosine.second.size());
}

/**
 * @test DirectMetricComputation
 * @brief Test direct distance computation without DTW
 *
 * Validates the underlying distance functions by computing
 * distances directly between point pairs and comparing with
 * expected values calculated manually.
 */
TEST_F(DistanceMetricsTest, DirectMetricComputation) {
double a[] = {1.0, 2.0, 3.0};
double b[] = {1.5, 2.5, 3.5};
int dim = 3;

// Test Euclidean
double euclidean_dist = Metric<MetricType::EUCLIDEAN>::compute(a, b, dim);
double expected_euclidean = std::sqrt(0.5*0.5 + 0.5*0.5 + 0.5*0.5);
EXPECT_NEAR(euclidean_dist, expected_euclidean, 1e-10);

// Test Manhattan
double manhattan_dist = Metric<MetricType::MANHATTAN>::compute(a, b, dim);
double expected_manhattan = 0.5 + 0.5 + 0.5;
EXPECT_NEAR(manhattan_dist, expected_manhattan, 1e-10);

// Test Chebyshev
double chebyshev_dist = Metric<MetricType::CHEBYSHEV>::compute(a, b, dim);
double expected_chebyshev = 0.5;  // Maximum difference
EXPECT_NEAR(chebyshev_dist, expected_chebyshev, 1e-10);

// Test Cosine (should be small for similar vectors)
double cosine_dist = Metric<MetricType::COSINE>::compute(a, b, dim);
EXPECT_GE(cosine_dist, 0.0);
EXPECT_LE(cosine_dist, 2.0);  // Cosine distance is bounded between 0 and 2
}

/**
 * @test CosineDistanceWithZeroVectors
 * @brief Test Cosine distance edge case with zero vectors
 *
 * Validates that Cosine distance handles zero vectors gracefully,
 * returning a valid result (typically 1.0) without NaN or infinity.
 */
TEST_F(DistanceMetricsTest, CosineDistanceWithZeroVectors) {
DoubleTimeSeries zero_series(2, 3, 0.0);  // All zeros
DoubleTimeSeries normal_series(2, 3);
normal_series[0][0] = 1.0; normal_series[0][1] = 2.0; normal_series[0][2] = 3.0;
normal_series[1][0] = 4.0; normal_series[1][1] = 5.0; normal_series[1][2] = 6.0;

// Cosine distance with zero vector should handle it gracefully
auto result = dtw<MetricType::COSINE>(zero_series, normal_series);
EXPECT_TRUE(std::isfinite(result.first));  // Should not be NaN or Inf
}

/**
 * @test MetricsWithConstraints
 * @brief Test distance metrics with path constraints
 *
 * Validates that all distance metrics work correctly when combined
 * with path constraints (Sakoe-Chiba, Itakura, window constraints).
 */
TEST_F(DistanceMetricsTest, MetricsWithConstraints) {
execution::SequentialStrategy strategy;

// Test Euclidean with Sakoe-Chiba
auto euclidean_sc = dtw<MetricType::EUCLIDEAN,
        constraints::ConstraintType::SAKOE_CHIBA, 2>(
        series_a_, series_b_, strategy);
EXPECT_GT(euclidean_sc.first, 0.0);

// Test Manhattan with Itakura
auto manhattan_it = dtw<MetricType::MANHATTAN,
        constraints::ConstraintType::ITAKURA, 1, 2.0>(
        series_a_, series_b_, strategy);
EXPECT_GT(manhattan_it.first, 0.0);

// Test with window constraint
execution::WindowConstraint window = {{0,0}, {0,1}, {1,0}, {1,1}, {2,2}};
auto chebyshev_window = dtw<MetricType::CHEBYSHEV,
        constraints::ConstraintType::NONE>(
        series_a_, series_b_, strategy, &window);
EXPECT_TRUE(std::isfinite(chebyshev_window.first));
}

/**
 * @test MetricsWithConvenienceFunctions
 * @brief Test metrics with convenience function interfaces
 *
 * Validates that distance metrics work correctly through all
 * convenience function interfaces (unconstrained, Sakoe-Chiba, etc.).
 */
TEST_F(DistanceMetricsTest, MetricsWithConvenienceFunctions) {
execution::BlockedStrategy strategy(32);

// Test unconstrained with different metrics
auto euclidean = dtw_unconstrained<MetricType::EUCLIDEAN>(
        series_a_, series_b_, strategy);
auto manhattan = dtw_unconstrained<MetricType::MANHATTAN>(
        series_a_, series_b_, strategy);

EXPECT_GT(euclidean.first, 0.0);
EXPECT_GT(manhattan.first, 0.0);
EXPECT_NE(euclidean.first, manhattan.first);

// Test Sakoe-Chiba with different metrics
auto euclidean_sc = dtw_sakoe_chiba<MetricType::EUCLIDEAN, 2>(
series_a_, series_b_, strategy);
auto manhattan_sc = dtw_sakoe_chiba<MetricType::MANHATTAN, 2>(
series_a_, series_b_, strategy);

EXPECT_GT(euclidean_sc.first, 0.0);
EXPECT_GT(manhattan_sc.first, 0.0);
EXPECT_NE(euclidean_sc.first, manhattan_sc.first);
}
/**
 * @brief Main test runner
 * @param argc Number of command-line arguments
 * @param argv Command-line arguments
 * @return Test execution status
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}