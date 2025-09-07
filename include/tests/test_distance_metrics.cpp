#include "dtw_accelerator/dtw_accelerator.hpp"
#include "gtest/gtest.h"
#include <cmath>

using namespace dtw_accelerator;
using namespace dtw_accelerator::distance;

// Test fixture for distance metrics
class DistanceMetricsTest : public ::testing::Test {
protected:
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

    DoubleTimeSeries series_a_;
    DoubleTimeSeries series_b_;
};

// Test Euclidean distance
TEST_F(DistanceMetricsTest, EuclideanDistance) {
auto result = dtw<MetricType::EUCLIDEAN>(series_a_, series_b_);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Euclidean distance between identical series should be 0
auto identical_result = dtw<MetricType::EUCLIDEAN>(series_a_, series_a_);
EXPECT_NEAR(identical_result.first, 0.0, 1e-10);
}

// Test Manhattan distance
TEST_F(DistanceMetricsTest, ManhattanDistance) {
auto result = dtw<MetricType::MANHATTAN>(series_a_, series_b_);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Manhattan distance between identical series should be 0
auto identical_result = dtw<MetricType::MANHATTAN>(series_a_, series_a_);
EXPECT_NEAR(identical_result.first, 0.0, 1e-10);
}

// Test Chebyshev distance
TEST_F(DistanceMetricsTest, ChebyshevDistance) {
auto result = dtw<MetricType::CHEBYSHEV>(series_a_, series_b_);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Chebyshev distance between identical series should be 0
auto identical_result = dtw<MetricType::CHEBYSHEV>(series_a_, series_a_);
EXPECT_NEAR(identical_result.first, 0.0, 1e-10);
}

// Test Cosine distance
TEST_F(DistanceMetricsTest, CosineDistance) {
auto result = dtw<MetricType::COSINE>(series_a_, series_b_);

EXPECT_GE(result.first, 0.0);  // Cosine distance is always >= 0
EXPECT_FALSE(result.second.empty());

// Cosine distance between identical series should be close to 0
auto identical_result = dtw<MetricType::COSINE>(series_a_, series_a_);
EXPECT_NEAR(identical_result.first, 0.0, 1e-10);
}

// Test that different metrics produce different results
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

// Test metric computation directly
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

// Test with zero vectors for cosine distance
TEST_F(DistanceMetricsTest, CosineDistanceWithZeroVectors) {
DoubleTimeSeries zero_series(2, 3, 0.0);  // All zeros
DoubleTimeSeries normal_series(2, 3);
normal_series[0][0] = 1.0; normal_series[0][1] = 2.0; normal_series[0][2] = 3.0;
normal_series[1][0] = 4.0; normal_series[1][1] = 5.0; normal_series[1][2] = 6.0;

// Cosine distance with zero vector should handle it gracefully
auto result = dtw<MetricType::COSINE>(zero_series, normal_series);
EXPECT_TRUE(std::isfinite(result.first));  // Should not be NaN or Inf
}

// Test distance metrics with different constraint types
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

// Test that metrics work correctly with convenience functions
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}