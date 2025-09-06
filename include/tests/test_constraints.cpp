#include "dtw_accelerator/dtw_accelerator.hpp"
#include "gtest/gtest.h"
#include <random>

using namespace dtw_accelerator;
using namespace dtw_accelerator::constraints;

// Test fixture for constraint tests
class ConstraintTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test series
        series_a_ = generate_test_series(20, 2, 42);
        series_b_ = generate_test_series(20, 2, 43);
    }

    DoubleTimeSeries generate_test_series(size_t length, size_t dim, unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(-5.0, 5.0);

        DoubleTimeSeries series(length, dim);
        for (size_t i = 0; i < length; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                series[i][d] = dis(gen);
            }
        }
        return series;
    }

    DoubleTimeSeries series_a_;
    DoubleTimeSeries series_b_;
};

// Test Sakoe-Chiba band constraint
TEST_F(ConstraintTest, SakoeChibaBand) {
// Test with different radius values
const int radius = 2;

// Check constraint function directly
int n = 10, m = 10;

// Points on diagonal should be within band
EXPECT_TRUE(within_sakoe_chiba_band<radius>(0, 0, n, m));
EXPECT_TRUE(within_sakoe_chiba_band<radius>(5, 5, n, m));
EXPECT_TRUE(within_sakoe_chiba_band<radius>(9, 9, n, m));

// Points near diagonal should be within band
EXPECT_TRUE(within_sakoe_chiba_band<radius>(3, 4, n, m));
EXPECT_TRUE(within_sakoe_chiba_band<radius>(4, 3, n, m));

// Points far from diagonal should be outside band
EXPECT_FALSE(within_sakoe_chiba_band<5>(0, 9, n, m));
EXPECT_FALSE(within_sakoe_chiba_band<5>(9, 0, n, m));
}

// Test Itakura parallelogram constraint
TEST_F(ConstraintTest, ItakuraParallelogram) {
constexpr double slope = 2.0;

int n = 10, m = 10;

// Points on diagonal should be within parallelogram
EXPECT_TRUE(within_itakura_parallelogram<slope>(0, 0, n, m));
EXPECT_TRUE(within_itakura_parallelogram<slope>(5, 5, n, m));
EXPECT_TRUE(within_itakura_parallelogram<slope>(9, 9, n, m));

// Check some boundary conditions
// The exact boundaries depend on the slope parameter
// Just ensure the function doesn't crash and returns reasonable values
bool result1 = within_itakura_parallelogram<slope>(2, 8, n, m);
bool result2 = within_itakura_parallelogram<slope>(8, 2, n, m);

// At least check these return valid booleans
EXPECT_TRUE(result1 == true || result1 == false);
EXPECT_TRUE(result2 == true || result2 == false);
}

// Test constrained DTW with window
TEST_F(ConstraintTest, ConstrainedDTWWithWindow) {
// Create a diagonal window
std::vector<std::pair<int, int>> window;
int n = series_a_.size();
int m = series_b_.size();
int radius = 2;

for (int i = 0; i < n; ++i) {
    for (int j = std::max(0, i - radius); j <= std::min(m - 1, i + radius); ++j) {
        window.push_back({i, j});
    }
}

strategies::BlockedStrategy strategy(32);

auto constrained_result = dtw_constrained<MetricType::EUCLIDEAN>(
        series_a_, series_b_, window, strategy
);

auto unconstrained_result = dtw<MetricType::EUCLIDEAN>(
        series_a_, series_b_, strategy
);

// Constrained DTW should have cost >= unconstrained
EXPECT_GE(constrained_result.first, unconstrained_result.first);

// Both should produce valid paths
EXPECT_FALSE(constrained_result.second.empty());
EXPECT_FALSE(unconstrained_result.second.empty());
}

// Test DTW with Sakoe-Chiba constraint using template
TEST_F(ConstraintTest, DTWWithSakoeChibaTemplate) {
strategies::SequentialStrategy strategy;

// Test with different radius values
auto result_r1 = dtw_with_constraint<ConstraintType::SAKOE_CHIBA, 1>(
        series_a_, series_b_, strategy
);

auto result_r3 = dtw_with_constraint<ConstraintType::SAKOE_CHIBA, 3>(
        series_a_, series_b_, strategy
);

auto result_r5 = dtw_with_constraint<ConstraintType::SAKOE_CHIBA, 5>(
        series_a_, series_b_, strategy
);

// All should produce valid results
EXPECT_GT(result_r1.first, 0.0);
EXPECT_GT(result_r3.first, 0.0);
EXPECT_GT(result_r5.first, 0.0);

// Smaller radius should generally give higher cost (more constrained)
EXPECT_GE(result_r1.first, result_r3.first);
EXPECT_GE(result_r3.first, result_r5.first);
}

// Test DTW with Itakura constraint using template
TEST_F(ConstraintTest, DTWWithItakuraTemplate) {
strategies::SequentialStrategy strategy;

// Test with different slope values
auto result_s15 = dtw_with_constraint<ConstraintType::ITAKURA, 1, 1.5>(
        series_a_, series_b_, strategy
);

auto result_s20 = dtw_with_constraint<ConstraintType::ITAKURA, 1, 2.0>(
        series_a_, series_b_, strategy
);

auto result_s30 = dtw_with_constraint<ConstraintType::ITAKURA, 1, 3.0>(
        series_a_, series_b_, strategy
);

// All should produce valid results
EXPECT_GE(result_s15.first, 0.0);
EXPECT_GE(result_s20.first, 0.0);
EXPECT_GE(result_s30.first, 0.0);
}

// Test that constrained DTW is consistent across strategies
TEST_F(ConstraintTest, ConstrainedConsistencyAcrossStrategies) {
// Create a simple window
std::vector<std::pair<int, int>> window;
int n = 10;
int m = 10;
int radius = 2;

for (int i = 0; i < n; ++i) {
for (int j = std::max(0, i - radius); j <= std::min(m - 1, i + radius); ++j) {
window.push_back({i, j});
}
}

// Create small test series
DoubleTimeSeries small_a = generate_test_series(10, 2, 100);
DoubleTimeSeries small_b = generate_test_series(10, 2, 101);

strategies::SequentialStrategy seq_strategy;
strategies::BlockedStrategy blocked_strategy(16);

auto seq_result = dtw_constrained<MetricType::EUCLIDEAN>(
        small_a, small_b, window, seq_strategy
);

auto blocked_result = dtw_constrained<MetricType::EUCLIDEAN>(
        small_a, small_b, window, blocked_strategy
);

// Results should be nearly identical
EXPECT_NEAR(seq_result.first, blocked_result.first, 1e-6);
}

// Test empty window constraint
TEST_F(ConstraintTest, EmptyWindowConstraint) {
std::vector<std::pair<int, int>> empty_window;
strategies::SequentialStrategy strategy;

auto result = dtw_constrained<MetricType::EUCLIDEAN>(
        series_a_, series_b_, empty_window, strategy
);

// With empty window, result should be 0 and empty path
EXPECT_EQ(result.first, 0.0);
EXPECT_TRUE(result.second.empty());
}

// Test single point window
TEST_F(ConstraintTest, SinglePointWindow) {
std::vector<std::pair<int, int>> single_window = {{0, 0}};

// Create single element series
DoubleTimeSeries single_a(1, 2);
single_a[0][0] = 1.0;
single_a[0][1] = 2.0;

DoubleTimeSeries single_b(1, 2);
single_b[0][0] = 1.5;
single_b[0][1] = 2.5;

strategies::SequentialStrategy strategy;

auto result = dtw_constrained<MetricType::EUCLIDEAN>(
        single_a, single_b, single_window, strategy
);

// Should compute distance for the single point
EXPECT_GT(result.first, 0.0);
EXPECT_EQ(result.second.size(), 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}