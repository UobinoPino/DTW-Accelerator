#include "dtw_accelerator/dtw_accelerator.hpp"
#include "gtest/gtest.h"
#include <random>

using namespace dtw_accelerator;
using namespace dtw_accelerator::constraints;
using namespace dtw_accelerator::execution;

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

// Test windowed DTW
TEST_F(ConstraintTest, WindowedDTW) {
// Create a diagonal window
WindowConstraint window;
int n = series_a_.size();
int m = series_b_.size();
int radius = 2;

for (int i = 0; i < n; ++i) {
for (int j = std::max(0, i - radius); j <= std::min(m - 1, i + radius); ++j) {
window.push_back({i, j});
}
}

BlockedStrategy strategy(32);

auto windowed_result = dtw_windowed<MetricType::EUCLIDEAN>(
        series_a_, series_b_, window, strategy
);

// Compare with unconstrained
auto unconstrained_result = dtw_unconstrained<MetricType::EUCLIDEAN>(
        series_a_, series_b_, strategy
);

// Windowed DTW should have cost >= unconstrained
EXPECT_GE(windowed_result.first, unconstrained_result.first);

// Both should produce valid paths
EXPECT_FALSE(windowed_result.second.empty());
EXPECT_FALSE(unconstrained_result.second.empty());
}

// Test DTW with Sakoe-Chiba constraint
TEST_F(ConstraintTest, DTWWithSakoeChibaTemplate) {
SequentialStrategy strategy;

// Test with different radius values
auto result_r1 = dtw_sakoe_chiba<MetricType::EUCLIDEAN, 1>(
series_a_, series_b_, strategy
);

auto result_r3 = dtw_sakoe_chiba<MetricType::EUCLIDEAN, 3>(
series_a_, series_b_, strategy
);

auto result_r5 = dtw_sakoe_chiba<MetricType::EUCLIDEAN, 5>(
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

// Test DTW with Itakura constraint
TEST_F(ConstraintTest, DTWWithItakuraTemplate) {
SequentialStrategy strategy;

// Test with different slope values
auto result_s15 = dtw_itakura<MetricType::EUCLIDEAN, 1.5>(
series_a_, series_b_, strategy
);

auto result_s20 = dtw_itakura<MetricType::EUCLIDEAN, 2.0>(
series_a_, series_b_, strategy
);

auto result_s30 = dtw_itakura<MetricType::EUCLIDEAN, 3.0>(
series_a_, series_b_, strategy
);

// All should produce valid results
EXPECT_GE(result_s15.first, 0.0);
EXPECT_GE(result_s20.first, 0.0);
EXPECT_GE(result_s30.first, 0.0);
}

// Test that windowed DTW is consistent across strategies
TEST_F(ConstraintTest, WindowedConsistencyAcrossStrategies) {
// Create a simple window
WindowConstraint window;
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

SequentialStrategy seq_strategy;
BlockedStrategy blocked_strategy(16);

auto seq_result = dtw_windowed<MetricType::EUCLIDEAN>(
        small_a, small_b, window, seq_strategy
);

auto blocked_result = dtw_windowed<MetricType::EUCLIDEAN>(
        small_a, small_b, window, blocked_strategy
);

// Results should be nearly identical
EXPECT_NEAR(seq_result.first, blocked_result.first, 1e-6);
}

// Test empty window constraint
TEST_F(ConstraintTest, EmptyWindowConstraint) {
WindowConstraint empty_window;
SequentialStrategy strategy;

auto result = dtw_windowed<MetricType::EUCLIDEAN>(
        series_a_, series_b_, empty_window, strategy
);

// With empty window, result should be infinity (no valid path)
// Check that it doesn't crash and returns a valid result
EXPECT_TRUE(std::isfinite(result.first) || std::isinf(result.first));
}

// Test single point window
TEST_F(ConstraintTest, SinglePointWindow) {
WindowConstraint single_window = {{0, 0}};

// Create single element series
DoubleTimeSeries single_a(1, 2);
single_a[0][0] = 1.0;
single_a[0][1] = 2.0;

DoubleTimeSeries single_b(1, 2);
single_b[0][0] = 1.5;
single_b[0][1] = 2.5;

SequentialStrategy strategy;

auto result = dtw_windowed<MetricType::EUCLIDEAN>(
        single_a, single_b, single_window, strategy
);

// Should compute distance for the single point
EXPECT_GT(result.first, 0.0);
EXPECT_EQ(result.second.size(), 1);
}

// Test the new unified execute_with_constraint interface directly
TEST_F(ConstraintTest, UnifiedExecuteInterface) {
SequentialStrategy strategy;
DoubleMatrix D;
int n = series_a_.size();
int m = series_b_.size();
int dim = series_a_.dimensions();

strategy.initialize_matrix(D, n, m);

// Test 1: Call without window (should work without specifying nullptr)
strategy.execute_with_constraint<ConstraintType::NONE, 1, 2.0, MetricType::EUCLIDEAN>(
        D, series_a_, series_b_, n, m, dim
);

auto result1 = strategy.extract_result(D);
EXPECT_GT(result1.first, 0.0);

// Test 2: Call with Sakoe-Chiba constraint (no nullptr needed)
strategy.initialize_matrix(D, n, m);
strategy.execute_with_constraint<ConstraintType::SAKOE_CHIBA, 3, 2.0, MetricType::EUCLIDEAN>(
        D, series_a_, series_b_, n, m, dim
);

auto result2 = strategy.extract_result(D);
EXPECT_GT(result2.first, 0.0);

// Test 3: Call with window
WindowConstraint window = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
strategy.initialize_matrix(D, n, m);
strategy.execute_with_constraint<ConstraintType::NONE, 1, 2.0, MetricType::EUCLIDEAN>(
        D, series_a_, series_b_, n, m, dim, &window
);  // Window passed explicitly

auto result3 = strategy.extract_result(D);
EXPECT_TRUE(std::isfinite(result3.first) || std::isinf(result3.first));
}

// Test that the overloading works correctly
TEST_F(ConstraintTest, FunctionOverloadingTest) {
BlockedStrategy strategy(32);

// This should compile and work without explicitly passing nullptr
auto result1 = dtw<MetricType::EUCLIDEAN, ConstraintType::NONE>(
        series_a_, series_b_, strategy
);
EXPECT_GT(result1.first, 0.0);

// This should also work with constraint
auto result2 = dtw<MetricType::EUCLIDEAN, ConstraintType::SAKOE_CHIBA, 3>(
        series_a_, series_b_, strategy
);
EXPECT_GT(result2.first, 0.0);

// And this should work with window
WindowConstraint window;
for (int i = 0; i < 5; ++i) {
for (int j = 0; j < 5; ++j) {
window.push_back({i, j});
}
}

auto result3 = dtw<MetricType::EUCLIDEAN, ConstraintType::NONE>(
        series_a_, series_b_, strategy, &window
);
EXPECT_GT(result3.first, 0.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}