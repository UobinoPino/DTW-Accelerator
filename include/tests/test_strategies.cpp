#include "dtw_accelerator/dtw_accelerator.hpp"
#include "gtest/gtest.h"
#include <random>

using namespace dtw_accelerator;
using namespace dtw_accelerator::strategies;

// Test fixture for strategy tests
class StrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate test data
        small_a_ = generate_test_series(50, 2, 42);
        small_b_ = generate_test_series(50, 2, 43);

        medium_a_ = generate_test_series(200, 3, 44);
        medium_b_ = generate_test_series(180, 3, 45);
    }

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

    bool results_equal(const std::pair<double, std::vector<std::pair<int, int>>>& r1,
    const std::pair<double, std::vector<std::pair<int, int>>>& r2,
    double tolerance = 1e-6) {
        return std::abs(r1.first - r2.first) < tolerance;
    }

    DoubleTimeSeries small_a_;
    DoubleTimeSeries small_b_;
    DoubleTimeSeries medium_a_;
    DoubleTimeSeries medium_b_;
};

// Test Sequential Strategy
TEST_F(StrategyTest, SequentialStrategy) {
SequentialStrategy strategy;

EXPECT_EQ(strategy.name(), "Sequential");
EXPECT_FALSE(strategy.is_parallel());

auto result = dtw<MetricType::EUCLIDEAN>(small_a_, small_b_, strategy);
EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());
}

// Test Blocked Strategy
TEST_F(StrategyTest, BlockedStrategy) {
BlockedStrategy strategy(32);

EXPECT_EQ(strategy.name(), "Blocked");
EXPECT_FALSE(strategy.is_parallel());
EXPECT_EQ(strategy.get_block_size(), 32);

strategy.set_block_size(64);
EXPECT_EQ(strategy.get_block_size(), 64);

auto result = dtw<MetricType::EUCLIDEAN>(medium_a_, medium_b_, strategy);
EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Compare with sequential for correctness
SequentialStrategy seq_strategy;
auto seq_result = dtw<MetricType::EUCLIDEAN>(medium_a_, medium_b_, seq_strategy);

EXPECT_TRUE(results_equal(result, seq_result));
}

#ifdef USE_OPENMP
// Test OpenMP Strategy
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
    
    auto result = dtw<MetricType::EUCLIDEAN>(medium_a_, medium_b_, strategy);
    EXPECT_GT(result.first, 0.0);
    EXPECT_FALSE(result.second.empty());
    
    // Compare with sequential for correctness
    SequentialStrategy seq_strategy;
    auto seq_result = dtw<MetricType::EUCLIDEAN>(medium_a_, medium_b_, seq_strategy);
    
    EXPECT_TRUE(results_equal(result, seq_result));
}

// Test OpenMP convenience function
TEST_F(StrategyTest, OpenMPConvenienceFunction) {
    auto result = dtw_openmp<MetricType::EUCLIDEAN>(medium_a_, medium_b_, 2, 32);
    EXPECT_GT(result.first, 0.0);
    EXPECT_FALSE(result.second.empty());
}
#endif

// Test Auto Strategy
TEST_F(StrategyTest, AutoStrategy) {
AutoStrategy strategy(medium_a_.size(), medium_b_.size());

auto result = dtw<MetricType::EUCLIDEAN>(medium_a_, medium_b_, strategy);
EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Check that the strategy name is not empty
EXPECT_FALSE(strategy.name().empty());
}

// Test convenience functions
TEST_F(StrategyTest, ConvenienceFunctions) {
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

// Test strategy consistency across different metrics
TEST_F(StrategyTest, StrategyConsistencyAcrossMetrics) {
SequentialStrategy seq_strategy;
BlockedStrategy blocked_strategy(32);

// Test Euclidean
auto seq_euclidean = dtw<MetricType::EUCLIDEAN>(small_a_, small_b_, seq_strategy);
auto blocked_euclidean = dtw<MetricType::EUCLIDEAN>(small_a_, small_b_, blocked_strategy);
EXPECT_TRUE(results_equal(seq_euclidean, blocked_euclidean));

// Test Manhattan
auto seq_manhattan = dtw<MetricType::MANHATTAN>(small_a_, small_b_, seq_strategy);
auto blocked_manhattan = dtw<MetricType::MANHATTAN>(small_a_, small_b_, blocked_strategy);
EXPECT_TRUE(results_equal(seq_manhattan, blocked_manhattan));

// Test Chebyshev
auto seq_chebyshev = dtw<MetricType::CHEBYSHEV>(small_a_, small_b_, seq_strategy);
auto blocked_chebyshev = dtw<MetricType::CHEBYSHEV>(small_a_, small_b_, blocked_strategy);
EXPECT_TRUE(results_equal(seq_chebyshev, blocked_chebyshev));
}

// Test custom strategy function
TEST_F(StrategyTest, CustomStrategyFunction) {
SequentialStrategy custom_strategy;

auto result = dtw_custom<MetricType::EUCLIDEAN>(small_a_, small_b_, custom_strategy);
EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());

// Should match standard sequential result
auto seq_result = dtw_sequential<MetricType::EUCLIDEAN>(small_a_, small_b_);
EXPECT_TRUE(results_equal(result, seq_result));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}