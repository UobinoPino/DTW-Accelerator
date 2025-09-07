#include "dtw_accelerator/dtw_accelerator.hpp"
#include "gtest/gtest.h"
#include <random>
#include <numeric>

using namespace dtw_accelerator;
using namespace dtw_accelerator::strategies;

// Test fixture for DTW tests
class DTWCoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple test series
        simple_a_ = DoubleTimeSeries(3, 2);
        simple_a_[0][0] = 1.0; simple_a_[0][1] = 2.0;
        simple_a_[1][0] = 3.0; simple_a_[1][1] = 4.0;
        simple_a_[2][0] = 5.0; simple_a_[2][1] = 6.0;

        simple_b_ = DoubleTimeSeries(3, 2);
        simple_b_[0][0] = 1.5; simple_b_[0][1] = 2.5;
        simple_b_[1][0] = 3.5; simple_b_[1][1] = 4.5;
        simple_b_[2][0] = 5.5; simple_b_[2][1] = 6.5;

        // Create larger test series
        larger_a_ = generate_test_series(100, 3, 42);
        larger_b_ = generate_test_series(90, 3, 43);
    }

    DoubleTimeSeries generate_test_series(size_t length, size_t dim, unsigned seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(-1.0, 1.0);

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

    DoubleTimeSeries simple_a_;
    DoubleTimeSeries simple_b_;
    DoubleTimeSeries larger_a_;
    DoubleTimeSeries larger_b_;
};

// Test basic DTW computation
TEST_F(DTWCoreTest, BasicDTW) {
// Using the default unconstrained DTW
auto result = dtw(simple_a_, simple_b_);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());
EXPECT_EQ(result.second.size(), 3);
}

// Test DTW with identical series
TEST_F(DTWCoreTest, IdenticalSeries) {
auto result = dtw(simple_a_, simple_a_);

EXPECT_NEAR(result.first, 0.0, 1e-10);
EXPECT_FALSE(result.second.empty());
}

// Test empty series
TEST_F(DTWCoreTest, EmptySeries) {
DoubleTimeSeries empty_a(0, 0);
DoubleTimeSeries empty_b(0, 0);

auto result = dtw(empty_a, empty_b);

EXPECT_EQ(result.first, 0.0);
EXPECT_TRUE(result.second.empty());
}

// Test single element series
TEST_F(DTWCoreTest, SingleElementSeries) {
DoubleTimeSeries single_a(1, 2);
single_a[0][0] = 1.0;
single_a[0][1] = 2.0;

DoubleTimeSeries single_b(1, 2);
single_b[0][0] = 1.5;
single_b[0][1] = 2.5;

auto result = dtw(single_a, single_b);

EXPECT_GT(result.first, 0.0);
EXPECT_EQ(result.second.size(), 1);
}

// Test different length series
TEST_F(DTWCoreTest, DifferentLengthSeries) {
DoubleTimeSeries series_a(5, 2);
DoubleTimeSeries series_b(3, 2);

for (size_t i = 0; i < 5; ++i) {
series_a[i][0] = i;
series_a[i][1] = i * 2;
}

for (size_t i = 0; i < 3; ++i) {
series_b[i][0] = i * 2;
series_b[i][1] = i * 4;
}

auto result = dtw(series_a, series_b);

EXPECT_GT(result.first, 0.0);
EXPECT_FALSE(result.second.empty());
}

// Test FastDTW
TEST_F(DTWCoreTest, FastDTW) {
auto standard_result = dtw(larger_a_, larger_b_);
auto fast_result = fastdtw(larger_a_, larger_b_, 2, 10);

// FastDTW should produce a valid result
EXPECT_GT(fast_result.first, 0.0);
EXPECT_FALSE(fast_result.second.empty());

// FastDTW result should be close to standard DTW (within some tolerance)
double relative_error = std::abs(fast_result.first - standard_result.first) / standard_result.first;
EXPECT_LT(relative_error, 0.3);  // Less than 30% error
}

// Test new convenience functions
TEST_F(DTWCoreTest, NewConvenienceFunctions) {
SequentialStrategy strategy;

// Test unconstrained DTW
auto unconstrained = dtw_unconstrained<MetricType::EUCLIDEAN>(
        simple_a_, simple_b_, strategy);
EXPECT_GT(unconstrained.first, 0.0);

// Test Sakoe-Chiba constrained DTW
auto sakoe_chiba = dtw_sakoe_chiba<MetricType::EUCLIDEAN, 2>(
simple_a_, simple_b_, strategy);
EXPECT_GT(sakoe_chiba.first, 0.0);

// Test Itakura constrained DTW
auto itakura = dtw_itakura<MetricType::EUCLIDEAN, 2.0>(
simple_a_, simple_b_, strategy);
EXPECT_GT(itakura.first, 0.0);
}

// Test FastDTW convenience functions
TEST_F(DTWCoreTest, FastDTWConvenienceFunctions) {
// Test sequential FastDTW
auto seq_result = fastdtw_sequential<MetricType::EUCLIDEAN>(
        larger_a_, larger_b_, 2, 10);
EXPECT_GT(seq_result.first, 0.0);

// Test blocked FastDTW
auto blocked_result = fastdtw_blocked<MetricType::EUCLIDEAN>(
        larger_a_, larger_b_, 2, 10, 64);
EXPECT_GT(blocked_result.first, 0.0);

// Results should be similar
EXPECT_TRUE(results_equal(seq_result, blocked_result, 1e-4));
}

// Test that the unified interface works without specifying nullptr
TEST_F(DTWCoreTest, UnifiedInterfaceNoNullptr) {
SequentialStrategy strategy;

// This should compile and work without specifying nullptr
auto result1 = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::NONE>(
        simple_a_, simple_b_, strategy);
EXPECT_GT(result1.first, 0.0);

// With Sakoe-Chiba constraint - no nullptr needed
auto result2 = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::SAKOE_CHIBA, 2>(
        simple_a_, simple_b_, strategy);
EXPECT_GT(result2.first, 0.0);

// With Itakura constraint - no nullptr needed
auto result3 = dtw<MetricType::EUCLIDEAN, constraints::ConstraintType::ITAKURA, 1, 2.0>(
        simple_a_, simple_b_, strategy);
EXPECT_GT(result3.first, 0.0);
}

// Test recommendation function
TEST_F(DTWCoreTest, RecommendStrategy) {
std::string small_rec = recommend_strategy(50, 50);
std::string medium_rec = recommend_strategy(100, 100);
std::string large_rec = recommend_strategy(1000, 1000);

// Just check that recommendations are returned
EXPECT_FALSE(small_rec.empty());
EXPECT_FALSE(medium_rec.empty());
EXPECT_FALSE(large_rec.empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}