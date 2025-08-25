#ifndef DTW_ACCELERATOR_DTW_CONCEPTS_HPP
#define DTW_ACCELERATOR_DTW_CONCEPTS_HPP

#include <concepts>
#include <vector>
#include <utility>
#include <type_traits>
#include <string_view>
#include "dtw_accelerator/core/distance_metrics.hpp"
#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/matrix.hpp"

namespace dtw_accelerator {
    namespace concepts {

// Concept for types that can be used as DTW matrices
        template<typename T>
        concept DTWMatrix = requires(T matrix, size_t i, size_t j, double value) {
    { matrix.resize(i, j, value) } -> std::same_as<void>;
{ matrix(i, j) } -> std::convertible_to<double>;
{ matrix(i, j) = value } -> std::same_as<double&>;
{ matrix.rows() } -> std::convertible_to<size_t>;
{ matrix.cols() } -> std::convertible_to<size_t>;
};

// Base concept for all execution strategies
template<typename Strategy>
concept ExecutionStrategy = requires(
        Strategy strategy,
DoubleMatrix& D,
const DoubleTimeSeries& A,
const DoubleTimeSeries& B,
int n, int m, int dim)
{
// Required: Initialize the DTW matrix
{ strategy.initialize_matrix(D, n, m) } -> std::same_as<void>;

// Required: Execute DTW computation
{ strategy.template execute<distance::MetricType::EUCLIDEAN>(
        D, A, B, n, m, dim) } -> std::same_as<void>;

// Required: Extract result and path
{ strategy.extract_result(D) } ->
std::convertible_to<std::pair<double, std::vector<std::pair<int, int>>>>;

// Required: Strategy identification
{ strategy.name() } -> std::convertible_to<std::string_view>;
{ strategy.is_parallel() } -> std::convertible_to<bool>;
};

// Concept for strategies that support constrained DTW
template<typename Strategy>
concept ConstrainedExecutionStrategy = ExecutionStrategy<Strategy> &&
                                       requires(
                                               Strategy strategy,
DoubleMatrix& D,
const DoubleTimeSeries& A,
const DoubleTimeSeries& B,
const std::vector<std::pair<int, int>>& window,
int n, int m, int dim)
{
// Required: Execute constrained DTW
{ strategy.template execute_constrained<distance::MetricType::EUCLIDEAN>(
        D, A, B, window, n, m, dim) } -> std::same_as<void>;
};

// Concept for strategies that support constraint templates
template<typename Strategy>
concept TemplatedConstraintStrategy = ExecutionStrategy<Strategy> &&
                                      requires(
                                              Strategy strategy,
DoubleMatrix& D,
const DoubleTimeSeries& A,
const DoubleTimeSeries& B,
int n, int m, int dim)
{
// Support Sakoe-Chiba and Itakura constraints
{ strategy.template execute_with_constraint<
        constraints::ConstraintType::SAKOE_CHIBA, 1, 2.0,
        distance::MetricType::EUCLIDEAN>(D, A, B, n, m, dim) } -> std::same_as<void>;
};

// Concept for parallel strategies with configurable resources
template<typename Strategy>
concept ConfigurableParallelStrategy = ExecutionStrategy<Strategy> &&
                                       requires(Strategy strategy, int value)
{
{ strategy.set_num_threads(value) } -> std::same_as<void>;
{ strategy.get_num_threads() } -> std::convertible_to<int>;
};

// Concept for block-based strategies
template<typename Strategy>
concept BlockBasedStrategy = ExecutionStrategy<Strategy> &&
                             requires(Strategy strategy, int block_size)
{
{ strategy.set_block_size(block_size) } -> std::same_as<void>;
{ strategy.get_block_size() } -> std::convertible_to<int>;
};

// Type trait to check if a strategy supports a specific distance metric
template<typename Strategy, distance::MetricType M>
struct supports_metric : std::false_type {};

template<typename Strategy, distance::MetricType M>
requires requires(Strategy s, DoubleMatrix& D,
                  const DoubleTimeSeries& A,
                  const DoubleTimeSeries& B,
                  int n, int m, int dim) {
    { s.template execute<M>(D, A, B, n, m, dim) } -> std::same_as<void>;
}
struct supports_metric<Strategy, M> : std::true_type {};

template<typename Strategy, distance::MetricType M>
inline constexpr bool supports_metric_v = supports_metric<Strategy, M>::value;

} // namespace concepts
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_DTW_CONCEPTS_HPP