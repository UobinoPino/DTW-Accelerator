/**
 * @file dtw_concepts.hpp
 * @brief C++20 concepts for DTW execution strategies
 * @author UobinoPino
 * @date 2024
 *
 * This file defines C++20 concepts that establish the interface
 * requirements for DTW execution strategies, enabling compile-time
 * polymorphism and type safety.
 */

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

        /// @brief Type alias for window constraint representation
        using WindowConstraint = std::vector<std::pair<int, int>>;

        /**
         * @brief Core concept for all DTW execution strategies
         * @tparam Strategy The strategy type to check
         *
         * This concept defines the required interface that all execution
         * strategies must implement. It ensures type safety and enables
         * compile-time polymorphism for different execution backends.
         */
        template<typename Strategy>
        concept ExecutionStrategy = requires(
                Strategy strategy,
        DoubleMatrix& D,
        const DoubleTimeSeries& A,
        const DoubleTimeSeries& B,
        int n, int m, int dim,
        const WindowConstraint* window)
        {

        /**
         * @brief Initialize the DTW cost matrix
         */
        { strategy.initialize_matrix(D, n, m) } -> std::same_as<void>;

        /**
         * @brief Execute DTW computation with constraints
         */
        { strategy.template execute_with_constraint<
                constraints::ConstraintType::NONE, 1, 2.0,
                distance::MetricType::EUCLIDEAN>(
                D, A, B, n, m, dim, window) } -> std::same_as<void>;

        /**
         * @brief Extract final result and optimal path
         */
        { strategy.extract_result(D) } ->
        std::convertible_to<std::pair<double, std::vector<std::pair<int, int>>>>;

        /**
         * @brief Get strategy name for identification
         */
        { strategy.name() } -> std::convertible_to<std::string_view>;

        /**
         * @brief Check if strategy uses parallel execution
         */
        { strategy.is_parallel() } -> std::convertible_to<bool>;
        };


        /**
         * @brief Type trait to check if a strategy supports a specific distance metric
         * @tparam Strategy The strategy type
         * @tparam M The metric type
         */
        template<typename Strategy, distance::MetricType M>
        struct supports_metric : std::false_type {};

        /**
         * @brief Specialization for strategies that support the metric
         */
        template<typename Strategy, distance::MetricType M>
        requires requires(Strategy s, DoubleMatrix& D,
                          const DoubleTimeSeries& A,
                          const DoubleTimeSeries& B,
                          int n, int m, int dim,
                          const WindowConstraint* window) {
            { s.template execute_with_constraint<
                        constraints::ConstraintType::NONE, 1, 2.0, M>(
                        D, A, B, n, m, dim, window) } -> std::same_as<void>;
        }
        struct supports_metric<Strategy, M> : std::true_type {};

        /**
         * @brief Helper variable template for supports_metric
         */
        template<typename Strategy, distance::MetricType M>
        inline constexpr bool supports_metric_v = supports_metric<Strategy, M>::value;

    } // namespace concepts
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_DTW_CONCEPTS_HPP