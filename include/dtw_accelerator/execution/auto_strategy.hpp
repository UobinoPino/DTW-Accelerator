/**
 * @file auto_strategy.hpp
 * @brief Automatic strategy selection based on problem characteristics
 * @author UobinoPino
 * @date 2024
 *
 * This file implements an automatic strategy selector that chooses
 * the optimal execution backend based on problem size and available
 * hardware resources.
 */

#ifndef DTWACCELERATOR_AUTO_STRATEGY_HPP
#define DTWACCELERATOR_AUTO_STRATEGY_HPP

#include "dtw_accelerator/core/dtw_concepts.hpp"
#include "dtw_accelerator/core/distance_metrics.hpp"
#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/dtw_utils.hpp"
#include "dtw_accelerator/core/matrix.hpp"
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <string_view>
#include <memory>
#include "dtw_accelerator/execution/base_strategy.hpp"
#include "dtw_accelerator/execution/sequential/standard_strategy.hpp"
#include "dtw_accelerator/execution/sequential/blocked_strategy.hpp"
#include "dtw_accelerator/execution/parallel/openmp/openmp_strategy.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace dtw_accelerator {
    namespace execution {


        /**
        * @brief Auto-selecting strategy based on problem size and available backends
        *
        * This strategy automatically selects the most appropriate execution
        * backend based on the problem dimensions and available hardware.
        * The selection heuristics are:
        * - n,m >= 100: Use OpenMP if available
        * - n,m >= 50: Use blocked strategy for cache optimization
        * - Otherwise: Use simple sequential strategy
        */
        class AutoStrategy : public BaseStrategy<AutoStrategy> {
        private:
            /// @brief Cached strategy instance
            mutable std::unique_ptr<SequentialStrategy> strategy_;

            /// @brief Problem dimensions for strategy selection
            int n_, m_;

            /**
             * @brief Select appropriate strategy based on problem size
             *
             * This method implements the heuristics for automatic
             * strategy selection. Called lazily on first use.
             */
            void select_strategy() const {
                if (strategy_) return;

#ifdef USE_OPENMP
                if (n_ >= 100 && m_ >= 100) {
            strategy_ = std::make_unique<SequentialStrategy>();
            return;
        }
#endif

                if (n_ >= 50 && m_ >= 50) {
                    // For medium problems, use blocked strategy
                    strategy_ = std::make_unique<SequentialStrategy>();
                    return;
                }

                // For small problems, use sequential
                strategy_ = std::make_unique<SequentialStrategy>();
            }

        public:

            /**
             * @brief Construct auto-strategy with problem dimensions
             * @param n First series length (optional)
             * @param m Second series length (optional)
             */
            explicit AutoStrategy(int n = 0, int m = 0) : n_(n), m_(m) {}

            /**
             * @brief Initialize DTW matrix
             * @param D Matrix to initialize
             * @param n Number of rows
             * @param m Number of columns
             */
            void initialize_matrix(DoubleMatrix& D, int n, int m) const {
                select_strategy();
                strategy_->initialize_matrix(D, n, m);
            }

            /**
             * @brief Execute DTW computation
             * @tparam M Distance metric type
             * @param D Cost matrix
             * @param A First time series
             * @param B Second time series
             * @param n Series A length
             * @param m Series B length
             * @param dim Dimensions per point
             */
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            void execute_with_constraint(DoubleMatrix& D,
                                         const DoubleTimeSeries& A,
                                         const DoubleTimeSeries& B,
                                         int n, int m, int dim,
                                         const WindowConstraint* window = nullptr) const {
                select_strategy();
                strategy_->template execute_with_constraint<CT, R, S, M>(
                        D, A, B, n, m, dim, window);
            }

            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result(const DoubleMatrix& D) const {
                return strategy_->extract_result(D);
            }

            std::string_view name() const {
                select_strategy();
                return strategy_->name();
            }

            bool is_parallel() const {
                select_strategy();
                return strategy_->is_parallel();
            }
        };

    } // namespace execution
} // namespace dtw_accelerator

#endif //DTWACCELERATOR_AUTO_STRATEGY_HPP