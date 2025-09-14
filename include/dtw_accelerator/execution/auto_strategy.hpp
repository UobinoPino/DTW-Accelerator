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
#include <variant>
#include "dtw_accelerator/execution/base_strategy.hpp"
#include "dtw_accelerator/execution/sequential/standard_strategy.hpp"
#include "dtw_accelerator/execution/sequential/blocked_strategy.hpp"

#ifdef USE_OPENMP
#include "dtw_accelerator/execution/parallel/openmp/openmp_strategy.hpp"
#endif

#ifdef USE_MPI
#include "dtw_accelerator/execution/parallel/mpi/mpi_strategy.hpp"
#endif

#ifdef USE_CUDA
#include "dtw_accelerator/execution/parallel/cuda/execution/cuda_strategy.hpp"
#endif

namespace dtw_accelerator {
    namespace execution {

        /**
         * @brief Auto-selecting strategy based on problem size and available backends
         *
         * This strategy automatically selects the most appropriate execution
         * backend based on the problem dimensions and available hardware.
         * The selection heuristics are:
         * - n,m >= 1000 and CUDA available: Use CUDA
         * - n,m >= 100 and OpenMP available: Use OpenMP
         * - n,m >= 50: Use blocked strategy for cache optimization
         * - Otherwise: Use simple sequential strategy
         */
        class AutoStrategy : public BaseStrategy<AutoStrategy> {
        private:
            /// @brief Problem dimensions for strategy selection
            int n_, m_;

            /// @brief Selected strategy name for reporting
            mutable std::string_view selected_name_;

            /// @brief Flag for parallel execution
            mutable bool is_parallel_;

            /// @brief Strategy variant to hold any strategy type
            using StrategyVariant = std::variant<
                    SequentialStrategy,
                    BlockedStrategy
#ifdef USE_OPENMP
                    , OpenMPStrategy
#endif
#ifdef USE_MPI
                    , MPIStrategy
#endif
#ifdef USE_CUDA
                    , parallel::cuda::CUDAStrategy
#endif
            >;

            /// @brief The actual selected strategy
            mutable std::unique_ptr<StrategyVariant> strategy_;

            /**
             * @brief Select appropriate strategy based on problem size
             *
             * This method implements the heuristics for automatic
             * strategy selection. Called lazily on first use.
             */
            void select_strategy() const {
                if (strategy_) return;  // Already selected

#ifdef USE_CUDA
                // Check CUDA availability for large problems
                if (n_ >= 1000 && m_ >= 1000) {
                    try {
                        if (parallel::cuda::CUDAStrategy::is_available()) {
                            strategy_ = std::make_unique<StrategyVariant>(
                                parallel::cuda::CUDAStrategy(256)  // Default tile size
                            );
                            selected_name_ = "CUDA-Auto";
                            is_parallel_ = true;
                            return;
                        }
                    } catch (...) {
                        // CUDA initialization failed, fall through to next option
                    }
                }
#endif

#ifdef USE_MPI
                // Check for MPI - typically not auto-selected unless explicitly in MPI context
                // MPI requires special initialization, so we generally don't auto-select it
                // unless we detect we're already in an MPI environment
                int flag = 0;
                MPI_Initialized(&flag);
                if (flag && n_ >= 500 && m_ >= 500) {
                    int size;
                    MPI_Comm_size(MPI_COMM_WORLD, &size);
                    if (size > 1) {  // Only use MPI if we have multiple processes
                        strategy_ = std::make_unique<StrategyVariant>(
                            MPIStrategy(64, 0, MPI_COMM_WORLD)
                        );
                        selected_name_ = "MPI-Auto";
                        is_parallel_ = true;
                        return;
                    }
                }
#endif

#ifdef USE_OPENMP
                // Use OpenMP for medium to large problems
                if (n_ >= 100 && m_ >= 100) {
                    int num_threads = omp_get_max_threads();
                    if (num_threads > 1) {  // Only use if we have multiple threads available
                        strategy_ = std::make_unique<StrategyVariant>(
                            OpenMPStrategy(0, 64)  // 0 means auto-detect threads
                        );
                        selected_name_ = "OpenMP-Auto";
                        is_parallel_ = true;
                        return;
                    }
                }
#endif

                // For medium problems, use blocked strategy for cache optimization
                if (n_ >= 50 && m_ >= 50) {
                    strategy_ = std::make_unique<StrategyVariant>(
                            BlockedStrategy(64)
                    );
                    selected_name_ = "Blocked-Auto";
                    is_parallel_ = false;
                    return;
                }

                // Default to sequential for small problems
                strategy_ = std::make_unique<StrategyVariant>(SequentialStrategy{});
                selected_name_ = "Sequential-Auto";
                is_parallel_ = false;
            }

        public:
            /**
             * @brief Construct auto-strategy with problem dimensions
             * @param n First series length
             * @param m Second series length
             */
            explicit AutoStrategy(int n = 0, int m = 0)
                    : n_(n), m_(m), selected_name_("Unknown"), is_parallel_(false) {}

            /**
             * @brief Initialize DTW matrix
             * @param D Matrix to initialize
             * @param n Number of rows
             * @param m Number of columns
             */
            void initialize_matrix(DoubleMatrix& D, int n, int m) const {
                // Update dimensions if they were not set in constructor


                select_strategy();

                std::visit([&D, n, m](auto& strategy) {
                    strategy.initialize_matrix(D, n, m);
                }, *strategy_);
            }

            /**
             * @brief Execute DTW computation with constraints
             * @tparam CT Constraint type
             * @tparam R Sakoe-Chiba radius
             * @tparam S Itakura slope
             * @tparam M Distance metric type
             * @param D Cost matrix
             * @param A First time series
             * @param B Second time series
             * @param n Series A length
             * @param m Series B length
             * @param dim Dimensions per point
             * @param window Optional window constraint
             */
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            void execute_with_constraint(DoubleMatrix& D,
                                         const DoubleTimeSeries& A,
                                         const DoubleTimeSeries& B,
                                         int n, int m, int dim,
                                         const WindowConstraint* window = nullptr) const {
                // Update dimensions if needed


                select_strategy();

                std::visit([&](auto& strategy) {
                    strategy.template execute_with_constraint<CT, R, S, M>(
                            D, A, B, n, m, dim, window);
                }, *strategy_);
            }

            /**
             * @brief Extract result from the cost matrix
             * @param D Computed cost matrix
             * @return Pair of (distance, path)
             */
            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result(const DoubleMatrix& D) const {
                select_strategy();

                return std::visit([&D](auto& strategy) {
                    return strategy.extract_result(D);
                }, *strategy_);
            }

            /**
             * @brief Get the name of the selected strategy
             * @return Strategy name
             */
            std::string_view name() const {
                select_strategy();
                return selected_name_;
            }

            /**
             * @brief Check if selected strategy uses parallel execution
             * @return True if parallel
             */
            bool is_parallel() const {
                select_strategy();
                return is_parallel_;
            }
        };

    } // namespace execution
} // namespace dtw_accelerator

#endif //DTWACCELERATOR_AUTO_STRATEGY_HPP