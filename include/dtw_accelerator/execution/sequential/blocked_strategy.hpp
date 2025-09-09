/**
 * @file blocked_strategy.hpp
 * @brief Cache-optimized blocked DTW execution strategy
 * @author UobinoPino
 * @date 2024
 *
 * This file implements a cache-aware blocked DTW algorithm that
 * processes the cost matrix in cache-friendly blocks to improve
 * memory access patterns and performance.
 */

#ifndef DTWACCELERATOR_BLOCKED_STRATEGY_HPP
#define DTWACCELERATOR_BLOCKED_STRATEGY_HPP

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

namespace dtw_accelerator {
    namespace execution {

        /// @brief Type alias for window constraints
        using WindowConstraint = std::vector<std::pair<int, int>>;

        /**
         * @brief Blocked execution strategy for better cache locality
         *
         * This strategy divides the DTW matrix into square blocks and
         * processes them in wavefront order to maintain dependencies
         * while improving cache utilization. The block size should be
         * tuned based on the cache size of the target processor.
         *
         * Benefits:
         * - Better temporal locality within blocks
         * - Reduced cache misses for large matrices
         * - Improved memory bandwidth utilization
         */
        class BlockedStrategy : public BaseStrategy<BlockedStrategy> {
        private:
            /// @brief Size of cache-friendly blocks
            int block_size_;

        public:
            /**
             * @brief Construct blocked strategy with specified block size
             * @param block_size Size of cache blocks (default: 64)
             */
            explicit BlockedStrategy(int block_size = 64) : block_size_(block_size) {}

            /**
             * @brief Execute DTW with blocking and constraints
             * @tparam CT Constraint type
             * @tparam R Sakoe-Chiba band radius
             * @tparam S Itakura parallelogram slope
             * @tparam M Distance metric type
             * @param D Cost matrix to fill
             * @param A First time series
             * @param B Second time series
             * @param n Length of series A
             * @param m Length of series B
             * @param dim Number of dimensions per point
             * @param window Optional window constraint
             *
             * Processes the DTW matrix in cache-friendly blocks using
             * wavefront parallelism pattern to maintain dependencies.
             */
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            void execute_with_constraint(DoubleMatrix& D,
                                         const DoubleTimeSeries& A,
                                         const DoubleTimeSeries& B,
                                         int n, int m, int dim,
                                         const WindowConstraint* window = nullptr) const {



                if (window != nullptr) {
                    // For sparse windows (FastDTW), don't use blocking
                    // Direct iteration is more efficient
                    for (const auto& [i, j] : *window) {
                        int i_idx = i + 1;
                        int j_idx = j + 1;
                        D(i_idx, j_idx) = utils::compute_cell_cost<M>(
                                A[i], B[j], dim,
                                D(i_idx-1, j_idx-1),
                                D(i_idx, j_idx-1),
                                D(i_idx-1, j_idx)
                        );
                    }
                    return;
                }

                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                // Process blocks in wavefront order
                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            this->process_block_with_constraint<CT, R, S, M>(
                                    D, A, B, bi, bj, n, m, dim, block_size_, window);
                        }
                    }
                }
            }

            /**
             * @brief Set the block size
             * @param block_size New block size
             */
            void set_block_size(int block_size) { block_size_ = block_size; }

            /**
             * @brief Get the current block size
             * @return Current block size
             */
            int get_block_size() const { return block_size_; }

            /**
             * @brief Get strategy name
             * @return "Blocked"
             */
            std::string_view name() const { return "Blocked"; }

            /**
             * @brief Check if strategy is parallel
             * @return False (sequential blocked execution)
             */
            bool is_parallel() const { return false; }
        };

    } // namespace execution
} // namespace dtw_accelerator
#endif