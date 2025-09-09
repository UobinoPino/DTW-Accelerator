/**
 * @file openmp_strategy.hpp
 * @brief OpenMP parallel DTW execution strategy
 * @author UobinoPino
 * @date 2024
 *
 * This file implements a parallel DTW algorithm using OpenMP
 * with wavefront parallelization pattern for multi-core CPUs.
 */

#ifndef DTWACCELERATOR_OPENMP_STRATEGY_HPP
#define DTWACCELERATOR_OPENMP_STRATEGY_HPP

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

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace dtw_accelerator {
    namespace execution {

        /// @brief Type alias for window constraints
        using WindowConstraint = std::vector<std::pair<int, int>>;


        /**
         * @brief OpenMP parallel strategy for multi-core execution
         *
         * This strategy uses OpenMP to parallelize DTW computation
         * across multiple CPU cores. It employs two different
         * parallelization patterns:
         *
         * 1. **Wavefront parallelization**: For standard DTW, processes
         *    blocks in wavefront order with parallel execution within
         *    each wave.
         *
         * 2. **Diagonal parallelization**: For windowed DTW (FastDTW),
         *    groups cells by diagonal and processes each diagonal in
         *    parallel.
         *
         */
        class OpenMPStrategy : public BaseStrategy<OpenMPStrategy> {
        private:
            /// @brief Number of OpenMP threads (0 for auto)
            int num_threads_;

            /// @brief Block size for wavefront processing
            int block_size_;

        public:
            /**
             * @brief Construct OpenMP strategy
             * @param num_threads Number of threads (0 for auto-detect)
             * @param block_size Block size for cache optimization
             */
            explicit OpenMPStrategy(int num_threads = 0, int block_size = 64)
                    : num_threads_(num_threads), block_size_(block_size) {}
            /**
             * @brief Execute parallel DTW with constraints
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
             * Uses OpenMP parallelization with dynamic scheduling
             * for load balancing across threads.
             */
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            void execute_with_constraint(DoubleMatrix& D,
                                         const DoubleTimeSeries& A,
                                         const DoubleTimeSeries& B,
                                         int n, int m, int dim,
                                         const WindowConstraint* window = nullptr) const {

#ifdef USE_OPENMP
                if (num_threads_ > 0) {
            omp_set_num_threads(num_threads_);
        }
#endif

                if (window != nullptr) {
                    // For windowed DTW, use diagonal-based parallelization
                    // This is more efficient for sparse windows

                    // Group window cells by their diagonal (i+j)
                    std::map<int, std::vector<std::pair<int, int>>> diagonals;
                    for (const auto& [i, j] : *window) {
                        diagonals[i + j].push_back({i, j});
                    }

                    // Process diagonals in order (ensures dependencies are met)
                    for (const auto& [diag_sum, cells] : diagonals) {
                        // Parallelize within each diagonal (no dependencies)
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
                        for (size_t idx = 0; idx < cells.size(); ++idx) {
                            int i = cells[idx].first;
                            int j = cells[idx].second;
                            int i_idx = i + 1;
                            int j_idx = j + 1;

                            D(i_idx, j_idx) = utils::compute_cell_cost<M>(
                                    A[i], B[j], dim,
                                    D(i_idx-1, j_idx-1),
                                    D(i_idx, j_idx-1),
                                    D(i_idx-1, j_idx)
                            );
                        }
                    }
                    return;
                }
                // Standard blocked wavefront parallelization
                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                // Process blocks in wavefront order with parallel execution
                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
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
             * @brief Set number of OpenMP threads
             * @param threads Number of threads (0 for auto)
             */
            void set_num_threads(int threads) { num_threads_ = threads; }

            /**
             * @brief Get current number of threads
             * @return Number of threads
             */
            int get_num_threads() const { return num_threads_; }

            /**
             * @brief Set block size
             * @param block_size New block size
             */
            void set_block_size(int block_size) { block_size_ = block_size; }

            /**
             * @brief Get current block size
             * @return Block size
             */
            int get_block_size() const { return block_size_; }

            /**
             * @brief Get strategy name
             * @return "OpenMP"
             */
            std::string_view name() const { return "OpenMP"; }

            /**
             * @brief Check if strategy is parallel
             * @return True (parallel execution)
             */
            bool is_parallel() const { return true; }
        };

    } // namespace execution
} // namespace dtw_accelerator
#endif
