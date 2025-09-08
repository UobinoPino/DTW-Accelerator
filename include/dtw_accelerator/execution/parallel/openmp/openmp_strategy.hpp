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

        using WindowConstraint = std::vector<std::pair<int, int>>;


        // OpenMP parallel strategy
        class OpenMPStrategy : public BaseStrategy<OpenMPStrategy> {
        private:
            int num_threads_;
            int block_size_;

        public:
            explicit OpenMPStrategy(int num_threads = 0, int block_size = 64)
                    : num_threads_(num_threads), block_size_(block_size) {}

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
                    // For windowed DTW (FastDTW case), use diagonal-based parallelization
                    // This is much more for sparse windows

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

            void set_num_threads(int threads) { num_threads_ = threads; }
            int get_num_threads() const { return num_threads_; }
            void set_block_size(int block_size) { block_size_ = block_size; }
            int get_block_size() const { return block_size_; }

            std::string_view name() const { return "OpenMP"; }
            bool is_parallel() const { return true; }
        };

    } // namespace execution
} // namespace dtw_accelerator
#endif
