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

        // Blocked execution strategy for better cache locality
        class BlockedStrategy : public BaseStrategy<BlockedStrategy> {
        private:
            int block_size_;

        public:
            explicit BlockedStrategy(int block_size = 64) : block_size_(block_size) {}

            void initialize_matrix(DoubleMatrix& D, int n, int m) const {
                initialize_matrix_impl(D, n, m);
            }

            template<distance::MetricType M>
            void execute(DoubleMatrix& D,
                         const std::vector<std::vector<double>>& A,
                         const std::vector<std::vector<double>>& B,
                         int n, int m, int dim) const {

                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                // Process blocks in wavefront order for correct dependencies
                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            process_block<M>(D, A, B, bi, bj, n, m, dim);
                        }
                    }
                }
            }

            template<distance::MetricType M>
            void execute_constrained(DoubleMatrix& D,
                                     const std::vector<std::vector<double>>& A,
                                     const std::vector<std::vector<double>>& B,
                                     const std::vector<std::pair<int, int>>& window,
                                     int n, int m, int dim) const {

                auto in_window = utils::create_window_mask(window, n, m);
                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            process_block_constrained<M>(D, A, B, bi, bj, n, m, dim, in_window);
                        }
                    }
                }
            }

            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            void execute_with_constraint(DoubleMatrix& D,
                                         const std::vector<std::vector<double>>& A,
                                         const std::vector<std::vector<double>>& B,
                                         int n, int m, int dim) const {

                auto constraint_mask = utils::generate_constraint_mask<CT, R, S>(n, m);
                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            process_block_constrained<M>(D, A, B, bi, bj, n, m, dim, constraint_mask);
                        }
                    }
                }
            }

            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result(const DoubleMatrix& D) const {
                return extract_result_impl(D);
            }

            void set_block_size(int block_size) { block_size_ = block_size; }
            int get_block_size() const { return block_size_; }

            std::string_view name() const { return "Blocked"; }
            bool is_parallel() const { return false; }

        private:
            template<distance::MetricType M>
            void process_block(DoubleMatrix& D,
                               const std::vector<std::vector<double>>& A,
                               const std::vector<std::vector<double>>& B,
                               int bi, int bj, int n, int m, int dim) const {

                int i_start = bi * block_size_ + 1;
                int i_end = std::min((bi + 1) * block_size_, n);
                int j_start = bj * block_size_ + 1;
                int j_end = std::min((bj + 1) * block_size_, m);

                for (int i = i_start; i <= i_end; ++i) {
                    for (int j = j_start; j <= j_end; ++j) {
                        D(i, j) = utils::compute_cell_cost<M>(
                                A[i-1].data(), B[j-1].data(), dim,
                                D(i-1, j-1), D(i, j-1), D(i-1, j)
                        );
                    }
                }
            }

            template<distance::MetricType M>
            void process_block_constrained(DoubleMatrix& D,
                                           const std::vector<std::vector<double>>& A,
                                           const std::vector<std::vector<double>>& B,
                                           int bi, int bj, int n, int m, int dim,
                                           const BoolMatrix& mask) const {

                int i_start = bi * block_size_ + 1;
                int i_end = std::min((bi + 1) * block_size_, n);
                int j_start = bj * block_size_ + 1;
                int j_end = std::min((bj + 1) * block_size_, m);

                for (int i = i_start; i <= i_end; ++i) {
                    for (int j = j_start; j <= j_end; ++j) {
                        if (i-1 < n && j-1 < m && mask(i-1, j-1)) {
                            D(i, j) = utils::compute_cell_cost<M>(
                                    A[i-1].data(), B[j-1].data(), dim,
                                    D(i-1, j-1), D(i, j-1), D(i-1, j)
                            );
                        }
                    }
                }
            }
        };



    }
}

#endif //DTWACCELERATOR_BLOCKED_STRATEGY_HPP