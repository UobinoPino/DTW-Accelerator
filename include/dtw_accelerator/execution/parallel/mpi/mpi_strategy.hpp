/**
 * @file mpi_strategy.hpp
 * @brief MPI distributed DTW execution strategy
 * @author UobinoPino
 * @date 2024
 *
 * This file implements a distributed DTW algorithm using MPI
 * for execution across multiple nodes in a cluster.
 */

#ifndef DTWACCELERATOR_MPI_STRATEGY_HPP
#define DTWACCELERATOR_MPI_STRATEGY_HPP

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
#include "dtw_accelerator/execution/sequential/blocked_strategy.hpp"
#include "dtw_accelerator/execution/parallel/openmp/openmp_strategy.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <mpi.h>

namespace dtw_accelerator {
    namespace execution {

        /// @brief Type alias for window constraints
        using WindowConstraint = std::vector<std::pair<int, int>>;

        /**
        * @brief MPI distributed strategy for cluster execution
        *
        * This strategy distributes DTW computation across multiple
        * MPI processes, potentially on different nodes. It uses:
        *
        * - **Block distribution**: The DTW matrix is divided into blocks
        *   that are distributed across processes
        * - **Wavefront synchronization**: Processes synchronize after
        *   each wave to exchange boundary values
        * - **Hybrid parallelism**: Can use OpenMP within each MPI process
        *   for multi-level parallelism
        *
        * The strategy is most effective for very large DTW problems
        * that don't fit in a single node's memory or require more
        * computational power than a single node can provide.
        */
        class MPIStrategy : public BaseStrategy<MPIStrategy> {
        private:
            /// @brief Block size for distribution
            int block_size_;

            /// @brief OpenMP threads per MPI process
            int threads_per_process_;

            /// @brief MPI communicator
            MPI_Comm communicator_;

            /// @brief Cached MPI rank
            mutable int cached_rank_ = -1;

            /// @brief Cached MPI size
            mutable int cached_size_ = -1;

            /**
             * @brief Broadcast block boundaries between processes
             * @param D Cost matrix
             * @param bi Block row index
             * @param bj Block column index
             * @param n Total rows
             * @param m Total columns
             * @param owner Process that owns this block
             * @param row_buffer Buffer for row communication
             * @param col_buffer Buffer for column communication
             *
             * Exchanges boundary values between processes to maintain
             * data dependencies across block boundaries.
             */
            void broadcast_block_boundaries(DoubleMatrix& D,
                                            int bi, int bj, int n, int m, int owner,
                                            std::vector<double>& row_buffer,
                                            std::vector<double>& col_buffer) const {

                int rank = cached_rank_;
                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                // Broadcast last row
                if (bi < n_blocks - 1) {
                    int i_boundary = std::min((bi + 1) * block_size_, n);
                    int j_start = bj * block_size_;
                    int j_end = std::min((bj + 1) * block_size_, m);

                    if (rank == owner) {
                        for (int j = j_start; j <= j_end; ++j) {
                            row_buffer[j - j_start] = D(i_boundary, j);
                        }
                    }

                    MPI_Bcast(row_buffer.data(), j_end - j_start + 1,
                              MPI_DOUBLE, owner, communicator_);

                    if (rank != owner) {
                        for (int j = j_start; j <= j_end; ++j) {
                            D(i_boundary, j) = row_buffer[j - j_start];
                        }
                    }
                }

                // Broadcast last column
                if (bj < m_blocks - 1) {
                    int j_boundary = std::min((bj + 1) * block_size_, m);
                    int i_start = bi * block_size_;
                    int i_end = std::min((bi + 1) * block_size_, n);

                    if (rank == owner) {
                        for (int i = i_start; i <= i_end; ++i) {
                            col_buffer[i - i_start] = D(i, j_boundary);
                        }
                    }

                    MPI_Bcast(col_buffer.data(), i_end - i_start + 1,
                              MPI_DOUBLE, owner, communicator_);

                    if (rank != owner) {
                        for (int i = i_start; i <= i_end; ++i) {
                            D(i, j_boundary) = col_buffer[i - i_start];
                        }
                    }
                }
            }

        public:
            /**
             * @brief Construct MPI strategy
             * @param block_size Block size for distribution
             * @param threads OpenMP threads per process (0 for none)
             * @param comm MPI communicator
             */
            explicit MPIStrategy(int block_size = 64, int threads = 0, MPI_Comm comm = MPI_COMM_WORLD)
                    : block_size_(block_size), threads_per_process_(threads), communicator_(comm) {
                MPI_Comm_rank(communicator_, &cached_rank_);
                MPI_Comm_size(communicator_, &cached_size_);
            }

            /**
             * @brief Execute distributed DTW with constraints
             * @tparam CT Constraint type
             * @tparam R Sakoe-Chiba band radius
             * @tparam S Itakura parallelogram slope
             * @tparam M Distance metric type
             * @param D Cost matrix (complete on all processes)
             * @param A First time series
             * @param B Second time series
             * @param n Length of series A
             * @param m Length of series B
             * @param dim Number of dimensions per point
             * @param window Optional window constraint
             *
             * Distributes blocks across MPI processes and coordinates
             * execution with boundary exchanges.
             */
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            void execute_with_constraint(DoubleMatrix& D,
                                         const DoubleTimeSeries& A,
                                         const DoubleTimeSeries& B,
                                         int n, int m, int dim,
                                         const WindowConstraint* window = nullptr) const {

                int rank = cached_rank_;
                int size = cached_size_;

                if (n <= 0 || m <= 0 || dim <= 0) {
                    return;
                }

                // Single process fallback
                if (size == 1) {
#ifdef USE_OPENMP
                    if (threads_per_process_ > 0) {
                OpenMPStrategy omp_strat(threads_per_process_, block_size_);
                omp_strat.template execute_with_constraint<CT, R, S, M>(
                    D, A, B, n, m, dim, window);
            } else {
                BlockedStrategy blocked(block_size_);
                blocked.template execute_with_constraint<CT, R, S, M>(
                    D, A, B, n, m, dim, window);
            }
#else
                    BlockedStrategy blocked(block_size_);
                    blocked.template execute_with_constraint<CT, R, S, M>(
                            D, A, B, n, m, dim, window);
#endif
                    return;
                }

#ifdef USE_OPENMP
                if (threads_per_process_ > 0) {
            omp_set_num_threads(threads_per_process_);
        }
#endif

                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                // Pre-allocate communication buffers
                const int buffer_size = std::max(m + 1, n + 1) + 100;
                std::vector<double> row_buffer(buffer_size, 0.0);
                std::vector<double> col_buffer(buffer_size, 0.0);

                // Process blocks in wavefront order
                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

                    // Distribute blocks within the wave across processes
                    std::vector<std::pair<int, int>> my_blocks;
                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            // Round-robin distribution of blocks
                            int block_idx = bi - start_bi;
                            if (block_idx % size == rank) {
                                my_blocks.push_back({bi, bj});
                            }
                        }
                    }

                    // Process assigned blocks with OpenMP parallelization
                    if (!my_blocks.empty()) {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) if(threads_per_process_ > 0)
#endif
                        for (size_t idx = 0; idx < my_blocks.size(); ++idx) {
                            int bi = my_blocks[idx].first;
                            int bj = my_blocks[idx].second;
                            this->process_block_with_constraint<CT, R, S, M>(
                                    D, A, B, bi, bj, n, m, dim, block_size_, window);
                        }
                    }

                    // Exchange boundaries using MPI_Bcast
                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            int block_owner = (bi - start_bi) % size;
                            broadcast_block_boundaries(D, bi, bj, n, m, block_owner,
                                                       row_buffer, col_buffer);
                        }
                    }
                }
            }

            /**
            * @brief Extract result (only valid on rank 0)
            * @param D Computed cost matrix
            * @return Pair of (total cost, optimal path)
            *
            * Only rank 0 returns the valid result; other ranks
            * return empty results.
            */
            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result(const DoubleMatrix& D) const {
                int rank = cached_rank_;

                if (rank != 0) {
                    return {0.0, {}};  // Empty path, cost will be ignored
                }
                return extract_result_impl(D);
            }

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
             * @return "MPI"
             */
            std::string_view name() const { return "MPI"; }

            /**
             * @brief Check if strategy is parallel
             * @return True (distributed parallel execution)
             */
            bool is_parallel() const { return true; }
        };

    } // namespace execution
} // namespace dtw_accelerator

#endif //DTWACCELERATOR_MPI_STRATEGY_HPP