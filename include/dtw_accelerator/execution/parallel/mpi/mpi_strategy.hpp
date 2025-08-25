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

//#ifdef USE_MPI
#include <mpi.h>
//#endif


namespace dtw_accelerator {
    namespace execution {

        class MPIStrategy : public BaseStrategy<MPIStrategy> {
        private:
            int block_size_;
            int threads_per_process_;
            MPI_Comm communicator_;

            // Cache for avoiding repeated MPI calls
            mutable int cached_rank_ = -1;
            mutable int cached_size_ = -1;

        public:
            explicit MPIStrategy(int block_size = 64, int threads = 0, MPI_Comm comm = MPI_COMM_WORLD)
                    : block_size_(block_size), threads_per_process_(threads), communicator_(comm) {
                MPI_Comm_rank(communicator_, &cached_rank_);
                MPI_Comm_size(communicator_, &cached_size_);
            }

            void initialize_matrix(DoubleMatrix& D, int n, int m) const {
                initialize_matrix_impl(D, n, m);
            }

            template<distance::MetricType M>
            void execute(DoubleMatrix& D,
                         const DoubleTimeSeries& A,
                         const DoubleTimeSeries& B,
                         int n, int m, int dim) const {

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
                        omp_strat.template execute<M>(D, A, B, n, m, dim);
                    } else {
                        BlockedStrategy blocked(block_size_);
                        blocked.template execute<M>(D, A, B, n, m, dim);
                    }
#else
                    BlockedStrategy blocked(block_size_);
                    blocked.template execute<M>(D, A, B, n, m, dim);
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
#pragma omp parallel for schedule(dynamic, 1)
#endif
                        for (size_t idx = 0; idx < my_blocks.size(); ++idx) {
                            int bi = my_blocks[idx].first;
                            int bj = my_blocks[idx].second;
                            process_block<M>(D, A, B, bi, bj, n, m, dim);
                        }
                    }

                    // Efficient synchronization - share boundaries needed for next wave
                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            int block_owner = (bi - start_bi) % size;

                            // Share last row of block (if not last block row)
                            if (bi < n_blocks - 1) {
                                int i_boundary = std::min((bi + 1) * block_size_, n);
                                int j_start = bj * block_size_;
                                int j_end = std::min((bj + 1) * block_size_ -1, m - 1);

                                if (rank == block_owner) {
                                    // Copy row data from Matrix to buffer
                                    for (int j = j_start; j <= j_end; ++j) {
                                        row_buffer[j - j_start] = D(i_boundary, j);
                                    }
                                }

                                MPI_Bcast(row_buffer.data(), j_end - j_start + 1,
                                          MPI_DOUBLE, block_owner, communicator_);

                                if (rank != block_owner) {
                                    // Copy buffer data back to Matrix
                                    for (int j = j_start; j <= j_end; ++j) {
                                        D(i_boundary, j) = row_buffer[j - j_start];
                                    }
                                }
                            }

                            // Share last column of block (if not last block column)
                            if (bj < m_blocks - 1) {
                                int j_boundary = std::min((bj + 1) * block_size_, m);
                                int i_start = bi * block_size_;
                                int i_end = std::min((bi + 1) * block_size_ - 1, n - 1);

                                if (rank == block_owner) {
                                    for (int i = i_start; i <= i_end; ++i) {
                                        col_buffer[i - i_start] = D(i, j_boundary);
                                    }
                                }

                                MPI_Bcast(col_buffer.data(), i_end - i_start + 1,
                                          MPI_DOUBLE, block_owner, communicator_);

                                if (rank != block_owner) {
                                    for (int i = i_start; i <= i_end; ++i) {
                                        D(i, j_boundary) = col_buffer[i - i_start];
                                    }
                                }
                            }
                        }
                    }

                }
            }

            template<distance::MetricType M>
            void execute_constrained(DoubleMatrix& D,
                                     const DoubleTimeSeries& A,
                                     const DoubleTimeSeries& B,
                                     const std::vector<std::pair<int, int>>& window,
                                     int n, int m, int dim) const {

                int rank = cached_rank_;
                int size = cached_size_;

                if (size == 1) {
                    BlockedStrategy blocked(block_size_);
                    blocked.template execute_constrained<M>(D, A, B, window, n, m, dim);
                    return;
                }

                auto in_window = utils::create_window_mask(window, n, m);
                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                std::vector<double> row_buffer(m + 1);
                std::vector<double> col_buffer(n + 1);

                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

                    std::vector<std::pair<int, int>> my_blocks;
                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            int block_idx = bi - start_bi;
                            if (block_idx % size == rank) {
                                my_blocks.push_back({bi, bj});
                            }
                        }
                    }

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) if(threads_per_process_ > 0)
#endif
                    for (size_t idx = 0; idx < my_blocks.size(); ++idx) {
                        int bi = my_blocks[idx].first;
                        int bj = my_blocks[idx].second;
                        process_block_constrained<M>(D, A, B, bi, bj, n, m, dim, in_window);
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

            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result(const DoubleMatrix& D) const {
                int rank = cached_rank_;

                if (rank != 0) {
                    return {0.0, {}};  // Empty path, cost will be ignored
                }

                if (rank == 0) {
                    // Before extraction, ensure the last element D[n][m] exists and is valid
                    int n = D.rows() - 1;
                    int m = D.cols() - 1;

                    // Check if D[n][m] is accessible
                    if (n >= 0 && m >= 0) {
                        double final_cost = D(n, m);
                        // Extract the path
                        auto path = utils::backtrack_path(D);
                        return {final_cost, path};
                    } else {
                        return {0.0, {}};
                    }
                }

                return {0.0, {}};  // Default return, shouldn't be reached
            }

            void set_block_size(int block_size) { block_size_ = block_size; }
            int get_block_size() const { return block_size_; }

            std::string_view name() const { return "MPI_V1_Fixed"; }
            bool is_parallel() const { return true; }

        private:
            template<distance::MetricType M>
            void process_block(DoubleMatrix& D,
                               const DoubleTimeSeries& A,
                               const DoubleTimeSeries& B,
                               int bi, int bj, int n, int m, int dim) const {

                int i_start = bi * block_size_ + 1;
                int i_end = std::min((bi + 1) * block_size_, n);
                int j_start = bj * block_size_ + 1;
                int j_end = std::min((bj + 1) * block_size_, m);

                for (int i = i_start; i <= i_end; ++i) {
                    for (int j = j_start; j <= j_end; ++j) {
                        D(i, j) = utils::compute_cell_cost<M>(
                                A[i-1], B[j-1], dim,
                                D(i-1, j-1), D(i, j-1), D(i-1, j)
                        );
                    }
                }
            }

            template<distance::MetricType M>
            void process_block_constrained(DoubleMatrix& D,
                                           const DoubleTimeSeries& A,
                                           const DoubleTimeSeries& B,
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
                                    A[i-1], B[j-1], dim,
                                    D(i-1, j-1), D(i, j-1), D(i-1, j)
                            );
                        }
                    }
                }
            }

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
        };



    }
}

#endif //DTWACCELERATOR_MPI_STRATEGY_HPP