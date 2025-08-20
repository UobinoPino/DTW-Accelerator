#ifndef DTWACCELERATOR_MPI_DEBUG_STRATEGY_HPP
#define DTWACCELERATOR_MPI_DEBUG_STRATEGY_HPP



#include "dtw_accelerator/core/dtw_concepts.hpp"
#include "dtw_accelerator/core/distance_metrics.hpp"
#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/dtw_utils.hpp"
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

        class MPIStrategyDebug : public BaseStrategy<MPIStrategyDebug> {
        private:
            int block_size_;
            int threads_per_process_;
            MPI_Comm communicator_;

            // Cache for avoiding repeated MPI calls
            mutable int cached_rank_ = -1;
            mutable int cached_size_ = -1;

        public:
            explicit MPIStrategyDebug(int block_size = 64, int threads = 0, MPI_Comm comm = MPI_COMM_WORLD)
                    : block_size_(block_size), threads_per_process_(threads), communicator_(comm) {
                MPI_Comm_rank(communicator_, &cached_rank_);
                MPI_Comm_size(communicator_, &cached_size_);

                printf("[DEBUG Rank %d] MPIStrategy constructor - block_size=%d, threads=%d\n",
                       cached_rank_, block_size_, threads_per_process_);
            }

            void initialize_matrix(std::vector<std::vector<double>>& D, int n, int m) const {
                printf("[DEBUG Rank %d] Initializing matrix: n=%d, m=%d\n", cached_rank_, n, m);
                initialize_matrix_impl(D, n, m);
                printf("[DEBUG Rank %d] Matrix initialized: D.size=%zu, D[0].size=%zu\n",
                       cached_rank_, D.size(), D[0].size());
            }

            template<distance::MetricType M>
            void execute(std::vector<std::vector<double>>& D,
                         const std::vector<std::vector<double>>& A,
                         const std::vector<std::vector<double>>& B,
                         int n, int m, int dim) const {

                int rank = cached_rank_;
                int size = cached_size_;

                printf("[DEBUG Rank %d] Starting execute: n=%d, m=%d, dim=%d, size=%d\n",
                       rank, n, m, dim, size);

                if (n <= 0 || m <= 0 || dim <= 0) {
                    printf("[DEBUG Rank %d] Invalid dimensions, returning early\n", rank);
                    return;
                }

                printf("[DEBUG Rank %d] A.size=%zu, B.size=%zu\n", rank, A.size(), B.size());
                if (!A.empty()) {
                    printf("[DEBUG Rank %d] A[0].size=%zu\n", rank, A[0].size());
                }
                if (!B.empty()) {
                    printf("[DEBUG Rank %d] B[0].size=%zu\n", rank, B[0].size());
                }

                // Single process fallback
                if (size == 1) {
                    printf("[DEBUG Rank %d] Using single-process fallback\n", rank);
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
                    printf("[DEBUG Rank %d] Setting OpenMP threads to %d\n", rank, threads_per_process_);
                    omp_set_num_threads(threads_per_process_);
                }
#endif

                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                printf("[DEBUG Rank %d] n_blocks=%d, m_blocks=%d, block_size=%d\n",
                       rank, n_blocks, m_blocks, block_size_);

                // Pre-allocate communication buffers
                const int buffer_size = std::max(m + 1, n + 1) + 100;
                std::vector<double> row_buffer(buffer_size, 0.0);
                std::vector<double> col_buffer(buffer_size, 0.0);

                printf("[DEBUG Rank %d] Communication buffers allocated: size=%d\n",
                       rank, buffer_size);

                // Process blocks in wavefront order
                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    printf("[DEBUG Rank %d] Processing wave %d of %d\n",
                           rank, wave, n_blocks + m_blocks - 2);

                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

                    printf("[DEBUG Rank %d] Wave %d: start_bi=%d, end_bi=%d\n",
                           rank, wave, start_bi, end_bi);

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

                    printf("[DEBUG Rank %d] Wave %d: assigned %zu blocks\n",
                           rank, wave, my_blocks.size());

                    // Process assigned blocks with OpenMP parallelization
                    if (!my_blocks.empty()) {
                        printf("[DEBUG Rank %d] Processing %zu blocks\n", rank, my_blocks.size());
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                        for (size_t idx = 0; idx < my_blocks.size(); ++idx) {
                            int bi = my_blocks[idx].first;
                            int bj = my_blocks[idx].second;

                            // Thread-safe debug within OpenMP region
#pragma omp critical
                            {
                                printf("[DEBUG Rank %d] Processing block (%d,%d)\n", rank, bi, bj);
                            }

                            process_block<M>(D, A, B, bi, bj, n, m, dim);

#pragma omp critical
                            {
                                printf("[DEBUG Rank %d] Finished block (%d,%d)\n", rank, bi, bj);
                            }
                        }
                    }

                    printf("[DEBUG Rank %d] Wave %d: Starting boundary exchange\n", rank, wave);
                    // Efficient synchronization - share boundaries needed for next wave
                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            int block_owner = (bi - start_bi) % size;

                            printf("[DEBUG Rank %d] Block (%d,%d) owner=%d\n", rank, bi, bj, block_owner);

                            // Share last row of block (if not last block row)
                            if (bi < n_blocks - 1) {
                                int i_boundary = std::min((bi + 1) * block_size_, n);
                                int j_start = bj * block_size_;
                                int j_end = std::min((bj + 1) * block_size_ -1, m - 1);

                                printf("[DEBUG Rank %d] Row boundary for block (%d,%d): i=%d, j_start=%d, j_end=%d\n",
                                       rank, bi, bj, i_boundary, j_start, j_end);

                                if (rank == block_owner) {
                                    printf("[DEBUG Rank %d] Copying row data for broadcast\n", rank);
                                    // Check if j_end is valid
                                    if (j_end - j_start + 1 > 0 && j_end < D[i_boundary].size()) {
                                        std::copy(D[i_boundary].begin() + j_start,
                                                  D[i_boundary].begin() + j_end + 1,
                                                  row_buffer.begin());
                                        printf("[DEBUG Rank %d] Row data copied, length=%d\n", rank, j_end - j_start + 1);
                                    } else {
                                        printf("[ERROR Rank %d] Invalid row bounds: j_start=%d, j_end=%d, D[%d].size=%zu\n",
                                               rank, j_start, j_end, i_boundary, D[i_boundary].size());
                                    }
                                }

                                printf("[DEBUG Rank %d] Broadcasting row data, count=%d, root=%d\n",
                                       rank, j_end - j_start + 1, block_owner);

                                // Check if broadcast count is valid
                                if (j_end - j_start + 1 > 0) {
                                    MPI_Bcast(row_buffer.data(), j_end - j_start + 1,
                                              MPI_DOUBLE, block_owner, communicator_);
                                    printf("[DEBUG Rank %d] Row broadcast complete\n", rank);
                                } else {
                                    printf("[ERROR Rank %d] Invalid broadcast count: %d\n", rank, j_end - j_start + 1);
                                }

                                if (rank != block_owner) {
                                    printf("[DEBUG Rank %d] Receiving row data from rank %d\n", rank, block_owner);
                                    if (j_end - j_start + 1 > 0 && j_end < D[i_boundary].size()) {
                                        std::copy(row_buffer.begin(),
                                                  row_buffer.begin() + (j_end - j_start + 1),
                                                  D[i_boundary].begin() + j_start);
                                        printf("[DEBUG Rank %d] Row data received and copied\n", rank);
                                    } else {
                                        printf("[ERROR Rank %d] Invalid row copy bounds: j_start=%d, j_end=%d, D[%d].size=%zu\n",
                                               rank, j_start, j_end, i_boundary, D[i_boundary].size());
                                    }
                                }
                            }

                            // Share last column of block (if not last block column)
                            if (bj < m_blocks - 1) {
                                int j_boundary = std::min((bj + 1) * block_size_, m);
                                int i_start = bi * block_size_;
                                int i_end = std::min((bi + 1) * block_size_ - 1, n - 1);

                                printf("[DEBUG Rank %d] Column boundary for block (%d,%d): j=%d, i_start=%d, i_end=%d\n",
                                       rank, bi, bj, j_boundary, i_start, i_end);

                                if (rank == block_owner) {
                                    printf("[DEBUG Rank %d] Copying column data for broadcast\n", rank);
                                    if (i_end - i_start + 1 > 0 && j_boundary < D[i_start].size()) {
                                        for (int i = i_start; i <= i_end; ++i) {
                                            if (j_boundary >= D[i].size()) {
                                                printf("[ERROR Rank %d] j_boundary=%d >= D[%d].size=%zu\n",
                                                       rank, j_boundary, i, D[i].size());
                                            } else {
                                                col_buffer[i - i_start] = D[i][j_boundary];
                                            }
                                        }
                                        printf("[DEBUG Rank %d] Column data copied, length=%d\n", rank, i_end - i_start + 1);
                                    } else {
                                        printf("[ERROR Rank %d] Invalid column bounds: i_start=%d, i_end=%d, j_boundary=%d\n",
                                               rank, i_start, i_end, j_boundary);
                                    }
                                }

                                printf("[DEBUG Rank %d] Broadcasting column data, count=%d, root=%d\n",
                                       rank, i_end - i_start + 1, block_owner);

                                // Check if broadcast count is valid
                                if (i_end - i_start + 1 > 0) {
                                    MPI_Bcast(col_buffer.data(), i_end - i_start + 1,
                                              MPI_DOUBLE, block_owner, communicator_);
                                    printf("[DEBUG Rank %d] Column broadcast complete\n", rank);
                                } else {
                                    printf("[ERROR Rank %d] Invalid broadcast count: %d\n", rank, i_end - i_start + 1);
                                }

                                if (rank != block_owner) {
                                    printf("[DEBUG Rank %d] Receiving column data from rank %d\n", rank, block_owner);
                                    if (i_end - i_start + 1 > 0) {
                                        for (int i = i_start; i <= i_end; ++i) {
                                            if (j_boundary >= D[i].size()) {
                                                printf("[ERROR Rank %d] j_boundary=%d >= D[%d].size=%zu\n",
                                                       rank, j_boundary, i, D[i].size());
                                            } else {
                                                D[i][j_boundary] = col_buffer[i - i_start];
                                            }
                                        }
                                        printf("[DEBUG Rank %d] Column data received and copied\n", rank);
                                    } else {
                                        printf("[ERROR Rank %d] Invalid column copy: i_start=%d, i_end=%d\n",
                                               rank, i_start, i_end);
                                    }
                                }
                            }
                        }
                    }
                    printf("[DEBUG Rank %d] Wave %d completed\n", rank, wave);
                }

                printf("[DEBUG Rank %d] Execute method complete\n", rank);
            }

            template<distance::MetricType M>
            void execute_constrained(std::vector<std::vector<double>>& D,
                                     const std::vector<std::vector<double>>& A,
                                     const std::vector<std::vector<double>>& B,
                                     const std::vector<std::pair<int, int>>& window,
                                     int n, int m, int dim) const {

                int rank = cached_rank_;
                int size = cached_size_;

                printf("[DEBUG Rank %d] Starting execute_constrained\n", rank);

                if (size == 1) {
                    BlockedStrategy blocked(block_size_);
                    blocked.template execute_constrained<M>(D, A, B, window, n, m, dim);
                    return;
                }

                auto in_window = utils::create_window_mask(window, n, m);
                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                printf("[DEBUG Rank %d] n_blocks=%d, m_blocks=%d\n", rank, n_blocks, m_blocks);

                std::vector<double> row_buffer(m + 1);
                std::vector<double> col_buffer(n + 1);

                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    printf("[DEBUG Rank %d] Processing constrained wave %d\n", rank, wave);
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

                    printf("[DEBUG Rank %d] Wave %d: assigned %zu blocks\n", rank, wave, my_blocks.size());

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic, 1) if(threads_per_process_ > 0)
#endif
                    for (size_t idx = 0; idx < my_blocks.size(); ++idx) {
                        int bi = my_blocks[idx].first;
                        int bj = my_blocks[idx].second;

                        // Thread-safe debug within OpenMP region
#pragma omp critical
                        {
                            printf("[DEBUG Rank %d] Processing constrained block (%d,%d)\n", rank, bi, bj);
                        }

                        process_block_constrained<M>(D, A, B, bi, bj, n, m, dim, in_window);

#pragma omp critical
                        {
                            printf("[DEBUG Rank %d] Finished constrained block (%d,%d)\n", rank, bi, bj);
                        }
                    }

                    // Exchange boundaries using MPI_Bcast
                    printf("[DEBUG Rank %d] Wave %d: Starting boundary exchange\n", rank, wave);
                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            int block_owner = (bi - start_bi) % size;
                            printf("[DEBUG Rank %d] Broadcasting block boundaries for (%d,%d), owner=%d\n",
                                   rank, bi, bj, block_owner);
                            broadcast_block_boundaries(D, bi, bj, n, m, block_owner,
                                                       row_buffer, col_buffer);
                        }
                    }
                    printf("[DEBUG Rank %d] Wave %d completed\n", rank, wave);
                }

                printf("[DEBUG Rank %d] Execute_constrained complete\n", rank);
            }

            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result(const std::vector<std::vector<double>>& D) const {
                printf("[DEBUG Rank %d] Extracting result\n", cached_rank_);
                int rank = cached_rank_;

                if (rank != 0) {
                    // Non-root ranks return a dummy result to avoid segfault
                    printf("[DEBUG Rank %d] Non-root rank returning dummy result\n", rank);
                    return {0.0, {}};  // Empty path, cost will be ignored
                }
                if (rank == 0) {
                    printf("[DEBUG Rank %d] Root rank extracting complete path\n", rank);

                    // Before extraction, ensure the last element D[n][m] exists and is valid
                    int n = D.size() - 1;
                    int m = D[0].size() - 1;

                    printf("[DEBUG Rank %d] Matrix size: %dx%d\n", rank, n+1, m+1);

                    // Check if D[n][m] is accessible
                    if (n >= 0 && m >= 0) {
                        double final_cost = D[n][m];
                        printf("[DEBUG Rank %d] Final cost at D[%d][%d] = %f\n", rank, n, m, final_cost);

                        // Extract the path
                        auto path = utils::backtrack_path(D);
                        printf("[DEBUG Rank %d] Path extracted, length=%zu\n", rank, path.size());

                        return {final_cost, path};
                    } else {
                        printf("[ERROR Rank %d] Invalid matrix dimensions for path extraction\n", rank);
                        return {0.0, {}};
                    }
                }

                return {0.0, {}};  // Default return, shouldn't be reached


                // return extract_result_impl(D);
            }

            void set_block_size(int block_size) { block_size_ = block_size; }
            int get_block_size() const { return block_size_; }

            std::string_view name() const { return "MPI_V1_Fixed"; }
            bool is_parallel() const { return true; }

        private:
            template<distance::MetricType M>
            void process_block(std::vector<std::vector<double>>& D,
                               const std::vector<std::vector<double>>& A,
                               const std::vector<std::vector<double>>& B,
                               int bi, int bj, int n, int m, int dim) const {

                int i_start = bi * block_size_ + 1;
                int i_end = std::min((bi + 1) * block_size_, n);
                int j_start = bj * block_size_ + 1;
                int j_end = std::min((bj + 1) * block_size_, m);

                // Thread-safe debug within process_block
#ifdef USE_OPENMP
#pragma omp critical
#endif
                {
                    printf("[DEBUG Rank %d] Processing block (%d,%d): i=[%d,%d], j=[%d,%d], n=%d, m=%d\n",
                           cached_rank_, bi, bj, i_start, i_end, j_start, j_end, n, m);
                }

                for (int i = i_start; i <= i_end; ++i) {
                    for (int j = j_start; j <= j_end; ++j) {
                        // Check array bounds
                        if (i >= D.size() || j >= D[i].size() || i-1 >= A.size() || j-1 >= B.size()) {
#ifdef USE_OPENMP
#pragma omp critical
#endif
                            {
                                printf("[ERROR Rank %d] Index out of bounds: i=%d, j=%d, D.size=%zu, A.size=%zu, B.size=%zu\n",
                                       cached_rank_, i, j, D.size(), A.size(), B.size());
                                if (i < D.size()) {
                                    printf("[ERROR Rank %d] D[%d].size=%zu\n", cached_rank_, i, D[i].size());
                                }
                            }
                            continue;
                        }

                        try {
                            D[i][j] = utils::compute_cell_cost<M>(
                                    A[i-1].data(), B[j-1].data(), dim,
                                    D[i-1][j-1], D[i][j-1], D[i-1][j]
                            );
                        } catch (const std::exception& e) {
#ifdef USE_OPENMP
#pragma omp critical
#endif
                            {
                                printf("[ERROR Rank %d] Exception in compute_cell_cost: %s\n", cached_rank_, e.what());
                            }
                        }
                    }
                }

#ifdef USE_OPENMP
#pragma omp critical
#endif
                {
                    printf("[DEBUG Rank %d] Block (%d,%d) processing complete\n", cached_rank_, bi, bj);
                }
            }

            template<distance::MetricType M>
            void process_block_constrained(std::vector<std::vector<double>>& D,
                                           const std::vector<std::vector<double>>& A,
                                           const std::vector<std::vector<double>>& B,
                                           int bi, int bj, int n, int m, int dim,
                                           const std::vector<std::vector<bool>>& mask) const {

                int i_start = bi * block_size_ + 1;
                int i_end = std::min((bi + 1) * block_size_, n);
                int j_start = bj * block_size_ + 1;
                int j_end = std::min((bj + 1) * block_size_, m);

#ifdef USE_OPENMP
#pragma omp critical
#endif
                {
                    printf("[DEBUG Rank %d] Processing constrained block (%d,%d): i=[%d,%d], j=[%d,%d]\n",
                           cached_rank_, bi, bj, i_start, i_end, j_start, j_end);
                }

                for (int i = i_start; i <= i_end; ++i) {
                    for (int j = j_start; j <= j_end; ++j) {
                        if (i-1 < n && j-1 < m && mask[i-1][j-1]) {
                            try {
                                D[i][j] = utils::compute_cell_cost<M>(
                                        A[i-1].data(), B[j-1].data(), dim,
                                        D[i-1][j-1], D[i][j-1], D[i-1][j]
                                );
                            } catch (const std::exception& e) {
#ifdef USE_OPENMP
#pragma omp critical
#endif
                                {
                                    printf("[ERROR Rank %d] Exception in constrained compute_cell_cost: %s\n",
                                           cached_rank_, e.what());
                                }
                            }
                        }
                    }
                }

#ifdef USE_OPENMP
#pragma omp critical
#endif
                {
                    printf("[DEBUG Rank %d] Constrained block (%d,%d) processing complete\n", cached_rank_, bi, bj);
                }
            }

            void broadcast_block_boundaries(std::vector<std::vector<double>>& D,
                                            int bi, int bj, int n, int m, int owner,
                                            std::vector<double>& row_buffer,
                                            std::vector<double>& col_buffer) const {

                int rank = cached_rank_;
                int n_blocks = (n + block_size_ - 1) / block_size_;
                int m_blocks = (m + block_size_ - 1) / block_size_;

                printf("[DEBUG Rank %d] Broadcasting boundaries for block (%d,%d), owner=%d\n",
                       rank, bi, bj, owner);

                // Broadcast last row
                if (bi < n_blocks - 1) {
                    int i_boundary = std::min((bi + 1) * block_size_, n);
                    int j_start = bj * block_size_;
                    int j_end = std::min((bj + 1) * block_size_, m);

                    printf("[DEBUG Rank %d] Row boundary: i=%d, j=[%d,%d]\n", rank, i_boundary, j_start, j_end);

                    if (rank == owner) {
                        // Check array bounds
                        if (i_boundary < D.size() && j_end < D[i_boundary].size()) {
                            std::copy(D[i_boundary].begin() + j_start,
                                      D[i_boundary].begin() + j_end + 1,
                                      row_buffer.begin());
                        } else {
                            printf("[ERROR Rank %d] Row boundary out of bounds: D.size=%zu, i_boundary=%d\n",
                                   rank, D.size(), i_boundary);
                            if (i_boundary < D.size()) {
                                printf("[ERROR Rank %d] D[%d].size=%zu, j_end=%d\n",
                                       rank, i_boundary, D[i_boundary].size(), j_end);
                            }
                        }
                    }

                    printf("[DEBUG Rank %d] Broadcasting row, count=%d\n", rank, j_end - j_start + 1);
                    MPI_Bcast(row_buffer.data(), j_end - j_start + 1,
                              MPI_DOUBLE, owner, communicator_);

                    if (rank != owner) {
                        printf("[DEBUG Rank %d] Received row data, copying to D\n", rank);
                        // Check array bounds
                        if (i_boundary < D.size() && j_end < D[i_boundary].size()) {
                            std::copy(row_buffer.begin(),
                                      row_buffer.begin() + (j_end - j_start + 1),
                                      D[i_boundary].begin() + j_start);
                        } else {
                            printf("[ERROR Rank %d] Cannot copy row data: D.size=%zu, i_boundary=%d\n",
                                   rank, D.size(), i_boundary);
                            if (i_boundary < D.size()) {
                                printf("[ERROR Rank %d] D[%d].size=%zu, j_end=%d\n",
                                       rank, i_boundary, D[i_boundary].size(), j_end);
                            }
                        }
                    }
                }

                // Broadcast last column
                if (bj < m_blocks - 1) {
                    int j_boundary = std::min((bj + 1) * block_size_, m);
                    int i_start = bi * block_size_;
                    int i_end = std::min((bi + 1) * block_size_, n);

                    printf("[DEBUG Rank %d] Column boundary: j=%d, i=[%d,%d]\n", rank, j_boundary, i_start, i_end);

                    if (rank == owner) {
                        for (int i = i_start; i <= i_end; ++i) {
                            // Check array bounds
                            if (i < D.size() && j_boundary < D[i].size()) {
                                col_buffer[i - i_start] = D[i][j_boundary];
                            } else {
                                printf("[ERROR Rank %d] Column data access out of bounds: i=%d, j_boundary=%d\n",
                                       rank, i, j_boundary);
                                if (i < D.size()) {
                                    printf("[ERROR Rank %d] D[%d].size=%zu\n", rank, i, D[i].size());
                                }
                            }
                        }
                    }

                    printf("[DEBUG Rank %d] Broadcasting column, count=%d\n", rank, i_end - i_start + 1);
                    MPI_Bcast(col_buffer.data(), i_end - i_start + 1,
                              MPI_DOUBLE, owner, communicator_);

                    if (rank != owner) {
                        printf("[DEBUG Rank %d] Received column data, copying to D\n", rank);
                        for (int i = i_start; i <= i_end; ++i) {
                            // Check array bounds
                            if (i < D.size() && j_boundary < D[i].size()) {
                                D[i][j_boundary] = col_buffer[i - i_start];
                            } else {
                                printf("[ERROR Rank %d] Cannot copy column data: i=%d, j_boundary=%d\n",
                                       rank, i, j_boundary);
                                if (i < D.size()) {
                                    printf("[ERROR Rank %d] D[%d].size=%zu\n", rank, i, D[i].size());
                                }
                            }
                        }
                    }
                }

                printf("[DEBUG Rank %d] Block boundary exchange completed for (%d,%d)\n", rank, bi, bj);
            }
        };





    }
}

#endif //DTWACCELERATOR_MPI_DEBUG_STRATEGY_HPP
