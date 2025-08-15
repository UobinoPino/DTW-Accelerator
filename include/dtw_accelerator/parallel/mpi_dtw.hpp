#ifndef DTW_ACCELERATOR_PARALLEL_MPI_DTW_HPP
#define DTW_ACCELERATOR_PARALLEL_MPI_DTW_HPP

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cstring>

#include "../distance_metrics.hpp"
#include "../constraints.hpp"
#include "../path_processing.hpp"
#include "../core_dtw.hpp"
#include "../fast_dtw.hpp"
#include "../dtw_utils.hpp"
#include "../dtw_accelerator.hpp"
#include "openmp_dtw.hpp"

namespace dtw_accelerator {
    namespace parallel {
        namespace mpi {

            // MPI DTW implementation
            template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_mpi(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    int block_size = 64,
                    int omp_threads_per_process = 0,
                    MPI_Comm comm = MPI_COMM_WORLD) {

                int rank, size;
                MPI_Comm_rank(comm, &rank);
                MPI_Comm_size(comm, &size);

                if (omp_threads_per_process > 0) {
                    omp_set_num_threads(omp_threads_per_process);
                }

                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                if (size == 1) {
                    auto result = parallel::omp::dtw_omp_blocked<M>(A, B, block_size, omp_threads_per_process);
                    return result;

                }

                const double INF = std::numeric_limits<double>::infinity();

                MPI_Bcast(&n, 1, MPI_INT, 0, comm);
                MPI_Bcast(&m, 1, MPI_INT, 0, comm);
                MPI_Bcast(&dim, 1, MPI_INT, 0, comm);

                std::vector<std::vector<double>> A_local(n, std::vector<double>(dim));
                std::vector<std::vector<double>> B_local(m, std::vector<double>(dim));

                std::vector<double> buffer(n * dim + m * dim);
                if (rank == 0) {
                    for (int i = 0; i < n; ++i) {
                        std::copy(A[i].begin(), A[i].end(), buffer.begin() + i * dim);
                    }
                    for (int j = 0; j < m; ++j) {
                        std::copy(B[j].begin(), B[j].end(), buffer.begin() + n * dim + j * dim);
                    }
                }

                MPI_Bcast(buffer.data(), buffer.size(), MPI_DOUBLE, 0, comm);

                for (int i = 0; i < n; ++i) {
                    std::copy(buffer.begin() + i * dim, buffer.begin() + (i + 1) * dim, A_local[i].begin());
                }
                for (int j = 0; j < m; ++j) {
                    std::copy(buffer.begin() + n * dim + j * dim,
                              buffer.begin() + n * dim + (j + 1) * dim, B_local[j].begin());
                }

                std::vector<std::vector<double>> D(n + 1, std::vector<double>(m + 1, INF));
                utils::init_dtw_matrix(D);

                int n_blocks = (n + block_size - 1) / block_size;
                int m_blocks = (m + block_size - 1) / block_size;


                // Pre-allocate communication buffers
                std::vector<double> row_buffer(m + 1);
                std::vector<double> col_buffer(n + 1);

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

                    if (!my_blocks.empty()) {

                        // Process my blocks with OpenMP
                        #pragma omp parallel for schedule(dynamic, 1)
                        for (size_t idx = 0; idx < my_blocks.size(); ++idx) {
                            int bi = my_blocks[idx].first;
                            int bj = my_blocks[idx].second;

                            int i_start = bi * block_size + 1;
                            int i_end = std::min((bi + 1) * block_size, n);
                            int j_start = bj * block_size + 1;
                            int j_end = std::min((bj + 1) * block_size, m);

                            for (int i = i_start; i <= i_end; ++i) {
                                for (int j = j_start; j <= j_end; ++j) {
                                    D[i][j] = utils::compute_cell_cost<M>(
                                            A_local[i-1].data(), B_local[j-1].data(), dim,
                                            D[i-1][j-1], D[i][j-1], D[i-1][j]
                                    );
                                }
                            }
                        }

                    }

                    // Synchronization - only share boundaries needed for next wave

                    // Share only the last row and column of each block
                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            int block_owner = (bi - start_bi) % size;

                            // Share last row of block (if not last block row)
                            if (bi < n_blocks - 1) {
                                int i_boundary = std::min((bi + 1) * block_size, n);
                                int j_start = bj * block_size;
                                int j_end = std::min((bj + 1) * block_size, m);

                                if (rank == block_owner) {
                                    std::copy(D[i_boundary].begin() + j_start,
                                              D[i_boundary].begin() + j_end + 1,
                                              row_buffer.begin());
                                }

                                MPI_Bcast(row_buffer.data(), j_end - j_start + 1,
                                          MPI_DOUBLE, block_owner, comm);

                                if (rank != block_owner) {
                                    std::copy(row_buffer.begin(),
                                              row_buffer.begin() + (j_end - j_start + 1),
                                              D[i_boundary].begin() + j_start);
                                }
                            }

                            // Share last column of block (if not last block column)
                            if (bj < m_blocks - 1) {
                                int j_boundary = std::min((bj + 1) * block_size, m);
                                int i_start = bi * block_size;
                                int i_end = std::min((bi + 1) * block_size, n);

                                if (rank == block_owner) {
                                    for (int i = i_start; i <= i_end; ++i) {
                                        col_buffer[i - i_start] = D[i][j_boundary];
                                    }
                                }

                                MPI_Bcast(col_buffer.data(), i_end - i_start + 1,
                                          MPI_DOUBLE, block_owner, comm);

                                if (rank != block_owner) {
                                    for (int i = i_start; i <= i_end; ++i) {
                                        D[i][j_boundary] = col_buffer[i - i_start];
                                    }
                                }
                            }
                        }
                    }

                }

                std::vector<std::pair<int, int>> path;
                if (rank == 0) {
                    path = utils::backtrack_path(D);
                }
                return {D[n][m], path};
            }



        } // namespace mpi
    } // namespace parallel
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_PARALLEL_MPI_DTW_HPP