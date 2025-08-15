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


            // MPI DTW constrained implementation
            template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_constrained_mpi(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    const std::vector<std::pair<int, int>>& window,
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
                int window_size = window.size();

                if (size == 1) {
                    auto result = core::dtw_constrained<M>(A, B, window);
                    return result;
                }

                const double INF = std::numeric_limits<double>::infinity();

                // Broadcast dimensions and window size
                MPI_Bcast(&n, 1, MPI_INT, 0, comm);
                MPI_Bcast(&m, 1, MPI_INT, 0, comm);
                MPI_Bcast(&dim, 1, MPI_INT, 0, comm);
                MPI_Bcast(&window_size, 1, MPI_INT, 0, comm);

                // Broadcast data
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

                // Broadcast window data
                std::vector<std::pair<int, int>> window_local(window_size);
                std::vector<int> window_buffer(window_size * 2);
                if (rank == 0) {
                    for (int i = 0; i < window_size; ++i) {
                        window_buffer[i * 2] = window[i].first;
                        window_buffer[i * 2 + 1] = window[i].second;
                    }
                }

                MPI_Bcast(window_buffer.data(), window_size * 2, MPI_INT, 0, comm);

                for (int i = 0; i < window_size; ++i) {
                    window_local[i].first = window_buffer[i * 2];
                    window_local[i].second = window_buffer[i * 2 + 1];
                }

                // Initialize DTW matrix
                std::vector<std::vector<double>> D(n + 1, std::vector<double>(m + 1, INF));
                utils::init_dtw_matrix(D);

                // Create window mask
                auto in_window = utils::create_window_mask(window_local, n, m);

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
                                    // Only compute if cell is in window
                                    if (i-1 < n && j-1 < m && in_window[i-1][j-1]) {
                                        D[i][j] = utils::compute_cell_cost<M>(
                                                A_local[i-1].data(), B_local[j-1].data(), dim,
                                                D[i-1][j-1], D[i][j-1], D[i-1][j]
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Synchronization - share boundaries needed for next wave
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

            // MPI DTW with constraint templates implementation
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_with_constraint_mpi(
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
                    auto result = core::dtw_with_constraint<CT, R, S, M>(A, B);
                    return result;
                }

                const double INF = std::numeric_limits<double>::infinity();

                // Broadcast dimensions
                MPI_Bcast(&n, 1, MPI_INT, 0, comm);
                MPI_Bcast(&m, 1, MPI_INT, 0, comm);
                MPI_Bcast(&dim, 1, MPI_INT, 0, comm);

                // Broadcast data
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

                // Initialize DTW matrix
                std::vector<std::vector<double>> D(n + 1, std::vector<double>(m + 1, INF));
                utils::init_dtw_matrix(D);

                // Generate constraint mask (same on all processes)
                auto constraint_mask = utils::generate_constraint_mask<CT, R, S>(n, m);

                // Get ordered points that satisfy the constraint
                std::vector<std::pair<int, int>> ordered_points;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < m; ++j) {
                        if (constraint_mask[i][j]) {
                            ordered_points.emplace_back(i, j);
                        }
                    }
                }

                std::sort(ordered_points.begin(), ordered_points.end(),
                          [](const auto& a, const auto& b) {
                              return a.first + a.second < b.first + b.second;
                          });

                // Initialize first cell if valid
                if (!ordered_points.empty() && ordered_points[0].first == 0 && ordered_points[0].second == 0) {
                    if (rank == 0) {
                        D[1][1] = distance::Metric<M>::compute(A_local[0].data(), B_local[0].data(), dim);
                    }
                    // Broadcast the first cell to all processes
                    MPI_Bcast(&D[1][1], 1, MPI_DOUBLE, 0, comm);
                }

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
                                    // Only compute if cell satisfies constraint
                                    if (i-1 < n && j-1 < m && constraint_mask[i-1][j-1]) {
                                        double min_prev = utils::min_prev_cost(D[i-1][j-1], D[i][j-1], D[i-1][j]);

                                        // Update cell if we found a valid path to it
                                        if (min_prev != INF) {
                                            double cost = distance::Metric<M>::compute(A_local[i-1].data(), B_local[j-1].data(), dim);
                                            D[i][j] = cost + min_prev;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Synchronization - share boundaries needed for next wave
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

                // Check if we have a valid path to the end
                if (D[n][m] == INF) {
                    // No valid path found within constraints
                    return {INF, {}};
                }

                std::vector<std::pair<int, int>> path;
                if (rank == 0) {
                    path = utils::backtrack_path(D);
                }
                return {D[n][m], path};
            }

            // MPI FastDTW implementation
            template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
            inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_mpi(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    int radius = 1,
                    int min_size = 100,
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

                // Fallback to OpenMP version for single process
                if (size == 1) {
                    return parallel::omp::fastdtw_omp(A, B, radius, min_size, omp_threads_per_process);
                }

                // Base case 1: if time series are small enough, use standard DTW
                if (n <= min_size && m <= min_size) {
                    return dtw_mpi<M>(A, B, block_size, omp_threads_per_process, comm);
                }

                // Base case 2: if we can't downsample further
                if (n <= 2 || m <= 2) {
                    return dtw_mpi<M>(A, B, block_size, omp_threads_per_process, comm);
                }

                // Recursive case - all done sequentially on rank 0, then broadcast final window
                std::vector<std::pair<int, int>> window;

                if (rank == 0) {
                    // 1. Coarsen the time series
                    auto A_coarse = path::downsample(A);
                    auto B_coarse = path::downsample(B);

                    // 2. Recursively compute FastDTW on the coarsened data (sequential)
                    auto [cost_coarse, path_coarse] = fast::fastdtw_cpu(A_coarse, B_coarse, radius, min_size);

                    // 3. Project the low-resolution path to a higher resolution
                    auto projected_path = path::expand_path(path_coarse, n, m);

                    // 4. Create a search window around the projected path
                    window = path::get_window(projected_path, n, m, radius);
                }

                // 5. Compute constrained DTW within the window using MPI parallelization
                return dtw_constrained_mpi<M>(A, B, window, block_size, omp_threads_per_process, comm);
            }



        } // namespace mpi
    } // namespace parallel
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_PARALLEL_MPI_DTW_HPP