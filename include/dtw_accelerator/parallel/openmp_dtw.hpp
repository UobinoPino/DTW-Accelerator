#ifndef DTW_ACCELERATOR_PARALLEL_OPENMP_DTW_HPP
#define DTW_ACCELERATOR_PARALLEL_OPENMP_DTW_HPP

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <immintrin.h>

#include "../distance_metrics.hpp"
#include "../constraints.hpp"
#include "../path_processing.hpp"
#include "../core_dtw.hpp"
#include "../fast_dtw.hpp"

namespace dtw_accelerator {
    namespace parallel {
        namespace omp {

            // OpenMP DTW implementation using wavefront/diagonal parallelization
            template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_omp(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    int num_threads = 0) {

                if (num_threads > 0) {
                    omp_set_num_threads(num_threads);
                }

                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                const double INF = std::numeric_limits<double>::infinity();

                // Allocate cost matrix
                std::vector<std::vector<double>> D(n + 1, std::vector<double>(m + 1, INF));

                utils::init_dtw_matrix(D);

                // Wavefront parallelization - process anti-diagonals
                int total_diagonals = n + m - 1;

                for (int diag = 0; diag < total_diagonals; ++diag) {
                    // Calculate the range of valid cells in this diagonal
                    int start_i = std::max(1, diag - m + 2);
                    int end_i = std::min(n, diag + 1);

                    #pragma omp parallel for schedule(dynamic)
                    for (int i = start_i; i <= end_i; ++i) {
                        int j = diag - i + 2;
                        if (j >= 1 && j <= m) {
                            D[i][j] = utils::compute_cell_cost<M>(
                                    A[i-1].data(), B[j-1].data(), dim,
                                    D[i-1][j-1], D[i][j-1], D[i-1][j]
                            );
                        }
                    }
                }

                // Backtrack to find path (sequential)
                auto path = utils::backtrack_path(D);
                return {D[n][m], path};
            }

            // Alternative blocked parallelization for better cache locality
            template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_omp_blocked(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    int block_size = 64,
                    int num_threads = 0) {

                if (num_threads > 0) {
                    omp_set_num_threads(num_threads);
                }

                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                const double INF = std::numeric_limits<double>::infinity();

                // Allocate cost matrix
                std::vector<std::vector<double>> D(n + 1, std::vector<double>(m + 1, INF));

                utils::init_dtw_matrix(D);

                // Process in blocks for better cache locality
                int n_blocks = (n + block_size - 1) / block_size;
                int m_blocks = (m + block_size - 1) / block_size;

                // Process blocks in wavefront order
                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

                    #pragma omp parallel for schedule(dynamic)
                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            // Process block (bi, bj)
                            int i_start = bi * block_size + 1;
                            int i_end = std::min((bi + 1) * block_size, n);
                            int j_start = bj * block_size + 1;
                            int j_end = std::min((bj + 1) * block_size, m);

                            for (int i = i_start; i <= i_end; ++i) {
                                for (int j = j_start; j <= j_end; ++j) {
                                    D[i][j] = utils::compute_cell_cost<M>(
                                            A[i-1].data(), B[j-1].data(), dim,
                                            D[i-1][j-1], D[i][j-1], D[i-1][j]
                                    );
                                }
                            }
                        }
                    }
                }

                // Backtrack to find path
                auto path = utils::backtrack_path(D);
                return {D[n][m], path};
            }


            // OpenMP DTW with constraints using blocked implementation
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_omp_with_constraint(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    int block_size = 64,
                    int num_threads = 0) {

                using namespace constraints;

                if (num_threads > 0) {
                    omp_set_num_threads(num_threads);
                }

                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                const double INF = std::numeric_limits<double>::infinity();

                // Pre-compute constraint mask
                std::vector<std::vector<bool>> constraint_mask(n, std::vector<bool>(m, false));

                #pragma omp parallel for collapse(2)
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < m; ++j) {
                        bool in_constraint = false;

                        if constexpr (CT == ConstraintType::NONE) {
                            in_constraint = true;
                        }
                        else if constexpr (CT == ConstraintType::SAKOE_CHIBA) {
                            in_constraint = within_sakoe_chiba_band<R>(i, j, n, m);
                        }
                        else if constexpr (CT == ConstraintType::ITAKURA) {
                            in_constraint = within_itakura_parallelogram<S>(i, j, n, m);
                        }

                        constraint_mask[i][j] = in_constraint;
                    }
                }

                // Allocate cost matrix
                std::vector<std::vector<double>> D(n + 1, std::vector<double>(m + 1, INF));
                D[0][0] = 0;

                // Process in blocks for better cache locality
                int n_blocks = (n + block_size - 1) / block_size;
                int m_blocks = (m + block_size - 1) / block_size;

                // Process blocks in wavefront order
                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

                    #pragma omp parallel for schedule(dynamic)
                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            // Process block (bi, bj)
                            int i_start = bi * block_size + 1;
                            int i_end = std::min((bi + 1) * block_size, n);
                            int j_start = bj * block_size + 1;
                            int j_end = std::min((bj + 1) * block_size, m);

                            for (int i = i_start; i <= i_end; ++i) {
                                for (int j = j_start; j <= j_end; ++j) {
                                    // Only compute cells within the constraint
                                    if (i-1 >= 0 && j-1 >= 0 && i-1 < n && j-1 < m && constraint_mask[i-1][j-1]) {
                                        D[i][j] = utils::compute_cell_cost<M>(
                                                A[i-1].data(), B[j-1].data(), dim,
                                                D[i-1][j-1], D[i][j-1], D[i-1][j]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Check if we have a valid path
                if (D[n][m] == INF) {
                    return {INF, {}};
                }

                // Backtrack to find path
                auto path = utils::backtrack_path(D);
                return {D[n][m], path};
            }

            // OpenMP constrained DTW with blocked implementation (used by FastDTW)
            template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
            inline std::pair<double, std::vector<std::pair<int, int>>> dtw_constrained_omp(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    const std::vector<std::pair<int, int>>& window,
                    int block_size = 64,
                    int num_threads = 0) {

                if (num_threads > 0) {
                    omp_set_num_threads(num_threads);
                }

                int n = A.size();
                int m = B.size();
                int dim = A.empty() ? 0 : A[0].size();

                const double INF = std::numeric_limits<double>::infinity();

                // Create window mask for efficient lookup
                std::vector<std::vector<bool>> in_window(n, std::vector<bool>(m, false));

                #pragma omp parallel for
                for (size_t idx = 0; idx < window.size(); ++idx) {
                    const auto& [i, j] = window[idx];
                    if (i >= 0 && i < n && j >= 0 && j < m) {
                        in_window[i][j] = true;
                    }
                }

                // Initialize cost matrix
                std::vector<std::vector<double>> D(n + 1, std::vector<double>(m + 1, INF));
                D[0][0] = 0;

                // Process in blocks for better cache locality
                int n_blocks = (n + block_size - 1) / block_size;
                int m_blocks = (m + block_size - 1) / block_size;

                // Process blocks in wavefront order
                for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                    int start_bi = std::max(0, wave - m_blocks + 1);
                    int end_bi = std::min(n_blocks - 1, wave);

                    #pragma omp parallel for schedule(dynamic)
                    for (int bi = start_bi; bi <= end_bi; ++bi) {
                        int bj = wave - bi;
                        if (bj >= 0 && bj < m_blocks) {
                            // Process block (bi, bj)
                            int i_start = bi * block_size + 1;
                            int i_end = std::min((bi + 1) * block_size, n);
                            int j_start = bj * block_size + 1;
                            int j_end = std::min((bj + 1) * block_size, m);

                            for (int i = i_start; i <= i_end; ++i) {
                                for (int j = j_start; j <= j_end; ++j) {
                                    // Only compute cells within the constraint window
                                    if (in_window[i-1][j-1]) {
                                        D[i][j] = utils::compute_cell_cost<M>(
                                                A[i-1].data(), B[j-1].data(), dim,
                                                D[i-1][j-1], D[i][j-1], D[i-1][j]
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // Backtrack to find path
                auto path = utils::backtrack_path(D);
                return {D[n][m], path};
            }

            // OpenMP FastDTW implementation
            inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_omp(
                    const std::vector<std::vector<double>>& A,
                    const std::vector<std::vector<double>>& B,
                    int radius = 1,
                    int min_size = 100,
                    int num_threads = 0) {

                using namespace path;

                if (num_threads > 0) {
                    omp_set_num_threads(num_threads);
                }

                int n = A.size();
                int m = B.size();

                // Base case: if time series are small enough, use standard DTW
                if (n <= min_size && m <= min_size) {
                    return dtw_omp(A, B, num_threads);
                }

                // Base case: if we can't downsample further
                if (n <= 2 || m <= 2) {
                    return dtw_omp(A, B, num_threads);
                }

                // Recursive case:
                // 1. Coarsen the time series
                auto A_coarse = downsample(A);
                auto B_coarse = downsample(B);

                // 2. Recursively compute FastDTW on the coarsened data
                auto [cost_coarse, path_coarse] = fastdtw_omp(
                        A_coarse, B_coarse, radius, min_size, num_threads);

                // 3. Project the low-resolution path to a higher resolution
                auto projected_path = expand_path(path_coarse, n, m);

                // 4. Create a search window around the projected path
                auto window = get_window(projected_path, n, m, radius);

                // 5. Compute constrained DTW within the window
                return dtw_constrained_omp(A, B, window, num_threads);
            }



        } // namespace omp
    } // namespace parallel
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_PARALLEL_OPENMP_DTW_HPP