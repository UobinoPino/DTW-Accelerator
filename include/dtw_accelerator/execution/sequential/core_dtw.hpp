#ifndef DTW_ACCELERATOR_CORE_DTW_HPP
#define DTW_ACCELERATOR_CORE_DTW_HPP

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>

#include "dtw_accelerator/core/distance_metrics.hpp"
#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/dtw_utils.hpp"
#include <map>


namespace dtw_accelerator {
    namespace core {

        // DTW implementation (used for base case in fast_dtw)
        template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
        inline std::pair<double, std::vector<std::pair<int,int>>> dtw_cpu(
                const std::vector<std::vector<double>>& A,
                const std::vector<std::vector<double>>& B)
        {
            int n = (int)A.size(), m = (int)B.size(), dim = (int)A[0].size();
            const double INF = std::numeric_limits<double>::infinity();

            std::vector<std::vector<double>> D(n+1, std::vector<double>(m+1, INF));
            utils::init_dtw_matrix(D);

            for(int i=1;i<=n;++i) {
                for(int j=1;j<=m;++j) {
                    D[i][j] = utils::compute_cell_cost<M>(
                            A[i-1].data(), B[j-1].data(), dim,
                            D[i-1][j-1], D[i][j-1], D[i-1][j]
                    );
                }
            }

            auto path = utils::backtrack_path(D);
            return { D[n][m], path };
        }


        template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
        std::pair<double, std::vector<std::pair<int,int>>> dtw_blocked(
                const std::vector<std::vector<double>>& A,
                const std::vector<std::vector<double>>& B,
                int block_size = 64,
                int num_threads = 0) {

            int n = A.size();
            int m = B.size();
            int dim = A.empty() ? 0 : A[0].size();
            const double INF = std::numeric_limits<double>::infinity();
            std::vector<std::vector<double>> D(n + 1, std::vector<double>(m + 1, INF));
            dtw_accelerator::utils::init_dtw_matrix(D);

            int n_blocks = (n + block_size - 1) / block_size;
            int m_blocks = (m + block_size - 1) / block_size;

            for (int wave = 0; wave < n_blocks + m_blocks - 1; ++wave) {
                int start_bi = std::max(0, wave - m_blocks + 1);
                int end_bi = std::min(n_blocks - 1, wave);

                for (int bi = start_bi; bi <= end_bi; ++bi) {
                    int bj = wave - bi;
                    if (bj >= 0 && bj < m_blocks) {
                        int i_start = bi * block_size + 1;
                        int i_end = std::min((bi + 1) * block_size, n);
                        int j_start = bj * block_size + 1;
                        int j_end = std::min((bj + 1) * block_size, m);

                        for (int i = i_start; i <= i_end; ++i) {
                            for (int j = j_start; j <= j_end; ++j) {
                                D[i][j] = dtw_accelerator::utils::compute_cell_cost<M>(
                                        A[i-1].data(), B[j-1].data(), dim,
                                        D[i-1][j-1], D[i][j-1], D[i-1][j]
                                );
                            }
                        }
                    }
                }
            }

            auto path = utils::backtrack_path(D);

            return {D[n][m], path};
        }

        //Constrained DTW that only computes cells within the window
        template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
        inline std::pair<double, std::vector<std::pair<int, int>>> dtw_constrained(
                const std::vector<std::vector<double>>& A,
                const std::vector<std::vector<double>>& B,
                const std::vector<std::pair<int, int>>& window) {

            int n = A.size(), m = B.size(), dim = A[0].size();
            const double INF = std::numeric_limits<double>::infinity();

            // Initialize cost matrix with infinity
            std::vector<std::vector<double>> D(n+1, std::vector<double>(m+1, INF));
            utils::init_dtw_matrix(D);

            // Create a window mask for an easy lookup
            auto in_window = utils::create_window_mask(window, n, m);

            // Fill the cost matrix, but only for cells in the window
            for (int i = 1; i <= n; ++i) {
                for (int j = 1; j <= m; ++j) {
                    if (i-1 < n && j-1 < m && in_window[i-1][j-1]) {
                        D[i][j] = utils::compute_cell_cost<M>(
                                A[i-1].data(), B[j-1].data(), dim,
                                D[i-1][j-1], D[i][j-1], D[i-1][j]
                        );
                    }
                }
            }

            // Backtrack to find the path
            auto path = utils::backtrack_path(D);

            return {D[n][m], path};
        }

        // Helper function to check constraints
        template<constraints::ConstraintType CT, int R, double S>
        constexpr bool within_constraint(int i, int j, int n, int m) {
            if constexpr (CT == constraints::ConstraintType::SAKOE_CHIBA) {
                return constraints::within_sakoe_chiba_band<R>(i, j, n, m);
            } else if constexpr (CT == constraints::ConstraintType::ITAKURA) {
                return constraints::within_itakura_parallelogram<S>(i, j, n, m);
            }
            return true;
        }

        // Helper function to process a single cell
        template<distance::MetricType M>
        inline void process_cell(std::vector<std::vector<double>>& D,
                                 const std::vector<std::vector<double>>& A,
                                 const std::vector<std::vector<double>>& B,
                                 int i, int j, int dim) {
            const double INF = std::numeric_limits<double>::infinity();
            int i_cost = i + 1;
            int j_cost = j + 1;

            double min_prev = utils::min_prev_cost(
                    D[i_cost-1][j_cost-1],
                    D[i_cost][j_cost-1],
                    D[i_cost-1][j_cost]
            );

            if (min_prev != INF) {
                double cost = distance::Metric<M>::compute(A[i].data(), B[j].data(), dim);
                D[i_cost][j_cost] = cost + min_prev;
            }
        }




        template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                distance::MetricType M = distance::MetricType::EUCLIDEAN>
        inline std::pair<double, std::vector<std::pair<int, int>>> dtw_with_constraint(
                const std::vector<std::vector<double>>& A,
                const std::vector<std::vector<double>>& B) {

            using namespace constraints;

            int n = A.size(), m = B.size(), dim = A[0].size();
            const double INF = std::numeric_limits<double>::infinity();

            // Initialize cost matrix with infinity
            std::vector<std::vector<double>> D(n+1, std::vector<double>(m+1, INF));
            utils::init_dtw_matrix(D);

            if constexpr (CT == ConstraintType::NONE) {
                D[1][1] = distance::Metric<M>::compute(A[0].data(), B[0].data(), dim);
            } else {
                if (within_constraint<CT, R, S>(0, 0, n, m)) {
                    D[1][1] = distance::Metric<M>::compute(A[0].data(), B[0].data(), dim);
                }
            }

            // Process diagonals in order (sum = i + j)
            // This naturally maintains DTW dependencies
            for (int sum = 1; sum < n + m - 1; ++sum) {
                int start_i = std::max(0, sum - (m - 1));
                int end_i = std::min(n - 1, sum);

                for (int i = start_i; i <= end_i; ++i) {
                    int j = sum - i;
                    if (j >= 0 && j < m) {
                        // Check constraint on-the-fly
                        if constexpr (CT == ConstraintType::NONE) {
                            process_cell<M>(D, A, B, i, j, dim);
                        } else if (within_constraint<CT, R, S>(i, j, n, m)) {
                            process_cell<M>(D, A, B, i, j, dim);
                        }
                    }
                }
            }

            // Check if we have a valid path to the end
            if (D[n][m] == INF) {
                // No valid path found within constraints
                return {INF, {}};
            }

            // Backtrack to find the path
            auto path = utils::backtrack_path(D);
            return {D[n][m], path};
        }


    } // namespace core
} // namespace dtw_accelerator


#endif // DTW_ACCELERATOR_CORE_DTW_HPP