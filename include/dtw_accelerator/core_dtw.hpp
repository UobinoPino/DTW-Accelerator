#ifndef DTW_ACCELERATOR_CORE_DTW_HPP
#define DTW_ACCELERATOR_CORE_DTW_HPP

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>

#include "distance_metrics.hpp"
#include "constraints.hpp"
#include "dtw_utils.hpp"
#include <map>


namespace dtw_accelerator {
    namespace core {

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

            // Create a constraint mask based on the specified constraint type
            auto constraint_mask = utils::generate_constraint_mask<CT, R, S>(n, m);

            // Process points in order of increasing sum of coordinates (i + j)
            // This ensures dependencies are met (cells to the left and above are computed first)
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

            // Initialize first cell
            if (!ordered_points.empty()) {
                D[1][1] = distance::Metric<M>::compute(A[0].data(), B[0].data(), dim);
            }

            // Process constraint points directly
            for (const auto& [i, j] : ordered_points) {
                if (i == 0 && j == 0) continue; // Skip the origin

                int i_cost = i + 1; // +1 for cost matrix indexing
                int j_cost = j + 1;

                double min_prev = utils::min_prev_cost(D[i_cost-1][j_cost-1], D[i_cost][j_cost-1], D[i_cost-1][j_cost]);

                // Update cell if we found a valid path to it
                if (min_prev != INF) {
                    double cost = distance::Metric<M>::compute(A[i].data(), B[j].data(), dim);
                    D[i_cost][j_cost] = cost + min_prev;
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

    } // namespace core
} // namespace dtw_accelerator


#endif // DTW_ACCELERATOR_CORE_DTW_HPP