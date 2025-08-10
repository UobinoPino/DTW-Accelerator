#ifndef DTW_ACCELERATOR_CORE_DTW_HPP
#define DTW_ACCELERATOR_CORE_DTW_HPP

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>

#include "distance_metrics.hpp"
#include "constraints.hpp"
#include <map>


namespace dtw_accelerator {
    namespace core {

        //Constrained DTW that only computes cells within the window
        inline std::pair<double, std::vector<std::pair<int, int>>> dtw_constrained(
                const std::vector<std::vector<double>>& A,
                const std::vector<std::vector<double>>& B,
                const std::vector<std::pair<int, int>>& window) {

            int n = A.size(), m = B.size(), dim = A[0].size();
            const double INF = std::numeric_limits<double>::infinity();

            // Initialize cost matrix with infinity
            std::vector<std::vector<double>> D(n+1, std::vector<double>(m+1, INF));
            D[0][0] = 0;

            // Create a window mask for an easy lookup
            std::vector<std::vector<bool>> in_window(n, std::vector<bool>(m, false));
            for (const auto& [i, j] : window) {
                in_window[i][j] = true;
            }

            // Fill the cost matrix, but only for cells in the window
            for (int i = 1; i <= n; ++i) {
                for (int j = 1; j <= m; ++j) {
                    if (i-1 < n && j-1 < m && in_window[i-1][j-1]) {
                        double cost = distance::euclidean_dist(A[i-1].data(), B[j-1].data(), dim);
                        double best = D[i-1][j];
                        if (D[i][j-1] < best) best = D[i][j-1];
                        if (D[i-1][j-1] < best) best = D[i-1][j-1];
                        D[i][j] = cost + best;
                    }
                }
            }

            // Backtrack to find the path
            int i = n, j = m;
            std::vector<std::pair<int, int>> path;
            while (i > 0 || j > 0) {
                path.emplace_back(i-1, j-1);
                double d0 = (i > 0 && j > 0) ? D[i-1][j-1] : INF;
                double d1 = (i > 0) ? D[i-1][j] : INF;
                double d2 = (j > 0) ? D[i][j-1] : INF;
                if (d0 <= d1 && d0 <= d2) { --i; --j; }
                else if (d1 < d2) { --i; }
                else { --j; }
            }
            std::reverse(path.begin(), path.end());

            return {D[n][m], path};
        }




        template<constraints::ConstraintType CT, int R = 1, double S = 2.0>
        inline std::pair<double, std::vector<std::pair<int, int>>> dtw_with_constraint(
                const std::vector<std::vector<double>>& A,
                const std::vector<std::vector<double>>& B) {

            using namespace constraints;

            int n = A.size(), m = B.size(), dim = A[0].size();
            const double INF = std::numeric_limits<double>::infinity();

            // Initialize cost matrix with infinity
            std::vector<std::vector<double>> D(n+1, std::vector<double>(m+1, INF));
            D[0][0] = 0;

            // Directly generate constraint regions (instead of creating a boolean mask)
            std::vector<std::pair<int, int>> constraint_points;
            constraint_points.reserve(std::min(n, m) * (CT == ConstraintType::SAKOE_CHIBA ? 2*R+1 : 5)); // Appropriate size estimate

            // Always include start and end points
            constraint_points.emplace_back(0, 0);
            constraint_points.emplace_back(n-1, m-1);

            if constexpr (CT == ConstraintType::NONE) {
                // For no constraint, include all points
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        constraint_points.emplace_back(i, j);
                    }
                }
            }
            else if constexpr (CT == ConstraintType::SAKOE_CHIBA) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        if (within_sakoe_chiba_band<R>(i, j, n, m)) {
                            constraint_points.emplace_back(i, j);
                        }
                    }
                }
            }
            else if constexpr (CT == ConstraintType::ITAKURA) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        if (within_itakura_parallelogram<S>(i, j, n, m)) {
                            constraint_points.emplace_back(i, j);
                        }
                    }
                }

            }

            // Remove duplicates
            std::sort(constraint_points.begin(), constraint_points.end());
            constraint_points.erase(std::unique(constraint_points.begin(), constraint_points.end()), constraint_points.end());

            // Use efficient lookup for constraint points
            std::unordered_map<int, std::vector<int>> constraint_map;
            for (const auto& [i, j] : constraint_points) {
                constraint_map[i].push_back(j);
            }

            // Initialize first cell
            if (!constraint_points.empty()) {
                D[1][1] = distance::euclidean_dist(A[0].data(), B[0].data(), dim);
            }

            // Process points in order of increasing sum of coordinates (i+j)
            // This ensures dependencies are met (cells to the left and above are computed first)
            std::sort(constraint_points.begin(), constraint_points.end(),
                      [](const auto& a, const auto& b) {
                          return a.first + a.second < b.first + b.second;
                      });

            // Process constraint points directly
            for (const auto& [i, j] : constraint_points) {
                if (i == 0 && j == 0) continue; // Skip the origin

                int i_cost = i + 1; // +1 for cost matrix indexing
                int j_cost = j + 1;

                double cost = distance::euclidean_dist(A[i].data(), B[j].data(), dim);

                // Find minimum from three adjacent cells
                double min_prev = INF;

                // Check above (i-1, j)
                if (i > 0 && D[i_cost-1][j_cost] != INF) {
                    min_prev = D[i_cost-1][j_cost];
                }

                // Check left (i, j-1)
                if (j > 0 && D[i_cost][j_cost-1] != INF && D[i_cost][j_cost-1] < min_prev) {
                    min_prev = D[i_cost][j_cost-1];
                }

                // Check diagonal (i-1, j-1)
                if (i > 0 && j > 0 && D[i_cost-1][j_cost-1] != INF && D[i_cost-1][j_cost-1] < min_prev) {
                    min_prev = D[i_cost-1][j_cost-1];
                }

                // Update cell if we found a valid path to it
                if (min_prev != INF) {
                    D[i_cost][j_cost] = cost + min_prev;
                }
            }

            // Check if we have a valid path to the end
            if (D[n][m] == INF) {
                // No valid path found within constraints
                return {INF, {}};
            }

            // Backtrack to find the path
            int i = n, j = m;
            std::vector<std::pair<int, int>> path;

            while (i > 0 || j > 0) {
                path.emplace_back(i-1, j-1);

                if (i == 0) {
                    j--;
                } else if (j == 0) {
                    i--;
                } else {
                    double d_diag = D[i-1][j-1];
                    double d_up = D[i-1][j];
                    double d_left = D[i][j-1];

                    if (d_diag <= d_up && d_diag <= d_left) {
                        i--; j--;
                    } else if (d_up <= d_left) {
                        i--;
                    } else {
                        j--;
                    }
                }
            }
            std::reverse(path.begin(), path.end());

            return {D[n][m], path};
        }

// Original DTW implementation (used for base case in fast_dtw)
        inline std::pair<double, std::vector<std::pair<int,int>>> dtw_cpu(
                const std::vector<std::vector<double>>& A,
                const std::vector<std::vector<double>>& B)
        {
            int n = (int)A.size(), m = (int)B.size(), dim = (int)A[0].size();
            const double INF = std::numeric_limits<double>::infinity();
            std::vector<std::vector<double>> D(n+1, std::vector<double>(m+1, INF));
            D[0][0] = 0;
            for(int i=1;i<=n;++i) D[i][0]=INF;
            for(int j=1;j<=m;++j) D[0][j]=INF;

            for(int i=1;i<=n;++i) {
                for(int j=1;j<=m;++j) {
                    double cost = distance::euclidean_dist(A[i-1].data(), B[j-1].data(), dim);
                    double best = D[i-1][j];
                    if(D[i][j-1] < best) best = D[i][j-1];
                    if(D[i-1][j-1] < best) best = D[i-1][j-1];
                    D[i][j] = cost + best;
                }
            }

            // backtrack
            int i = n, j = m;
            std::vector<std::pair<int,int>> path;
            while(i>0 || j>0) {
                path.emplace_back(i-1, j-1);
                double d0 = (i>0 && j>0)? D[i-1][j-1] : INF;
                double d1 = (i>0)?       D[i-1][j]   : INF;
                double d2 = (j>0)?       D[i][j-1]   : INF;
                if(d0 <= d1 && d0 <= d2) { --i; --j; }
                else if(d1 < d2)         { --i; }
                else                     { --j; }
            }
            std::reverse(path.begin(), path.end());
            return { D[n][m], path };
        }

    } // namespace core
} // namespace dtw_accelerator


#endif // DTW_ACCELERATOR_CORE_DTW_HPP