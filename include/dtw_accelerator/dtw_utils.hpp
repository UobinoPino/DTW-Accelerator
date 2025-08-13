#ifndef DTW_ACCELERATOR_DTW_UTILS_HPP
#define DTW_ACCELERATOR_DTW_UTILS_HPP

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include "distance_metrics.hpp"

namespace dtw_accelerator {
    namespace utils {

        // Initialize DTW cost matrix
        inline void init_dtw_matrix(std::vector<std::vector<double>>& D) {
            const double INF = std::numeric_limits<double>::infinity();
            int n = D.size() - 1;
            int m = D[0].size() - 1;

            D[0][0] = 0.0;
            for (int i = 1; i <= n; ++i) D[i][0] = INF;
            for (int j = 1; j <= m; ++j) D[0][j] = INF;
        }


        // Calculate optimal cost for a DTW cell
        template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
        inline double compute_cell_cost(
                const double* a_point,
                const double* b_point,
                int dim,
                double cost_diag,
                double cost_left,
                double cost_up) {

            double cost = distance::Metric<M>::compute(a_point, b_point, dim);
            double best = std::min({cost_diag, cost_left, cost_up});
            return cost + best;
        }

        // Find minimum of previous cells
        inline double min_prev_cost(double diag, double left, double up) {
            const double INF = std::numeric_limits<double>::infinity();
            double min_val = INF;

            if (diag != INF) min_val = diag;
            if (left != INF && left < min_val) min_val = left;
            if (up != INF && up < min_val) min_val = up;

            return min_val;
        }

        // Backtrack through DTW matrix to find an optimal path
        inline std::vector<std::pair<int, int>> backtrack_path(
                const std::vector<std::vector<double>>& D) {

            const double INF = std::numeric_limits<double>::infinity();
            int i = D.size() - 1;
            int j = D[0].size() - 1;
            std::vector<std::pair<int, int>> path;

            while (i > 0 || j > 0) {
                path.emplace_back(i-1, j-1);

                double d_diag = (i > 0 && j > 0) ? D[i-1][j-1] : INF;
                double d_up = (i > 0) ? D[i-1][j] : INF;
                double d_left = (j > 0) ? D[i][j-1] : INF;

                if (d_diag <= d_up && d_diag <= d_left) {
                    --i; --j;
                } else if (d_up < d_left) {
                    --i;
                } else {
                    --j;
                }
            }

            std::reverse(path.begin(), path.end());
            return path;
        }



        // Create window mask from window points
        inline std::vector<std::vector<bool>> create_window_mask(
                const std::vector<std::pair<int, int>>& window,
                int n, int m) {

            std::vector<std::vector<bool>> in_window(n, std::vector<bool>(m, false));
            for (const auto& [i, j] : window) {
                if (i >= 0 && i < n && j >= 0 && j < m) {
                    in_window[i][j] = true;
                }
            }
            return in_window;
        }

        // Generate constraint mask based on constraint type
        template<constraints::ConstraintType CT, int R = 1, double S = 2.0>
        inline std::vector<std::vector<bool>> generate_constraint_mask(int n, int m) {
            using namespace constraints;

            std::vector<std::vector<bool>> mask(n, std::vector<bool>(m, false));

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    if constexpr (CT == ConstraintType::NONE) {
                        mask[i][j] = true;
                    }
                    else if constexpr (CT == ConstraintType::SAKOE_CHIBA) {
                        mask[i][j] = within_sakoe_chiba_band<R>(i, j, n, m);
                    }
                    else if constexpr (CT == ConstraintType::ITAKURA) {
                        mask[i][j] = within_itakura_parallelogram<S>(i, j, n, m);
                    }
                }
            }

            return mask;
        }
    } // namespace utils
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_DTW_UTILS_HPP