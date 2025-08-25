#ifndef DTW_ACCELERATOR_DTW_UTILS_HPP
#define DTW_ACCELERATOR_DTW_UTILS_HPP

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include "dtw_accelerator/core/distance_metrics.hpp"
#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/matrix.hpp"
#include <omp.h>

namespace dtw_accelerator {
    namespace utils {

        inline void init_dtw_matrix(DoubleMatrix& D) {
            const double INF = std::numeric_limits<double>::infinity();
            D.fill(INF);
            D(0, 0) = 0.0;
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

        inline std::vector<std::pair<int, int>> backtrack_path(const DoubleMatrix& D) {
            const double INF = std::numeric_limits<double>::infinity();
            int i = D.rows() - 1;
            int j = D.cols() - 1;
            std::vector<std::pair<int, int>> path;

            while (i > 0 || j > 0) {
                path.emplace_back(i-1, j-1);

                double d_diag = (i > 0 && j > 0) ? D(i-1, j-1) : INF;
                double d_up = (i > 0) ? D(i-1, j) : INF;
                double d_left = (j > 0) ? D(i, j-1) : INF;

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


        inline BoolMatrix create_window_mask(const std::vector<std::pair<int, int>>& window,
                                             int n, int m) {
            BoolMatrix mask(n, m, false);
            for (const auto& [i, j] : window) {
                if (i >= 0 && i < n && j >= 0 && j < m) {
                    mask(i, j) = true;
                }
            }
            return mask;
        }

        template<constraints::ConstraintType CT, int R = 1, double S = 2.0>
        inline BoolMatrix generate_constraint_mask(int n, int m) {
            using namespace constraints;
            BoolMatrix mask(n, m, false);

            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    if constexpr (CT == ConstraintType::NONE) {
                        mask(i, j) = true;
                    } else if constexpr (CT == ConstraintType::SAKOE_CHIBA) {
                        mask(i, j) = within_sakoe_chiba_band<R>(i, j, n, m);
                    } else if constexpr (CT == ConstraintType::ITAKURA) {
                        mask(i, j) = within_itakura_parallelogram<S>(i, j, n, m);
                    }
                }
            }
            return mask;
        }
    } // namespace utils
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_DTW_UTILS_HPP