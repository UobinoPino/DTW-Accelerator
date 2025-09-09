/**
 * @file dtw_utils.hpp
 * @brief Utility functions for DTW computations
 * @author UobinoPino
 * @date 2024
 *
 * This file contains common utility functions used across different
 * DTW execution strategies including matrix initialization, cost
 * computation, and path backtracking.
 */

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

    /**
    * @namespace utils
    * @brief Utility functions for DTW algorithm implementations
    */
    namespace utils {

        /**
         * @brief Initialize DTW cost matrix with infinity values
         * @param D Matrix to initialize
         *
         * Sets all matrix elements to infinity except D(0,0) = 0.0
         * which serves as the starting point for DTW computation.
         */
        inline void init_dtw_matrix(DoubleMatrix& D) {
            const double INF = std::numeric_limits<double>::infinity();
            D.fill(INF);
            D(0, 0) = 0.0;
        }


        /**
         * @brief Calculate optimal cost for a DTW cell
         * @tparam M Distance metric type to use
         * @param a_point Point from first time series
         * @param b_point Point from second time series
         * @param dim Number of dimensions per point
         * @param cost_diag Cost from diagonal predecessor (i-1, j-1)
         * @param cost_left Cost from left predecessor (i, j-1)
         * @param cost_up Cost from upper predecessor (i-1, j)
         * @return Optimal cost for cell (i,j)
         *
         * Computes the distance between two points and adds it to the
         * minimum cost among the three possible predecessors.
         */
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

        /**
        * @brief Find minimum cost among predecessor cells
        * @param diag Cost from diagonal predecessor
        * @param left Cost from left predecessor
        * @param up Cost from upper predecessor
        * @return Minimum cost among valid predecessors
        *
        * Handles infinity values properly to find the minimum
        * among potentially invalid predecessor cells.
        */
        inline double min_prev_cost(double diag, double left, double up) {
            const double INF = std::numeric_limits<double>::infinity();
            double min_val = INF;

            if (diag != INF) min_val = diag;
            if (left != INF && left < min_val) min_val = left;
            if (up != INF && up < min_val) min_val = up;

            return min_val;
        }

        /**
         * @brief Backtrack the optimal warping path through the DTW matrix
         * @param D Computed DTW cost matrix
         * @return Vector of (i,j) pairs representing the optimal path
         *
         * Starting from the bottom-right corner of the matrix, traces back
         * the optimal path by following the minimum cost predecessors.
         * The path is returned in forward order (from start to end).
         */
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


        /**
         * @brief Create a boolean mask from window constraints
         * @param window Vector of valid (i,j) cell coordinates
         * @param n Number of rows in the mask
         * @param m Number of columns in the mask
         * @return Boolean matrix with true for valid cells
         *
         * Converts a sparse window representation to a dense boolean
         * matrix for efficient constraint checking.
         */
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

        /**
         * @brief Generate constraint mask based on constraint type
         * @tparam CT Constraint type (NONE, SAKOE_CHIBA, ITAKURA)
         * @tparam R Sakoe-Chiba band radius
         * @tparam S Itakura parallelogram slope
         * @param n Number of rows
         * @param m Number of columns
         * @return Boolean matrix indicating valid cells under the constraint
         *
         * Creates a mask matrix where true values indicate cells that
         * satisfy the specified global path constraint.
         */
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