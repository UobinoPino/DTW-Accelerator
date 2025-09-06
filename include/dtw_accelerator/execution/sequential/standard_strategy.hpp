#ifndef DTWACCELERATOR_SEQUENTIAL_STRATEGY_HPP
#define DTWACCELERATOR_SEQUENTIAL_STRATEGY_HPP

#include "dtw_accelerator/core/dtw_concepts.hpp"
#include "dtw_accelerator/core/distance_metrics.hpp"
#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/dtw_utils.hpp"
#include "dtw_accelerator/core/matrix.hpp"
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <string_view>
#include <memory>
#include "dtw_accelerator/execution/base_strategy.hpp"


namespace dtw_accelerator {
    namespace execution {

        // Sequential CPU strategy
        class SequentialStrategy : public BaseStrategy<SequentialStrategy> {
        public:

            template<distance::MetricType M>
            void execute(DoubleMatrix& D,
                         const DoubleTimeSeries& A,
                         const DoubleTimeSeries& B,
                         int n, int m, int dim) const {

                for (int i = 1; i <= n; ++i) {
                    for (int j = 1; j <= m; ++j) {
                        D(i, j) = utils::compute_cell_cost<M>(
                                A[i-1], B[j-1], dim,
                                D(i-1, j-1), D(i, j-1), D(i-1, j)
                        );
                    }
                }
            }

            template<distance::MetricType M>
            void execute_constrained(DoubleMatrix& D,
                                     const DoubleTimeSeries& A,
                                     const DoubleTimeSeries& B,
                                     const std::vector<std::pair<int, int>>& window,
                                     int n, int m, int dim) const {

                // Iterate over the window pairs directly
                for (const auto& [i, j] : window) {
                    // Convert to 1-based indexing for the matrix
                    int i_idx = i + 1;
                    int j_idx = j + 1;

                    // Compute DTW cost for this cell
                    D(i_idx, j_idx) = utils::compute_cell_cost<M>(
                            A[i], B[j], dim,
                            D(i_idx-1, j_idx-1),
                            D(i_idx, j_idx-1),
                            D(i_idx-1, j_idx)
                    );
                }
            }

            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            void execute_with_constraint(DoubleMatrix& D,
                                         const DoubleTimeSeries& A,
                                         const DoubleTimeSeries& B,
                                         int n, int m, int dim) const {

                if constexpr (CT == constraints::ConstraintType::NONE) {
                    // No constraint - standard DTW
                    for (int i = 1; i <= n; ++i) {
                        for (int j = 1; j <= m; ++j) {
                            D(i, j) = utils::compute_cell_cost<M>(
                                    A[i-1], B[j-1], dim,
                                    D(i-1, j-1), D(i, j-1), D(i-1, j)
                            );
                        }
                    }
                }
                else if constexpr (CT == constraints::ConstraintType::SAKOE_CHIBA) {
                    // Sakoe-Chiba: Only iterate within the band
                    for (int i = 1; i <= n; ++i) {
                        // Calculate the valid j range for this i
                        // The band constraint in normalized coordinates is |i/n - j/m| <= R/max(n,m)
                        double ni = static_cast<double>(i-1) / n;
                        double max_nm = std::max(n, m);

                        // Convert back to j coordinates
                        int j_min = std::max(1, static_cast<int>(
                                std::floor((ni - R/max_nm) * m + 1)));
                        int j_max = std::min(m, static_cast<int>(
                                std::ceil((ni + R/max_nm) * m + 1)));

                        // Only iterate through valid j values
                        for (int j = j_min; j <= j_max; ++j) {
                            D(i, j) = utils::compute_cell_cost<M>(
                                    A[i-1], B[j-1], dim,
                                    D(i-1, j-1), D(i, j-1), D(i-1, j)
                            );
                        }
                    }
                }
                else if constexpr (CT == constraints::ConstraintType::ITAKURA) {
                    // Itakura: Calculate valid j range based on parallelogram
                    for (int i = 1; i <= n; ++i) {
                        double di = static_cast<double>(i - 1);
                        double dn = static_cast<double>(n - 1);

                        if (dn <= 0) {
                            // Edge case: single point
                            D(i, 1) = utils::compute_cell_cost<M>(
                                    A[i-1], B[0], dim,
                                    D(i-1, 0), D(i, 0), D(i-1, 1)
                            );
                            continue;
                        }

                        double ni = di / dn;
                        double dm = static_cast<double>(m - 1);

                        // Calculate j bounds from Itakura constraints
                        double lower_bound = std::max(ni / S, S * ni - (S - 1.0));
                        double upper_bound = std::min(S * ni, ni / S + (1.0 - 1.0/S));

                        int j_min = std::max(1, static_cast<int>(std::floor(lower_bound * dm + 1)));
                        int j_max = std::min(m, static_cast<int>(std::ceil(upper_bound * dm + 1)));

                        // Only iterate through valid j values
                        for (int j = j_min; j <= j_max; ++j) {
                            D(i, j) = utils::compute_cell_cost<M>(
                                    A[i-1], B[j-1], dim,
                                    D(i-1, j-1), D(i, j-1), D(i-1, j)
                            );
                        }
                    }
                }
            }

            std::string_view name() const { return "Sequential"; }
            bool is_parallel() const { return false; }
        };



    }
}

#endif //DTWACCELERATOR_SEQUENTIAL_STRATEGY_HPP