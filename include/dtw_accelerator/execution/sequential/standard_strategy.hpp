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

        using WindowConstraint = std::vector<std::pair<int, int>>;


        // Sequential CPU strategy with unified interface
        class SequentialStrategy : public BaseStrategy<SequentialStrategy> {
        public:
            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            void execute_with_constraint(DoubleMatrix& D,
                                         const DoubleTimeSeries& A,
                                         const DoubleTimeSeries& B,
                                         int n, int m, int dim,
                                         const WindowConstraint* window = nullptr) const {

                // Process with custom window if provided (used by FastDTW)
                if (window != nullptr) {
                    for (const auto& [i, j] : *window) {
                        int i_idx = i + 1;
                        int j_idx = j + 1;
                        D(i_idx, j_idx) = utils::compute_cell_cost<M>(
                                A[i], B[j], dim,
                                D(i_idx-1, j_idx-1),
                                D(i_idx, j_idx-1),
                                D(i_idx-1, j_idx)
                        );
                    }
                    return;
                }

                // No custom window - use selected constraint type
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
                        double ni = static_cast<double>(i-1) / n;
                        double max_nm = std::max(n, m);

                        int j_min = std::max(1, static_cast<int>(
                                std::floor((ni - R/max_nm) * m + 1)));
                        int j_max = std::min(m, static_cast<int>(
                                std::ceil((ni + R/max_nm) * m + 1)));

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
                            D(i, 1) = utils::compute_cell_cost<M>(
                                    A[i-1], B[0], dim,
                                    D(i-1, 0), D(i, 0), D(i-1, 1)
                            );
                            continue;
                        }

                        double ni = di / dn;
                        double dm = static_cast<double>(m - 1);
                        double lower_bound = std::max(ni / S, S * ni - (S - 1.0));
                        double upper_bound = std::min(S * ni, ni / S + (1.0 - 1.0/S));

                        int j_min = std::max(1, static_cast<int>(std::floor(lower_bound * dm + 1)));
                        int j_max = std::min(m, static_cast<int>(std::ceil(upper_bound * dm + 1)));

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

    } // namespace execution
} // namespace dtw_accelerator
#endif