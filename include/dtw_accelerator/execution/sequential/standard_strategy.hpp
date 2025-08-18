#ifndef DTWACCELERATOR_SEQUENTIAL_STRATEGY_HPP
#define DTWACCELERATOR_SEQUENTIAL_STRATEGY_HPP

#include "dtw_accelerator/core/dtw_concepts.hpp"
#include "dtw_accelerator/core/distance_metrics.hpp"
#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/dtw_utils.hpp"
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
            void initialize_matrix(std::vector<std::vector<double>>& D, int n, int m) const {
                initialize_matrix_impl(D, n, m);
            }

            template<distance::MetricType M>
            void execute(std::vector<std::vector<double>>& D,
                         const std::vector<std::vector<double>>& A,
                         const std::vector<std::vector<double>>& B,
                         int n, int m, int dim) const {

                for (int i = 1; i <= n; ++i) {
                    for (int j = 1; j <= m; ++j) {
                        D[i][j] = utils::compute_cell_cost<M>(
                                A[i-1].data(), B[j-1].data(), dim,
                                D[i-1][j-1], D[i][j-1], D[i-1][j]
                        );
                    }
                }
            }

            template<distance::MetricType M>
            void execute_constrained(std::vector<std::vector<double>>& D,
                                     const std::vector<std::vector<double>>& A,
                                     const std::vector<std::vector<double>>& B,
                                     const std::vector<std::pair<int, int>>& window,
                                     int n, int m, int dim) const {

                auto in_window = utils::create_window_mask(window, n, m);

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
            }

            template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
                    distance::MetricType M = distance::MetricType::EUCLIDEAN>
            void execute_with_constraint(std::vector<std::vector<double>>& D,
                                         const std::vector<std::vector<double>>& A,
                                         const std::vector<std::vector<double>>& B,
                                         int n, int m, int dim) const {

                auto constraint_mask = utils::generate_constraint_mask<CT, R, S>(n, m);

                for (int i = 1; i <= n; ++i) {
                    for (int j = 1; j <= m; ++j) {
                        if (constraint_mask[i-1][j-1]) {
                            D[i][j] = utils::compute_cell_cost<M>(
                                    A[i-1].data(), B[j-1].data(), dim,
                                    D[i-1][j-1], D[i][j-1], D[i-1][j]
                            );
                        }
                    }
                }
            }

            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result(const std::vector<std::vector<double>>& D) const {
                return extract_result_impl(D);
            }

            std::string_view name() const { return "Sequential"; }
            bool is_parallel() const { return false; }
        };



    }
}

#endif //DTWACCELERATOR_SEQUENTIAL_STRATEGY_HPP
