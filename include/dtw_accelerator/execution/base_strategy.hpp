#ifndef DTWACCELERATOR_BASE_STRATEGY_HPP
#define DTWACCELERATOR_BASE_STRATEGY_HPP
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
#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif


namespace dtw_accelerator {
    namespace execution {

// Base class for common functionality (CRTP pattern for static polymorphism)
        template<typename Derived>
        class BaseStrategy {
        protected:
            // Common initialization for all CPU strategies
            void initialize_matrix_impl(DoubleMatrix& D, int n, int m) const {
                D.resize(n + 1, m + 1, std::numeric_limits<double>::infinity());
                D(0, 0) = 0.0;
            }

            // Common result extraction
            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result_impl(const DoubleMatrix& D) const {
                int n = D.rows() - 1;
                int m = D.cols() - 1;
                auto path = utils::backtrack_path(D);
                return {D(n, m), path};
            }


            template<distance::MetricType M>
            void process_block(DoubleMatrix& D,
                               const DoubleTimeSeries& A,
                               const DoubleTimeSeries& B,
                               int bi, int bj, int n, int m, int dim,
                               int block_size) const {
                int i_start = bi * block_size + 1;
                int i_end = std::min((bi + 1) * block_size, n);
                int j_start = bj * block_size + 1;
                int j_end = std::min((bj + 1) * block_size, m);

                for (int i = i_start; i <= i_end; ++i) {
                    for (int j = j_start; j <= j_end; ++j) {
                        D(i, j) = utils::compute_cell_cost<M>(
                                A[i-1], B[j-1], dim,
                                D(i-1, j-1), D(i, j-1), D(i-1, j)
                        );
                    }
                }
            }

            template<distance::MetricType M>
            void process_block_constrained(DoubleMatrix& D,
                                      const DoubleTimeSeries& A,
                                      const DoubleTimeSeries& B,
                                      int bi, int bj, int n, int m, int dim,
                                      const std::vector<std::pair<int, int>>& window,
                                      int block_size) const {

                int i_start = bi * block_size;
                int i_end = std::min((bi + 1) * block_size, n);
                int j_start = bj * block_size;
                int j_end = std::min((bj + 1) * block_size, m);

                // Process only window cells that fall within this block
                for (const auto& [i, j] : window) {
                    if (i >= i_start && i < i_end && j >= j_start && j < j_end) {
                        int i_idx = i + 1;  // Convert to 1-based matrix indexing
                        int j_idx = j + 1;

                        D(i_idx, j_idx) = utils::compute_cell_cost<M>(
                                A[i], B[j], dim,
                                D(i_idx-1, j_idx-1),
                                D(i_idx, j_idx-1),
                                D(i_idx-1, j_idx)
                        );
                    }
                }
            }

            template<constraints::ConstraintType CT, int R, double S,
                    distance::MetricType M>
            void process_block_with_constraint(DoubleMatrix& D,
                                               const DoubleTimeSeries& A,
                                               const DoubleTimeSeries& B,
                                               int bi, int bj, int n, int m, int dim,
                                               int block_size) const {

                int i_start = bi * block_size + 1;
                int i_end = std::min((bi + 1) * block_size, n);
                int j_start = bj * block_size + 1;
                int j_end = std::min((bj + 1) * block_size, m);

                if constexpr (CT == constraints::ConstraintType::NONE) {
                    // No constraint - process all cells in block
                    for (int i = i_start; i <= i_end; ++i) {
                        for (int j = j_start; j <= j_end; ++j) {
                            D(i, j) = utils::compute_cell_cost<M>(
                                    A[i-1], B[j-1], dim,
                                    D(i-1, j-1), D(i, j-1), D(i-1, j)
                            );
                        }
                    }
                }
                else if constexpr (CT == constraints::ConstraintType::SAKOE_CHIBA) {
                    // Process only cells within the Sakoe-Chiba band
                    for (int i = i_start; i <= i_end; ++i) {
                        // Calculate valid j range for Sakoe-Chiba band
                        double ni = static_cast<double>(i-1) / n;
                        double max_nm = std::max(n, m);

                        int j_min_band = std::max(1, static_cast<int>(
                                std::floor((ni - static_cast<double>(R)/max_nm) * m + 1)));
                        int j_max_band = std::min(m, static_cast<int>(
                                std::ceil((ni + static_cast<double>(R)/max_nm) * m + 1)));

                        // Intersect with block boundaries
                        int j_min = std::max(j_start, j_min_band);
                        int j_max = std::min(j_end, j_max_band);

                        // Process valid cells in this row
                        for (int j = j_min; j <= j_max; ++j) {
                            D(i, j) = utils::compute_cell_cost<M>(
                                    A[i-1], B[j-1], dim,
                                    D(i-1, j-1), D(i, j-1), D(i-1, j)
                            );
                        }
                    }
                }
                else if constexpr (CT == constraints::ConstraintType::ITAKURA) {
                    // Process only cells within the Itakura parallelogram
                    for (int i = i_start; i <= i_end; ++i) {
                        double di = static_cast<double>(i - 1);
                        double dn = static_cast<double>(n - 1);

                        if (dn <= 0) {
                            // Edge case: single point
                            if (j_start == 1) {
                                D(i, 1) = utils::compute_cell_cost<M>(
                                        A[i-1], B[0], dim,
                                        D(i-1, 0), D(i, 0), D(i-1, 1)
                                );
                            }
                            continue;
                        }

                        double ni = di / dn;
                        double dm = static_cast<double>(m - 1);

                        // Calculate j bounds from Itakura constraints
                        double lower_bound = std::max(ni / S, S * ni - (S - 1.0));
                        double upper_bound = std::min(S * ni, ni / S + (1.0 - 1.0/S));

                        int j_min_para = std::max(1, static_cast<int>(
                                std::floor(lower_bound * dm + 1)));
                        int j_max_para = std::min(m, static_cast<int>(
                                std::ceil(upper_bound * dm + 1)));

                        // Intersect with block boundaries
                        int j_min = std::max(j_start, j_min_para);
                        int j_max = std::min(j_end, j_max_para);

                        // Process valid cells in this row
                        for (int j = j_min; j <= j_max; ++j) {
                            D(i, j) = utils::compute_cell_cost<M>(
                                    A[i-1], B[j-1], dim,
                                    D(i-1, j-1), D(i, j-1), D(i-1, j)
                            );
                        }
                    }
                }
            }



        public:
            // Default implementations that can be overridden
            void initialize_matrix(DoubleMatrix& D, int n, int m) const {
                initialize_matrix_impl(D, n, m);
            }

            std::pair<double, std::vector<std::pair<int, int>>>
            extract_result(const DoubleMatrix& D) const {
                return extract_result_impl(D);
            }


        };
    }
}

#endif //DTWACCELERATOR_BASE_STRATEGY_HPP