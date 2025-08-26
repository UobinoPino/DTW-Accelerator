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
                                           const BoolMatrix& mask,
                                           int block_size) const {
                int i_start = bi * block_size + 1;
                int i_end = std::min((bi + 1) * block_size, n);
                int j_start = bj * block_size + 1;
                int j_end = std::min((bj + 1) * block_size, m);

                for (int i = i_start; i <= i_end; ++i) {
                    for (int j = j_start; j <= j_end; ++j) {
                        if (i-1 < n && j-1 < m && mask(i-1, j-1)) {
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