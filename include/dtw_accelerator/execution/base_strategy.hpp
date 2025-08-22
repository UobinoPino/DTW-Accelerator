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
        };
    }
}

#endif //DTWACCELERATOR_BASE_STRATEGY_HPP