#ifndef DTWACCELERATOR_DTW_WITH_CONSTRAINT_ALGORITHM_HPP
#define DTWACCELERATOR_DTW_WITH_CONSTRAINT_ALGORITHM_HPP

#include "dtw_accelerator/core/dtw_concepts.hpp"
#include "dtw_accelerator/execution/execution_strategies.hpp"
#include "dtw_accelerator/core/path_processing.hpp"
#include "dtw_accelerator/core/matrix.hpp"
#include <vector>
#include <utility>
#include <type_traits>

namespace dtw_accelerator {


// Generic DTW with templated constraints
    template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
            distance::MetricType M = distance::MetricType::EUCLIDEAN,
            concepts::TemplatedConstraintStrategy Strategy>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_with_constraint(
            const std::vector<std::vector<double>>& A,
            const std::vector<std::vector<double>>& B,
            Strategy&& strategy) {

        int n = A.size();
        int m = B.size();
        int dim = A.empty() ? 0 : A[0].size();

        // Validate input
        if (n == 0 || m == 0) {
            return {0.0, {}};
        }

        // Initialize DTW matrix
        DoubleMatrix D;
        strategy.initialize_matrix(D, n, m);

        // Execute DTW with constraint
        strategy.template execute_with_constraint<CT, R, S, M>(D, A, B, n, m, dim);

        // Extract result and path
        return strategy.extract_result(D);
    }


} // namespace dtw_accelerator
#endif //DTWACCELERATOR_DTW_WITH_CONSTRAINT_ALGORITHM_HPP