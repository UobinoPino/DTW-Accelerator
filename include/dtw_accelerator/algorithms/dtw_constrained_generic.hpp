#ifndef DTWACCELERATOR_DTW_CONSTRAINED_GENERIC_HPP
#define DTWACCELERATOR_DTW_CONSTRAINED_GENERIC_HPP

#include "dtw_accelerator/core/dtw_concepts.hpp"
#include "dtw_accelerator/execution/execution_strategies.hpp"
#include "dtw_accelerator/core/path_processing.hpp"
#include <vector>
#include <utility>
#include <type_traits>

namespace dtw_accelerator {

// Generic constrained DTW algorithm
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN,
            concepts::ConstrainedExecutionStrategy Strategy>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_constrained(
            const std::vector<std::vector<double>>& A,
            const std::vector<std::vector<double>>& B,
            const std::vector<std::pair<int, int>>& window,
            Strategy&& strategy) {

        int n = A.size();
        int m = B.size();
        int dim = A.empty() ? 0 : A[0].size();

        // Validate input
        if (n == 0 || m == 0 || window.empty()) {
            return {0.0, {}};
        }

        // Initialize DTW matrix
        std::vector<std::vector<double>> D;
        strategy.initialize_matrix(D, n, m);

        // Execute constrained DTW algorithm
        strategy.template execute_constrained<M>(D, A, B, window, n, m, dim);

        // Extract result and path
        return strategy.extract_result(D);
    }

} // namespace dtw_accelerator
#endif //DTWACCELERATOR_DTW_CONSTRAINED_GENERIC_HPP
