#ifndef DTW_ACCELERATOR_DTW_ALGORITHM_HPP
#define DTW_ACCELERATOR_DTW_ALGORITHM_HPP

#include "dtw_accelerator/core/dtw_concepts.hpp"
#include "dtw_accelerator/execution/execution_strategies.hpp"
#include "dtw_accelerator/core/path_processing.hpp"
#include "dtw_accelerator/core/matrix.hpp"
#include <vector>
#include <utility>
#include <type_traits>

namespace dtw_accelerator {

// Generic DTW algorithm for any execution strategy satisfying the concept
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN,
            concepts::ExecutionStrategy Strategy>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            Strategy&& strategy) {

        static_assert(concepts::supports_metric_v<std::decay_t<Strategy>, M>,
                      "Strategy does not support the specified distance metric");

        int n = A.size();
        int m = B.size();
        int dim = A.dimensions();

        // Validate input
        if (n == 0 || m == 0) {
            return {0.0, {}};
        }

        // Initialize DTW matrix
        DoubleMatrix D;
        strategy.initialize_matrix(D, n, m);

        // Execute DTW algorithm
        strategy.template execute<M>(D, A, B, n, m, dim);

        // Extract result and path
        return strategy.extract_result(D);
    }

// Convenience functions with default strategies

// Sequential DTW
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_sequential(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B) {
        return dtw<M>(A, B, execution::SequentialStrategy{});
    }

// Blocked DTW
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_blocked(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int block_size = 64) {
        return dtw<M>(A, B, execution::BlockedStrategy{block_size});
    }

#ifdef USE_OPENMP
    // OpenMP DTW
template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
inline std::pair<double, std::vector<std::pair<int, int>>> dtw_openmp(
        const DoubleTimeSeries& A,
        const DoubleTimeSeries& B,
        int num_threads = 0,
        int block_size = 64) {
    return dtw<M>(A, B, execution::OpenMPStrategy{num_threads, block_size});
}
#endif

#ifdef USE_MPI
    // MPI DTW
template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
inline std::pair<double, std::vector<std::pair<int, int>>> dtw_mpi(
        const DoubleTimeSeries& A,
        const DoubleTimeSeries& B,
        int block_size = 64,
        int threads_per_process = 0,
        MPI_Comm comm = MPI_COMM_WORLD) {
    return dtw<M>(A, B, execution::MPIStrategy{block_size, threads_per_process, comm});
}
#endif

// Auto-selecting DTW
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_auto(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B) {

        int n = A.size();
        int m = B.size();

        // CUDA check for large sequences
#ifdef USE_CUDA
        if (n >= 1000 && m >= 1000) {
        // Delegate to specialized CUDA implementation
       // return parallel::cuda::dtw_cuda<M>(A, B);
    }
#endif

        // Use CPU auto-strategy for everything else
        return dtw<M>(A, B, execution::AutoStrategy{n, m});
    }


} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_DTW_ALGORITHM_HPP