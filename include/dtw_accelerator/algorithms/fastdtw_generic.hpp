#ifndef DTWACCELERATOR_FASTDTW_GENERIC_HPP
#define DTWACCELERATOR_FASTDTW_GENERIC_HPP

#include "dtw_accelerator/core/dtw_concepts.hpp"
#include "dtw_accelerator/execution/execution_strategies.hpp"
#include "dtw_accelerator/core/path_processing.hpp"
#include "dtw_accelerator/core/matrix.hpp"
#include <vector>
#include <utility>
#include <type_traits>

namespace dtw_accelerator {


// FastDTW implementation using concepts
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN,
            concepts::ExecutionStrategy Strategy>
    inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int radius,
            int min_size,
            Strategy&& strategy) {

        int n = A.size();
        int m = B.size();

        // Base case: small sequences use standard DTW
        if (n <= min_size && m <= min_size) {
            return dtw_unconstrained<M>(A, B, std::forward<Strategy>(strategy));
        }

        // Base case: can't downsample further
        if (n <= 2 || m <= 2) {
            return dtw_unconstrained<M>(A, B, std::forward<Strategy>(strategy));
        }

        // Recursive case:
        // 1. Coarsen the time series
        auto A_coarse = path::downsample(A);
        auto B_coarse = path::downsample(B);

        // 2. Recursively compute FastDTW on coarsened data
        auto [cost_coarse, path_coarse] = fastdtw<M>(
                A_coarse, B_coarse, radius, min_size, std::forward<Strategy>(strategy)
        );

        // 3. Project the low-resolution path to higher resolution
        auto projected_path = path::expand_path(path_coarse, n, m);

        // 4. Create search window around projected path
        auto window = path::get_window(projected_path, n, m, radius);

        // 5. Compute constrained DTW within window using unified interface
        return dtw_windowed<M>(A, B, window, std::forward<Strategy>(strategy));
    }

// Convenience functions with default strategies

// Sequential FastDTW
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_sequential(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int radius = 1,
            int min_size = 100) {
        return fastdtw<M>(A, B, radius, min_size, execution::SequentialStrategy{});
    }

// Blocked FastDTW
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_blocked(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int radius = 1,
            int min_size = 100,
            int block_size = 64) {
        return fastdtw<M>(A, B, radius, min_size, execution::BlockedStrategy{block_size});
    }

#ifdef USE_OPENMP
    // OpenMP FastDTW
template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_openmp(
        const DoubleTimeSeries& A,
        const DoubleTimeSeries& B,
        int radius = 1,
        int min_size = 100,
        int num_threads = 0,
        int block_size = 64) {
    return fastdtw<M>(A, B, radius, min_size,
                      execution::OpenMPStrategy{num_threads, block_size});
}
#endif

#ifdef USE_MPI
    // MPI FastDTW
template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_mpi(
        const DoubleTimeSeries& A,
        const DoubleTimeSeries& B,
        int radius = 1,
        int min_size = 100,
        int block_size = 64,
        int threads_per_process = 0,
        MPI_Comm comm = MPI_COMM_WORLD) {
    return fastdtw<M>(A, B, radius, min_size,
                      execution::MPIStrategy{block_size, threads_per_process, comm});
}
#endif

// Auto-selecting FastDTW
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_auto(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int radius = 1,
            int min_size = 100) {

        int n = A.size();
        int m = B.size();

        // CUDA check for large sequences
#ifdef USE_CUDA
        if (n >= 1000 && m >= 1000) {
        // Delegate to specialized CUDA implementation if available
        // return parallel::cuda::fastdtw_cuda<M>(A, B, radius, min_size);
    }
#endif

        // Use CPU auto-strategy for everything else
        return fastdtw<M>(A, B, radius, min_size, execution::AutoStrategy{n, m});
    }

} // namespace dtw_accelerator

#endif //DTWACCELERATOR_FASTDTW_GENERIC_HPP