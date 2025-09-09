/**
 * @file fastdtw_generic.hpp
 * @brief FastDTW algorithm implementation with concepts
 * @author UobinoPino
 * @date 2024
 *
 * This file contains the FastDTW algorithm implementation that provides
 * an O(n) approximation to standard DTW through recursive coarsening
 * and refinement with constrained search windows.
 */

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


    /**
     * @brief FastDTW algorithm implementation using execution strategies
     * @tparam M Distance metric type
     * @tparam Strategy Execution strategy type (must satisfy ExecutionStrategy concept)
     * @param A First time series
     * @param B Second time series
     * @param radius Search radius for refinement at each level
     * @param min_size Minimum size to stop recursion and use standard DTW
     * @param strategy Execution strategy instance
     * @return Pair of (approximate DTW distance, warping path)
     *
     * FastDTW provides an O(n) approximation to DTW by:
     * 1. Recursively downsampling the input series
     * 2. Computing DTW on the coarsest level
     * 3. Projecting the path to finer resolutions
     * 4. Refining within a constrained window
     *
     * The radius parameter controls the trade-off between accuracy
     * and computational cost. Larger radius values give more accurate
     * results but increase computation time.
     */
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

        // 5. Compute constrained DTW within window
        return dtw_windowed<M>(A, B, window, std::forward<Strategy>(strategy));
    }


    /**
     * @brief Sequential FastDTW with default parameters
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @param radius Search radius (default: 1)
     * @param min_size Minimum size for recursion (default: 100)
     * @return Pair of (approximate DTW distance, warping path)
     */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_sequential(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int radius = 1,
            int min_size = 100) {
        return fastdtw<M>(A, B, radius, min_size, execution::SequentialStrategy{});
    }

    /**
     * @brief Blocked FastDTW for improved cache performance
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @param radius Search radius (default: 1)
     * @param min_size Minimum size for recursion (default: 100)
     * @param block_size Cache block size (default: 64)
     * @return Pair of (approximate DTW distance, warping path)
     */
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
    /**
     * @brief OpenMP parallel FastDTW
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @param radius Search radius (default: 1)
     * @param min_size Minimum size for recursion (default: 100)
     * @param num_threads Number of OpenMP threads (0 for auto)
     * @param block_size Cache block size (default: 64)
     * @return Pair of (approximate DTW distance, warping path)
     */
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
    /**
     * @brief MPI distributed FastDTW
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @param radius Search radius (default: 1)
     * @param min_size Minimum size for recursion (default: 100)
     * @param block_size Distribution block size (default: 64)
     * @param threads_per_process OpenMP threads per MPI process
     * @param comm MPI communicator
     * @return Pair of (approximate DTW distance, warping path)
     */
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

    /**
    * @brief Auto-selecting FastDTW with optimal backend
    * @tparam M Distance metric type
    * @param A First time series
    * @param B Second time series
    * @param radius Search radius (default: 1)
    * @param min_size Minimum size for recursion (default: 100)
    * @return Pair of (approximate DTW distance, warping path)
    *
    * Automatically selects the best execution strategy based on
    * problem size and available hardware accelerators.
    */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_auto(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int radius = 1,
            int min_size = 100) {

        int n = A.size();
        int m = B.size();

        // Use CPU auto-strategy
        return fastdtw<M>(A, B, radius, min_size, execution::AutoStrategy{n, m});
    }

} // namespace dtw_accelerator

#endif //DTWACCELERATOR_FASTDTW_GENERIC_HPP