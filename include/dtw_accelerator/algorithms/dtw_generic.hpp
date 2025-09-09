/**
 * @file dtw_generic.hpp
 * @brief Generic DTW algorithm implementation with concepts
 * @author UobinoPino
 * @date 2024
 *
 * This file contains the main DTW algorithm that works with any
 * execution strategy satisfying the ExecutionStrategy concept.
 */

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

    /**
    * @brief Generic DTW algorithm for any execution strategy
    * @tparam M Distance metric type
    * @tparam CT Constraint type
    * @tparam R Sakoe-Chiba band radius
    * @tparam S Itakura parallelogram slope
    * @tparam Strategy Execution strategy type (must satisfy ExecutionStrategy concept)
    * @param A First time series
    * @param B Second time series
    * @param strategy Execution strategy instance
    * @param window Optional window constraint
    * @return Pair of (DTW distance, optimal warping path)
    *
    * This is the main entry point for DTW computation. It accepts any
    * execution strategy that satisfies the ExecutionStrategy concept,
    * enabling seamless switching between sequential, parallel, and
    * accelerated implementations.
    */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN,
            constraints::ConstraintType CT = constraints::ConstraintType::NONE,
            int R = 1, double S = 2.0,
            concepts::ExecutionStrategy Strategy>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            Strategy&& strategy,
            const execution::WindowConstraint* window = nullptr) {

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

        // Execute DTW algorithm with unified method
        strategy.template execute_with_constraint<CT, R, S, M>(
                D, A, B, n, m, dim, window);

        // Extract result and path
        return strategy.extract_result(D);
    }

    /**
     * @brief Convenience function for unconstrained DTW
     * @tparam M Distance metric type
     * @tparam Strategy Execution strategy type
     * @param A First time series
     * @param B Second time series
     * @param strategy Execution strategy instance
     * @return Pair of (DTW distance, optimal warping path)
     */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN,
            concepts::ExecutionStrategy Strategy>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_unconstrained(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            Strategy&& strategy) {
        return dtw<M, constraints::ConstraintType::NONE>(
                A, B, std::forward<Strategy>(strategy), nullptr);
    }

    /**
     * @brief Convenience function for windowed DTW (used by FastDTW)
     * @tparam M Distance metric type
     * @tparam Strategy Execution strategy type
     * @param A First time series
     * @param B Second time series
     * @param window Window constraint defining valid cells
     * @param strategy Execution strategy instance
     * @return Pair of (DTW distance, optimal warping path)
     */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN,
            concepts::ExecutionStrategy Strategy>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_windowed(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            const execution::WindowConstraint& window,
            Strategy&& strategy) {
        return dtw<M, constraints::ConstraintType::NONE>(
                A, B, std::forward<Strategy>(strategy), &window);
    }

    /**
     * @brief Convenience function for Sakoe-Chiba constrained DTW
     * @tparam M Distance metric type
     * @tparam R Band radius
     * @tparam Strategy Execution strategy type
     * @param A First time series
     * @param B Second time series
     * @param strategy Execution strategy instance
     * @return Pair of (DTW distance, optimal warping path)
     */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN,
            int R = 1,
            concepts::ExecutionStrategy Strategy>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_sakoe_chiba(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            Strategy&& strategy) {
        return dtw<M, constraints::ConstraintType::SAKOE_CHIBA, R>(
                A, B, std::forward<Strategy>(strategy), nullptr);
    }

    /**
     * @brief Convenience function for Itakura constrained DTW
     * @tparam M Distance metric type
     * @tparam S Maximum slope parameter
     * @tparam Strategy Execution strategy type
     * @param A First time series
     * @param B Second time series
     * @param strategy Execution strategy instance
     * @return Pair of (DTW distance, optimal warping path)
     */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN,
            double S = 2.0,
            concepts::ExecutionStrategy Strategy>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_itakura(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            Strategy&& strategy) {
        return dtw<M, constraints::ConstraintType::ITAKURA, 1, S>(
                A, B, std::forward<Strategy>(strategy), nullptr);
    }

    /**
     * @brief Sequential DTW computation
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @return Pair of (DTW distance, optimal warping path)
     */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_sequential(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B) {
        return dtw_unconstrained<M>(A, B, execution::SequentialStrategy{});
    }

    /**
     * @brief Blocked DTW computation for better cache locality
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @param block_size Size of cache-friendly blocks
     * @return Pair of (DTW distance, optimal warping path)
     */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_blocked(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int block_size = 64) {
        return dtw_unconstrained<M>(A, B, execution::BlockedStrategy{block_size});
    }

#ifdef USE_OPENMP
    /**
     * @brief OpenMP parallel DTW computation
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @param num_threads Number of OpenMP threads (0 for auto)
     * @param block_size Size of blocks for wavefront processing
     * @return Pair of (DTW distance, optimal warping path)
     */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_openmp(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int num_threads = 0,
            int block_size = 64) {
        return dtw_unconstrained<M>(A, B, execution::OpenMPStrategy{num_threads, block_size});
    }
#endif

#ifdef USE_MPI
    /**
     * @brief MPI distributed DTW computation
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @param block_size Size of blocks for distribution
     * @param threads_per_process OpenMP threads per MPI process
     * @param comm MPI communicator
     * @return Pair of (DTW distance, optimal warping path)
     */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_mpi(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int block_size = 64,
            int threads_per_process = 0,
            MPI_Comm comm = MPI_COMM_WORLD) {
        return dtw_unconstrained<M>(A, B, execution::MPIStrategy{block_size, threads_per_process, comm});
    }
#endif

    /**
    * @brief Auto-selecting DTW with optimal backend selection
    * @tparam M Distance metric type
    * @param A First time series
    * @param B Second time series
    * @return Pair of (DTW distance, optimal warping path)
    *
    * Automatically selects the best execution strategy based on
    * problem size and available hardware accelerators.
    */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_auto(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B) {

        int n = A.size();
        int m = B.size();

        // Use CPU auto-strategy
        return dtw_unconstrained<M>(A, B, execution::AutoStrategy{n, m});
    }

} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_DTW_ALGORITHM_HPP