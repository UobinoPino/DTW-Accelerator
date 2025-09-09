/**
 * @file dtw_accelerator.hpp
 * @brief Main header file for DTW Accelerator library
 * @author UobinoPino
 * @date 2024
 *
 * This is the main include file for the DTW Accelerator library.
 * It provides the complete public API including DTW algorithms,
 * execution strategies, and utility functions.
 *
 * @mainpage DTW Accelerator Documentation
 *
 * DTW Accelerator is a high-performance Dynamic Time Warping library
 * with modern C++20 concepts-based architecture supporting multiple
 * parallel backends.
 *
 * @section features Key Features
 * - Modern C++20 concepts for type-safe interfaces
 * - Multiple execution backends (Sequential, OpenMP, MPI, CUDA)
 * - FastDTW implementation for O(n) approximation
 * - Global path constraints (Sakoe-Chiba, Itakura)
 * - Multiple distance metrics (Euclidean, Manhattan, etc.)
 */

#ifndef DTW_ACCELERATOR_HPP
#define DTW_ACCELERATOR_HPP

#include <chrono>
#include "dtw_accelerator/core/distance_metrics.hpp"
#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/path_processing.hpp"
#include "dtw_accelerator/core/dtw_utils.hpp"

#include "dtw_accelerator/core/dtw_concepts.hpp"
#include "dtw_accelerator/execution/execution_strategies.hpp"
#include "dtw_accelerator/algorithms/dtw_generic.hpp"
#include "dtw_accelerator/algorithms/fastdtw_generic.hpp"

#ifdef USE_CUDA
#include "dtw_accelerator/execution/parallel/cuda/cuda_dtw.hpp"
#endif



namespace dtw_accelerator {

// ============================================================================
// Public API - Concepts-based interface
// ============================================================================

    /// @brief Re-export distance metric types
    using distance::MetricType;

    /// @brief Re-export constraint types
    using constraints::ConstraintType;

    /// @brief Type alias for window constraints
    using WindowConstraint = std::vector<std::pair<int, int>>;

    /// @brief Re-export double-precision time series
    using DoubleTimeSeries = TimeSeries<double>;

    /// @brief Re-export float-precision time series
    using FloatTimeSeries = TimeSeries<float>;

    /**
     * @namespace strategies
     * @brief Execution strategy types
     */
    namespace strategies {
        using execution::SequentialStrategy;
        using execution::BlockedStrategy;
        using execution::AutoStrategy;

    #ifdef USE_OPENMP
            using execution::OpenMPStrategy;
    #endif

    #ifdef USE_MPI
            using execution::MPIStrategy;
    #endif

    #ifdef USE_CUDA
            using execution::CUDAStrategy;
    #endif
    }

// ============================================================================
// High-level convenience functions
// ============================================================================

    /**
     * @brief Simple DTW with automatic backend selection
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @return Pair of (DTW distance, optimal warping path)
     */
    template<MetricType M = MetricType::EUCLIDEAN>
    inline auto dtw(const DoubleTimeSeries& A,
                    const DoubleTimeSeries& B) {
        return dtw_auto<M>(A, B);
    }

    /**
     * @brief FastDTW with automatic backend selection
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @param radius Search window radius
     * @param min_size Minimum size for recursion
     * @return Pair of (approximate DTW distance, warping path)
     */
    template<MetricType M = MetricType::EUCLIDEAN>
    inline auto fastdtw(const DoubleTimeSeries& A,
                        const DoubleTimeSeries& B,
                        int radius = 1,
                        int min_size = 100) {
        return fastdtw_auto<M>(A, B, radius, min_size);
    }

// ============================================================================
// Backend-specific interfaces
// ============================================================================

    /**
     * @namespace cpu
     * @brief CPU-based DTW implementations
     */
    namespace cpu {
        /**
         * @brief Sequential CPU implementation
         */
        template<MetricType M = MetricType::EUCLIDEAN>
        inline auto dtw(const DoubleTimeSeries& A,
                        const DoubleTimeSeries& B) {
            return dtw_sequential<M>(A, B);
        }

        /**
         * @brief Cache-optimized blocked implementation
         */
        template<MetricType M = MetricType::EUCLIDEAN>
        inline auto dtw_blocked(const DoubleTimeSeries& A,
                                const DoubleTimeSeries& B,
                                int block_size = 64) {
            return dtw_accelerator::dtw_blocked<M>(A, B, block_size);
        }
    }

    #ifdef USE_OPENMP
        /**
         * @namespace openmp
         * @brief OpenMP parallel DTW implementations
         */
        namespace openmp {
        /**
         * @brief OpenMP parallel DTW
         */
        template<MetricType M = MetricType::EUCLIDEAN>
        inline auto dtw(const DoubleTimeSeries& A,
                       const DoubleTimeSeries& B,
                       int num_threads = 0,
                       int block_size = 64) {
            return dtw_openmp<M>(A, B, num_threads, block_size);
        }

        /**
         * @brief OpenMP parallel FastDTW
         */
        template<MetricType M = MetricType::EUCLIDEAN>
        inline auto fastdtw(const DoubleTimeSeries& A,
                           const DoubleTimeSeries& B,
                           int radius = 1,
                           int min_size = 100,
                           int num_threads = 0,
                           int block_size = 64) {
            return fastdtw_openmp<M>(A, B, radius, min_size, num_threads, block_size);
        }
    }
    #endif

    #ifdef USE_MPI

        /**
         * @namespace mpi
         * @brief MPI distributed DTW implementations
         */
        namespace mpi {
        /**
         * @brief MPI distributed DTW
         */
        template<MetricType M = MetricType::EUCLIDEAN>
        inline auto dtw(const DoubleTimeSeries& A,
                       const DoubleTimeSeries& B,
                       int block_size = 64,
                       int threads_per_process = 0,
                       MPI_Comm comm = MPI_COMM_WORLD) {
            return dtw_mpi<M>(A, B, block_size, threads_per_process, comm);
        }
    }
    #endif

    #ifdef USE_CUDA

        /**
         * @namespace cuda
         * @brief CUDA GPU-accelerated DTW implementations
         */
        namespace cuda {
        using parallel::cuda::dtw_cuda;
        using parallel::cuda::CUDAStrategy;

        /**
         * @brief Check CUDA availability
         * @return True if CUDA devices are available
         */
        inline bool is_available() {
            return CUDAStrategy::is_available();
        }
        /**
         * @brief Get CUDA device information
         * @return String describing available CUDA devices
         */
        inline std::string device_info() {
            return CUDAStrategy::get_device_info();
        }
    }
    #endif

// ============================================================================
// Advanced interface with custom strategies
// ============================================================================

    /**
     * @brief DTW with custom execution strategy
     * @tparam M Distance metric type
     * @tparam Strategy Custom strategy type
     * @param A First time series
     * @param B Second time series
     * @param strategy Custom execution strategy
     * @return Pair of (DTW distance, optimal warping path)
     */
    template<MetricType M = MetricType::EUCLIDEAN, typename Strategy>
    requires concepts::ExecutionStrategy<Strategy>
    inline auto dtw_custom(const DoubleTimeSeries& A,
                           const DoubleTimeSeries& B,
                           Strategy&& strategy) {
        return dtw<M>(A, B, std::forward<Strategy>(strategy));
    }



} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_HPP