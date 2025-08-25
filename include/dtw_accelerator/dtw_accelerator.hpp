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
#include "dtw_accelerator/algorithms/dtw_constrained_generic.hpp"
#include "dtw_accelerator/algorithms/dtw_with_constraint_generic.hpp"


// CUDA integration (not yet available)
#ifdef USE_CUDA
#include "dtw_accelerator/execution/parallel/cuda/cuda_strategy.hpp"
#endif



namespace dtw_accelerator {

// ============================================================================
// Public API - Concepts-based interface
// ============================================================================

// Re-export main types
    using distance::MetricType;
    using constraints::ConstraintType;

    // Re-export TimeSeries types
    using DoubleTimeSeries = TimeSeries<double>;
    using FloatTimeSeries = TimeSeries<float>;

// Re-export execution strategies
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

// Simple DTW with automatic backend selection
    template<MetricType M = MetricType::EUCLIDEAN>
    inline auto dtw(const DoubleTimeSeries& A,
                    const DoubleTimeSeries& B) {
        return dtw_auto<M>(A, B);
    }

// FastDTW with automatic backend selection
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

    namespace cpu {
        // Sequential implementation
        template<MetricType M = MetricType::EUCLIDEAN>
        inline auto dtw(const DoubleTimeSeries& A,
                        const DoubleTimeSeries& B) {
            return dtw_sequential<M>(A, B);
        }

        // Blocked implementation
        template<MetricType M = MetricType::EUCLIDEAN>
        inline auto dtw_blocked(const DoubleTimeSeries& A,
                                const DoubleTimeSeries& B,
                                int block_size = 64) {
            return dtw_accelerator::dtw_blocked<M>(A, B, block_size);
        }
    }

#ifdef USE_OPENMP
    namespace openmp {
    template<MetricType M = MetricType::EUCLIDEAN>
    inline auto dtw(const DoubleTimeSeries& A,
                   const DoubleTimeSeries& B,
                   int num_threads = 0,
                   int block_size = 64) {
        return dtw_openmp<M>(A, B, num_threads, block_size);
    }

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
    namespace mpi {
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
    namespace cuda {
    using parallel::cuda::dtw_cuda;
    using parallel::cuda::dtw_constrained_cuda;
    using parallel::cuda::fastdtw_cuda;
    using parallel::cuda::CUDAStrategy;

    // Helper to check CUDA availability
    inline bool is_available() {
        return CUDAStrategy::is_available();
    }

    inline std::string device_info() {
        return CUDAStrategy::get_device_info();
    }
}
#endif

// ============================================================================
// Advanced interface with custom strategies
// ============================================================================

    template<MetricType M = MetricType::EUCLIDEAN, typename Strategy>
    requires concepts::ExecutionStrategy<Strategy>
    inline auto dtw_custom(const DoubleTimeSeries& A,
                           const DoubleTimeSeries& B,
                           Strategy&& strategy) {
        return dtw<M>(A, B, std::forward<Strategy>(strategy));
    }

    template<MetricType M = MetricType::EUCLIDEAN, typename Strategy>
    requires concepts::ConstrainedExecutionStrategy<Strategy>
    inline auto dtw_constrained_custom(const DoubleTimeSeries& A,
                                       const DoubleTimeSeries& B,
                                       const std::vector<std::pair<int, int>>& window,
                                       Strategy&& strategy) {
        return dtw_constrained<M>(A, B, window, std::forward<Strategy>(strategy));
    }

// ============================================================================
// Constraint-based DTW
// ============================================================================

    template<ConstraintType CT, int R = 1, double S = 2.0,
            MetricType M = MetricType::EUCLIDEAN, typename Strategy>
    inline auto dtw_with_constraint(const DoubleTimeSeries& A,
                                    const DoubleTimeSeries& B,
                                    Strategy&& strategy) {
        // Use auto-strategy by default
        return dtw_with_constraint<CT, R, S, M>(A, B, std::forward<Strategy>(strategy));
    }



// ============================================================================
// Utility functions
// ============================================================================

// Get optimal strategy for given problem size
    inline std::string recommend_strategy(size_t n, size_t m) {
#ifdef USE_CUDA
        if (n >= 1000 && m >= 1000 && cuda::is_available()) {
        return "CUDA";
    }
#endif

#ifdef USE_MPI
        int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized && n >= 500 && m >= 500) {
        return "MPI";
    }
#endif

#ifdef USE_OPENMP
        if (n >= 100 && m >= 100) {
        return "OpenMP";
    }
#endif

        if (n >= 50 && m >= 50) {
            return "Blocked";
        }

        return "Sequential";
    }

// Performance benchmarking helper
    template<typename Func>
    auto benchmark_dtw(Func&& func, int iterations = 10) {
        using clock = std::chrono::high_resolution_clock;

        auto start = clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        auto end = clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        return duration.count() / static_cast<double>(iterations);
    }

} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_HPP