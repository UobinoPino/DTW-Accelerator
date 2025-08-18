#ifndef DTW_ACCELERATOR_DTW_ACCELERATOR_HPP
#define DTW_ACCELERATOR_DTW_ACCELERATOR_HPP

// Include all components
#include "dtw_accelerator/core/distance_metrics.hpp"
#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/path_processing.hpp"
#include "dtw_accelerator/core/dtw_utils.hpp"
#include "dtw_accelerator/execution/sequential/core_dtw.hpp"
#include "dtw_accelerator/execution/sequential/fast_dtw.hpp"
#include "dtw_accelerator.hpp"

// Include parallel implementations if available
#ifdef USE_OPENMP
#include "dtw_accelerator/parallel/openmp_dtw.hpp"
#endif

#ifdef USE_MPI
#include "dtw_accelerator/parallel/mpi_dtw.hpp"
#endif

#ifdef USE_CUDA
#include "parallel/cuda_dtw.cuh"
#endif


// Namespace wrapper for all components
namespace dtw_accelerator {
    // Re-export key types and functions for convenience
    using constraints::ConstraintType;

    using core::dtw_cpu;
    using core::dtw_constrained;
    using core::dtw_with_constraint;

    using fast::fastdtw_cpu;

#ifdef USE_OPENMP
    // Re-export OpenMP functions
    namespace parallel {
        namespace omp {
            using dtw_accelerator::parallel::omp::dtw_omp;
            using dtw_accelerator::parallel::omp::dtw_omp_with_constraint;
            using dtw_accelerator::parallel::omp::dtw_constrained_omp;
            using dtw_accelerator::parallel::omp::fastdtw_omp;
            using dtw_accelerator::parallel::omp::dtw_omp_blocked;
        }
    }

#endif

#ifdef USE_MPI
    // Re-export MPI functions
    namespace parallel {
        namespace mpi {
            using dtw_accelerator::parallel::mpi::dtw_mpi;
            using dtw_accelerator::parallel::mpi::dtw_with_constraint_mpi;
            using dtw_accelerator::parallel::mpi::dtw_constrained_mpi;
//            using dtw_accelerator::parallel::mpi::fastdtw_mpi;


        }
    }
#endif

#ifdef USE_CUDA
    // Re-export CUDA functions
    namespace parallel {
        namespace cuda {
            using dtw_accelerator::parallel::cuda::dtw_cuda;
            using dtw_accelerator::parallel::cuda::dtw_cuda_with_constraint;
            using dtw_accelerator::parallel::cuda::dtw_constrained_cuda;
            using dtw_accelerator::parallel::cuda::fastdtw_cuda;
        }
    }

#endif
    // Unified interface for automatic backend selection
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_auto(
            const std::vector<std::vector<double>>& A,
            const std::vector<std::vector<double>>& B) {

        int n = A.size();
        int m = B.size();

#ifdef USE_CUDA
        // Use CUDA for large sequences
        if (n >= 1000 && m >= 1000) {
            return parallel::cuda::dtw_cuda<M>(A, B);
        }
#endif

#ifdef USE_OPENMP
        // Use OpenMP for medium sequences
        if (n >= 100 && m >= 100) {
            return parallel::omp::dtw_omp<M>(A, B);
        }
#endif

        // Fall back to sequential CPU
        return core::dtw_cpu<M>(A, B);
    }

    // Unified FastDTW interface
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_auto(
            const std::vector<std::vector<double>>& A,
            const std::vector<std::vector<double>>& B,
            int radius = 1,
            int min_size = 100) {

        int n = A.size();
        int m = B.size();

#ifdef USE_CUDA
        // Use CUDA for large sequences
        if (n >= 1000 && m >= 1000) {
            return parallel::cuda::fastdtw_cuda<M>(A, B, radius, min_size);
        }
#endif

#ifdef USE_OPENMP
        // Use OpenMP for medium sequences
        if (n >= 100 && m >= 100) {
            return parallel::omp::fastdtw_omp(A, B, radius, min_size);
        }
#endif

        // Fall back to sequential CPU
        return fast::fastdtw_cpu(A, B, radius, min_size);
    }
}
#endif //DTW_ACCELERATOR_DTW_ACCELERATOR_HPP