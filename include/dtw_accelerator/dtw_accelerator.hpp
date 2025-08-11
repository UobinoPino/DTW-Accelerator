#ifndef DTW_ACCELERATOR_DTW_ACCELERATOR_HPP
#define DTW_ACCELERATOR_DTW_ACCELERATOR_HPP

// Include all components
#include "distance_metrics.hpp"
#include "constraints.hpp"
#include "path_processing.hpp"
#include "core_dtw.hpp"
#include "fast_dtw.hpp"

// Include parallel implementations if available
#ifdef USE_OPENMP
#include "dtw_accelerator/parallel/openmp_dtw.hpp"
#endif

#ifdef USE_MPI
#include "dtw_accelerator/parallel/mpi_dtw.hpp"
#endif

#ifdef USE_CUDA

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

#endif

#ifdef USE_MPI
    // Re-export MPI functions
    namespace parallel {
        namespace mpi {
            using dtw_accelerator::parallel::mpi::dtw_mpi;

        }
    }
#endif

#ifdef USE_CUDA

#endif
}
#endif //DTW_ACCELERATOR_DTW_ACCELERATOR_HPP