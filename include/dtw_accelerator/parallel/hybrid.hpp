#ifndef DTW_ACCELERATOR_HYBRID_DTW_HPP
#define DTW_ACCELERATOR_HYBRID_DTW_HPP

#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <mpi.h>
#include <omp.h>

#include "../distance_metrics.hpp"
#include "../constraints.hpp"
#include "../path_processing.hpp"
#include "../core_dtw.hpp"

namespace dtw_accelerator {
    namespace parallel {
        namespace hybrid {

            // MPI + OpenMP implementation

        } // namespace hybrid
    } // namespace parallel
} // namespace dtw_accelerator
#endif // DTW_ACCELERATOR_HYBRID_DTW_HPP