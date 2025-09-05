#ifndef DTW_ACCELERATOR_DTW_CORE_KERNELS_CUH
#define DTW_ACCELERATOR_DTW_CORE_KERNELS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../../../core/distance_metrics.hpp"

namespace dtw_accelerator {
namespace parallel {
namespace cuda {
namespace kernels {

using distance::MetricType;

template<MetricType M, int TILE_SIZE = 64>
__global__ void dtw_tile_wavefront(
    double* __restrict__ D,
    const double* __restrict__ A,
    const double* __restrict__ B,
    int n, int m, int dim,
    int wave, int n_tiles, int m_tiles);

} // namespace kernels
} // namespace cuda
} // namespace parallel
} // namespace dtw_accelerator

#endif // __CUDACC__
#endif // DTW_ACCELERATOR_DTW_CORE_KERNELS_CUH