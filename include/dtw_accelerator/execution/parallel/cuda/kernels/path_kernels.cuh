#ifndef DTW_ACCELERATOR_PATH_KERNELS_CUH
#define DTW_ACCELERATOR_PATH_KERNELS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dtw_accelerator {
namespace parallel {
namespace cuda {
namespace kernels {

__global__ void backtrack_path(
    const double* __restrict__ D,
    int* __restrict__ path_i,
    int* __restrict__ path_j,
    int* __restrict__ path_length,
    int n, int m);

__global__ void reverse_path(
    int* __restrict__ path_i,
    int* __restrict__ path_j,
    int path_length);

} // namespace kernels
} // namespace cuda
} // namespace parallel
} // namespace dtw_accelerator

#endif // __CUDACC__
#endif // DTW_ACCELERATOR_PATH_KERNELS_CUH