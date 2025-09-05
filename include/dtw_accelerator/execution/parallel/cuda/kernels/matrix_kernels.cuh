#ifndef DTW_ACCELERATOR_MATRIX_KERNELS_CUH
#define DTW_ACCELERATOR_MATRIX_KERNELS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dtw_accelerator {
namespace parallel {
namespace cuda {
namespace kernels {

__global__ void init_cost_matrix(double* D, int n, int m);

} // namespace kernels
} // namespace cuda
} // namespace parallel
} // namespace dtw_accelerator

#endif // __CUDACC__
#endif // DTW_ACCELERATOR_MATRIX_KERNELS_CUH