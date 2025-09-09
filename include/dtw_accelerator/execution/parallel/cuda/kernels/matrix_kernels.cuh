/**
 * @file matrix_kernels.cuh
 * @brief CUDA kernel declarations for matrix operations
 * @author UobinoPino
 * @date 2024
 *
 * This file declares CUDA kernels for matrix operations used
 * in DTW computations.
 */

#ifndef DTW_ACCELERATOR_MATRIX_KERNELS_CUH
#define DTW_ACCELERATOR_MATRIX_KERNELS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dtw_accelerator {
namespace parallel {
namespace cuda {
    /**
     * @namespace kernels
     * @brief CUDA kernel implementations
     */
    namespace kernels {
        /**
         * @brief Initialize DTW cost matrix on GPU
         * @param D Cost matrix in device memory
         * @param n Number of rows (excluding padding)
         * @param m Number of columns (excluding padding)
         *
         * Sets all elements to infinity except D[0][0] = 0.0
         * This kernel should be launched with enough threads to
         * cover the entire matrix: (n+1) * (m+1) elements.
         */
        __global__ void init_cost_matrix(double* D, int n, int m);

    } // namespace kernels
} // namespace cuda
} // namespace parallel
} // namespace dtw_accelerator

#endif // __CUDACC__
#endif // DTW_ACCELERATOR_MATRIX_KERNELS_CUH