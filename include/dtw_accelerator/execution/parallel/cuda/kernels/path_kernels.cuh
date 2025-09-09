/**
 * @file path_kernels.cuh
 * @brief CUDA kernel declarations for optimal path extraction
 * @author UobinoPino
 * @date 2024
 *
 * This file declares CUDA kernels for extracting and processing
 * the optimal warping path from DTW cost matrices.
 */

#ifndef DTW_ACCELERATOR_PATH_KERNELS_CUH
#define DTW_ACCELERATOR_PATH_KERNELS_CUH

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
         * @brief Extract optimal warping path through backtracking
         * @param D Cost matrix in device memory
         * @param path_i Output array for path row indices
         * @param path_j Output array for path column indices
         * @param path_length Output for actual path length
         * @param n Number of rows (excluding padding)
         * @param m Number of columns (excluding padding)
         *
         * Performs sequential backtracking to find the optimal path.
         * Should be launched with a single thread (<<<1, 1>>>).
         * The path is initially stored in reverse order.
         */
        __global__ void backtrack_path(
            const double* __restrict__ D,
            int* __restrict__ path_i,
            int* __restrict__ path_j,
            int* __restrict__ path_length,
            int n, int m);

        /**
         * @brief Reverse path arrays to get forward order
         * @param path_i Path row indices to reverse in-place
         * @param path_j Path column indices to reverse in-place
         * @param path_length Length of the path
         *
         * Reverses the path from backward order to forward order.
         * Can be launched with multiple threads for parallel reversal.
         * Use (path_length/2 + 255) / 256 blocks with 256 threads each.
         */
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