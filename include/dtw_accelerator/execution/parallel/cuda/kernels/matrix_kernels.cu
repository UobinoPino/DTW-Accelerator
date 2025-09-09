/**
 * @file matrix_kernels.cu
 * @brief CUDA kernel implementations for matrix operations
 * @author UobinoPino
 * @date 2024
 *
 * This file implements CUDA kernels for matrix operations needed
 * in DTW computation, including matrix initialization.
 */

#include "matrix_kernels.cuh"
#include <float.h>

namespace dtw_accelerator {
    namespace parallel {
        namespace cuda {
            namespace kernels {

                /**
                 * @brief Initialize cost matrix with infinity values
                 * @param D Cost matrix in device memory
                 * @param n Number of rows (excluding padding)
                 * @param m Number of columns (excluding padding)
                 *
                 * Initializes all matrix elements to DBL_MAX (representing infinity)
                 * except D[0][0] which is set to 0.0 as the starting point.
                 * The matrix has dimensions (n+1) x (m+1) to accommodate padding.
                 */
                __global__ void init_cost_matrix(double* D, int n, int m) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    int total_size = (n + 1) * (m + 1);

                    if (idx < total_size) {
                        D[idx] = DBL_MAX;
                    }

                    if (idx == 0) {
                        D[0] = 0.0;
                    }
                }

            } // namespace kernels
        } // namespace cuda
    } // namespace parallel
} // namespace dtw_accelerator