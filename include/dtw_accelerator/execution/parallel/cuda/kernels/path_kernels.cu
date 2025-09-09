/**
 * @file path_kernels.cu
 * @brief CUDA kernel implementations for optimal path extraction
 * @author UobinoPino
 * @date 2024
 *
 * This file implements CUDA kernels for extracting the optimal warping
 * path from the computed DTW cost matrix through backtracking.
 */

#include "path_kernels.cuh"

namespace dtw_accelerator {
    namespace parallel {
        namespace cuda {
            namespace kernels {
                /**
                * @brief Backtrack through cost matrix to find optimal path
                * @param D Cost matrix in device memory
                * @param path_i Output array for path row indices
                * @param path_j Output array for path column indices
                * @param path_length Output for actual path length
                * @param n Number of rows (excluding padding)
                * @param m Number of columns (excluding padding)
                *
                * This kernel performs sequential backtracking from D[n][m] to D[0][0]
                * to extract the optimal warping path. It should be launched with a
                * single thread as backtracking is inherently sequential.
                *
                * The path is initially stored in reverse order (from end to start).
                */
                __global__ void backtrack_path(
                        const double* __restrict__ D,
                        int* __restrict__ path_i,
                        int* __restrict__ path_j,
                        int* __restrict__ path_length,
                        int n, int m) {

                    if (threadIdx.x == 0 && blockIdx.x == 0) {
                        int i = n;
                        int j = m;
                        int k = 0;
                        int max_path_length = n + m;

                        while (i > 0 || j > 0) {
                            path_i[k] = i - 1;
                            path_j[k] = j - 1;
                            k++;

                            if (k >= max_path_length) break;

                            if (i == 0) {
                                j--;
                            } else if (j == 0) {
                                i--;
                            } else {
                                double cost_diag = D[(i - 1) * (m + 1) + (j - 1)];
                                double cost_left = D[i * (m + 1) + (j - 1)];
                                double cost_up = D[(i - 1) * (m + 1) + j];

                                if (cost_diag <= cost_left && cost_diag <= cost_up) {
                                    i--; j--;
                                } else if (cost_left <= cost_up) {
                                    j--;
                                } else {
                                    i--;
                                }
                            }
                        }
                        *path_length = k;
                    }
                }

                /**
                 * @brief Reverse the path array to get forward order
                 * @param path_i Path row indices to reverse
                 * @param path_j Path column indices to reverse
                 * @param path_length Length of the path
                 *
                 * This kernel reverses the path arrays in-place to convert
                 * from backward order (end to start) to forward order (start to end).
                 * It can be launched with multiple threads for parallel reversal.
                 */
                __global__ void reverse_path(
                        int* __restrict__ path_i,
                        int* __restrict__ path_j,
                        int path_length) {

                    int idx = blockIdx.x * blockDim.x + threadIdx.x;

                    if (idx < path_length / 2) {
                        int other_idx = path_length - 1 - idx;

                        int temp_i = path_i[idx];
                        path_i[idx] = path_i[other_idx];
                        path_i[other_idx] = temp_i;

                        int temp_j = path_j[idx];
                        path_j[idx] = path_j[other_idx];
                        path_j[other_idx] = temp_j;
                    }
                }

            } // namespace kernels
        } // namespace cuda
    } // namespace parallel
} // namespace dtw_accelerator