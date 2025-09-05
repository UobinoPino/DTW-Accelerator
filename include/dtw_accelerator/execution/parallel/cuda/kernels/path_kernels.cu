#include "path_kernels.cuh"

namespace dtw_accelerator {
    namespace parallel {
        namespace cuda {
            namespace kernels {

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