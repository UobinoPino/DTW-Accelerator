#include "matrix_kernels.cuh"
#include <float.h>

namespace dtw_accelerator {
    namespace parallel {
        namespace cuda {
            namespace kernels {

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