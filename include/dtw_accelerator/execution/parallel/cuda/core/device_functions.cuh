#ifndef DTW_ACCELERATOR_CUDA_DEVICE_FUNCTIONS_CUH
#define DTW_ACCELERATOR_CUDA_DEVICE_FUNCTIONS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../../../core/distance_metrics.hpp"

namespace dtw_accelerator {
namespace parallel {
namespace cuda {
namespace device {

using distance::MetricType;

template<MetricType M>
__device__ __forceinline__ double compute_distance(const double* a, const double* b, int dim) {
    if constexpr (M == MetricType::EUCLIDEAN) {
        double sum = 0.0;
#pragma unroll 4
        for (int i = 0; i < dim; ++i) {
            double diff = a[i] - b[i];
            sum = fma(diff, diff, sum);
        }
        return sqrt(sum);
    }
    else if constexpr (M == MetricType::MANHATTAN) {
        double sum = 0.0;
#pragma unroll 4
        for (int i = 0; i < dim; ++i) {
            sum += fabs(a[i] - b[i]);
        }
        return sum;
    }
    else if constexpr (M == MetricType::CHEBYSHEV) {
        double max_diff = 0.0;
#pragma unroll 4
        for (int i = 0; i < dim; ++i) {
            max_diff = fmax(max_diff, fabs(a[i] - b[i]));
        }
        return max_diff;
    }
    else if constexpr (M == MetricType::COSINE) {
        double dot_product = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;
#pragma unroll 4
        for (int i = 0; i < dim; ++i) {
            dot_product = fma(a[i], b[i], dot_product);
            norm_a = fma(a[i], a[i], norm_a);
            norm_b = fma(b[i], b[i], norm_b);
        }
        if (norm_a == 0.0 || norm_b == 0.0) return 1.0;
        return 1.0 - (dot_product * rsqrt(norm_a) * rsqrt(norm_b));
    }
}

} // namespace device
} // namespace cuda
} // namespace parallel
} // namespace dtw_accelerator

#endif // __CUDACC__
#endif // DTW_ACCELERATOR_CUDA_DEVICE_FUNCTIONS_CUH