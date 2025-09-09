/**
 * @file device_functions.cuh
 * @brief CUDA device functions for distance metric computations
 * @author UobinoPino
 * @date 2024
 *
 * This file contains optimized CUDA device functions for computing various
 * distance metrics on the GPU. These functions are designed to be called
 * from within CUDA kernels.
 */

#ifndef DTW_ACCELERATOR_CUDA_DEVICE_FUNCTIONS_CUH
#define DTW_ACCELERATOR_CUDA_DEVICE_FUNCTIONS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../../../core/distance_metrics.hpp"

namespace dtw_accelerator {
namespace parallel {
namespace cuda {
    /**
     * @namespace device
     * @brief CUDA device functions for kernel computations
     */
    namespace device {

    using distance::MetricType;

    /**
     * @brief Compute distance between two points on GPU device
     * @tparam M Distance metric type to use
     * @param a First point coordinates
     * @param b Second point coordinates
     * @param dim Number of dimensions
     * @return Computed distance using the specified metric
     *
     * This device function provides optimized distance computations
     * for various metrics using GPU-specific intrinsics and functions
     * like fma() for fused multiply-add and rsqrt() for reciprocal square root.
     */
    template<MetricType M>
    __device__ __forceinline__ double compute_distance(const double* a, const double* b, int dim) {
        if constexpr (M == MetricType::EUCLIDEAN) {
            double sum = 0.0;
    #pragma unroll 4 // Unroll loop
            for (int i = 0; i < dim; ++i) {
                double diff = a[i] - b[i];
                sum = fma(diff, diff, sum); // Fused multiply-add
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