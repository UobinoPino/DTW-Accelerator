/**
 * @file dtw_core_kernels.cuh
 * @brief CUDA kernel declarations for core DTW computations
 * @author UobinoPino
 * @date 2024
 *
 * This file declares the CUDA kernels for DTW computation,
 * including the main tiled wavefront kernel.
 */

#ifndef DTW_ACCELERATOR_DTW_CORE_KERNELS_CUH
#define DTW_ACCELERATOR_DTW_CORE_KERNELS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../../../core/distance_metrics.hpp"

namespace dtw_accelerator {
namespace parallel {
namespace cuda {
    /**
     * @namespace kernels
     * @brief CUDA kernel implementations for DTW
     */
    namespace kernels {

    using distance::MetricType;
    /**
     * @brief Tiled wavefront kernel for DTW computation
     * @tparam M Distance metric type
     * @tparam TILE_SIZE Size of tiles (default 64)
     * @param D Cost matrix (device memory)
     * @param A First time series (device memory)
     * @param B Second time series (device memory)
     * @param n Length of first series
     * @param m Length of second series
     * @param dim Number of dimensions
     * @param wave Current wave index
     * @param n_tiles Number of tiles in row dimension
     * @param m_tiles Number of tiles in column dimension
     *
     * Processes DTW in tiles using wavefront parallelization.
     * Each thread block processes one tile of the cost matrix.
     */
    template<MetricType M, int TILE_SIZE = 64>
    __global__ void dtw_tile_wavefront(
        double* __restrict__ D,
        const double* __restrict__ A,
        const double* __restrict__ B,
        int n, int m, int dim,
        int wave, int n_tiles, int m_tiles);

    } // namespace kernels
} // namespace cuda
} // namespace parallel
} // namespace dtw_accelerator

#endif // __CUDACC__
#endif // DTW_ACCELERATOR_DTW_CORE_KERNELS_CUH