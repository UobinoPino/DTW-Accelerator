#include "dtw_core_kernels.cuh"
#include "../core/device_functions.cuh"
#include <algorithm>

namespace dtw_accelerator {
    namespace parallel {
        namespace cuda {
            namespace kernels {

                template<MetricType M, int TILE_SIZE>
                __global__ void dtw_tile_wavefront(
                        double* __restrict__ D,
                        const double* __restrict__ A,
                        const double* __restrict__ B,
                        int n, int m, int dim,
                        int wave, int n_tiles, int m_tiles) {

                    extern __shared__ double shared_mem[];
                    double* shared_left = shared_mem;
                    double* shared_top = &shared_mem[TILE_SIZE + 1];

                    int tile_idx = blockIdx.x;
                    int ti = 0, tj = 0;

                    for (int i = 0, count = 0; i < n_tiles; ++i) {
                        int j = wave - i;
                        if (j >= 0 && j < m_tiles) {
                            if (count++ == tile_idx) {
                                ti = i; tj = j;
                                break;
                            }
                        }
                    }

                    int i_start = ti * TILE_SIZE;
                    int j_start = tj * TILE_SIZE;
                    int i_end = min(i_start + TILE_SIZE, n);
                    int j_end = min(j_start + TILE_SIZE, m);

                    int tid = threadIdx.x;
                    int num_threads = blockDim.x;

                    if (tid < TILE_SIZE + 1) {
                        int global_j = j_start + tid;
                        if (global_j <= m) {
                            shared_top[tid] = D[i_start * (m + 1) + global_j];
                        }
                        int global_i = i_start + tid;
                        if (global_i <= n) {
                            shared_left[tid] = D[global_i * (m + 1) + j_start];
                        }
                    }
                    __syncthreads();

                    int tile_height = i_end - i_start;
                    int tile_width = j_end - j_start;
                    int num_waves = tile_height + tile_width - 1;

                    for (int mini_wave = 0; mini_wave < num_waves; ++mini_wave) {
                        int wave_start = max(0, mini_wave - tile_width + 1);
                        int wave_end = min(mini_wave + 1, tile_height);
                        int cells_in_wave = wave_end - wave_start;

                        for (int cell = tid; cell < cells_in_wave; cell += num_threads) {
                            int local_i = wave_start + cell;
                            int local_j = mini_wave - local_i;

                            if (local_i >= 0 && local_i < tile_height &&
                                local_j >= 0 && local_j < tile_width) {

                                int global_i = i_start + local_i + 1;
                                int global_j = j_start + local_j + 1;

                                double dist = device::compute_distance<M>(
                                        &A[(global_i - 1) * dim],
                                        &B[(global_j - 1) * dim],
                                        dim
                                );

                                double cost_diag = (local_i == 0 && local_j == 0) ? shared_left[0] :
                                                   (local_i == 0) ? shared_top[local_j] :
                                                   (local_j == 0) ? shared_left[local_i] :
                                                   D[(global_i - 1) * (m + 1) + (global_j - 1)];

                                double cost_left = (local_j == 0) ? shared_left[local_i + 1] :
                                                   D[global_i * (m + 1) + (global_j - 1)];

                                double cost_up = (local_i == 0) ? shared_top[local_j + 1] :
                                                 D[(global_i - 1) * (m + 1) + global_j];

                                double min_prev = fmin(fmin(cost_diag, cost_left), cost_up);
                                D[global_i * (m + 1) + global_j] = dist + min_prev;

                                if (local_j == tile_width - 1 && local_i < TILE_SIZE) {
                                    shared_left[local_i + 1] = dist + min_prev;
                                }
                                if (local_i == tile_height - 1 && local_j < TILE_SIZE) {
                                    shared_top[local_j + 1] = dist + min_prev;
                                }
                            }
                        }
                        __syncthreads();
                    }
                }

// Explicit instantiations
                template __global__ void dtw_tile_wavefront<MetricType::EUCLIDEAN, 32>
                        (double*, const double*, const double*, int, int, int, int, int, int);
                template __global__ void dtw_tile_wavefront<MetricType::EUCLIDEAN, 64>
                        (double*, const double*, const double*, int, int, int, int, int, int);
                template __global__ void dtw_tile_wavefront<MetricType::MANHATTAN, 32>
                        (double*, const double*, const double*, int, int, int, int, int, int);
                template __global__ void dtw_tile_wavefront<MetricType::MANHATTAN, 64>
                        (double*, const double*, const double*, int, int, int, int, int, int);
                template __global__ void dtw_tile_wavefront<MetricType::CHEBYSHEV, 32>
                        (double*, const double*, const double*, int, int, int, int, int, int);
                template __global__ void dtw_tile_wavefront<MetricType::CHEBYSHEV, 64>
                        (double*, const double*, const double*, int, int, int, int, int, int);
                template __global__ void dtw_tile_wavefront<MetricType::COSINE, 32>
                        (double*, const double*, const double*, int, int, int, int, int, int);
                template __global__ void dtw_tile_wavefront<MetricType::COSINE, 64>
                        (double*, const double*, const double*, int, int, int, int, int, int);

            } // namespace kernels
        } // namespace cuda
    } // namespace parallel
} // namespace dtw_accelerator