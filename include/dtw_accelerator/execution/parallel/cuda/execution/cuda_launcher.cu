/**
 * @file cuda_launcher.cu
 * @brief CUDA kernel launcher implementation for DTW algorithms
 * @author UobinoPino
 * @date 2024
 *
 * This file implements the host-side launcher functions that configure
 * and launch CUDA kernels for DTW computations. It handles memory transfers,
 * kernel configuration, and result retrieval.
 */

#include "cuda_launcher.hpp"
#include "../core/cuda_memory.hpp"
#include "../kernels/matrix_kernels.cuh"
#include "../kernels/dtw_core_kernels.cuh"
#include "../kernels/path_kernels.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace dtw_accelerator {
    namespace parallel {
        namespace cuda {

            using memory::DeviceBuffer;

            /**
             * @brief Template implementation of CUDA DTW computation
             * @tparam M Distance metric type
             * @param A First time series
             * @param B Second time series
             * @param tile_size Size of tiles for blocked processing
             * @return DTW result containing distance and optimal path
             *
             * This function orchestrates the entire CUDA DTW computation:
             * 1. Allocates device memory
             * 2. Transfers data to GPU
             * 3. Launches kernels in wavefront order
             * 4. Retrieves results from GPU
             */
            template<distance::MetricType M>
            DTWResult dtw_cuda_impl_template(
                    const DoubleTimeSeries& A,
                    const DoubleTimeSeries& B,
                    int tile_size) {

                int n = A.size();
                int m = B.size();
                int dim = A.dimensions();

                DeviceBuffer<double> d_A(n * dim);
                DeviceBuffer<double> d_B(m * dim);
                DeviceBuffer<double> d_D((n + 1) * (m + 1));

                int max_path_length = n + m;
                DeviceBuffer<int> d_path_i(max_path_length);
                DeviceBuffer<int> d_path_j(max_path_length);
                DeviceBuffer<int> d_path_length(1);

                cudaMemcpy(d_A.data, A.data(), n * dim * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_B.data, B.data(), m * dim * sizeof(double), cudaMemcpyHostToDevice);

                constexpr int BLOCK_SIZE = 256;
                int grid_size = ((n+1)*(m+1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kernels::init_cost_matrix<<<grid_size, BLOCK_SIZE>>>(d_D.data, n, m);

                int n_tiles = (n + tile_size - 1) / tile_size;
                int m_tiles = (m + tile_size - 1) / tile_size;
                int total_waves = n_tiles + m_tiles - 1;

                size_t shared_mem_size = 2 * (tile_size + 1) * sizeof(double);

                for (int wave = 0; wave < total_waves; ++wave) {
                    int tiles_in_wave = 0;

                    for (int ti = 0; ti < n_tiles; ++ti) {
                        int tj = wave - ti;
                        if (tj >= 0 && tj < m_tiles) {
                            tiles_in_wave++;
                        }
                    }

                    if (tiles_in_wave > 0) {
                        constexpr int THREADS_PER_BLOCK = 256;
                        dim3 grid(tiles_in_wave, 1, 1);
                        dim3 block(THREADS_PER_BLOCK, 1, 1);

                        if (tile_size == 32) {
                            kernels::dtw_tile_wavefront<M, 32>
                            <<<grid, block, shared_mem_size>>>(
                                    d_D.data, d_A.data, d_B.data, n, m, dim,
                                            wave, n_tiles, m_tiles);
                        } else if (tile_size == 64) {
                            kernels::dtw_tile_wavefront<M, 64>
                            <<<grid, block, shared_mem_size>>>(
                                    d_D.data, d_A.data, d_B.data, n, m, dim,
                                            wave, n_tiles, m_tiles);
                        } else {
                            kernels::dtw_tile_wavefront<M, 32>
                            <<<grid, block, shared_mem_size>>>(
                                    d_D.data, d_A.data, d_B.data, n, m, dim,
                                            wave, n_tiles, m_tiles);
                        }
                    }
                }

                kernels::backtrack_path<<<1, 1>>>(
                        d_D.data, d_path_i.data, d_path_j.data,
                                d_path_length.data, n, m);

                int h_path_length;
                cudaMemcpyAsync(&h_path_length, d_path_length.data, sizeof(int),
                                cudaMemcpyDeviceToHost);

                if (h_path_length > 1) {
                    int reverse_blocks = (h_path_length / 2 + 255) / 256;
                    kernels::reverse_path<<<reverse_blocks, 256>>>(
                            d_path_i.data, d_path_j.data, h_path_length);
                }

                double final_distance;
                cudaMemcpyAsync(&final_distance, &d_D.data[n * (m + 1) + m],
                                sizeof(double), cudaMemcpyDeviceToHost);

                std::vector<int> h_path_i(h_path_length);
                std::vector<int> h_path_j(h_path_length);

                cudaMemcpy(h_path_i.data(), d_path_i.data,
                           h_path_length * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_path_j.data(), d_path_j.data,
                           h_path_length * sizeof(int), cudaMemcpyDeviceToHost);

                std::vector<std::pair<int, int>> path;
                path.reserve(h_path_length);
                for (int k = 0; k < h_path_length; ++k) {
                    path.emplace_back(h_path_i[k], h_path_j[k]);
                }

                return {final_distance, std::move(path)};
            }

            /**
             * @brief Main CUDA DTW implementation dispatcher
             * @param A First time series
             * @param B Second time series
             * @param metric Distance metric to use
             * @param tile_size Size of tiles for blocked processing
             * @return DTW result containing distance and optimal path
             *
             * Dispatches to the appropriate template instantiation based
             * on the runtime metric selection.
             */
            DTWResult dtw_cuda_impl(
                    const DoubleTimeSeries& A,
                    const DoubleTimeSeries& B,
                    distance::MetricType metric,
                    int tile_size) {

                switch(metric) {
                    case distance::MetricType::EUCLIDEAN:
                        return dtw_cuda_impl_template<distance::MetricType::EUCLIDEAN>(A, B, tile_size);
                    case distance::MetricType::MANHATTAN:
                        return dtw_cuda_impl_template<distance::MetricType::MANHATTAN>(A, B, tile_size);
                    case distance::MetricType::CHEBYSHEV:
                        return dtw_cuda_impl_template<distance::MetricType::CHEBYSHEV>(A, B, tile_size);
                    case distance::MetricType::COSINE:
                        return dtw_cuda_impl_template<distance::MetricType::COSINE>(A, B, tile_size);
                    default:
                        return dtw_cuda_impl_template<distance::MetricType::EUCLIDEAN>(A, B, tile_size);
                }
            }

            /**
             * @brief Check if CUDA is available on the system
             * @return True if at least one CUDA device is available
             */
            bool is_cuda_available() {
                int device_count = 0;
                cudaError_t error = cudaGetDeviceCount(&device_count);
                return error == cudaSuccess && device_count > 0;
            }

            /**
             * @brief Get information about available CUDA devices
             * @return String containing device properties and capabilities
             */
            std::string get_cuda_device_info() {
                int device_count = 0;
                cudaGetDeviceCount(&device_count);

                if (device_count == 0) {
                    return "No CUDA devices available";
                }

                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);

                std::string info = "CUDA Device: ";
                info += prop.name;
                info += ", Compute Capability: ";
                info += std::to_string(prop.major) + "." + std::to_string(prop.minor);
                info += ", SMs: ";
                info += std::to_string(prop.multiProcessorCount);
                info += ", Max Threads/Block: ";
                info += std::to_string(prop.maxThreadsPerBlock);
                info += ", Shared Mem/Block: ";
                info += std::to_string(prop.sharedMemPerBlock) + " bytes";

                return info;
            }

        } // namespace cuda
    } // namespace parallel
} // namespace dtw_accelerator

