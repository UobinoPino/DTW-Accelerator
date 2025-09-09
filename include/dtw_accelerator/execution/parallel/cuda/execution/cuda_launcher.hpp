/**
 * @file cuda_launcher.hpp
 * @brief CUDA kernel launcher interface for DTW algorithms
 * @author UobinoPino
 * @date 2024
 *
 * This file defines the interface for launching CUDA kernels for DTW
 * computations, including the main entry points and result structures.
 */

#ifndef DTW_ACCELERATOR_CUDA_LAUNCHER_HPP
#define DTW_ACCELERATOR_CUDA_LAUNCHER_HPP

#include <vector>
#include <utility>
#include <string>
#include "../../../../core/distance_metrics.hpp"
#include "../../../../core/time_series.hpp"

namespace dtw_accelerator {
    namespace parallel {
        namespace cuda {

            /**
             * @brief Structure to hold DTW computation results
             */
            struct DTWResult {
                /// @brief Total DTW distance between the series
                double distance;

                /// @brief Optimal warping path as sequence of (i,j) pairs
                std::vector<std::pair<int, int>> path;
            };

            /// @brief Default tile size for blocked CUDA processing
            constexpr int DEFAULT_TILE_SIZE = 64;

            /**
             * @brief Launch CUDA DTW computation
             * @param A First time series
             * @param B Second time series
             * @param metric Distance metric to use
             * @param tile_size Size of tiles for blocked processing
             * @return DTW result containing distance and path
             *
             * Main entry point for CUDA-accelerated DTW computation.
             * Automatically handles device memory management and kernel configuration.
             */
            DTWResult dtw_cuda_impl(
                    const DoubleTimeSeries& A,
                    const DoubleTimeSeries& B,
                    distance::MetricType metric = distance::MetricType::EUCLIDEAN,
                    int tile_size = DEFAULT_TILE_SIZE);

            /**
             * @brief Check if CUDA is available on the system
             * @return True if at least one CUDA device is available
             */
            bool is_cuda_available();

            /**
             * @brief Get detailed information about CUDA devices
             * @return String containing device names, compute capabilities, and specs
             */
            std::string get_cuda_device_info();

        } // namespace cuda
    } // namespace parallel
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_CUDA_LAUNCHER_HPP