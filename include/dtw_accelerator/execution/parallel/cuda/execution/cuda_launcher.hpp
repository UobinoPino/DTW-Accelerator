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

            struct DTWResult {
                double distance;
                std::vector<std::pair<int, int>> path;
            };

            constexpr int DEFAULT_TILE_SIZE = 64;

            DTWResult dtw_cuda_impl(
                    const DoubleTimeSeries& A,
                    const DoubleTimeSeries& B,
                    distance::MetricType metric = distance::MetricType::EUCLIDEAN,
                    int tile_size = DEFAULT_TILE_SIZE);

            bool is_cuda_available();
            std::string get_cuda_device_info();

        } // namespace cuda
    } // namespace parallel
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_CUDA_LAUNCHER_HPP