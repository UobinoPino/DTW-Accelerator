#ifndef DTW_ACCELERATOR_FAST_DTW_HPP
#define DTW_ACCELERATOR_FAST_DTW_HPP

#include <vector>
#include <utility>

#include "dtw_accelerator/core/constraints.hpp"
#include "dtw_accelerator/core/path_processing.hpp"
#include "dtw_accelerator/execution/sequential/core_dtw.hpp"

namespace dtw_accelerator {
    namespace fast {

// Main FastDTW implementation
        inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_cpu(
                const DoubleTimeSeries& A,
                const DoubleTimeSeries& B,
                int radius = 1,
                int min_size = 100) {

            int n = A.size(), m = B.size();

            // Base case 1: if time series are small enough, use standard DTW
            if (n <= min_size && m <= min_size) {
                return core::dtw_cpu(A, B);
            }

            // Base case 2: if we can't downsample further
            if (n <= 2 || m <= 2) {
                return core::dtw_cpu(A, B);
            }

            // Recursive case:
            // 1. Coarsen the time series
            auto A_coarse = path::downsample(A);
            auto B_coarse = path::downsample(B);

            // 2. Compute FastDTW on the coarsened data
            auto [cost, path_result] = fastdtw_cpu(A_coarse, B_coarse, radius, min_size);

            // 3. Project the low-resolution path to a higher resolution
            auto projected_path = path::expand_path(path_result, n, m);

            // 4. Create a search window around the projected path
            auto window = path::get_window(projected_path, n, m, radius);

            // 5. Compute constrained DTW within the window
            return core::dtw_constrained(A, B, window);
        }

    } // namespace fast
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_FAST_DTW_HPP


