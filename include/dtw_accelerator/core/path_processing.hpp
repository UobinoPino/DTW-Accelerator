/**
 * @file path_processing.hpp
 * @brief Path processing utilities for DTW algorithms
 * @author UobinoPino
 * @date 2024
 *
 * This file contains utilities for processing DTW paths including
 * downsampling for FastDTW, path expansion, and window generation.
 */

#ifndef DTW_ACCELERATOR_PATH_PROCESSING_HPP
#define DTW_ACCELERATOR_PATH_PROCESSING_HPP

#include "time_series.hpp"
#include <vector>
#include <utility>
#include <algorithm>
#include <omp.h>

namespace dtw_accelerator {

    /**
     * @namespace path
     * @brief Path processing utilities for DTW algorithms
     */
    namespace path {

        /**
         * @brief Downsample a time series by averaging consecutive points
         * @tparam T Data type of the time series elements
         * @param series Input time series to downsample
         * @return Downsampled time series with approximately half the length
         *
         * This function is used in FastDTW to create coarser representations
         * of time series. Each pair of consecutive points is averaged to
         * produce a single point in the output.
         */
        template<typename T = double>
        inline TimeSeries<T> downsample(const TimeSeries<T>& series) {
            if (series.length() <= 2) return series;

            size_t n = series.length();
            size_t dim = series.dimensions();
            size_t new_size = (n + 1) / 2;

            TimeSeries<T> result(new_size, dim, 0.0);

         //   #pragma omp parallel for
            for (size_t i = 0; i < new_size; ++i) {
                for (size_t d = 0; d < dim; ++d) {
                    if (2*i + 1 < n) {
                        // Average two points
                        result[i][d] = (series[2*i][d] + series[2*i+1][d]) / 2.0;
                    } else {
                        // Last point if odd length
                        result[i][d] = series[2*i][d];
                    }
                }
            }

            return result;
        }

        /**
         * @brief Expand a path from coarse resolution to higher resolution
         * @param path Low-resolution path to expand
         * @param higher_n Target number of rows (series A length)
         * @param higher_m Target number of columns (series B length)
         * @return Expanded path with additional intermediate points
         *
         * Used in FastDTW to project a path computed on downsampled series
         * back to the original resolution. Each point in the low-resolution
         * path maps to up to 4 points in the higher resolution.
         */
        inline std::vector<std::pair<int, int>> expand_path(
                const std::vector<std::pair<int, int>>& path,
                int higher_n,
                int higher_m) {
            std::vector<std::pair<int, int>> expanded;

            for (const auto& [i, j] : path) {
                int i_high = std::min(2*i, higher_n-1);
                int j_high = std::min(2*j, higher_m-1);

                expanded.emplace_back(i_high, j_high);

                // Add additional points if not at boundary
                if (i_high + 1 < higher_n) expanded.emplace_back(i_high + 1, j_high);
                if (j_high + 1 < higher_m) expanded.emplace_back(i_high, j_high + 1);
                if (i_high + 1 < higher_n && j_high + 1 < higher_m)
                    expanded.emplace_back(i_high + 1, j_high + 1);
            }

            // Remove duplicates
            std::sort(expanded.begin(), expanded.end());
            expanded.erase(std::unique(expanded.begin(), expanded.end()), expanded.end());

            return expanded;
        }

        /**
        * @brief Create a search window around a projected path
        * @param path Base path to expand around
        * @param n Number of rows in the DTW matrix
        * @param m Number of columns in the DTW matrix
        * @param radius Expansion radius around the path
        * @return Window containing all valid cells within radius of the path
        *
        * Used in FastDTW to constrain the search space in higher resolutions.
        * The window includes all cells within Manhattan distance 'radius'
        * from any point on the given path.
        */
        inline std::vector<std::pair<int, int>> get_window(
                const std::vector<std::pair<int, int>>& path,
                int n, int m, int radius) {

            std::vector<std::pair<int, int>> window = path;

            // Expand window by radius
            for (const auto& [i, j] : path) {
                for (int di = -radius; di <= radius; ++di) {
                    for (int dj = -radius; dj <= radius; ++dj) {
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni >= 0 && ni < n && nj >= 0 && nj < m) {
                            window.emplace_back(ni, nj);
                        }
                    }
                }
            }

            // Remove duplicates
            std::sort(window.begin(), window.end());
            window.erase(std::unique(window.begin(), window.end()), window.end());

            return window;
        }

    } // namespace path
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_PATH_PROCESSING_HPP