/**
 * @file constraints.hpp
 * @brief DTW path constraint implementations
 * @author UobinoPino
 * @date 2024
 *
 * This file contains implementations of global path constraints
 * for DTW including Sakoe-Chiba band and Itakura parallelogram.
 */

#ifndef DTW_ACCELERATOR_CONSTRAINTS_HPP
#define DTW_ACCELERATOR_CONSTRAINTS_HPP

#include <cmath>

namespace dtw_accelerator {
    namespace constraints {

        /**
         * @brief Enumeration of supported constraint types
         */
        enum class ConstraintType {
            NONE,           ///< No constraint (full matrix)
            SAKOE_CHIBA,    ///< Sakoe-Chiba band constraint
            ITAKURA         ///< Itakura parallelogram constraint
        };

        /**
         * @brief Check if a cell is within the Sakoe-Chiba band constraint
         * @tparam R Radius of the band (template parameter for compile-time optimization)
         * @param i Row index (0-based)
         * @param j Column index (0-based)
         * @param n Number of rows
         * @param m Number of columns
         * @return True if cell (i,j) is within the band
         *
         * The Sakoe-Chiba band constrains the warping path to stay within
         * a fixed distance R from the diagonal when normalized.
         */
        template<int R>
        constexpr bool within_sakoe_chiba_band(int i, int j, int n, int m) {
            // Normalize indices to account for different lengths
            double ni = static_cast<double>(i) / n;
            double nj = static_cast<double>(j) / m;

            // Check if within band of width R around the diagonal
            return std::abs(ni - nj) * std::max(n, m) <= R;
        }


        /**
        * @brief Check if a cell is within the Itakura parallelogram constraint
        * @tparam S Maximum slope constraint (must be > 1.0)
        * @param i Row index (0-based)
        * @param j Column index (0-based)
        * @param n Number of rows
        * @param m Number of columns
        * @return True if cell (i,j) is within the parallelogram
        *
        * The Itakura parallelogram constrains the warping path slope
        * to be between 1/S and S, preventing excessive stretching or compression.
        */
        template<double S>
        constexpr bool within_itakura_parallelogram(int i, int j, int n, int m) {
            static_assert(S > 1.0, "Itakura slope constraint must be greater than 1");

            double di = static_cast<double>(i);
            double dj = static_cast<double>(j);
            double dn = static_cast<double>(n - 1);
            double dm = static_cast<double>(m - 1);

            // Handle edge cases efficiently
            if (dn <= 0 || dm <= 0) return true;

            // Normalize to [0,1] range
            double ni = di / dn;
            double nj = dj / dm;

            // Itakura parallelogram constraints:
            // The path must satisfy both forward and backward slope constraints
            // max(1/S * ni, S * ni - (S - 1)) ≤ nj ≤ min(S * ni, 1/S * ni + (1 - 1/S))

            double lower_bound = std::max(ni / S, S * ni - (S - 1.0));
            double upper_bound = std::min(S * ni, ni / S + (1.0 - 1.0/S));

            return nj >= lower_bound && nj <= upper_bound;

        }



    } // namespace constraints
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_CONSTRAINTS_HPP


