#ifndef DTW_ACCELERATOR_CONSTRAINTS_HPP
#define DTW_ACCELERATOR_CONSTRAINTS_HPP

#include <cmath>

namespace dtw_accelerator {
    namespace constraints {

// Define constraint types
        enum class ConstraintType {
            NONE,           // No constraint
            SAKOE_CHIBA,    // Sakoe-Chiba band constraint
            ITAKURA         // Itakura parallelogram constraint
        };

// Check if a cell is within the Sakoe-Chiba band constraint
        template<int R>
        constexpr bool within_sakoe_chiba_band(int i, int j, int n, int m) {
            // Normalize indices to account for different lengths
            double ni = static_cast<double>(i) / n;
            double nj = static_cast<double>(j) / m;

            // This computes a band of width R around the diagonal
            return std::abs(ni - nj) * std::max(n, m) <= R;
        }


// Check if a cell is within the Itakura parallelogram constraint
        template<double S>
        constexpr bool within_itakura_parallelogram(int i, int j, int n, int m) {
            static_assert(S > 1.0, "Itakura slope constraint must be greater than 1");
            // Normalize indices to account for different lengths
            double ni = static_cast<double>(i) / n;
            double nj = static_cast<double>(j) / m;
            // This checks if the point is within the Itakura parallelogram
            return (nj >= ni / S) && (nj <= (1.0 - ni) / S) &&
                   (nj <= ni * S) && (nj >= (1.0 - ni) * S);

        }



    } // namespace constraints
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_CONSTRAINTS_HPP


