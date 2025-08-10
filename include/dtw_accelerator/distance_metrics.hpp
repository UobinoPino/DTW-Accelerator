#ifndef DTW_ACCELERATOR_DISTANCE_METRICS_HPP
#define DTW_ACCELERATOR_DISTANCE_METRICS_HPP

#include <cmath>

namespace dtw_accelerator {
    namespace distance {

        // Euclidean distance between two vectors
        inline double euclidean_dist(const double* a, const double* b, int dim) {
            double sum = 0.0;
            for (int i = 0; i < dim; ++i) {
                double diff = a[i] - b[i];
                sum += diff * diff;
            }
            return std::sqrt(sum);
        }

        // Manhattan distance between two vectors
        inline double manhattan_dist(const double* a, const double* b, int dim) {
            double sum = 0.0;
            for (int i = 0; i < dim; ++i) {
                sum += std::abs(a[i] - b[i]);
            }
            return sum;
        }
        // Chebyshev distance between two vectors
        inline double chebyshev_dist(const double* a, const double* b, int dim) {
            double max_diff = 0.0;
            for (int i = 0; i < dim; ++i) {
                double diff = std::abs(a[i] - b[i]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
            return max_diff;
        }
        // Cosine distance between two vectors
        inline double cosine_dist(const double* a, const double* b, int dim) {
            double dot_product = 0.0;
            double norm_a = 0.0;
            double norm_b = 0.0;
            for (int i = 0; i < dim; ++i) {
                dot_product += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            if (norm_a == 0 || norm_b == 0) return 1.0; // Handle zero vectors
            return 1.0 - (dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b)));
        }


    } // namespace distance
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_DISTANCE_METRICS_HPP