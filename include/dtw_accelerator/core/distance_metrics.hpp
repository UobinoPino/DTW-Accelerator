#ifndef DTW_ACCELERATOR_DISTANCE_METRICS_HPP
#define DTW_ACCELERATOR_DISTANCE_METRICS_HPP

#include <cmath>

namespace dtw_accelerator {
    namespace distance {

        // Distance metric types
        enum class MetricType {
            EUCLIDEAN,
            MANHATTAN,
            CHEBYSHEV,
            COSINE
        };

        // Primary distance metric trait template
        template<MetricType M>
        struct Metric {
            // Default implementation will cause a compile error if used directly
            static_assert(M == MetricType::EUCLIDEAN || M == MetricType::MANHATTAN ||
                          M == MetricType::CHEBYSHEV || M == MetricType::COSINE,
                          "Unsupported distance metric");

            static inline double compute(const double* a, const double* b, int dim);
        };

        // Euclidean distance specialization
        template<>
        struct Metric<MetricType::EUCLIDEAN> {
            static inline double compute(const double* a, const double* b, int dim) {
                double sum = 0.0;
                for (int i = 0; i < dim; ++i) {
                    double diff = a[i] - b[i];
                    sum += diff * diff;
                }
                return std::sqrt(sum);
            }
        };

        // Manhattan distance specialization
        template<>
        struct Metric<MetricType::MANHATTAN> {
            static inline double compute(const double* a, const double* b, int dim) {
                double sum = 0.0;
                for (int i = 0; i < dim; ++i) {
                    sum += std::abs(a[i] - b[i]);
                }
                return sum;
            }
        };

        // Chebyshev distance specialization
        template<>
        struct Metric<MetricType::CHEBYSHEV> {
            static inline double compute(const double* a, const double* b, int dim) {
                double max_diff = 0.0;
                for (int i = 0; i < dim; ++i) {
                    double diff = std::abs(a[i] - b[i]);
                    if (diff > max_diff) {
                        max_diff = diff;
                    }
                }
                return max_diff;
            }
        };

        // Cosine distance specialization
        template<>
        struct Metric<MetricType::COSINE> {
            static inline double compute(const double* a, const double* b, int dim) {
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
        };


    } // namespace distance
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_DISTANCE_METRICS_HPP