/**
 * @file distance_metrics.hpp
 * @brief Distance metric implementations for DTW computations
 * @author UobinoPino
 * @date 2024
 *
 * This file contains template specializations for various distance
 * metrics used in DTW algorithms including Euclidean, Manhattan,
 * Chebyshev, and Cosine distances.
 */

#ifndef DTW_ACCELERATOR_DISTANCE_METRICS_HPP
#define DTW_ACCELERATOR_DISTANCE_METRICS_HPP

#include <cmath>

namespace dtw_accelerator {
    namespace distance {

        /**
         * @brief Enumeration of supported distance metric types
         */
        enum class MetricType {
            EUCLIDEAN, ///< Euclidean (L2) distance
            MANHATTAN, ///< Manhattan (L1) distance
            CHEBYSHEV, ///< Chebyshev (L∞) distance
            COSINE ///< Cosine distance (1 - cosine similarity)
        };

        /**
         * @brief Primary template for distance metric computation
         * @tparam M The metric type to use
         *
         * This template is specialized for each supported metric type.
         * Direct instantiation will cause a compile error.
         */
        template<MetricType M>
        struct Metric {
            static_assert(M == MetricType::EUCLIDEAN || M == MetricType::MANHATTAN ||
                          M == MetricType::CHEBYSHEV || M == MetricType::COSINE,
                          "Unsupported distance metric");
            /**
             * @brief Compute distance between two points
             * @param a First point
             * @param b Second point
             * @param dim Number of dimensions
             * @return The computed distance
             */
            static inline double compute(const double* a, const double* b, int dim);
        };

        /**
         * @brief Euclidean distance specialization
         *
         * Computes the L2 norm: sqrt(sum((a[i] - b[i])^2))
         */
        template<>
        struct Metric<MetricType::EUCLIDEAN> {
            /**
             * @brief Compute Euclidean distance
             * @param a First point coordinates
             * @param b Second point coordinates
             * @param dim Number of dimensions
             * @return Euclidean distance between points
             */
            static inline double compute(const double* a, const double* b, int dim) {
                double sum = 0.0;
                for (int i = 0; i < dim; ++i) {
                    double diff = a[i] - b[i];
                    sum += diff * diff;
                }
                return std::sqrt(sum);
            }
        };

        /**
         * @brief Manhattan distance specialization
         *
         * Computes the L1 norm: sum(|a[i] - b[i]|)
         */
        template<>
        struct Metric<MetricType::MANHATTAN> {
            /**
             * @brief Compute Manhattan distance
             * @param a First point coordinates
             * @param b Second point coordinates
             * @param dim Number of dimensions
             * @return Manhattan distance between points
             */
            static inline double compute(const double* a, const double* b, int dim) {
                double sum = 0.0;
                for (int i = 0; i < dim; ++i) {
                    sum += std::abs(a[i] - b[i]);
                }
                return sum;
            }
        };

        /**
         * @brief Chebyshev distance specialization
         *
         * Computes the L∞ norm: max(|a[i] - b[i]|)
         */
        template<>
        struct Metric<MetricType::CHEBYSHEV> {
            /**
             * @brief Compute Chebyshev distance
             * @param a First point coordinates
             * @param b Second point coordinates
             * @param dim Number of dimensions
             * @return Maximum absolute difference across dimensions
             */
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

        /**
         * @brief Cosine distance specialization
         *
         * Computes 1 - cosine_similarity(a, b)
         * where cosine_similarity = dot(a,b) / (||a|| * ||b||)
         */
        template<>
        struct Metric<MetricType::COSINE> {
            /**
             * @brief Compute Cosine distance
             * @param a First point coordinates
             * @param b Second point coordinates
             * @param dim Number of dimensions
             * @return Cosine distance (1 - cosine similarity)
             *
             * Returns 1.0 if either vector has zero magnitude
             */
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