/**
 * @file cuda_strategy.hpp
 * @brief CUDA execution strategy for DTW algorithms
 * @author UobinoPino
 * @date 2024
 *
 * This file implements the CUDA execution strategy that conforms to the
 * ExecutionStrategy concept, providing GPU-accelerated DTW computations
 * while maintaining compatibility with the unified DTW interface.
 */

#ifndef DTW_ACCELERATOR_CUDA_STRATEGY_HPP
#define DTW_ACCELERATOR_CUDA_STRATEGY_HPP

#ifdef USE_CUDA

#include <vector>
#include <utility>
#include <memory>
#include <stdexcept>
#include "cuda_launcher.hpp"
#include "../../../../core/dtw_concepts.hpp"
#include "../../../../core/path_processing.hpp"
#include "../../../../core/dtw_utils.hpp"
#include "../../../../core/matrix.hpp"
#include "../../../../execution/sequential/standard_strategy.hpp"

namespace dtw_accelerator {
namespace parallel {
namespace cuda {

    /**
     * @brief CUDA execution strategy for GPU-accelerated DTW
     *
     * This class implements the ExecutionStrategy concept for CUDA,
     * providing GPU-accelerated DTW computations. It automatically
     * falls back to CPU execution for small sequences where GPU
     * overhead would outweigh benefits.
     */

    class CUDAStrategy {
    private:

        /// @brief Tile size for blocked GPU processing
        int tile_size_;

        /// @brief Flag indicating if CUDA device is initialized
        mutable bool device_initialized_;

        /// @brief Cache for the last computed path
        mutable std::vector<std::pair<int, int>> last_path_;

        /**
         * @brief Ensure CUDA device is initialized
         * @throws std::runtime_error if no CUDA devices are available
         */
        void ensure_device_initialized() const {
            if (!device_initialized_) {
                if (!is_cuda_available()) {
                    throw std::runtime_error("No CUDA devices available");
                }
                device_initialized_ = true;
            }
        }

    public:
        /**
         * @brief Construct a CUDA strategy with specified tile size
         * @param tile_size Size of tiles for blocked processing
         */
        explicit CUDAStrategy(int tile_size = DEFAULT_TILE_SIZE)
            : tile_size_(tile_size), device_initialized_(false) {}

        /**
         * @brief Initialize the DTW cost matrix
         * @param D Cost matrix (placeholder for CUDA)
         * @param n Number of rows
         * @param m Number of columns
         *
         * For CUDA, this is a minimal initialization as the actual
         * matrix is allocated on the device.
         */
        void initialize_matrix(DoubleMatrix& D, int n, int m) const {
            D.resize(1, 1, 0.0);
        }


        /**
         * @brief Execute DTW computation on GPU and return result
         * @tparam M Distance metric type
         * @param A First time series
         * @param B Second time series
         * @param n Length of first series
         * @param m Length of second series
         * @param dim Number of dimensions
         * @return Pair of (distance, path)
         *
         * Executes the complete DTW computation on GPU and returns
         * both the distance and optimal path.
         */
        template<distance::MetricType M>
        std::pair<double, std::vector<std::pair<int, int>>>
        execute_with_path(const DoubleTimeSeries& A,
                          const DoubleTimeSeries& B,
                          int n, int m, int dim) const {

            ensure_device_initialized();

            if (n < 100 || m < 100) {
                execution::SequentialStrategy seq_strategy;
                DoubleMatrix D;
                seq_strategy.initialize_matrix(D, n, m);
                seq_strategy.template execute_with_constraint<M>(D, A, B, n, m, dim);
                auto path = utils::backtrack_path(D);
                return {D(n, m), path};
            }

            auto result = dtw_cuda_impl(A, B, M, tile_size_);
            return {result.distance, result.path};
        }

        /**
         * @brief Execute DTW computation conforming to concept interface
         * @tparam M Distance metric type
         * @param D Cost matrix (updated with final cost)
         * @param A First time series
         * @param B Second time series
         * @param n Length of first series
         * @param m Length of second series
         * @param dim Number of dimensions
         */
        template<distance::MetricType M>
        void execute(DoubleMatrix& D,
                     const DoubleTimeSeries& A,
                     const DoubleTimeSeries& B,
                     int n, int m, int dim) const {

            auto [distance, path] = execute_with_path<M>(A, B, n, m, dim);

            D.resize(n + 1, m + 1, std::numeric_limits<double>::infinity());
            D(n, m) = distance;
            last_path_ = path;
        }

        /**
         * @brief Extract result from cost matrix
         * @param D Cost matrix
         * @return Pair of (distance, path)
         */
        std::pair<double, std::vector<std::pair<int, int>>>
        extract_result(const DoubleMatrix& D) const {
            int n = D.rows() - 1;
            int m = D.cols() - 1;
            return {D(n, m), last_path_};
        }

        /**
         * @brief Set tile size for blocked processing
         * @param tile_size New tile size
         */
        void set_tile_size(int tile_size) { tile_size_ = tile_size; }

        /**
         * @brief Get current tile size
         * @return Current tile size
         */
        int get_tile_size() const { return tile_size_; }

        /**
         * @brief Get strategy name
         * @return "CUDA-Optimized"
         */
        std::string_view name() const { return "CUDA-Optimized"; }

        /**
         * @brief Check if strategy uses parallel execution
         * @return True (CUDA is inherently parallel)
         */
        bool is_parallel() const { return true; }

         /**
         * @brief Check if CUDA is available on the system
         * @return True if at least one CUDA device is available
         */
        static bool is_available() {
            return is_cuda_available();
        }

        /**
         * @brief Get information about available CUDA devices
         * @return String containing device specifications
         */
        static std::string get_device_info() {
            return get_cuda_device_info();
        }
    };

    /**
     * @brief Convenience function for direct CUDA DTW computation
     * @tparam M Distance metric type
     * @param A First time series
     * @param B Second time series
     * @param tile_size Size of tiles for blocked processing
     * @return Pair of (distance, path)
     */
    template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
    inline std::pair<double, std::vector<std::pair<int, int>>> dtw_cuda(
            const DoubleTimeSeries& A,
            const DoubleTimeSeries& B,
            int tile_size = DEFAULT_TILE_SIZE) {

        auto result = dtw_cuda_impl(A, B, M, tile_size);
        return {result.distance, result.path};
    }

} // namespace cuda
} // namespace parallel

namespace execution {

    /// @brief Alias for CUDA strategy in execution namespace
    using CUDAStrategy = parallel::cuda::CUDAStrategy;
}

} // namespace dtw_accelerator

#endif // USE_CUDA
#endif // DTW_ACCELERATOR_CUDA_STRATEGY_HPP