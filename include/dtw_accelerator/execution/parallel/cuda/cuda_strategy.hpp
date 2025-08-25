#ifndef DTW_ACCELERATOR_CUDA_STRATEGY_HPP
#define DTW_ACCELERATOR_CUDA_STRATEGY_HPP

#ifdef USE_CUDA

#include <vector>
#include <utility>
#include <memory>
#include <stdexcept>
#include "cuda_wrapper.hpp"
#include "../../../core/dtw_concepts.hpp"
#include "../../../core/path_processing.hpp"
#include "../../../core/dtw_utils.hpp"
#include "../../../core/matrix.hpp"
#include "../../../execution/sequential/standard_strategy.hpp"

namespace dtw_accelerator {
namespace parallel {
namespace cuda {

// CUDA execution strategy
class CUDAStrategy {
private:
    int tile_size_;
    mutable bool device_initialized_;

    void ensure_device_initialized() const {
        if (!device_initialized_) {
            if (!is_cuda_available()) {
                throw std::runtime_error("No CUDA devices available");
            }
            device_initialized_ = true;
        }
    }

public:
    explicit CUDAStrategy(int tile_size = DEFAULT_TILE_SIZE)
        : tile_size_(tile_size), device_initialized_(false) {}

    void initialize_matrix(DoubleMatrix& D, int n, int m) const {
        D.resize(n + 1, m + 1, std::numeric_limits<double>::infinity());
        D(0, 0) = 0.0;
    }

    template<distance::MetricType M>
    void execute(DoubleMatrix& D,
                 const DoubleTimeSeries& A,
                 const DoubleTimeSeries& B,
                 int n, int m, int dim) const {

        ensure_device_initialized();

        // For small sequences, fallback to CPU
        if (n < 100 || m < 100) {
            execution::SequentialStrategy seq_strategy;
            seq_strategy.template execute<M>(D, A, B, n, m, dim);
            return;
        }

        // Delegate to CUDA implementation
        auto result = dtw_cuda_impl(A, B, M, tile_size_);

        // Copy result back to D matrix
        D = std::move(result.cost_matrix);
    }

    template<distance::MetricType M>
    void execute_constrained(DoubleMatrix& D,
                            const DoubleTimeSeries& A,
                            const DoubleTimeSeries& B,
                            const std::vector<std::pair<int, int>>& window,
                            int n, int m, int dim) const {

        ensure_device_initialized();

        // For small sequences, fallback to CPU
        if (n < 100 || m < 100) {
            execution::SequentialStrategy seq_strategy;
            seq_strategy.template execute_constrained<M>(D, A, B, window, n, m, dim);
            return;
        }

        // Delegate to CUDA implementation
        auto result = dtw_constrained_cuda_impl(A, B, window, M, tile_size_);

        // Copy result back to D matrix
        D = std::move(result.cost_matrix);
    }

    template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
             distance::MetricType M = distance::MetricType::EUCLIDEAN>
    void execute_with_constraint(DoubleMatrix& D,
                                 const DoubleTimeSeries& A,
                                 const DoubleTimeSeries& B,
                                 int n, int m, int dim) const {

        ensure_device_initialized();

        // For small sequences, fallback to CPU
        if (n < 100 || m < 100) {
            execution::SequentialStrategy seq_strategy;
            seq_strategy.template execute_with_constraint<CT, R, S, M>(D, A, B, n, m, dim);
            return;
        }

        // Delegate to CUDA implementation
        auto result = dtw_with_constraint_cuda_impl(A, B, CT, R, S, M, tile_size_);

        // Copy result back to D matrix
        D = std::move(result.cost_matrix);
    }

    std::pair<double, std::vector<std::pair<int, int>>>
    extract_result(const DoubleMatrix& D) const {
        int n = D.rows() - 1;
        int m = D.cols() - 1;
        auto path = utils::backtrack_path(D);
        return {D(n, m), path};
    }

    void set_tile_size(int tile_size) { tile_size_ = tile_size; }
    int get_tile_size() const { return tile_size_; }

    std::string_view name() const { return "CUDA"; }
    bool is_parallel() const { return true; }

    // Check if CUDA is available
    static bool is_available() {
        return is_cuda_available();
    }

    // Get device properties
    static std::string get_device_info() {
        return get_cuda_device_info();
    }
};

// Convenience template functions

template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
inline std::pair<double, std::vector<std::pair<int, int>>> dtw_cuda(
        const DoubleTimeSeries& A,
        const DoubleTimeSeries& B,
        int tile_size = DEFAULT_TILE_SIZE) {

    auto result = dtw_cuda_impl(A, B, M, tile_size);
    auto path = utils::backtrack_path(result.cost_matrix);
    return {result.distance, path};
}

template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
inline std::pair<double, std::vector<std::pair<int, int>>> dtw_constrained_cuda(
        const DoubleTimeSeries& A,
        const DoubleTimeSeries& B,
        const std::vector<std::pair<int, int>>& window,
        int tile_size = DEFAULT_TILE_SIZE) {

    auto result = dtw_constrained_cuda_impl(A, B, window, M, tile_size);
    auto path = utils::backtrack_path(result.cost_matrix);
    return {result.distance, path};
}

template<constraints::ConstraintType CT, int R = 1, double S = 2.0,
         distance::MetricType M = distance::MetricType::EUCLIDEAN>
inline std::pair<double, std::vector<std::pair<int, int>>> dtw_cuda_with_constraint(
        const DoubleTimeSeries& A,
        const DoubleTimeSeries& B,
        int tile_size = DEFAULT_TILE_SIZE) {

    auto result = dtw_with_constraint_cuda_impl(A, B, CT, R, S, M, tile_size);
    auto path = utils::backtrack_path(result.cost_matrix);
    return {result.distance, path};
}

// FastDTW implementation using CUDA
template<distance::MetricType M = distance::MetricType::EUCLIDEAN>
inline std::pair<double, std::vector<std::pair<int, int>>> fastdtw_cuda(
        const DoubleTimeSeries& A,
        const DoubleTimeSeries& B,
        int radius = 1,
        int min_size = 100,
        int tile_size = DEFAULT_TILE_SIZE) {

    int n = A.size();
    int m = B.size();

    // Base case 1: if time series are small enough, use standard DTW
    if (n <= min_size && m <= min_size) {
        return dtw_cuda<M>(A, B, tile_size);
    }

    // Base case 2: if we can't downsample further
    if (n <= 2 || m <= 2) {
        return dtw_cuda<M>(A, B, tile_size);
    }

    // Recursive case:
    // 1. Coarsen the time series
    auto A_coarse = path::downsample(A);
    auto B_coarse = path::downsample(B);

    // 2. Recursively compute FastDTW on the coarsened data
    auto [cost_coarse, path_coarse] = fastdtw_cuda<M>(
            A_coarse, B_coarse, radius, min_size, tile_size
    );

    // 3. Project the low-resolution path to a higher resolution
    auto projected_path = path::expand_path(path_coarse, n, m);

    // 4. Create a search window around the projected path
    auto window = path::get_window(projected_path, n, m, radius);

    // 5. Compute constrained DTW within the window
    return dtw_constrained_cuda<M>(A, B, window, tile_size);
}

} // namespace cuda
} // namespace parallel

// Make CUDA strategy available in execution namespace for consistency
namespace execution {
    using CUDAStrategy = parallel::cuda::CUDAStrategy;
}

} // namespace dtw_accelerator

#endif // USE_CUDA

#endif // DTW_ACCELERATOR_CUDA_STRATEGY_HPP