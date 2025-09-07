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

class CUDAStrategy {
private:
    int tile_size_;
    mutable bool device_initialized_;
    mutable std::vector<std::pair<int, int>> last_path_;

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
        D.resize(1, 1, 0.0);
    }

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

    std::pair<double, std::vector<std::pair<int, int>>>
    extract_result(const DoubleMatrix& D) const {
        int n = D.rows() - 1;
        int m = D.cols() - 1;
        return {D(n, m), last_path_};
    }

    void set_tile_size(int tile_size) { tile_size_ = tile_size; }
    int get_tile_size() const { return tile_size_; }

    std::string_view name() const { return "CUDA-Optimized"; }
    bool is_parallel() const { return true; }

    static bool is_available() {
        return is_cuda_available();
    }

    static std::string get_device_info() {
        return get_cuda_device_info();
    }
};

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
    using CUDAStrategy = parallel::cuda::CUDAStrategy;
}

} // namespace dtw_accelerator

#endif // USE_CUDA
#endif // DTW_ACCELERATOR_CUDA_STRATEGY_HPP