/**
 * @file cuda_dtw.hpp
 * @brief Main header for CUDA-accelerated DTW implementations
 * @author UobinoPino
 * @date 2024
 *
 * This file provides the main interface for CUDA-accelerated Dynamic Time Warping
 * computations, including convenience aliases and helper functions for CUDA support.
 */

#ifndef DTW_ACCELERATOR_CUDA_DTW_HPP
#define DTW_ACCELERATOR_CUDA_DTW_HPP

#ifdef USE_CUDA

#include "execution/cuda_strategy.hpp"
#include "execution/cuda_launcher.hpp"

namespace dtw_accelerator {

    /**
     * @namespace cuda
     * @brief CUDA-accelerated DTW implementations and utilities
     *
     * This namespace provides convenient access to CUDA-accelerated DTW
     * algorithms and related utilities through simplified aliases.
     */
    namespace cuda {


    /// @brief CUDA execution strategy for DTW algorithms
    using ::dtw_accelerator::parallel::cuda::CUDAStrategy;

    /// @brief Direct CUDA DTW computation function
    using ::dtw_accelerator::parallel::cuda::dtw_cuda;

    /// @brief Check CUDA availability on the system
    using ::dtw_accelerator::parallel::cuda::is_cuda_available;

    /// @brief Get information about available CUDA devices
    using ::dtw_accelerator::parallel::cuda::get_cuda_device_info;

    } // namespace cuda
} // namespace dtw_accelerator

#endif // USE_CUDA
#endif // DTW_ACCELERATOR_CUDA_DTW_HPP