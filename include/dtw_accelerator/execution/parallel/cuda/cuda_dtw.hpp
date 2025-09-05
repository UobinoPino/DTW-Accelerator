#ifndef DTW_ACCELERATOR_CUDA_DTW_HPP
#define DTW_ACCELERATOR_CUDA_DTW_HPP

#ifdef USE_CUDA

#include "execution/cuda_strategy.hpp"
#include "execution/cuda_launcher.hpp"

namespace dtw_accelerator {
namespace cuda {

using ::dtw_accelerator::parallel::cuda::CUDAStrategy;
using ::dtw_accelerator::parallel::cuda::dtw_cuda;
using ::dtw_accelerator::parallel::cuda::is_cuda_available;
using ::dtw_accelerator::parallel::cuda::get_cuda_device_info;

} // namespace cuda
} // namespace dtw_accelerator

#endif // USE_CUDA
#endif // DTW_ACCELERATOR_CUDA_DTW_HPP