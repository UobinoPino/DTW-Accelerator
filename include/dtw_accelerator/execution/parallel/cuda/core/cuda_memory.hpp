#ifndef DTW_ACCELERATOR_CUDA_MEMORY_HPP
#define DTW_ACCELERATOR_CUDA_MEMORY_HPP

#include <cuda_runtime.h>
#include <utility>

namespace dtw_accelerator {
    namespace parallel {
        namespace cuda {
            namespace memory {

                template<typename T>
                struct DeviceBuffer {
                    T* data = nullptr;
                    size_t size = 0;

                    explicit DeviceBuffer(size_t n) : size(n) {
                        if (n > 0) {
                            cudaMalloc(&data, n * sizeof(T));
                        }
                    }

                    ~DeviceBuffer() {
                        if (data) cudaFree(data);
                    }

                    DeviceBuffer(const DeviceBuffer&) = delete;
                    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
                    DeviceBuffer(DeviceBuffer&& other) noexcept
                            : data(std::exchange(other.data, nullptr)), size(other.size) {}
                };

            } // namespace memory
        } // namespace cuda
    } // namespace parallel
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_CUDA_MEMORY_HPP