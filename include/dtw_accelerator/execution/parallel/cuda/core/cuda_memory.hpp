/**
 * @file cuda_memory.hpp
 * @brief CUDA memory management utilities for DTW computations
 * @author UobinoPino
 * @date 2024
 *
 * This file provides RAII-compliant memory management utilities for CUDA
 * device memory, ensuring proper allocation and deallocation of GPU resources.
 */

#ifndef DTW_ACCELERATOR_CUDA_MEMORY_HPP
#define DTW_ACCELERATOR_CUDA_MEMORY_HPP

#include <cuda_runtime.h>
#include <utility>

namespace dtw_accelerator {
    namespace parallel {
        namespace cuda {

            /**
             * @namespace memory
             * @brief CUDA memory management utilities
             */
            namespace memory {

                /**
                 * @brief RAII wrapper for CUDA device memory
                 * @tparam T Data type to store in device memory
                 *
                 * This class provides automatic memory management for CUDA device
                 * memory using RAII principles. Memory is automatically freed when
                 * the object goes out of scope.
                 */

                template<typename T>
                struct DeviceBuffer {
                    /// @brief Pointer to device memory
                    T* data = nullptr;

                    /// @brief Number of elements allocated
                    size_t size = 0;

                    /**
                     * @brief Construct a device buffer with specified capacity
                     * @param n Number of elements to allocate
                     * @throws std::runtime_error if CUDA allocation fails
                     */
                    explicit DeviceBuffer(size_t n) : size(n) {
                        if (n > 0) {
                            cudaMalloc(&data, n * sizeof(T));
                        }
                    }

                    /**
                     * @brief Destructor - automatically frees device memory
                     */
                    ~DeviceBuffer() {
                        if (data) cudaFree(data);
                    }

                    /// @brief Deleted copy constructor to prevent double-free
                    DeviceBuffer(const DeviceBuffer&) = delete;

                    /// @brief Deleted copy assignment to prevent double-free
                    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

                    /**
                     * @brief Move constructor for efficient transfer of ownership
                     * @param other Buffer to move from
                     */
                    DeviceBuffer(DeviceBuffer&& other) noexcept
                            : data(std::exchange(other.data, nullptr)), size(other.size) {}
                };

            } // namespace memory
        } // namespace cuda
    } // namespace parallel
} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_CUDA_MEMORY_HPP