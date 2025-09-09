/**
 * @file time_series.hpp
 * @brief Time series data structure for DTW computations
 * @author UobinoPino
 * @date 2024
 *
 * This file contains the TimeSeries class template which provides
 * a contiguous memory representation of time series data optimized
 * for DTW algorithms and parallel processing.
 */

#ifndef DTW_ACCELERATOR_TIME_SERIES_HPP
#define DTW_ACCELERATOR_TIME_SERIES_HPP

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace dtw_accelerator {

    /**
     * @brief A contiguous memory time series container with row-major layout
     * @tparam T The data type of time series elements (default: double)
     *
     * This class provides an efficient representation of multivariate time series
     * data using contiguous memory storage. The data is stored in row-major order
     * where each time point's dimensions are stored consecutively.
     *
     * Memory layout: data[time_index * dimensions + dimension_index]
     */

    template<typename T = double>
    class TimeSeries {
    private:

        /// @brief Contiguous storage for time series data
        std::vector<T> data_;

        /// @brief Number of time points in the series
        size_t length_;

        /// @brief Number of dimensions per time point
        size_t dimensions_;

    public:
        /**
         * @brief Default constructor creating an empty time series
         */
        TimeSeries() : length_(0), dimensions_(0) {}

        /**
         * @brief Constructs a time series with specified dimensions
         * @param length Number of time points
         * @param dimensions Number of dimensions per time point
         * @param init_val Initial value for all elements
         */
        TimeSeries(size_t length, size_t dimensions, T init_val = T{})
                : data_(length * dimensions, init_val), length_(length), dimensions_(dimensions) {}


        /**
         * @brief Copy constructor
         * @param other The time series to copy from
         */
        TimeSeries(const TimeSeries& other) = default;

        /**
         * @brief Move constructor
         * @param other The time series to move from
         */
        TimeSeries(TimeSeries&& other) noexcept = default;

        /**
         * @brief Copy assignment operator
         * @param other The time series to copy from
         * @return Reference to this object
         */
        TimeSeries& operator=(const TimeSeries& other) = default;

        /**
         * @brief Move assignment operator
         * @param other The time series to move from
         * @return Reference to this object
         */
        TimeSeries& operator=(TimeSeries&& other) noexcept = default;

        /**
         * @brief Access a time point by index
         * @param i Time point index
         * @return Pointer to the first dimension of the time point
         *
         * Row-major layout: returns pointer to data_[i * dimensions_]
         */
        T* operator[](size_t i) {
            return &data_[i * dimensions_];
        }

        /**
         * @brief Access a time point by index (const version)
         * @param i Time point index
         * @return Const pointer to the first dimension of the time point
         */
        const T* operator[](size_t i) const {
            return &data_[i * dimensions_];
        }

        /**
         * @brief Safe element access with bounds checking
         * @param i Time point index
         * @param d Dimension index
         * @return Reference to the element
         * @throws std::out_of_range if indices are out of bounds
         */
        T& at(size_t i, size_t d) {
            if (i >= length_ || d >= dimensions_) {
                throw std::out_of_range("TimeSeries index out of range");
            }
            return data_[i * dimensions_ + d];
        }

        /**
        * @brief Safe element access with bounds checking (const version)
        * @param i Time point index
        * @param d Dimension index
        * @return Const reference to the element
        * @throws std::out_of_range if indices are out of bounds
        */
        const T& at(size_t i, size_t d) const {
            if (i >= length_ || d >= dimensions_) {
                throw std::out_of_range("TimeSeries index out of range");
            }
            return data_[i * dimensions_ + d];
        }

        /**
        * @brief Get raw pointer to underlying data
        * @return Pointer to the first element
        *
        * Useful for MPI/CUDA operations requiring direct memory access
        */
        T* data() { return data_.data(); }

        /**
         * @brief Get const raw pointer to underlying data
         * @return Const pointer to the first element
         */
        const T* data() const { return data_.data(); }

        /**
         * @brief Get the number of time points
         * @return Number of time points in the series
         */
        size_t length() const { return length_; }

        /**
         * @brief Alias for length()
         * @return Number of time points in the series
         */
        size_t size() const { return length_; }

        /**
        * @brief Get the number of dimensions per time point
        * @return Number of dimensions
        */
        size_t dimensions() const { return dimensions_; }

        /**
         * @brief Alias for dimensions()
         * @return Number of dimensions
         */
        size_t dim() const { return dimensions_; }

        /**
         * @brief Check if the time series is empty
         * @return True if the series has no time points
         */
        bool empty() const { return length_ == 0; }

        /**
         * @brief Resize the time series
         * @param new_length New number of time points
         * @param new_dimensions New number of dimensions
         * @param init_val Value to initialize new elements
         */
        void resize(size_t new_length, size_t new_dimensions, T init_val = T{}) {
            data_.resize(new_length * new_dimensions, init_val);
            length_ = new_length;
            dimensions_ = new_dimensions;
        }

        /**
         * @brief Clear all data from the time series
         *
         * After this operation, the series will be empty with zero length and dimensions
         */
        void clear() {
            data_.clear();
            length_ = 0;
            dimensions_ = 0;
        }
    };
    /// @brief Type alias for double-precision time series
    using DoubleTimeSeries = TimeSeries<double>;

    /// @brief Type alias for single-precision time series
    using FloatTimeSeries = TimeSeries<float>;

} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_TIME_SERIES_HPP