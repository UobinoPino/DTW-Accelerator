#ifndef DTW_ACCELERATOR_TIME_SERIES_HPP
#define DTW_ACCELERATOR_TIME_SERIES_HPP

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace dtw_accelerator {

    template<typename T = double>
    class TimeSeries {
    private:
        std::vector<T> data_;
        size_t length_;
        size_t dimensions_;

    public:
        // Constructors
        TimeSeries() : length_(0), dimensions_(0) {}

        TimeSeries(size_t length, size_t dimensions, T init_val = T{})
                : data_(length * dimensions, init_val), length_(length), dimensions_(dimensions) {}


        // Copy constructor
        TimeSeries(const TimeSeries& other) = default;

        // Move constructor
        TimeSeries(TimeSeries&& other) noexcept = default;

        // Assignment operators
        TimeSeries& operator=(const TimeSeries& other) = default;
        TimeSeries& operator=(TimeSeries&& other) noexcept = default;

        // Row-major layout: data_[i * dimensions_ + d] = series[i][d]
        T* operator[](size_t i) {
            return &data_[i * dimensions_];
        }

        const T* operator[](size_t i) const {
            return &data_[i * dimensions_];
        }

        // Element access
        T& at(size_t i, size_t d) {
            if (i >= length_ || d >= dimensions_) {
                throw std::out_of_range("TimeSeries index out of range");
            }
            return data_[i * dimensions_ + d];
        }

        const T& at(size_t i, size_t d) const {
            if (i >= length_ || d >= dimensions_) {
                throw std::out_of_range("TimeSeries index out of range");
            }
            return data_[i * dimensions_ + d];
        }

        // Direct data access for MPI/CUDA
        T* data() { return data_.data(); }
        const T* data() const { return data_.data(); }

        // Size accessors
        size_t length() const { return length_; }
        size_t size() const { return length_; }  // Alias for compatibility
        size_t dimensions() const { return dimensions_; }
        size_t dim() const { return dimensions_; }  // Alias
        bool empty() const { return length_ == 0; }

        // Resize
        void resize(size_t new_length, size_t new_dimensions, T init_val = T{}) {
            data_.resize(new_length * new_dimensions, init_val);
            length_ = new_length;
            dimensions_ = new_dimensions;
        }

        // Clear
        void clear() {
            data_.clear();
            length_ = 0;
            dimensions_ = 0;
        }
    };

    using DoubleTimeSeries = TimeSeries<double>;
    using FloatTimeSeries = TimeSeries<float>;

} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_TIME_SERIES_HPP