#ifndef DTW_ACCELERATOR_MATRIX_HPP
#define DTW_ACCELERATOR_MATRIX_HPP

#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace dtw_accelerator {

// Contiguous memory matrix implementation
    template<typename T>
    class Matrix {
    private:
        std::vector<T> data_;
        size_t rows_;
        size_t cols_;
        class BoolProxy {
        private:
            std::vector<bool>& data_;
            size_t index_;
        public:
            BoolProxy(std::vector<bool>& data, size_t index) : data_(data), index_(index) {}

            operator bool() const { return data_[index_]; }
            BoolProxy& operator=(bool value) {
                data_[index_] = value;
                return *this;
            }
        };

    public:
        // Constructors
        Matrix() : rows_(0), cols_(0) {}

        Matrix(size_t rows, size_t cols, T init_val = T{})
                : data_(rows * cols, init_val), rows_(rows), cols_(cols) {}

        // Copy constructor
        Matrix(const Matrix& other)
                : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {}

        // Move constructor
        Matrix(Matrix&& other) noexcept
                : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_) {
            other.rows_ = 0;
            other.cols_ = 0;
        }

        // Copy assignment
        Matrix& operator=(const Matrix& other) {
            if (this != &other) {
                data_ = other.data_;
                rows_ = other.rows_;
                cols_ = other.cols_;
            }
            return *this;
        }

        // Move assignment
        Matrix& operator=(Matrix&& other) noexcept {
            if (this != &other) {
                data_ = std::move(other.data_);
                rows_ = other.rows_;
                cols_ = other.cols_;
                other.rows_ = 0;
                other.cols_ = 0;
            }
            return *this;
        }

        // Element access

        template <typename U = T>
        typename std::enable_if<!std::is_same<U, bool>::value, T&>::type
        operator()(size_t i, size_t j) {
            return data_[i * cols_ + j];
        }

        // bool specialization
        template <typename U = T>
        typename std::enable_if<std::is_same<U, bool>::value, BoolProxy>::type
        operator()(size_t i, size_t j) {
            return BoolProxy(data_, i * cols_ + j);
        }



        inline const T& operator()(size_t i, size_t j) const {
            return data_[i * cols_ + j];
        }

        inline T& at(size_t i, size_t j) {
            if (i >= rows_ || j >= cols_) {
                throw std::out_of_range("Matrix index out of range");
            }
            return data_[i * cols_ + j];
        }

        inline const T& at(size_t i, size_t j) const {
            if (i >= rows_ || j >= cols_) {
                throw std::out_of_range("Matrix index out of range");
            }
            return data_[i * cols_ + j];
        }

        // Get raw pointer (for MPI/CUDA)
        T* data() { return data_.data(); }
        const T* data() const { return data_.data(); }

        // Get row pointer
        T* row(size_t i) { return &data_[i * cols_]; }
        const T* row(size_t i) const { return &data_[i * cols_]; }

        // Size accessors
        size_t rows() const { return rows_; }
        size_t cols() const { return cols_; }
        size_t size() const { return rows_ * cols_; }

        // Resize
        void resize(size_t new_rows, size_t new_cols, T init_val = T{}) {
            data_.resize(new_rows * new_cols, init_val);
            rows_ = new_rows;
            cols_ = new_cols;
        }

        // Fill
        void fill(T val) {
            std::fill(data_.begin(), data_.end(), val);
        }

        // Clear
        void clear() {
            data_.clear();
            rows_ = 0;
            cols_ = 0;
        }

    };

    using DoubleMatrix = Matrix<double>;
    using BoolMatrix = Matrix<bool>;

} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_MATRIX_HPP