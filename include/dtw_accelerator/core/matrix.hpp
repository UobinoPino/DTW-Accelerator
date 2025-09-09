/**
 * @file matrix.hpp
 * @brief Matrix data structure for DTW cost matrix storage
 * @author UobinoPino
 * @date 2024
 *
 * This file provides a contiguous memory matrix implementation
 * optimized for DTW algorithms with support for parallel operations.
 */

#ifndef DTW_ACCELERATOR_MATRIX_HPP
#define DTW_ACCELERATOR_MATRIX_HPP

#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace dtw_accelerator {

    /**
    * @brief Contiguous memory matrix implementation for DTW computations
    * @tparam T Element type (typically double for DTW cost matrices)
    *
    * This class provides a 2D matrix with contiguous memory storage,
    * optimized for cache performance and parallel operations.
    * Elements are stored in row-major order.
    */
    template<typename T>
    class Matrix {
    private:
        /// @brief Contiguous data storage
        std::vector<T> data_;

        /// @brief Number of rows in the matrix
        size_t rows_;

        /// @brief Number of columns in the matrix
        size_t cols_;

        /**
         * @brief Proxy class for handling bool specialization
         *
         * This proxy allows proper assignment to vector<bool> elements
         */
        class BoolProxy {
        private:
            std::vector<bool>& data_;
            size_t index_;
        public:
            /**
             * @brief Construct a bool proxy
             * @param data Reference to the underlying bool vector
             * @param index Index in the vector
             */
            BoolProxy(std::vector<bool>& data, size_t index) : data_(data), index_(index) {}

            /**
             * @brief Implicit conversion to bool
             */
            operator bool() const { return data_[index_]; }

            /**
             * @brief Assignment operator for bool values
             * @param value The bool value to assign
             * @return Reference to this proxy
             */
            BoolProxy& operator=(bool value) {
                data_[index_] = value;
                return *this;
            }
        };

    public:
        /**
         * @brief Default constructor creating an empty matrix
         */
        Matrix() : rows_(0), cols_(0) {}

        /**
         * @brief Construct a matrix with specified dimensions
         * @param rows Number of rows
         * @param cols Number of columns
         * @param init_val Initial value for all elements
         */
        Matrix(size_t rows, size_t cols, T init_val = T{})
                : data_(rows * cols, init_val), rows_(rows), cols_(cols) {}

        /**
         * @brief Copy constructor
         * @param other Matrix to copy from
         */
        Matrix(const Matrix& other)
                : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {}

        /**
         * @brief Move constructor
         * @param other Matrix to move from
         */
        Matrix(Matrix&& other) noexcept
                : data_(std::move(other.data_)), rows_(other.rows_), cols_(other.cols_) {
            other.rows_ = 0;
            other.cols_ = 0;
        }

        /**
         * @brief Copy assignment operator
         * @param other Matrix to copy from
         * @return Reference to this matrix
         */
        Matrix& operator=(const Matrix& other) {
            if (this != &other) {
                data_ = other.data_;
                rows_ = other.rows_;
                cols_ = other.cols_;
            }
            return *this;
        }

        /**
         * @brief Move assignment operator
         * @param other Matrix to move from
         * @return Reference to this matrix
         */
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

        /**
         * @brief Access matrix element (non-bool types)
         * @tparam U Element type (SFINAE for non-bool types)
         * @param i Row index
         * @param j Column index
         * @return Reference to the element
         */
        template <typename U = T>
        typename std::enable_if<!std::is_same<U, bool>::value, T&>::type
        operator()(size_t i, size_t j) {
            return data_[i * cols_ + j];
        }

        /**
         * @brief Access matrix element (bool specialization)
         * @tparam U Element type (SFINAE for bool type)
         * @param i Row index
         * @param j Column index
         * @return BoolProxy for proper bool handling
         */
        template <typename U = T>
        typename std::enable_if<std::is_same<U, bool>::value, BoolProxy>::type
        operator()(size_t i, size_t j) {
            return BoolProxy(data_, i * cols_ + j);
        }

        /**
         * @brief Const access to matrix element
         * @param i Row index
         * @param j Column index
         * @return Const reference to the element
         */
        inline const T& operator()(size_t i, size_t j) const {
            return data_[i * cols_ + j];
        }

        /**
         * @brief Safe element access with bounds checking
         * @param i Row index
         * @param j Column index
         * @return Reference to the element
         * @throws std::out_of_range if indices are out of bounds
         */
        inline T& at(size_t i, size_t j) {
            if (i >= rows_ || j >= cols_) {
                throw std::out_of_range("Matrix index out of range");
            }
            return data_[i * cols_ + j];
        }

        /**
         * @brief Safe const element access with bounds checking
         * @param i Row index
         * @param j Column index
         * @return Const reference to the element
         * @throws std::out_of_range if indices are out of bounds
         */
        inline const T& at(size_t i, size_t j) const {
            if (i >= rows_ || j >= cols_) {
                throw std::out_of_range("Matrix index out of range");
            }
            return data_[i * cols_ + j];
        }

        /**
         * @brief Get raw pointer to underlying data
         * @return Pointer to the first element
         *
         * Useful for MPI/CUDA operations
         */
        T* data() { return data_.data(); }

        /**
         * @brief Get const raw pointer to underlying data
         * @return Const pointer to the first element
         */
        const T* data() const { return data_.data(); }

        /**
         * @brief Get pointer to a specific row
         * @param i Row index
         * @return Pointer to the first element of the row
         */
        T* row(size_t i) { return &data_[i * cols_]; }

        /**
         * @brief Get const pointer to a specific row
         * @param i Row index
         * @return Const pointer to the first element of the row
         */
        const T* row(size_t i) const { return &data_[i * cols_]; }

        /**
         * @brief Get number of rows
         * @return Number of rows in the matrix
         */
        size_t rows() const { return rows_; }

        /**
         * @brief Get number of columns
         * @return Number of columns in the matrix
         */
        size_t cols() const { return cols_; }

        /**
         * @brief Get total number of elements
         * @return Total number of elements (rows * cols)
         */
        size_t size() const { return rows_ * cols_; }

        /**
         * @brief Resize the matrix
         * @param new_rows New number of rows
         * @param new_cols New number of columns
         * @param init_val Value for new elements
         */
        void resize(size_t new_rows, size_t new_cols, T init_val = T{}) {
            data_.resize(new_rows * new_cols, init_val);
            rows_ = new_rows;
            cols_ = new_cols;
        }

        /**
         * @brief Fill all elements with a value
         * @param val Value to fill the matrix with
         */
        void fill(T val) {
            std::fill(data_.begin(), data_.end(), val);
        }

        /**
         * @brief Clear the matrix
         *
         * After this operation, the matrix will be empty with zero dimensions
         */
        void clear() {
            data_.clear();
            rows_ = 0;
            cols_ = 0;
        }

    };

    /// @brief Type alias for double-precision matrices
    using DoubleMatrix = Matrix<double>;
    /// @brief Type alias for boolean matrices (for masks)
    using BoolMatrix = Matrix<bool>;

} // namespace dtw_accelerator

#endif // DTW_ACCELERATOR_MATRIX_HPP