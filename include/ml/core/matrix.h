#ifndef ML_CORE_MATRIX_H
#define ML_CORE_MATRIX_H

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <iostream>

// ==================== Macros ====================

// Macro to calculate 2D index in row-major order.
#define MATRIX2D_INDEX(row, col, ncols) ((row) * (ncols) + (col))

// Macro to calculate 3D index in a depth-major fashion (depth, row, col).
#define MATRIX3D_INDEX(depth, row, col, nrows, ncols) (((depth) * (nrows) * (ncols)) + ((row) * (ncols)) + (col))

// Macro to perform bounds checking and throw an exception with a custom message.
#define CHECK_BOUNDS(condition, msg) do { \
    if (!(condition)) { \
        throw std::out_of_range(msg); \
    } \
} while(0)

// Macro for optional debug logging (can be enabled/disabled by defining DEBUG_ENABLED).
#ifdef DEBUG_ENABLED
    #define DEBUG_MATRIX(msg) std::cerr << "DEBUG: " << msg << std::endl
#else
    #define DEBUG_MATRIX(msg)
#endif

// ==================== Namespace and Classes ====================

namespace ml {
namespace core {

/**
 * @class Matrix2D
 * @brief A 2D matrix class using row-major order.
 *
 * This class provides basic functionality for storing and manipulating 2D data.
 */
class Matrix2D {
public:
    // Default constructor creates an empty matrix.
    Matrix2D() : rows_(0), cols_(0) {}

    // Constructs a matrix with given rows and columns, initializing all elements to init_value.
    Matrix2D(std::size_t rows, std::size_t cols, double init_value = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, init_value) {
        DEBUG_MATRIX("Matrix2D constructed with dimensions " << rows << "x" << cols);
    }

    // Constructs a matrix with given dimensions and a provided data vector.
    Matrix2D(std::size_t rows, std::size_t cols, const std::vector<double>& data)
        : rows_(rows), cols_(cols), data_(data) {
        if(data.size() != rows * cols) {
            throw std::invalid_argument("Matrix2D constructor: data size does not match dimensions.");
        }
        DEBUG_MATRIX("Matrix2D constructed from external data.");
    }

    // Non-const element access with bounds checking.
    double& operator()(std::size_t row, std::size_t col) {
        CHECK_BOUNDS(row < rows_, "Matrix2D: row index out of bounds.");
        CHECK_BOUNDS(col < cols_, "Matrix2D: col index out of bounds.");
        return data_[MATRIX2D_INDEX(row, col, cols_)];
    }

    // Const element access with bounds checking.
    double operator()(std::size_t row, std::size_t col) const {
        CHECK_BOUNDS(row < rows_, "Matrix2D: row index out of bounds.");
        CHECK_BOUNDS(col < cols_, "Matrix2D: col index out of bounds.");
        return data_[MATRIX2D_INDEX(row, col, cols_)];
    }

    // Get number of rows.
    std::size_t rows() const { return rows_; }

    // Get number of columns.
    std::size_t cols() const { return cols_; }

    // Get a const reference to the underlying data.
    const std::vector<double>& data() const { return data_; }

    // Resize the matrix; optionally preserve overlapping data.
    void resize(std::size_t new_rows, std::size_t new_cols, bool preserve = false) {
        if (!preserve) {
            data_.assign(new_rows * new_cols, 0.0);
        } else {
            std::vector<double> new_data(new_rows * new_cols, 0.0);
            std::size_t min_rows = (new_rows < rows_) ? new_rows : rows_;
            std::size_t min_cols = (new_cols < cols_) ? new_cols : cols_;
            for (std::size_t r = 0; r < min_rows; ++r) {
                for (std::size_t c = 0; c < min_cols; ++c) {
                    new_data[MATRIX2D_INDEX(r, c, new_cols)] = (*this)(r, c);
                }
            }
            data_ = std::move(new_data);
        }
        rows_ = new_rows;
        cols_ = new_cols;
        DEBUG_MATRIX("Matrix2D resized to " << new_rows << "x" << new_cols);
    }

private:
    std::size_t rows_;            ///< Number of rows.
    std::size_t cols_;            ///< Number of columns.
    std::vector<double> data_;    ///< Data stored in row-major order.
};

/**
 * @class Matrix3D
 * @brief A 3D matrix class that stores data in a depth x rows x cols structure.
 *
 * Although it represents three-dimensional data, you can treat each "depth" layer
 * as a 2D matrix. This design is useful when you want to maintain a 2D appearance (e.g., for display)
 * while handling multiple layers or channels of data.
 */
class Matrix3D {
public:
    // Default constructor creates an empty 3D matrix.
    Matrix3D() : depth_(0), rows_(0), cols_(0) {}

    // Constructs a 3D matrix with given depth, rows, and columns; initializes all elements to init_value.
    Matrix3D(std::size_t depth, std::size_t rows, std::size_t cols, double init_value = 0.0)
        : depth_(depth), rows_(rows), cols_(cols), data_(depth * rows * cols, init_value) {
        DEBUG_MATRIX("Matrix3D constructed with dimensions " << depth << "x" << rows << "x" << cols);
    }

    // Constructs a 3D matrix with given dimensions and provided data.
    Matrix3D(std::size_t depth, std::size_t rows, std::size_t cols, const std::vector<double>& data)
        : depth_(depth), rows_(rows), cols_(cols), data_(data) {
        if (data.size() != depth * rows * cols) {
            throw std::invalid_argument("Matrix3D constructor: data size does not match dimensions.");
        }
        DEBUG_MATRIX("Matrix3D constructed from external data.");
    }

    // Non-const element access: depth, row, col.
    double& operator()(std::size_t d, std::size_t row, std::size_t col) {
        CHECK_BOUNDS(d < depth_, "Matrix3D: depth index out of bounds.");
        CHECK_BOUNDS(row < rows_, "Matrix3D: row index out of bounds.");
        CHECK_BOUNDS(col < cols_, "Matrix3D: col index out of bounds.");
        return data_[MATRIX3D_INDEX(d, row, col, rows_, cols_)];
    }

    // Const element access.
    double operator()(std::size_t d, std::size_t row, std::size_t col) const {
        CHECK_BOUNDS(d < depth_, "Matrix3D: depth index out of bounds.");
        CHECK_BOUNDS(row < rows_, "Matrix3D: row index out of bounds.");
        CHECK_BOUNDS(col < cols_, "Matrix3D: col index out of bounds.");
        return data_[MATRIX3D_INDEX(d, row, col, rows_, cols_)];
    }

    // Get the number of depth layers.
    std::size_t depth() const { return depth_; }

    // Get the number of rows.
    std::size_t rows() const { return rows_; }

    // Get the number of columns.
    std::size_t cols() const { return cols_; }

    // Get a const reference to the underlying data.
    const std::vector<double>& data() const { return data_; }

    // Resize the 3D matrix; optionally preserve overlapping data.
    void resize(std::size_t new_depth, std::size_t new_rows, std::size_t new_cols, bool preserve = false) {
        if (!preserve) {
            data_.assign(new_depth * new_rows * new_cols, 0.0);
        } else {
            std::vector<double> new_data(new_depth * new_rows * new_cols, 0.0);
            std::size_t min_depth = (new_depth < depth_) ? new_depth : depth_;
            std::size_t min_rows = (new_rows < rows_) ? new_rows : rows_;
            std::size_t min_cols = (new_cols < cols_) ? new_cols : cols_;
            for (std::size_t d = 0; d < min_depth; ++d) {
                for (std::size_t r = 0; r < min_rows; ++r) {
                    for (std::size_t c = 0; c < min_cols; ++c) {
                        new_data[MATRIX3D_INDEX(d, r, c, new_rows, new_cols)] = (*this)(d, r, c);
                    }
                }
            }
            data_ = std::move(new_data);
        }
        depth_ = new_depth;
        rows_ = new_rows;
        cols_ = new_cols;
        DEBUG_MATRIX("Matrix3D resized to " << new_depth << "x" << new_rows << "x" << new_cols);
    }

private:
    std::size_t depth_;           ///< Number of layers (depth).
    std::size_t rows_;            ///< Number of rows per layer.
    std::size_t cols_;            ///< Number of columns per layer.
    std::vector<double> data_;    ///< Underlying data stored in a flattened vector.
};

} // namespace core
} // namespace ml

#endif // ML_CORE_MATRIX_H
