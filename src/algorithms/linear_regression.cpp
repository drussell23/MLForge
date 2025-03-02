#include "ml/algorithms/linear_regression.h"
#include "ml/core/matrix.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

// ---------------------------------------------------------------------
// Define the matrix type for linear regression.
// By default, we use Matrix2D. To switch to Matrix3D, define this macro externally.
#ifndef LINEAR_REGRESSION_MATRIX_TYPE
    #define LINEAR_REGRESSION_MATRIX_TYPE ml::core::Matrix2D
#endif // LINEAR_REGRESSION_MATRIX_TYPE

// Alias for ease-of-use in function signatures.
#define LR_MATRIX_TYPE LINEAR_REGRESSION_MATRIX_TYPE

// ---------------------------------------------------------------------
// Helper Functions in an Anonymous Namespace
// ---------------------------------------------------------------------
namespace {

    // Transpose a matrix.
    LR_MATRIX_TYPE transpose(const LR_MATRIX_TYPE& A) {
        size_t rows = A.rows();
        size_t cols = A.cols();
        LR_MATRIX_TYPE result(cols, rows, 0.0);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; i < rows; ++i) {
                result(j, i) = A(i, j);
            }
        }
        return result;
    }

    // Multiply two matrices.
    LR_MATRIX_TYPE multiply(const LR_MATRIX_TYPE& A, const LR_MATRIX_TYPE& B) {
        if (A.cols() != B.rows()) {
            throw invalid_argument("Incompatible dimensions for matrix multiplication.");
        }

        size_t m = A.rows();
        size_t n = A.cols();
        size_t p = B.cols();
        LR_MATRIX_TYPE C(m, p, 0.0);

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < p; ++j) {
                double sum = 0.0;

                for (size_t k =0; k < n; ++k) {
                    sum += A(i, k) * B(k, j);
                }
                C(i, j) = sum;
            }
        }
        return C;
    }

    // Multiply a matrix (m x n) by a vector (n elements), returning a vector of m elements.  
    vector<double> multiply(const LR_MATRIX_TYPE& A, const vector<double>& v) {
        if (A.cols() != v.size()) {
            throw invalid_argument("Incompatible dimensions for matrix-vector multiplication.");
        }

        size_t m = A.rows();
        size_t n = A.cols();
        vector<double> result(m, 0.0);

        for (size_t i = 0; i < m; ++i) {
            double sum = 0.0;

            for (size_t j = 0; j < n; ++j) {
                sum += A(i, j) * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    // Invert a square matrix using Gauss-Jordan elimination. 
    LR_MATRIX_TYPE invertMatrix(const LR_MATRIX_TYPE& A) {
        size_t n = A.rows();

        if (A.cols() != n) {
            throw invalid_argument("Only square matrices can be inverted.");
        }

        // Create augmented matrix [A | I]
        LR_MATRIX_TYPE aug(n, 2 * n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                aug(i, j) = A(i, j);
            }
            for (size_t j = n; j < 2 * n; ++j) {
                aug(i, j) = (j - n == i) ? 1.0 : 0.0;
            }
        }

        // Perform Gauss-Jordan elimination. 
        for (size_t i = 0; i < n; ++i) {
            double pivot = aug(i, i);

            if (abs(pivot) < numeric_limits<double>::epsilon()) {
                throw runtime_error("Matrix is singular and cannot be inverted.");
            }

            for (size_t j = 0; j < 2 * n; ++j) {
                aug(i, j) /= pivot;
            }

            for (size_t k = 0; k < n; ++k) {
                if (k == i) continue;
                
                double factor = aug(k, i);

                for (size_t j = 0; j < 2 * n; ++j) {
                    aug(k, j) -= factor * aug(i, j);
                }
            }
        }

        // Extract the inverse matrix.
        LR_MATRIX_TYPE inv(n, n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                inv(i, j) = aug(i, j + n);
            }
        }
        return inv;
    }
}

namespace ml {
    namespace algorithms {
        LinearRegression::LinearRegression(double alpha, const string& regularization_type) : alpha_(alpha), regularization_type_(regularization_type) {
            coefficients_.clear();
        }

        LinearRegression::~LinearRegression() {
            // No dynamic memory to clean up.
        }

        void LinearRegression::fit(const LR_MATRIX_TYPE& X, const vector<double>& y) {
            computeCoefficients(X, y);
        }

        vector<double> LinearRegression::predict(const LR_MATRIX_TYPE& X) const {
            size_t n_samples = X.rows();
            size_t n_features = X.cols();
            vector<double> predictions;

            predictions.reserve(n_samples);

            // Predict by computing: intercept + sum(weights * feature)
            for (size_t i = 0; i < n_samples; ++i) {
                double pred = coefficients_[0]; // Intercept term.

                for (size_t j = 0; j < n_features; ++j) {
                    pred += coefficients_[j + 1] * X(i, j);
                }
                predictions.push_back(pred);
            }
            return predictions;
        }

        vector<double> LinearRegression::getCoefficients() const {
            return coefficients_;
        }

        void LinearRegression::setCoefficients(const vector<double>& coeffs) {
            coefficients_ = coeffs;
        }

        double LinearRegression::getAlpha() const {
            return alpha_;
        }

        string LinearRegression::getRegularizationType() const {
            return regularization_type_;
        }

        void LinearRegression::computeCoefficients(const LR_MATRIX_TYPE& X, const vector<double>& y) {
            // Augment X with an intercept column.
            size_t n_samples = X.rows();
            size_t n_features = X.cols();
            vector<double> data;
            
            data.reserve(n_samples * (n_features + 1));

            for (size_t i = 0; i < n_samples; ++i) {
                data.push_back(1.0); // Intercept.
                for (size_t j = 0; j < n_features; ++j) {
                    data.push_back(X(i, j));
                }
            }

            LR_MATRIX_TYPE X_aug(n_samples, n_features + 1, data);

            // Compute X_aug^T * X_aug.
            LR_MATRIX_TYPE X_trans = transpose(X_aug);
            LR_MATRIX_TYPE XTX = multiply(X_trans, X_aug);

            // Apply regularization (skipping intercept term).
            if (alpha_ > 0.0) {
                for (size_t i = 1; i < XTX.cols(); ++i) {
                    XTX(i, i) += alpha_;
                }
            }

            // Compute X_aug^T * y.
            vector<double> XTy = multiply(X_trans, y);

            // Invert XTX.
            LR_MATRIX_TYPE XTX_inv = invertMatrix(XTX);

            // Compute coefficients: (XTX_inv * XTy).
            coefficients_ = multiply(XTX_inv, XTy);
        }
    } // namespace algorithms
} // namespace ml