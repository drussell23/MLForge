#include "ml/algorithms/logistic_regression.h"
#include "ml/core/matrix.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>
#include <random>
#include <iostream>

// ---------------------------------------------------------------------
// Define the matrix type for logistic regression.
// By default, we use Matrix2D. To switch to Matrix3D, define this macro externally.
#ifndef LINEAR_REGRESSION_MATRIX_TYPE
    #define LINEAR_REGRESSION_MATRIX_TYPE ml::core::Matrix2D
#endif

// Alias for ease-of-use.
#define LR_MATRIX_TYPE LINEAR_REGRESSION_MATRIX_TYPE

namespace {

// ---------------------------------------------------------------------
// Helper Functions for Matrix Operations
// ---------------------------------------------------------------------

// Transpose a matrix.
LR_MATRIX_TYPE transpose(const LR_MATRIX_TYPE& A) {
    std::size_t rows = A.rows();
    std::size_t cols = A.cols();
    LR_MATRIX_TYPE result(cols, rows, 0.0);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            result(j, i) = A(i, j);
    return result;
}

// Multiply two matrices.
LR_MATRIX_TYPE multiply(const LR_MATRIX_TYPE& A, const LR_MATRIX_TYPE& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Incompatible dimensions for matrix multiplication.");
    }
    std::size_t m = A.rows();
    std::size_t n = A.cols();
    std::size_t p = B.cols();
    LR_MATRIX_TYPE C(m, p, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

// Multiply matrix A by vector v.
std::vector<double> multiply(const LR_MATRIX_TYPE& A, const std::vector<double>& v) {
    if (A.cols() != v.size()) {
        throw std::invalid_argument("Incompatible dimensions for matrix-vector multiplication.");
    }
    std::size_t m = A.rows();
    std::size_t n = A.cols();
    std::vector<double> result(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            sum += A(i, j) * v[j];
        }
        result[i] = sum;
    }
    return result;
}

// Invert a square matrix using Gauss-Jordan elimination.
LR_MATRIX_TYPE invertMatrix(const LR_MATRIX_TYPE& A) {
    std::size_t n = A.rows();
    if (A.cols() != n) {
        throw std::invalid_argument("Only square matrices can be inverted.");
    }
    LR_MATRIX_TYPE aug(n, 2*n, 0.0);
    // Create augmented matrix [A | I]
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            aug(i, j) = A(i, j);
        }
        for (std::size_t j = n; j < 2*n; ++j) {
            aug(i, j) = (j - n == i) ? 1.0 : 0.0;
        }
    }
    // Gauss-Jordan elimination.
    for (std::size_t i = 0; i < n; ++i) {
        double pivot = aug(i, i);
        if (std::abs(pivot) < std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }
        for (std::size_t j = 0; j < 2*n; ++j) {
            aug(i, j) /= pivot;
        }
        for (std::size_t k = 0; k < n; ++k) {
            if (k == i) continue;
            double factor = aug(k, i);
            for (std::size_t j = 0; j < 2*n; ++j) {
                aug(k, j) -= factor * aug(i, j);
            }
        }
    }
    LR_MATRIX_TYPE inv(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            inv(i, j) = aug(i, j+n);
    return inv;
}

// ---------------------------------------------------------------------
// Compute the Cross-Entropy Cost Function with L2 Regularization
// ---------------------------------------------------------------------
double computeCost(const LR_MATRIX_TYPE& X, const std::vector<double>& y,
                   const std::vector<double>& coeffs, double alpha) {
    std::size_t n_samples = X.rows();
    std::size_t n_params = X.cols();
    double cost = 0.0;
    for (std::size_t i = 0; i < n_samples; ++i) {
        double z = 0.0;
        for (std::size_t j = 0; j < n_params; ++j) {
            z += X(i, j) * coeffs[j];
        }
        double pred = 1.0 / (1.0 + std::exp(-z));
        // Clamp predictions to avoid log(0)
        pred = std::min(std::max(pred, 1e-10), 1.0 - 1e-10);
        cost += -y[i] * std::log(pred) - (1 - y[i]) * std::log(1 - pred);
    }
    cost /= n_samples;
    // L2 regularization (skip intercept at index 0)
    if (alpha > 0.0) {
        double reg_cost = 0.0;
        for (std::size_t j = 1; j < coeffs.size(); ++j) {
            reg_cost += coeffs[j] * coeffs[j];
        }
        cost += (alpha / (2.0 * n_samples)) * reg_cost;
    }
    return cost;
}

} // end anonymous namespace

namespace ml {
namespace algorithms {

LogisticRegression::LogisticRegression(double alpha, const std::string& regularization_type)
    : alpha_(alpha), regularization_type_(regularization_type), learning_rate_(0.01)
{
    coefficients_.clear();
}

LogisticRegression::~LogisticRegression() {
    // No dynamic memory to clean up.
}

void LogisticRegression::fit(const LR_MATRIX_TYPE& X, const std::vector<double>& y) {
    // Advanced training hyperparameters.
    int epochs = 1000;
    int batch_size = X.rows(); // Full batch by default; modify for mini-batch training.
    double tolerance = 1e-6;
    double decay_rate = 0.001;
    int patience = 50; // Early stopping patience.

    std::size_t n_samples = X.rows();
    std::size_t n_features = X.cols();

    // Augment X with an intercept term.
    std::vector<double> data;
    data.reserve(n_samples * (n_features + 1));
    for (std::size_t i = 0; i < n_samples; ++i) {
        data.push_back(1.0);
        for (std::size_t j = 0; j < n_features; ++j) {
            data.push_back(X(i, j));
        }
    }
    LR_MATRIX_TYPE X_aug(n_samples, n_features + 1, data);

    // Initialize coefficients to zero.
    coefficients_.assign(n_features + 1, 0.0);

    double prev_cost = std::numeric_limits<double>::infinity();
    int no_improve_epochs = 0;

    // Create a vector of indices for mini-batch shuffling.
    std::vector<std::size_t> indices(n_samples);
    for (std::size_t i = 0; i < n_samples; ++i) {
        indices[i] = i;
    }
    std::random_device rd;
    std::mt19937 gen(rd());

    // Training loop.
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle indices for mini-batch formation.
        std::shuffle(indices.begin(), indices.end(), gen);

        // Process mini-batches.
        for (std::size_t start = 0; start < n_samples; start += batch_size) {
            std::size_t end = std::min(start + static_cast<std::size_t>(batch_size), n_samples);
            std::size_t current_batch = end - start;

            // Build mini-batch matrix and labels.
            std::vector<double> batch_data;
            batch_data.reserve(current_batch * (n_features + 1));
            std::vector<double> batch_y;
            batch_y.reserve(current_batch);
            for (std::size_t i = start; i < end; ++i) {
                std::size_t idx = indices[i];
                for (std::size_t j = 0; j < n_features + 1; ++j) {
                    batch_data.push_back(X_aug(idx, j));
                }
                batch_y.push_back(y[idx]);
            }
            LR_MATRIX_TYPE X_batch(current_batch, n_features + 1, batch_data);
            std::vector<double> grad = computeGradient(X_batch, batch_y);
            updateCoefficients(grad, learning_rate_);
        }

        // Apply learning rate decay.
        learning_rate_ *= (1.0 / (1.0 + decay_rate * epoch));

        // Compute cost on the full dataset.
        double cost = computeCost(X_aug, y, coefficients_, alpha_);
        // Optional: Print cost for debugging.
        // std::cout << "Epoch " << epoch << " cost: " << cost << std::endl;

        if (std::abs(prev_cost - cost) < tolerance) {
            no_improve_epochs++;
            if (no_improve_epochs >= patience) {
                break; // Early stopping.
            }
        } else {
            no_improve_epochs = 0;
        }
        prev_cost = cost;
    }
}

std::vector<double> LogisticRegression::predictProbability(const LR_MATRIX_TYPE& X) const {
    std::size_t n_samples = X.rows();
    std::size_t n_features = X.cols();
    std::vector<double> probabilities;
    probabilities.reserve(n_samples);
    // Assume X is not augmented.
    for (std::size_t i = 0; i < n_samples; ++i) {
        double z = coefficients_[0];
        for (std::size_t j = 0; j < n_features; ++j) {
            z += coefficients_[j + 1] * X(i, j);
        }
        probabilities.push_back(sigmoid(z));
    }
    return probabilities;
}

std::vector<int> LogisticRegression::predict(const LR_MATRIX_TYPE& X) const {
    std::vector<double> probs = predictProbability(X);
    std::vector<int> predictions;
    predictions.reserve(probs.size());
    for (double p : probs) {
        predictions.push_back(p >= 0.5 ? 1 : 0);
    }
    return predictions;
}

std::vector<double> LogisticRegression::getCoefficients() const {
    return coefficients_;
}

void LogisticRegression::setCoefficients(const std::vector<double>& coeffs) {
    coefficients_ = coeffs;
}

double LogisticRegression::getAlpha() const {
    return alpha_;
}

std::string LogisticRegression::getRegularizationType() const {
    return regularization_type_;
}

double LogisticRegression::sigmoid(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

std::vector<double> LogisticRegression::computeGradient(const LR_MATRIX_TYPE& X, const std::vector<double>& y) const {
    std::size_t n_samples = X.rows();
    std::size_t n_params = X.cols();
    std::vector<double> gradient(n_params, 0.0);
    for (std::size_t i = 0; i < n_samples; ++i) {
        double z = 0.0;
        for (std::size_t j = 0; j < n_params; ++j) {
            z += X(i, j) * coefficients_[j];
        }
        double pred = sigmoid(z);
        double error = pred - y[i];
        for (std::size_t j = 0; j < n_params; ++j) {
            gradient[j] += error * X(i, j);
        }
    }
    for (std::size_t j = 0; j < n_params; ++j) {
        gradient[j] /= n_samples;
    }
    if (alpha_ > 0.0) {
        for (std::size_t j = 1; j < n_params; ++j) {
            gradient[j] += alpha_ * coefficients_[j] / n_samples;
        }
    }
    return gradient;
}

void LogisticRegression::updateCoefficients(const std::vector<double>& gradient, double learning_rate) {
    for (std::size_t j = 0; j < coefficients_.size(); ++j) {
        coefficients_[j] -= learning_rate * gradient[j];
    }
}

} // namespace algorithms
} // namespace ml
