#ifndef ML_ALGORITHMS_LOGISTIC_REGRESSION_H
#define ML_ALGORITHMS_LOGISTIC_REGRESSION_H

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "ml/core/matrix.h"

// ---------------------------------------------------------------------
// Macro to select the matrix type for logistic regression.
// By default, we use Matrix2D. Override this macro before including
// this header if you wish to work with a 3D matrix.
// ---------------------------------------------------------------------
#ifndef LOGISTIC_REGRESSION_MATRIX_TYPE
    #define LOGISTIC_REGRESSION_MATRIX_TYPE ml::core::Matrix2D
#endif

// Alias for ease-of-use in function signatures.
#define LR_MATRIX_TYPE LOGISTIC_REGRESSION_MATRIX_TYPE

namespace ml {
namespace algorithms {

/**
 * @class LogisticRegression
 * @brief Implements logistic regression for binary classification.
 *
 * This class provides methods to fit a logistic regression model using a gradient descent
 * based optimization (or similar iterative methods) and to make predictions. It optionally
 * supports a regularization term (Ridge or Lasso) if needed.
 *
 * The model uses input data in the form of a matrix defined by LOGISTIC_REGRESSION_MATRIX_TYPE,
 * which by default is a 2D matrix (ml::core::Matrix2D). You can override this to use a 3D matrix.
 */
class LogisticRegression {
public:
    /**
     * @brief Default constructor.
     * @param alpha Regularization parameter. (Set to 0.0 for no regularization.)
     * @param regularization_type The type of regularization ("none", "ridge", or "lasso").
     */
    LogisticRegression(double alpha = 0.0, const std::string& regularization_type = "none");

    /**
     * @brief Destructor.
     */
    ~LogisticRegression();

    /**
     * @brief Fit the logistic regression model to the provided data.
     *
     * @param X A matrix of size (n_samples x n_features) defined by LR_MATRIX_TYPE.
     * @param y A vector of size (n_samples) containing binary target values (0 or 1).
     *
     * This function optimizes the model parameters by maximizing the likelihood of the observed data.
     * If regularization is specified (alpha > 0), the corresponding penalty is applied.
     */
    void fit(const LR_MATRIX_TYPE& X, const std::vector<double>& y);

    /**
     * @brief Predict binary class labels for a given matrix of features.
     *
     * @param X A matrix of size (m_samples x n_features) defined by LR_MATRIX_TYPE.
     * @return A vector of predicted binary values (0 or 1) for each sample.
     */
    std::vector<int> predict(const LR_MATRIX_TYPE& X) const;

    /**
     * @brief Predict probabilities for the positive class for a given matrix of features.
     *
     * @param X A matrix of size (m_samples x n_features) defined by LR_MATRIX_TYPE.
     * @return A vector of probabilities (in the range [0,1]) for each sample.
     */
    std::vector<double> predictProbability(const LR_MATRIX_TYPE& X) const;

    /**
     * @brief Get the learned coefficients (including intercept).
     *
     * @return A vector of coefficients of size (n_features + 1), where the first element is the intercept.
     */
    std::vector<double> getCoefficients() const;

    /**
     * @brief Set the learned coefficients manually (useful for loading a pre-trained model).
     *
     * @param coeffs A vector of coefficients of size (n_features + 1).
     */
    void setCoefficients(const std::vector<double>& coeffs);

    /**
     * @brief Get the current regularization parameter (alpha).
     *
     * @return The regularization parameter.
     */
    double getAlpha() const;

    /**
     * @brief Get the current regularization type.
     *
     * @return A string representing the regularization type ("none", "ridge", or "lasso").
     */
    std::string getRegularizationType() const;

private:
    /**
     * @brief Internal method to compute the gradient of the cost function.
     *
     * @param X The feature matrix defined by LR_MATRIX_TYPE.
     * @param y The target vector.
     * @return A vector representing the gradient.
     */
    std::vector<double> computeGradient(const LR_MATRIX_TYPE& X, const std::vector<double>& y) const;

    /**
     * @brief Internal method to update coefficients using a gradient-based update.
     *
     * @param gradient The gradient vector.
     * @param learning_rate The learning rate.
     */
    void updateCoefficients(const std::vector<double>& gradient, double learning_rate);

    /**
     * @brief Sigmoid activation function.
     *
     * @param z Input value.
     * @return The sigmoid of z.
     */
    double sigmoid(double z) const;

private:
    std::vector<double> coefficients_; ///< Model coefficients: [intercept, w1, w2, ..., wn]
    double alpha_;                     ///< Regularization parameter.
    std::string regularization_type_;  ///< "none", "ridge", or "lasso"
    double learning_rate_;             ///< Learning rate for gradient descent. 
};

} // namespace algorithms
} // namespace ml

#endif // ML_ALGORITHMS_LOGISTIC_REGRESSION_H
