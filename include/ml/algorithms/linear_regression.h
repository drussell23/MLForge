#ifndef ML_ALGORITHMS_LINEAR_REGRESSION_H
#define ML_ALGORITHMS_LINEAR_REGRESSION_H

#include <vector>
#include <string>
#include "ml/core/matrix.h"

// ---------------------------------------------------------------------
// Macro to select the matrix type for the algorithm.
// By default, we use Matrix2D. You can define this macro to ml::core::Matrix3D
// before including this header if you wish to work with 3D data.
// ---------------------------------------------------------------------
#ifndef LINEAR_REGRESSION_MATRIX_TYPE
    #define LINEAR_REGRESSION_MATRIX_TYPE ml::core::Matrix2D
#endif

// ---------------------------------------------------------------------
// Optional macro to define method signatures using the chosen matrix type.
// This makes it easier to change the underlying data type in one spot.
// ---------------------------------------------------------------------
#define LR_MATRIX_TYPE LINEAR_REGRESSION_MATRIX_TYPE

namespace ml {
namespace algorithms {

/**
 * @class LinearRegression
 * @brief Implements a basic linear regression model using Ordinary Least Squares (OLS).
 *
 * This class provides methods to fit a linear model to data and to make predictions.
 * It optionally supports a regularization term (Ridge or Lasso) if you wish to extend it.
 * 
 * The model accepts data in the form of a matrix defined by the macro LINEAR_REGRESSION_MATRIX_TYPE,
 * which defaults to a 2D matrix. You can override this to use a 3D matrix if needed.
 */
class LinearRegression {
public:
    /**
     * @brief Default constructor.
     * @param alpha Regularization parameter. (Set to 0.0 for no regularization.)
     * @param regularization_type The type of regularization ("none", "ridge", or "lasso").
     */
    LinearRegression(double alpha = 0.0, const std::string& regularization_type = "none");

    /**
     * @brief Destructor.
     */
    ~LinearRegression();

    /**
     * @brief Fit the model to the provided data using Ordinary Least Squares or regularized OLS.
     *
     * @param X A matrix of size (n_samples x n_features) defined by LR_MATRIX_TYPE.
     * @param y A vector of size (n_samples) representing the target values.
     *
     * This function computes the model parameters (coefficients) that minimize
     * the residual sum of squares between the observed targets and predicted targets.
     * If `alpha > 0`, applies the specified regularization.
     */
    void fit(const LR_MATRIX_TYPE& X, const std::vector<double>& y);

    /**
     * @brief Predict target values for a given matrix of features.
     *
     * @param X A matrix of size (m_samples x n_features) defined by LR_MATRIX_TYPE.
     * @return A vector of predicted values of size (m_samples).
     */
    std::vector<double> predict(const LR_MATRIX_TYPE& X) const;

    /**
     * @brief Get the learned coefficients (including intercept).
     *
     * @return A vector of size (n_features + 1), where the first element is the intercept.
     */
    std::vector<double> getCoefficients() const;

    /**
     * @brief Set the learned coefficients manually (e.g., for loading a pre-trained model).
     *
     * @param coeffs A vector of size (n_features + 1).
     */
    void setCoefficients(const std::vector<double>& coeffs);

    /**
     * @brief Get the current regularization parameter (alpha).
     */
    double getAlpha() const;

    /**
     * @brief Get the current regularization type (e.g., "none", "ridge", or "lasso").
     */
    std::string getRegularizationType() const;

private:
    /**
     * @brief Internal method to compute the closed-form solution for OLS (and possibly regularized OLS).
     * @param X The feature matrix defined by LR_MATRIX_TYPE.
     * @param y The target vector.
     */
    void computeCoefficients(const LR_MATRIX_TYPE& X, const std::vector<double>& y);

private:
    std::vector<double> coefficients_; ///< Model coefficients: [intercept, w1, w2, ..., wn]
    double alpha_;                     ///< Regularization strength.
    std::string regularization_type_;  ///< "none", "ridge", or "lasso"
};

} // namespace algorithms
} // namespace ml

#endif // ML_ALGORITHMS_LINEAR_REGRESSION_H
