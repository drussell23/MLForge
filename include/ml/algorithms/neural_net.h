#ifndef ML_ALGORITHMS_NEURAL_NET_H
#define ML_ALGORITHMS_NEURAL_NET_H

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include "ml/core/matrix.h"

// ---------------------------------------------------------------------
// Macro to select the matrix type for the neural network.
// By default, we use Matrix2D. To use a different type (e.g., Matrix3D),
// define NEURAL_NET_MATRIX_TYPE before including this header.
// ---------------------------------------------------------------------
#ifndef NEURAL_NET_MATRIX_TYPE
    #define NEURAL_NET_MATRIX_TYPE ml::core::Matrix2D
#endif

// Alias for ease-of-use in function signatures.
#define NN_MATRIX_TYPE NEURAL_NET_MATRIX_TYPE

namespace ml {
namespace algorithms {

/**
 * @class NeuralNet
 * @brief Implements a simple feedforward neural network.
 *
 * This class provides methods to construct, train, and make predictions with
 * a neural network. The network supports a configurable number of layers and
 * neurons per layer, using standard activation functions (here, sigmoid).
 * The input data is accepted as a matrix defined by NN_MATRIX_TYPE.
 */
class NeuralNet {
public:
    /**
     * @brief Constructs a neural network with the specified architecture.
     *
     * @param layers A vector where each element specifies the number of neurons in that layer.
     * The first element is the input layer size, and the last is the output layer size.
     * @param learning_rate The learning rate for training (default is 0.01).
     */
    NeuralNet(const std::vector<int>& layers, double learning_rate = 0.01);

    /**
     * @brief Destructor.
     */
    ~NeuralNet();

    /**
     * @brief Trains the neural network using the provided training data.
     *
     * @param X A matrix of size (n_samples x n_features) defined by NN_MATRIX_TYPE.
     * @param y A vector of target values for each sample.
     * @param epochs The number of iterations over the training dataset.
     */
    void train(const NN_MATRIX_TYPE& X, const std::vector<double>& y, int epochs);

    /**
     * @brief Predicts output values for a given matrix of features.
     *
     * @param X A matrix of size (m_samples x n_features) defined by NN_MATRIX_TYPE.
     * @return A vector of predicted output values.
     */
    std::vector<double> predict(const NN_MATRIX_TYPE& X) const;

    /**
     * @brief Retrieves the current network weights.
     *
     * @return A vector of matrices representing the weights between layers.
     */
    std::vector<NN_MATRIX_TYPE> getWeights() const;

    /**
     * @brief Sets the network weights manually (useful for loading a pre-trained model).
     *
     * @param weights A vector of matrices representing the weights between layers.
     */
    void setWeights(const std::vector<NN_MATRIX_TYPE>& weights);

private:
    /**
     * @brief Performs a forward pass through the network for a single input sample.
     *
     * @param input A vector representing the input features.
     * @return A vector of vectors, where each inner vector contains the activations
     *         for one layer.
     */
    std::vector<std::vector<double>> forward(const std::vector<double>& input) const;

    /**
     * @brief Computes the sigmoid activation function.
     *
     * @param z The input value.
     * @return The sigmoid of z.
     */
    double activation(double z) const;

    /**
     * @brief Computes the derivative of the sigmoid activation function.
     *
     * @param z The input value.
     * @return The derivative of the sigmoid function at z.
     */
    double activationDerivative(double z) const;

private:
    std::vector<int> layers_;              ///< Number of neurons per layer.
    double learning_rate_;                 ///< Learning rate for training.
    std::vector<NN_MATRIX_TYPE> weights_;  ///< Weights for the connections between layers.
};

} // namespace algorithms
} // namespace ml

#endif // ML_ALGORITHMS_NEURAL_NET_H
