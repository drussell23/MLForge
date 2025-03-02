#include "ml/algorithms/neural_net.h"
#include "ml/core/matrix.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <iostream>

// Ensure NN_MATRIX_TYPE is defined (it should be set in the header)
#ifndef NEURAL_NET_MATRIX_TYPE
    #define NEURAL_NET_MATRIX_TYPE ml::core::Matrix2D
#endif
#define NN_MATRIX_TYPE NEURAL_NET_MATRIX_TYPE

namespace {

    // Helper: Multiply a matrix (of dimensions m x n) by a vector (of size n).
    std::vector<double> matVecMultiply(const NN_MATRIX_TYPE& W, const std::vector<double>& a) {
        std::size_t m = W.rows();
        std::size_t n = W.cols();
        if (a.size() != n) {
            throw std::invalid_argument("Dimension mismatch in matVecMultiply");
        }
        std::vector<double> result(m, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                sum += W(i, j) * a[j];
            }
            result[i] = sum;
        }
        return result;
    }

    // Helper: Append a bias term (value 1.0) to a vector.
    std::vector<double> appendBias(const std::vector<double>& a) {
        std::vector<double> result;
        result.reserve(a.size() + 1);
        result.push_back(1.0);
        result.insert(result.end(), a.begin(), a.end());
        return result;
    }

    // Helper: Generate a random double in [-limit, limit].
    double randomWeight(double limit = 0.5) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-limit, limit);
        return dist(gen);
    }

    // Helper: Initialize a weight matrix with random values.
    // Dimensions: rows x cols.
    NN_MATRIX_TYPE randomMatrix(std::size_t rows, std::size_t cols, double limit = 0.5) {
        std::vector<double> data(rows * cols);
        for (std::size_t i = 0; i < rows * cols; ++i) {
            data[i] = randomWeight(limit);
        }
        return NN_MATRIX_TYPE(rows, cols, data);
    }

} // end anonymous namespace

namespace ml {
namespace algorithms {

NeuralNet::NeuralNet(const std::vector<int>& layers, double learning_rate)
    : layers_(layers), learning_rate_(learning_rate)
{
    if (layers.empty()) {
        throw std::invalid_argument("NeuralNet: layers vector must not be empty.");
    }
    // Initialize weights for each layer.
    // For each connection from layer l-1 to layer l, create a weight matrix
    // with dimensions: [layers_[l] x (layers_[l-1] + 1)] to incorporate bias.
    weights_.clear();
    for (std::size_t l = 1; l < layers_.size(); ++l) {
        int rows = layers_[l];
        int cols = layers_[l-1] + 1;  // +1 for bias.
        weights_.push_back(randomMatrix(rows, cols));
    }
}

NeuralNet::~NeuralNet() {
    // No dynamic memory needs explicit deletion.
}

std::vector<std::vector<double>> NeuralNet::forward(const std::vector<double>& input) const {
    // Forward pass: compute activations for each layer.
    // a[0] = input with bias appended.
    std::vector<std::vector<double>> activations;
    // Append bias to input:
    std::vector<double> a = appendBias(input);
    activations.push_back(a);

    // For each layer, compute z = W * a, then a = activation(z).
    for (std::size_t l = 0; l < weights_.size(); ++l) {
        const NN_MATRIX_TYPE& W = weights_[l];
        std::vector<double> z = matVecMultiply(W, activations[l]); // z has size = layers_[l+1]
        std::vector<double> a_next;
        a_next.resize(z.size());
        // Apply activation function (sigmoid)
        for (std::size_t i = 0; i < z.size(); ++i) {
            a_next[i] = activation(z[i]);
        }
        // For hidden layers, append bias.
        if (l < weights_.size() - 1) {
            a_next = appendBias(a_next);
        }
        activations.push_back(a_next);
    }
    return activations;
}

void NeuralNet::train(const NN_MATRIX_TYPE& X, const std::vector<double>& y, int epochs) {
    std::size_t n_samples = X.rows();
    if (n_samples != y.size()) {
        throw std::invalid_argument("train: Number of samples in X must equal size of y.");
    }
    // Simple stochastic gradient descent (one sample per update).
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;
        // Loop over all training samples.
        for (std::size_t i = 0; i < n_samples; ++i) {
            // Extract the i-th sample as a vector.
            std::vector<double> sample;
            sample.resize(X.cols());
            for (std::size_t j = 0; j < X.cols(); ++j) {
                sample[j] = X(i, j);
            }
            // Forward pass.
            std::vector<std::vector<double>> activations = forward(sample);
            // Compute error at output layer.
            const std::vector<double>& output = activations.back();
            double error = output[0] - y[i]; // Assuming single output neuron.
            total_error += error * error;

            // Backpropagation:
            // delta for output layer.
            std::vector<double> delta;
            delta.resize(output.size());
            for (std::size_t j = 0; j < output.size(); ++j) {
                // derivative of sigmoid at z is a * (1 - a); output = a.
                delta[j] = error * output[j] * (1.0 - output[j]);
            }

            // Gradients for each layer's weights.
            std::vector<NN_MATRIX_TYPE> grad(weights_.size());
            // Backpropagate for layers L-1 down to 0.
            // Let activations[l] be the activation of layer l (with bias appended for l < L).
            std::vector<std::vector<double>> deltas(weights_.size());
            deltas[weights_.size() - 1] = delta;
            // Propagate backward:
            for (int l = weights_.size() - 2; l >= 0; --l) {
                // delta_l = ( (W_{l+1}^T * delta_{l+1}) (excluding bias) ) .* activationDerivative(a_l without bias)
                const NN_MATRIX_TYPE& W_next = weights_[l+1];
                const std::vector<double>& a = activations[l+1]; // This activation has bias at index 0.
                std::vector<double> delta_next = deltas[l+1];
                // Compute weighted error for layer l.
                std::vector<double> delta_curr;
                // Dimensions: current layer (excluding bias). a has size = layers_[l+1]+1, so ignore first element (bias).
                int n_neurons = layers_[l+1]; // actual neurons in layer l+1.
                delta_curr.resize(n_neurons, 0.0);
                // For each neuron in layer l+1 (excluding bias), compute sum_{k} (W_next[k][j] * delta_next[k])
                // Note: W_next dimensions: [layers_[l+2] x (layers_[l+1] + 1)] if l+1 is hidden.
                // However, here, for backprop we need to sum over neurons of next layer,
                // skipping the bias weight.
                // We'll compute for each neuron j in layer l+1:
                for (int j = 0; j < n_neurons; ++j) {
                    double sum = 0.0;
                    // Iterate over neurons in next layer (l+2).
                    int next_neurons = weights_[l+1].rows();
                    for (int k = 0; k < next_neurons; ++k) {
                        // Note: the first column of weights_[l+1] corresponds to bias in layer l+1.
                        // So the weight corresponding to neuron j in layer l+1 is at column (j+1).
                        sum += weights_[l+1](k, j+1) * delta_next[k];
                    }
                    // Remove bias from activation a: a[j+1] is the activation of neuron j in layer l+1.
                    double a_val = a[j+1];
                    delta_curr[j] = sum * a_val * (1.0 - a_val);
                }
                deltas[l] = delta_curr;
            }

            // Now, compute gradients for each layer.
            // For layer l, gradient = delta_{l} outer product activations[l] (from previous layer)
            for (std::size_t l = 0; l < weights_.size(); ++l) {
                const std::vector<double>& a_prev = activations[l]; // includes bias if l < last hidden.
                const std::vector<double>& delta_l = deltas[l];
                int rows = weights_[l].rows();     // neurons in current layer.
                int cols = weights_[l].cols();     // neurons in previous layer + bias.
                std::vector<double> grad_data(rows * cols, 0.0);
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        grad_data[i * cols + j] = delta_l[i] * a_prev[j];
                    }
                }
                grad[l] = NN_MATRIX_TYPE(rows, cols, grad_data);
            }

            // Update weights using gradients (stochastic update).
            for (std::size_t l = 0; l < weights_.size(); ++l) {
                int rows = weights_[l].rows();
                int cols = weights_[l].cols();
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        double updated = weights_[l](i, j) - learning_rate_ * grad[l](i, j);
                        weights_[l](i, j) = updated;
                    }
                }
            }
        } // End sample loop

        // Optionally, print average error per epoch.
        // std::cout << "Epoch " << epoch << " MSE: " << total_error / n_samples << std::endl;
    } // End epoch loop
}

std::vector<double> NeuralNet::predict(const NN_MATRIX_TYPE& X) const {
    std::size_t n_samples = X.rows();
    std::vector<double> predictions;
    predictions.reserve(n_samples);
    // For each sample, perform a forward pass and output the final activation.
    for (std::size_t i = 0; i < n_samples; ++i) {
        std::vector<double> sample;
        sample.resize(X.cols());
        for (std::size_t j = 0; j < X.cols(); ++j) {
            sample[j] = X(i, j);
        }
        std::vector<std::vector<double>> activations = forward(sample);
        // Output layer activation (assumes single output neuron)
        predictions.push_back(activations.back()[0]);
    }
    return predictions;
}

std::vector<NN_MATRIX_TYPE> NeuralNet::getWeights() const {
    return weights_;
}

void NeuralNet::setWeights(const std::vector<NN_MATRIX_TYPE>& weights) {
    weights_ = weights;
}

double NeuralNet::activation(double z) const {
    return 1.0 / (1.0 + std::exp(-z));
}

double NeuralNet::activationDerivative(double z) const {
    double a = activation(z);
    return a * (1 - a);
}

} // namespace algorithms
} // namespace ml
