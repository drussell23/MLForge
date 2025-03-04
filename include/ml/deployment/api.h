#ifndef ML_DEPLOYMENT_API_H
#define ML_DEPLOYMENT_API_H

#include <string>
#include <vector>
#include <memory>
#include <future>
#include <unordered_map>
#include "ml/core/matrix.h"

using namespace std;

namespace ml
{
    namespace deployment
    {
        /// @brief Enum representing different model types that can be served.
        enum class ModelType
        {
            LINEAR_REGRESSION,
            LOGISTIC_REGRESSION,
            DECISION_TREE,
            NEURAL_NET,
            // Extend with additional model types as needed.
        };

        // @brief Abstract interface for a model serving API.
        class API
        {
        public:
            virtual ~API() = default;

            /// @brief Loads a model from disk.
            /// @param modelPapth Path to the serialized model file.
            /// @return True if the model was loaded successfully.
            virtual bool loadModel(const string &modelPath) = 0;

            /// @brief Saves the current model to disk.
            /// @param modelPath Path to save the serialized model.
            /// @return True if the model was saved successfully.
            virtual bool saveModel(const string &modelPath) const = 0;

            /// @brief Makes synchronous predictions given an input data matrix.
            /// @param input A matrix of input features.
            /// @return A vector of predictions.
            virtual vector<double> predict(const ml::core::Matrix2D &input) const = 0;

            /// @brief Makes asynchronous predictions given an input data matrix.
            /// @param input A matrix of input features.
            /// @return A future that will contain the vector of predictions.
            virtual future<vector<double>> predictAsync(const ml::core::Matrix2D &input) const
            {
                // Default asynchronous implementation.
                return async(launch::async, [this, &input]()
                             { return this->predict(input); });
            }

            /// @brief Updates the model with new training data.
            /// @param X A matrix of input features.
            /// @param y A vector of target values.
            /// @return True if the update was successful.
            virtual bool updateModel(const ml::core::Matrix2D &X, const vector<double> &y) = 0;

            /// @brief Retrieves performance metrics (e.g., loss, accuracy) of the model.
            /// @return A map of metric names to their values.
            virtual unordered_map<string, double> getPerformanceMetrics() const
            {
                return {};
            }

            /// @brief Retrieves information about the model (e.g., version, architecture, hyperparameters).
            /// @return A string summarizing model information.
            virtual string getModelInfo() const
            {
                return "Model information not available.";
            }

            /// @brief Resets the model to its initial state.
            virtual void resetModel() = 0;

            /// @brief Dynamically updates hyperparameters.
            /// @param params A map of hyperparameter names to their new values.
            /// @return True if the update was successful.
            virtual bool setHyperparameters(const unordered_map<string, double> &params)
            {
                return false;
            }
        };

        /// @brief Factory function to create an API instance for a given model type.
        /// @param type The type of model to create an API for.
        /// @return A unique_ptr to the created API instance.
        std::unique_ptr<API> createAPI(ModelType type);

    } // namespace deployment
} // namespace ml

#endif // ML_DEPLOYMENT_API_H