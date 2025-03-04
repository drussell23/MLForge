#ifndef ML_DEPLOYMENT_MODEL_SERVER_H
#define ML_DEPLOYMENT_MODEL_SERVER_H

#include <string>
#include <vector>
#include <memory>
#include <future>
#include <unordered_map>
#include "ml/core/matrix.h"
#include "ml/deployment/api.h" // For ModelType enum.

namespace ml
{
    namespace deployment
    {

        /// @brief Abstract interface for a production-grade model server.
        class ModelServer
        {
        public:
            virtual ~ModelServer() = default;

            /// @brief Starts the model server using the provided configuration file.
            /// @param configFile Path to the server configuration file.
            /// @return True if the server started successfully.
            virtual bool startServer(const std::string &configFile) = 0;

            /// @brief Stops the model server.
            virtual void stopServer() = 0;

            /// @brief Makes synchronous predictions given an input data matrix.
            /// @param input A matrix of input features (using ml::core::Matrix2D).
            /// @return A vector of predictions.
            virtual std::vector<double> predict(const ml::core::Matrix2D &input) = 0;

            /// @brief Makes asynchronous predictions given an input data matrix.
            /// @param input A matrix of input features.
            /// @return A future that will contain the vector of predictions.
            virtual std::future<std::vector<double>> predictAsync(const ml::core::Matrix2D &input)
            {
                // Default implementation using std::async.
                return std::async(std::launch::async, [this, &input]()
                                  { return this->predict(input); });
            }

            /// @brief Updates the model with new training data.
            /// @param X A matrix of input features.
            /// @param y A vector of target values.
            /// @return True if the update was successful.
            virtual bool updateModel(const ml::core::Matrix2D &X, const std::vector<double> &y) = 0;

            /// @brief Retrieves performance metrics (e.g., loss, accuracy).
            /// @return A map of metric names to their current values.
            virtual std::unordered_map<std::string, double> getPerformanceMetrics() const = 0;

            /// @brief Retrieves model information (e.g., version, architecture, hyperparameters).
            /// @return A string summarizing model information.
            virtual std::string getModelInfo() const = 0;

            /// @brief Resets the model to its initial state.
            virtual void resetModel() = 0;

            /// @brief Dynamically updates hyperparameters.
            /// @param params A map of hyperparameter names to their new values.
            /// @return True if the update was successful.
            virtual bool setHyperparameters(const std::unordered_map<std::string, double> &params) = 0;
        };

        /// @brief Factory function to create a ModelServer instance for a given model type.
        std::unique_ptr<ModelServer> createModelServer(ModelType type);

    } // namespace deployment
} // namespace ml

#endif // ML_DEPLOYMENT_MODEL_SERVER_H
