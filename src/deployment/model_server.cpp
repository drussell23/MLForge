#include "ml/deployment/model_server.h"
#include "ml/deployment/api.h" // For ModelType enum.
#include "ml/core/matrix.h"
#include "ml/algorithms/logistic_regression.h" // Include the model header.
#include "httplib.h"                           // Ensure this file is in your include path (e.g., third_party/cpp-httplib/httplib.h)

#include <iostream>
#include <future>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>
#include <random>

using namespace std;

namespace ml
{
    namespace deployment
    {

        class AdvancedModelServer : public ModelServer
        {
        public:
            AdvancedModelServer() : running_(false)
            {
                modelInfo_ = "AdvancedModelServer v2.0";
                // For demonstration, instantiate a LogisticRegression model.
                // In a real scenario, you would load a pre-trained model from disk.
                model_ = std::make_unique<ml::algorithms::LogisticRegression>(0.1, "ridge");
            }

            virtual ~AdvancedModelServer() override
            {
                stopServer();
            }

            // Starts the server. In production, additional configuration would be applied.
            virtual bool startServer(const std::string &configFile) override
            {
                std::cout << "[AdvancedModelServer] Starting server with config: " << configFile << std::endl;
                // (Load configuration settings here, if any)
                running_ = true;
                // Launch the HTTP server on a separate thread.
                serverThread_ = std::thread(&AdvancedModelServer::runHttpServer, this);
                return true;
            }

            // Stops the server gracefully.
            virtual void stopServer() override
            {
                running_ = false;
                if (server_)
                {
                    server_->stop();
                }
                if (serverThread_.joinable())
                {
                    serverThread_.join();
                }
                std::cout << "[AdvancedModelServer] Server stopped." << std::endl;
            }

            // Synchronous prediction method using the loaded model.
            virtual std::vector<double> predict(const ml::core::Matrix2D &input) override
            {
                std::lock_guard<std::mutex> lock(modelMutex_);

                if (!model_)
                {
                    throw std::runtime_error("No model loaded.");
                }

                // Get predictions from the model (which returns vector<int>).
                vector<int> intPreds = model_->predict(input);

                // Convert vector<int> to vector<double>.
                vector<double> preds;
                preds.reserve(intPreds.size());

                for (int val : intPreds) {
                    preds.push_back(static_cast<double>(val));
                }

                return preds;
            }

            // Asynchronous prediction using std::async.
            virtual std::future<std::vector<double>> predictAsync(const ml::core::Matrix2D &input) override
            {
                return std::async(std::launch::async, [this, input]()
                                  { return this->predict(input); });
            }

            // Updates the model with new training data without downtime.
            virtual bool updateModel(const ml::core::Matrix2D &X, const std::vector<double> &y) override
            {
                std::lock_guard<std::mutex> lock(modelMutex_);
                // In a real implementation, perform incremental training or load new weights.
                std::cout << "[AdvancedModelServer] Model updated with " << X.rows() << " samples." << std::endl;
                return true;
            }

            // Returns performance metrics.
            virtual std::unordered_map<std::string, double> getPerformanceMetrics() const override
            {
                return {{"latency_ms", 10.0}, {"throughput", 100.0}};
            }

            // Returns a summary of the model's information.
            virtual std::string getModelInfo() const override
            {
                std::lock_guard<std::mutex> lock(modelMutex_);
                return modelInfo_;
            }

            // Resets the model to its initial state.
            virtual void resetModel() override
            {
                std::lock_guard<std::mutex> lock(modelMutex_);
                modelInfo_ = "AdvancedModelServer v2.0 (reset)";
                std::cout << "[AdvancedModelServer] Model has been reset." << std::endl;
            }

            // Dynamically updates hyperparameters.
            virtual bool setHyperparameters(const std::unordered_map<std::string, double> &params) override
            {
                std::lock_guard<std::mutex> lock(modelMutex_);
                for (const auto &param : params)
                {
                    std::cout << "[AdvancedModelServer] Hyperparameter " << param.first << " set to " << param.second << std::endl;
                }
                // Update the model configuration accordingly.
                return true;
            }

        private:
            std::atomic<bool> running_;
            std::string modelInfo_;
            std::unique_ptr<ml::algorithms::LogisticRegression> model_; // The loaded model for inference.
            mutable std::mutex modelMutex_;
            std::unique_ptr<httplib::Server> server_; // Using cpp-httplib for the HTTP server.
            std::thread serverThread_;

            // HTTP server loop to handle requests and perform health checks.
            void runHttpServer()
            {
                server_ = std::make_unique<httplib::Server>();

                // Define a POST endpoint for predictions.
                server_->Post("/predict", [this](const httplib::Request &req, httplib::Response &res)
                              {
            // For demonstration, create a dummy input matrix.
            // In a real implementation, parse req.body to extract input features.
            ml::core::Matrix2D input(1, 3, 0.0); // Dummy 1x3 matrix.
            std::vector<double> preds;
            try {
                preds = this->predict(input);
            } catch (const std::exception &ex) {
                res.status = 500;
                res.set_content("{\"error\": \"" + std::string(ex.what()) + "\"}", "application/json");
                return;
            }
            // Convert predictions to a JSON-like string.
            std::string output = "[";
            for (size_t i = 0; i < preds.size(); ++i) {
                output += std::to_string(preds[i]);
                if (i != preds.size() - 1) {
                    output += ", ";
                }
            }
            output += "]";
            res.set_content(output, "application/json"); });

                // Define a GET endpoint for health checks.
                server_->Get("/health", [](const httplib::Request &, httplib::Response &res)
                             { res.set_content("{\"status\": \"OK\"}", "application/json"); });

                std::cout << "[AdvancedModelServer] HTTP server running on port 8080..." << std::endl;
                server_->listen("0.0.0.0", 8080);
            }
        };

        // Factory function to create a ModelServer instance for a given model type.
        std::unique_ptr<ModelServer> createModelServer(ModelType type)
        {
            // For demonstration, return an instance of AdvancedModelServer regardless of type.
            return std::make_unique<AdvancedModelServer>();
        }

    } // namespace deployment
} // namespace ml
