#ifndef ML_CORE_UTILS_H
#define ML_CORE_UTILS_H

#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>
#include <utility>
#include <cmath>

using namespace std;

namespace ml {
    namespace core {
        /// Converts a vector of doubles to a string representation.
        inline string vectorToString(const vector<double>& v) {
            ostringstream oss;
            oss << "[";

            for (size_t i = 0; i < v.size(); ++i) {
                oss << v[i];
                if (i != v.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "]";
            return oss.str();
        }
        
        /// Computes the dot product between two vectors. 
        inline double dotProduct(const vector<double>& a, const vector<double>& b) {
            if (a.size() != b.size()) {
                throw invalid_argument("Vectors must be of the same size for dot product.");
            }
            return inner_product(a.begin(), a.end(), b.begin(), 0.0);
        }

        /// Computes the mean of elements in a vector. 
        inline double mean(const vector<double>& v) {
            if (v.empty()) {
                throw invalid_argument("Cannot compute mean of an empty vector.");
            }   

            double sum = accumulate(v.begin(), v.end(), 0.0);
            return sum / static_cast<double>(v.size());
        }

        // Computes the variance of the elements in a vector. 
        inline double variance(const vector<double>& v) {
            if (v.empty()) {
                throw invalid_argument("Cannot compute variance of an empty vector.");
            }
            double m = mean(v);
            double accum = 0.0;

            for (double x : v) {
                accum += (x - m) * (x - m);
            }
            return accum / static_cast<double>(v.size());
        }

        /// Normalize a vector using min-max scaling. 
        inline vector<double> normalize(const vector<double>& v) {
            if (v.empty()) {
                return v;
            }

            double min_val = *min_element(v.begin(), v.end());
            double max_val = *max_element(v.begin(), v.end());

            if (max_val - min_val == 0.0) {
                return v; // Avoid division by zero if all values are equal.
            }
            
            vector<double> norm(v.size());

            for (size_t i = 0; i < v.size(); ++i) {
                norm[i] = (v[i] - min_val) / (max_val - min_val);
            }
            return norm;
        }

        /// Splits a vector into a training and testing set based on the provided test ratio. 
        /// Returns a pair: (train_set, test_set)
        template <typename T>
        inline pair<vector<T>, vector<T>> trainTestSplit(const vector<T>& data, double test_ratio) {
            if (test_ratio < 0.0 || test_ratio > 1.0) {
                throw invalid_argument("Test ratio must be between 0 and 1.");
            }
            size_t test_size = static_cast<size_t>(data.size() * test_ratio);
            vector<T> test(data.begin(), data.begin() + test_size);
            vector<T> train(data.begin() + test_size, data.end());
        }

        /// Shuffles the elements of a vector in place.
        template <typename T>
        inline void shuffleVector(vector<T>& v) {
            random_device rd;
            mt19937 g(rd());
            shuffle(v.begin(), v.end(), g);
        }

        /// Computes the element-wise addition of two vectors. 
        inline vector<double> addVectors(const vector<double>& a, const vector<double>& b) {
            if (a.size() != b.size()) {
                throw invalid_argument("Vectors must be of the same size for element-wise addition.");
            }

            vector<double> result(a.size());

            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] + b[i];
            }

            return result;
        }

        /// Computes the element-wise substraction of two vectors. 
        inline vector<double> subtractVectors(const vector<double>& a, const vector<double>& b) {
            if (a.size() != b.size()) {
                throw invalid_argument("Vectors must be of the same size for element-wise subtraction.");
            }

            vector<double> result(a.size());

            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] - b[i];
            }

            return result;
        }

        /// Computes the element-wise multiplication of two vectors.
        inline vector<double> multiplyVectors(const vector<double>& a, const vector<double>& b) {
            if (a.size() != b.size()) {
                throw invalid_argument("Vectors must be of the same size for element-wise multiplication.");
            }

            vector<double> result(a.size());

            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] * b[i];
            }
            
            return result;
        }

        /// Applies a given funciton to each element of the vector. 
        template <typename T, typename Func> 
        inline vector<T> applyFunctions(const vector<T>& v, Func f) {
            vector<T> result(v.size());
            transform(v.begin(), v.end(), result.begin(), f);
            return result;
        }

    } // namespace core
} // namespace ml

#endif // ML_CORE_UTILS_H
