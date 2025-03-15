#include "ml/core/utils.h"

using namespace std;

namespace ml
{
    namespace core
    {
        string vectorToString(const vector<double> &v)
        {
            ostringstream oss;
            oss << "[";

            for (size_t i = 0; i < v.size(); ++i)
            {
                oss << v[i];

                if (i != v.size() - 1)
                {
                    oss << ", ";
                }
            }
            oss << "]";
            return oss.str();
        }

        double dotProduct(const vector<double> &a, const vector<double> &b)
        {
            if (a.size() != b.size())
            {
                throw invalid_argument("Vectors must be of the same size for dot product.");
            }
            return inner_product(a.begin(), a.end(), b.begin(), 0.0);
        }

        double mean(const vector<double> &v)
        {
            if (v.empty())
            {
                throw invalid_argument("Cannot compute mean of an empty vector.");
            }
            double sum = accumulate(v.begin(), v.end(), 0.0);
            return sum / static_cast<double>(v.size());
        }

        double variance(const vector<double> &v)
        {
            if (v.empty())
            {
                throw invalid_argument("Cannot compute variance of an empty vector.");
            }
            double m = mean(v);
            double accum = 0.0;
            for (double x : v)
            {
                accum += (x - m) * (x - m);
            }
            return accum / static_cast<double>(v.size());
        }

        vector<double> normalize(const vector<double> &v)
        {
            if (v.empty())
            {
                return v;
            }

            double min_val = *min_element(v.begin(), v.end());
            double max_val = *max_element(v.begin(), v.end());

            if (max_val - min_val == 0.0)
            {
                return v; // Avoid division by zero if all values are equal.
            }

            vector<double> norm(v.size());
            for (size_t i = 0; i < v.size(); ++i)
            {
                norm[i] = (v[i] - min_val) / (max_val - min_val);
            }
            return norm;
        }

        vector<double> addVectors(const vector<double> &a, const vector<double> &b)
        {
            if (a.size() != b.size())
            {
                throw invalid_argument("Vectors must be of the same size for element-wise addition.");
            }

            vector<double> result(a.size());
            for (size_t i = 0; i < a.size(); ++i)
            {
                result[i] = a[i] + b[i];
            }
            return result;
        }

        vector<double> substractVectors(const vector<double> &a, const vector<double> &b)
        {
            if (a.size() != b.size())
            {
                throw invalid_argument("Vectors must be of the same size for element-wise subtraction.");
            }

            vector<double> result(a.size());
            for (size_t i = 0; i < a.size(); ++i)
            {
                result[i] = a[i] - b[i];
            }
            return result;
        }

        vector<double> multiplyVectors(const vector<double> &a, const vector<double> &b)
        {
            if (a.size() != b.size())
            {
                throw invalid_argument("Vectors must be of the same size for element-wise multiplication.");
            }

            vector<double> result(a.size());
            for (size_t i = 0; i < a.size(); ++i)
            {
                result[i] = a[i] * b[i];
            }
            return result;
        }
    }
}