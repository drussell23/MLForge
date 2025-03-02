#include "ml/algorithms/decision_tree.h"
#include "ml/core/matrix.h"
#include <sstream>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <stdexcept>

// Ensure DT_MATRIX_TYPE is defined (if not defined in the header)
#ifndef DT_MATRIX_TYPE
    #define DT_MATRIX_TYPE ml::core::Matrix2D
#endif

// ---------------------------------------------------------------------
// Macro Definitions for Repetitive Tasks
// ---------------------------------------------------------------------

// Loop over samples: i from 0 to n-1.
#define FOR_EACH_SAMPLE(i, n) for (std::size_t i = 0; i < (n); ++i)

// Loop over features: j from 0 to n-1.
#define FOR_EACH_FEATURE(j, n) for (std::size_t j = 0; j < (n); ++j)

// Macro to compute the majority label from a vector of integer labels.
#define GET_MAJORITY_LABEL(y, label) do {                           \
    std::unordered_map<int, int> _count;                             \
    for (int _l : (y)) { _count[_l]++; }                              \
    int _majority = -1; int _max_count = 0;                            \
    for (const auto& kv : _count) {                                  \
        if (kv.second > _max_count) {                                \
            _max_count = kv.second; _majority = kv.first;            \
        }                                                            \
    }                                                                \
    label = _majority;                                               \
} while(0)

// Macro to build a matrix from a vector of rows (each row is a vector<double>).
#define BUILD_MATRIX_FROM_ROWS(data_vector, n_rows, n_cols, matrix) do { \
    std::vector<double> _flat;                                           \
    for (const auto& row : (data_vector)) {                              \
        _flat.insert(_flat.end(), row.begin(), row.end());               \
    }                                                                    \
    matrix = DT_MATRIX_TYPE(n_rows, n_cols, _flat);                      \
} while(0)

namespace ml {
namespace algorithms {

// ------------------------------------------------------------
// Node Constructor
// ------------------------------------------------------------
DecisionTree::Node::Node() 
    : is_leaf(false), feature_index(-1), threshold(0.0), prediction(-1), left(nullptr), right(nullptr) {}

// ------------------------------------------------------------
// DecisionTree Constructor & Destructor
// ------------------------------------------------------------
DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : root_(nullptr), max_depth_(max_depth), min_samples_split_(min_samples_split) {}

DecisionTree::~DecisionTree() {
    freeTree(root_);
}

// ------------------------------------------------------------
// Free Tree: Recursively delete nodes to free memory
// ------------------------------------------------------------
void DecisionTree::freeTree(Node* node) {
    if (node == nullptr) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}

// ------------------------------------------------------------
// Compute Impurity using Gini index
// ------------------------------------------------------------
double DecisionTree::computeImpurity(const std::vector<int>& y) const {
    if (y.empty()) return 0.0;
    std::unordered_map<int, int> count;
    for (int label : y) {
        count[label]++;
    }
    double impurity = 1.0;
    double total = static_cast<double>(y.size());
    for (const auto& kv : count) {
        double prob = kv.second / total;
        impurity -= prob * prob;
    }
    return impurity;
}

// ------------------------------------------------------------
// Find the Best Split for the Given Data
// ------------------------------------------------------------
void DecisionTree::findBestSplit(const DT_MATRIX_TYPE& X, const std::vector<int>& y,
                                 int& best_feature, double& best_threshold, double& best_impurity) const {
    std::size_t n_samples = X.rows();
    std::size_t n_features = X.cols();
    best_impurity = std::numeric_limits<double>::max();
    best_feature = -1;
    best_threshold = 0.0;
    double current_impurity = computeImpurity(y);
    if (current_impurity == 0.0) return; // Pure node; no split needed.

    FOR_EACH_FEATURE(feature, n_features) {
        // Gather candidate thresholds for the current feature.
        std::vector<double> thresholds;
        FOR_EACH_SAMPLE(i, n_samples) {
            thresholds.push_back(X(i, feature));
        }
        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

        // Evaluate each threshold.
        for (double threshold : thresholds) {
            std::vector<int> left_y, right_y;
            FOR_EACH_SAMPLE(i, n_samples) {
                if (X(i, feature) < threshold)
                    left_y.push_back(y[i]);
                else
                    right_y.push_back(y[i]);
            }
            if (left_y.empty() || right_y.empty()) continue;
            double left_impurity = computeImpurity(left_y);
            double right_impurity = computeImpurity(right_y);
            double weighted_impurity = (left_y.size() * left_impurity + right_y.size() * right_impurity) / n_samples;
            if (weighted_impurity < best_impurity) {
                best_impurity = weighted_impurity;
                best_feature = static_cast<int>(feature);
                best_threshold = threshold;
            }
        }
    }
}

// ------------------------------------------------------------
// Build Tree: Recursively create the decision tree
// ------------------------------------------------------------
DecisionTree::Node* DecisionTree::buildTree(const DT_MATRIX_TYPE& X, const std::vector<int>& y, int depth) {
    Node* node = new Node();
    std::size_t n_samples = X.rows();
    double impurity = computeImpurity(y);

    // Check stopping criteria.
    if (depth >= max_depth_ || n_samples < static_cast<std::size_t>(min_samples_split_) || impurity == 0.0) {
        node->is_leaf = true;
        GET_MAJORITY_LABEL(y, node->prediction);
        return node;
    }

    int best_feature;
    double best_threshold, best_impurity;
    findBestSplit(X, y, best_feature, best_threshold, best_impurity);

    // If no valid split was found, make a leaf node.
    if (best_feature == -1) {
        node->is_leaf = true;
        GET_MAJORITY_LABEL(y, node->prediction);
        return node;
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;

    // Partition data for left and right branches.
    std::vector<int> left_y, right_y;
    std::vector<std::vector<double>> left_data, right_data;
    FOR_EACH_SAMPLE(i, n_samples) {
        std::vector<double> sample;
        FOR_EACH_FEATURE(j, X.cols()) {
            sample.push_back(X(i, j));
        }
        if (sample[best_feature] < best_threshold) {
            left_y.push_back(y[i]);
            left_data.push_back(sample);
        } else {
            right_y.push_back(y[i]);
            right_data.push_back(sample);
        }
    }

    DT_MATRIX_TYPE left_X, right_X;
    if (!left_data.empty()) {
        BUILD_MATRIX_FROM_ROWS(left_data, left_data.size(), X.cols(), left_X);
    }
    if (!right_data.empty()) {
        BUILD_MATRIX_FROM_ROWS(right_data, right_data.size(), X.cols(), right_X);
    }

    // Recursively build subtrees.
    node->left = buildTree(left_X, left_y, depth + 1);
    node->right = buildTree(right_X, right_y, depth + 1);

    return node;
}

// ------------------------------------------------------------
// Fit: Train the decision tree using training data.
// ------------------------------------------------------------
void DecisionTree::fit(const DT_MATRIX_TYPE& X, const std::vector<int>& y) {
    freeTree(root_);
    root_ = buildTree(X, y, 0);
}

// ------------------------------------------------------------
// Predict a Single Sample by Traversing the Tree
// ------------------------------------------------------------
int DecisionTree::predictSample(const std::vector<double>& sample, Node* node) const {
    if (node->is_leaf) {
        return node->prediction;
    }
    if (sample[node->feature_index] < node->threshold) {
        return predictSample(sample, node->left);
    } else {
        return predictSample(sample, node->right);
    }
}

// ------------------------------------------------------------
// Predict: Returns predictions for each sample in the matrix.
// ------------------------------------------------------------
std::vector<int> DecisionTree::predict(const DT_MATRIX_TYPE& X) const {
    std::vector<int> predictions;
    std::size_t n_samples = X.rows();
    std::size_t n_features = X.cols();
    FOR_EACH_SAMPLE(i, n_samples) {
        std::vector<double> sample;
        FOR_EACH_FEATURE(j, n_features) {
            sample.push_back(X(i, j));
        }
        predictions.push_back(predictSample(sample, root_));
    }
    return predictions;
}

// ------------------------------------------------------------
// Get Tree Structure: Returns a string representation of the tree.
// ------------------------------------------------------------
std::string DecisionTree::getTreeStructure() const {
    std::ostringstream oss;
    std::function<void(Node*, int)> printTree = [&](Node* node, int depth) {
        if (!node) return;
        for (int i = 0; i < depth; ++i) {
            oss << "  ";
        }
        if (node->is_leaf) {
            oss << "Leaf: " << node->prediction << "\n";
        } else {
            oss << "Node: Feature " << node->feature_index << " < " << node->threshold << "\n";
            printTree(node->left, depth + 1);
            printTree(node->right, depth + 1);
        }
    };
    printTree(root_, 0);
    return oss.str();
}

// ---------------------------------------------------------------------
// Additional Advanced Features: Tree Analysis Functions
// ---------------------------------------------------------------------

namespace { // Anonymous namespace for helper functions.

    // Compute the maximum depth of the tree.
    int computeDepth(DecisionTree::Node* node) {
        if (!node) return 0;
        if (node->is_leaf) return 1;
        int left_depth = computeDepth(node->left);
        int right_depth = computeDepth(node->right);
        return 1 + std::max(left_depth, right_depth);
    }

    // Count the total number of nodes in the tree.
    int countNodesHelper(DecisionTree::Node* node) {
        if (!node) return 0;
        return 1 + countNodesHelper(node->left) + countNodesHelper(node->right);
    }

    // Compute feature usage frequency for internal (non-leaf) nodes.
    void featureUsageHelper(DecisionTree::Node* node, std::unordered_map<int, int>& usage) {
        if (!node) return;
        if (!node->is_leaf) {
            usage[node->feature_index]++;
            featureUsageHelper(node->left, usage);
            featureUsageHelper(node->right, usage);
        }
    }
}

int DecisionTree::getTreeDepth() const {
    return computeDepth(root_);
}

int DecisionTree::countNodes() const {
    return countNodesHelper(root_);
}

std::unordered_map<int, int> DecisionTree::getFeatureUsage() const {
    std::unordered_map<int, int> usage;
    featureUsageHelper(root_, usage);
    return usage;
}

} // namespace algorithms
} // namespace ml
