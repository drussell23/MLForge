#ifndef ML_ALGORITHMS_DECISION_TREE_H
#define ML_ALGORITHMS_DECISION_TREE_H

#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include "ml/core/matrix.h"

// ---------------------------------------------------------------------
// Macro to select the matrix type for the decision tree.
// By default, we use Matrix2D. You can override this macro to use a 3D matrix
// (e.g., ml::core::Matrix3D) if desired.
// ---------------------------------------------------------------------
#ifndef DECISION_TREE_MATRIX_TYPE
    #define DECISION_TREE_MATRIX_TYPE ml::core::Matrix2D
#endif

// Alias for ease-of-use in function signatures.
#define DT_MATRIX_TYPE DECISION_TREE_MATRIX_TYPE

namespace ml {
namespace algorithms {

/**
 * @class DecisionTree
 * @brief Implements a simple decision tree classifier.
 *
 * This class provides methods to train a decision tree on labeled data and
 * predict class labels for new samples. The decision tree is built recursively
 * by selecting the best feature and threshold to split the data.
 *
 * The model accepts input data as a matrix defined by DT_MATRIX_TYPE, which by
 * default is a 2D matrix (ml::core::Matrix2D). Override this macro to use a 3D
 * matrix if needed.
 */
class DecisionTree {
public:
    /**
     * @brief Node structure representing a node in the decision tree.
     */
    struct Node {
        bool is_leaf;
        int feature_index;
        double threshold;
        int prediction; // For leaf nodes.
        Node* left;
        Node* right;

        Node() : is_leaf(false), feature_index(-1), threshold(0.0), prediction(-1), left(nullptr), right(nullptr) {}
    };

    /**
     * @brief Constructor.
     * @param max_depth The maximum depth of the tree (default is 10).
     * @param min_samples_split The minimum number of samples required to split an internal node (default is 2).
     */
    DecisionTree(int max_depth = 10, int min_samples_split = 2);

    /**
     * @brief Destructor.
     */
    ~DecisionTree();

    /**
     * @brief Fit the decision tree classifier to the provided training data.
     *
     * @param X A matrix of size (n_samples x n_features) defined by DT_MATRIX_TYPE.
     * @param y A vector of size (n_samples) representing the target class labels.
     */
    void fit(const DT_MATRIX_TYPE& X, const std::vector<int>& y);

    /**
     * @brief Predict class labels for a given matrix of features.
     *
     * @param X A matrix of size (m_samples x n_features) defined by DT_MATRIX_TYPE.
     * @return A vector of predicted class labels for each sample.
     */
    std::vector<int> predict(const DT_MATRIX_TYPE& X) const;

    /**
     * @brief Get a string representation of the decision tree.
     *
     * @return A string describing the tree structure.
     */
    std::string getTreeStructure() const;

    // Advanced analysis functions:
    int getTreeDepth() const;
    int countNodes() const;
    std::unordered_map<int, int> getFeatureUsage() const;

private:
    Node* root_;                ///< Root node of the decision tree.
    int max_depth_;             ///< Maximum allowed depth for the tree.
    int min_samples_split_;     ///< Minimum number of samples to perform a split.

    // Helper functions:
    void freeTree(Node* node);
    double computeImpurity(const std::vector<int>& y) const;
    void findBestSplit(const DT_MATRIX_TYPE& X, const std::vector<int>& y,
                       int& best_feature, double& best_threshold, double& best_impurity) const;
    Node* buildTree(const DT_MATRIX_TYPE& X, const std::vector<int>& y, int depth);
    int predictSample(const std::vector<double>& sample, Node* node) const;
};

} // namespace algorithms
} // namespace ml

#endif // ML_ALGORITHMS_DECISION_TREE_H
