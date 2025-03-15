#ifndef KD_TREE_H
#define KD_TREE_H

#include <array>
#include <vector>
#include <memory>
#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <queue>

namespace ml
{
    namespace data
    {

        /**
         * @brief A KD-Tree implementation for k-dimensional space.
         *
         * This class provides methods to build a KD-Tree from a collection of points,
         * perform various types of searches (nearest neighbor, k-nearest, radius-based),
         * insert new points, and delete points. A custom distance metric can be provided.
         *
         * @tparam T The type of the coordinates (e.g., double, float).
         * @tparam k The dimensionality of the space.
         */
        template <typename T, std::size_t k>
        class KDTree
        {
        public:
            // Type alias for a point in k-dimensional space.
            using Point = std::array<T, k>;
            // Type alias for a custom distance function.
            using DistanceFunction = std::function<T(const Point &, const Point &)>;

            /**
             * @brief Constructs a KDTree from a set of points.
             *
             * @param points A vector of points.
             * @param distFunc Optional custom distance function. Defaults to squared Euclidean distance.
             */
            explicit KDTree(const std::vector<Point> &points, DistanceFunction distFunc = defaultDistance);

            /**
             * @brief Finds the nearest neighbor to the given query point.
             *
             * @param query The query point.
             * @return The nearest point in the tree.
             * @throws std::runtime_error if the tree is empty.
             */
            Point nearestNeighbor(const Point &query) const;

            /**
             * @brief Finds the k-nearest neighbors to the given query point.
             *
             * @param query The query point.
             * @param k_neighbors Number of nearest neighbors to return.
             * @return A vector of pairs containing the point and its distance to the query, sorted in ascending order.
             * @throws std::runtime_error if the tree is empty.
             */
            std::vector<std::pair<Point, T>> kNearestNeighbors(const Point &query, std::size_t k_neighbors) const;

            /**
             * @brief Finds all points within a given radius from the query point.
             *
             * @param query The query point.
             * @param radius The search radius.
             * @return A vector of points within the given radius.
             */
            std::vector<Point> radiusSearch(const Point &query, T radius) const;

            /**
             * @brief Inserts a new point into the KD-Tree.
             *
             * @param point The point to insert.
             */
            void insert(const Point &point);

            /**
             * @brief Removes a point from the KD-Tree.
             *
             * @param point The point to remove.
             * @return True if the point was found and removed, false otherwise.
             */
            bool remove(const Point &point);

        private:
            // Internal structure for tree nodes.
            struct Node
            {
                Point point;
                std::unique_ptr<Node> left;
                std::unique_ptr<Node> right;
                Node(const Point &pt) : point(pt), left(nullptr), right(nullptr) {}
            };

            // Root of the KD-Tree.
            std::unique_ptr<Node> root_;

            // Custom distance function.
            DistanceFunction distanceFunc_;

            // Default squared Euclidean distance function.
            static T defaultDistance(const Point &a, const Point &b)
            {
                T dist = 0;
                for (std::size_t i = 0; i < k; ++i)
                {
                    T d = a[i] - b[i];
                    dist += d * d;
                }
                return dist;
            }

            /**
             * @brief Recursively builds the KD-Tree.
             *
             * @param begin Iterator pointing to the beginning of the points range.
             * @param end Iterator pointing past the end of the points range.
             * @param depth Current depth in the tree.
             * @return A unique pointer to the constructed node.
             */
            std::unique_ptr<Node> buildTree(typename std::vector<Point>::iterator begin,
                                            typename std::vector<Point>::iterator end,
                                            int depth);

            /**
             * @brief Recursively performs nearest neighbor search.
             *
             * @param node Current node in the tree.
             * @param target The query point.
             * @param depth Current depth in the tree.
             * @param best Reference to the best node found so far.
             * @param bestDist Reference to the distance of the best node.
             */
            void nearestNeighborSearch(const Node *node,
                                       const Point &target,
                                       int depth,
                                       const Node *&best,
                                       T &bestDist) const;

            /**
             * @brief Recursively performs k-nearest neighbors search.
             *
             * @param node Current node in the tree.
             * @param target The query point.
             * @param depth Current depth in the tree.
             * @param maxHeap A max-heap (priority queue) that stores pairs of (distance, point).
             *                The top of the heap is the farthest among the current k-nearest.
             * @param k_neighbors Number of nearest neighbors to find.
             */
            void kNearestNeighborSearch(const Node *node,
                                        const Point &target,
                                        int depth,
                                        std::priority_queue<std::pair<T, Point>> &maxHeap,
                                        std::size_t k_neighbors) const;

            /**
             * @brief Recursively performs radius search.
             *
             * @param node Current node in the tree.
             * @param target The query point.
             * @param radius Squared radius for comparison.
             * @param depth Current depth in the tree.
             * @param results Vector to store points within the radius.
             */
            void radiusSearchRecursive(const Node *node,
                                       const Point &target,
                                       T radius,
                                       int depth,
                                       std::vector<Point> &results) const;

            /**
             * @brief Recursively inserts a new point into the tree.
             *
             * @param node Reference to the current node pointer.
             * @param point The point to insert.
             * @param depth Current depth in the tree.
             */
            void insertRecursive(std::unique_ptr<Node> &node, const Point &point, int depth);

            /**
             * @brief Recursively removes a point from the tree.
             *
             * @param node Reference to the current node pointer.
             * @param point The point to remove.
             * @param depth Current depth in the tree.
             * @return True if the point was removed, false otherwise.
             */
            bool removeRecursive(std::unique_ptr<Node> &node, const Point &point, int depth);

            /**
             * @brief Finds the node with the minimum value in the given axis within the subtree.
             *
             * @param node The root of the subtree.
             * @param axis The axis to consider.
             * @param depth Current depth in the tree.
             * @return A pointer to the node with the minimum value along the specified axis.
             */
            Node *findMin(Node *node, int axis, int depth) const;
        };

        template <typename T, std::size_t k>
        KDTree<T, k>::KDTree(const std::vector<Point> &points, DistanceFunction distFunc)
            : root_(nullptr), distanceFunc_(distFunc)
        {
            if (!points.empty())
            {
                std::vector<Point> pts = points;
                root_ = buildTree(pts.begin(), pts.end(), 0);
            }
        }

        template <typename T, std::size_t k>
        std::unique_ptr<typename KDTree<T, k>::Node>
        KDTree<T, k>::buildTree(typename std::vector<Point>::iterator begin,
                                typename std::vector<Point>::iterator end,
                                int depth)
        {
            if (begin >= end)
            {
                return nullptr;
            }
            int axis = depth % k;
            auto comparator = [axis](const Point &a, const Point &b)
            {
                return a[axis] < b[axis];
            };
            auto mid = begin + std::distance(begin, end) / 2;
            std::nth_element(begin, mid, end, comparator);
            auto node = std::make_unique<Node>(*mid);
            node->left = buildTree(begin, mid, depth + 1);
            node->right = buildTree(mid + 1, end, depth + 1);
            return node;
        }

        template <typename T, std::size_t k>
        typename KDTree<T, k>::Point KDTree<T, k>::nearestNeighbor(const Point &query) const
        {
            if (!root_)
            {
                throw std::runtime_error("KDTree is empty");
            }
            const Node *best = nullptr;
            T bestDist = std::numeric_limits<T>::max();
            nearestNeighborSearch(root_.get(), query, 0, best, bestDist);
            return best->point;
        }

        template <typename T, std::size_t k>
        void KDTree<T, k>::nearestNeighborSearch(const Node *node,
                                                 const Point &target,
                                                 int depth,
                                                 const Node *&best,
                                                 T &bestDist) const
        {
            if (!node)
                return;
            T d = distanceFunc_(node->point, target);
            if (d < bestDist)
            {
                bestDist = d;
                best = node;
            }
            int axis = depth % k;
            T diff = target[axis] - node->point[axis];
            const Node *nearBranch = diff < 0 ? node->left.get() : node->right.get();
            const Node *farBranch = diff < 0 ? node->right.get() : node->left.get();

            nearestNeighborSearch(nearBranch, target, depth + 1, best, bestDist);
            if (diff * diff < bestDist)
            {
                nearestNeighborSearch(farBranch, target, depth + 1, best, bestDist);
            }
        }

        template <typename T, std::size_t k>
        std::vector<std::pair<typename KDTree<T, k>::Point, T>>
        KDTree<T, k>::kNearestNeighbors(const Point &query, std::size_t k_neighbors) const
        {
            if (!root_)
            {
                throw std::runtime_error("KDTree is empty");
            }
            // Use a max-heap to store pairs of (distance, point).
            std::priority_queue<std::pair<T, Point>> maxHeap;
            kNearestNeighborSearch(root_.get(), query, 0, maxHeap, k_neighbors);
            std::vector<std::pair<Point, T>> result;
            while (!maxHeap.empty())
            {
                auto p = maxHeap.top();
                maxHeap.pop();
                result.push_back({p.second, p.first});
            }
            std::reverse(result.begin(), result.end());
            return result;
        }

        template <typename T, std::size_t k>
        void KDTree<T, k>::kNearestNeighborSearch(const Node *node,
                                                  const Point &target,
                                                  int depth,
                                                  std::priority_queue<std::pair<T, Point>> &maxHeap,
                                                  std::size_t k_neighbors) const
        {
            if (!node)
                return;
            T d = distanceFunc_(node->point, target);
            if (maxHeap.size() < k_neighbors)
            {
                maxHeap.push({d, node->point});
            }
            else if (d < maxHeap.top().first)
            {
                maxHeap.pop();
                maxHeap.push({d, node->point});
            }
            int axis = depth % k;
            T diff = target[axis] - node->point[axis];
            const Node *nearBranch = diff < 0 ? node->left.get() : node->right.get();
            const Node *farBranch = diff < 0 ? node->right.get() : node->left.get();

            kNearestNeighborSearch(nearBranch, target, depth + 1, maxHeap, k_neighbors);
            if (maxHeap.size() < k_neighbors || diff * diff < maxHeap.top().first)
            {
                kNearestNeighborSearch(farBranch, target, depth + 1, maxHeap, k_neighbors);
            }
        }

        template <typename T, std::size_t k>
        std::vector<typename KDTree<T, k>::Point>
        KDTree<T, k>::radiusSearch(const Point &query, T radius) const
        {
            std::vector<Point> results;
            T radiusSquared = radius * radius;
            radiusSearchRecursive(root_.get(), query, radiusSquared, 0, results);
            return results;
        }

        template <typename T, std::size_t k>
        void KDTree<T, k>::radiusSearchRecursive(const Node *node,
                                                 const Point &target,
                                                 T radiusSquared,
                                                 int depth,
                                                 std::vector<Point> &results) const
        {
            if (!node)
                return;
            T d = distanceFunc_(node->point, target);
            if (d <= radiusSquared)
            {
                results.push_back(node->point);
            }
            int axis = depth % k;
            T diff = target[axis] - node->point[axis];
            if (diff * diff <= radiusSquared)
            {
                radiusSearchRecursive(node->left.get(), target, radiusSquared, depth + 1, results);
                radiusSearchRecursive(node->right.get(), target, radiusSquared, depth + 1, results);
            }
            else if (diff < 0)
            {
                radiusSearchRecursive(node->left.get(), target, radiusSquared, depth + 1, results);
            }
            else
            {
                radiusSearchRecursive(node->right.get(), target, radiusSquared, depth + 1, results);
            }
        }

        template <typename T, std::size_t k>
        void KDTree<T, k>::insert(const Point &point)
        {
            insertRecursive(root_, point, 0);
        }

        template <typename T, std::size_t k>
        void KDTree<T, k>::insertRecursive(std::unique_ptr<Node> &node, const Point &point, int depth)
        {
            if (!node)
            {
                node = std::make_unique<Node>(point);
                return;
            }
            int axis = depth % k;
            if (point[axis] < node->point[axis])
            {
                insertRecursive(node->left, point, depth + 1);
            }
            else
            {
                insertRecursive(node->right, point, depth + 1);
            }
        }

        template <typename T, std::size_t k>
        bool KDTree<T, k>::remove(const Point &point)
        {
            return removeRecursive(root_, point, 0);
        }

        template <typename T, std::size_t k>
        bool KDTree<T, k>::removeRecursive(std::unique_ptr<Node> &node, const Point &point, int depth)
        {
            if (!node)
                return false;
            int axis = depth % k;
            if (node->point == point)
            {
                if (!node->right && !node->left)
                {
                    node.reset();
                    return true;
                }
                if (node->right)
                {
                    Node *minNode = findMin(node->right.get(), axis, depth + 1);
                    node->point = minNode->point;
                    return removeRecursive(node->right, minNode->point, depth + 1);
                }
                else
                {
                    Node *minNode = findMin(node->left.get(), axis, depth + 1);
                    node->point = minNode->point;
                    node->right = std::move(node->left);
                    node->left.reset();
                    return removeRecursive(node->right, minNode->point, depth + 1);
                }
            }
            else
            {
                if (point[axis] < node->point[axis])
                {
                    return removeRecursive(node->left, point, depth + 1);
                }
                else
                {
                    return removeRecursive(node->right, point, depth + 1);
                }
            }
        }

        template <typename T, std::size_t k>
        typename KDTree<T, k>::Node *KDTree<T, k>::findMin(Node *node, int axis, int depth) const
        {
            if (!node)
                return nullptr;
            int currentAxis = depth % k;
            if (currentAxis == axis)
            {
                if (node->left)
                {
                    return findMin(node->left.get(), axis, depth + 1);
                }
                return node;
            }
            Node *leftMin = findMin(node->left.get(), axis, depth + 1);
            Node *rightMin = findMin(node->right.get(), axis, depth + 1);
            Node *minNode = node;
            if (leftMin && leftMin->point[axis] < minNode->point[axis])
            {
                minNode = leftMin;
            }
            if (rightMin && rightMin->point[axis] < minNode->point[axis])
            {
                minNode = rightMin;
            }
            return minNode;
        }

    } // namespace data
} // namespace ml

#endif // KD_TREE_H
