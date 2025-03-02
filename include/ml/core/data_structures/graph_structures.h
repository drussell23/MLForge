#ifndef ML_CORE_DATA_STRUCTURES_GRAPH_H
#define ML_CORE_DATA_STRUCTURES_GRAPH_H

#include <unordered_map>
#include <vector>
#include <queue>
#include <stdexcept>
#include <algorithm>

using namespace std;

namespace ml
{
    namespace core
    {
        namespace data_structures
        {
            /// @brief A templated Graph data structure for representing weighted graphs.
            /// @tparam Vertex Type for vertices (must be hashable, e.g., int, string).
            /// @tparam Weight Type for edge weights (default is double).
            template <typename Vertex, typename Weight = double>
            class Graph
            {
            public:
                /// @brief Adds a vertex to the graph.
                /// @param v The vertex to add.
                void addVertex(const Vertex &v)
                {
                    if (adjList_.find(v) == adjList_.end())
                    {
                        adjList_[v] = vector<pair<Vertex, Weight>>();
                    }
                }

                /// @brief Adds an edge from src to dest with a given weight.
                /// @param src The source vertex.
                /// @param dest The destination vertex.
                /// @param weight The edge weight.
                /// @param undirected If true, also adds the reverse edge.
                void addEdge(const Vertex &src, const Vertex &dest, Weight weight, bool undirected = true)
                {
                    addVertex(src);
                    addVertex(dest);
                    adjList_[src].push_back(make_pair(dest, weight));

                    if (undirected)
                    {
                        adjList_[dest].push_back(make_pair(src, weight));
                    }
                }

                /// @brief Removes the edge from src to dest.
                /// @param src The source vertex.
                /// @param dest The destination vertex.
                /// @param undirected If true, also removes the reverse edge.
                /// @return True if an edge was removed, false otherwise.
                bool removeEdge(const Vertex &src, const Vertex &dest, bool undirected = true)
                {
                    bool removed = removeEdgeHelper(src, dest);

                    if (undirected)
                    {
                        removed = removeEdgeHelper(dest, src) || removed;
                    }
                    return removed;
                }

                /// @brief Checks if the graph contains a given vertex.
                /// @param v The vertex to check.
                /// @return True if the vertex exists, false otherwise.
                bool contains(const Vertex &v) const
                {
                    return adjList_.find(v) != adjList_.end();
                }

                /// @brief Returns the list of neighbors (vertex and weight pairs) for a given vertex.
                /// @param v The vertex whose neighbors to retrieve.
                /// @return A constant reference to a vector of (vertex, weight) pairs.
                const vector<pair<Vertex, Weight>> &getNeighbors(const Vertex &v) const
                {
                    auto it = adjList_.find(v);

                    if (it == adjList_.end())
                    {
                        throw invalid_argument("Vertex not found in grpah.");
                    }
                    return it->second;
                }

                // @brief Performs a breadth-first search (BFS) starting from the given vertex.
                /// @param source The starting vertex.
                /// @return A vector containing vertices in the order they were visited.
                vector<Vertex> bfs(const Vertex &source) const
                {
                    vector<Vertex> order;

                    if (adjList_.find(source) == adjList_.end())
                    {
                        return order;
                    }

                    unordered_map<Vertex, bool> visited;
                    queue<Vertex> q;
                    q.push(source);
                    visited[source] = true;

                    while (!q.empty())
                    {
                        Vertex current = q.front();
                        q.pop();
                        order.push_back(current);

                        for (const auto &neighbor : getNeighbors(current))
                        {
                            if (!visited[neighbor.first])
                            {
                                visited[neighbor.first] = true;
                                q.push(neighbor.first);
                            }
                        }
                    }
                    return order;
                }

                /// @brief Performs a depth-first search (DFS) starting from the given vertex.
                /// @param source The starting vertex.
                /// @return A vector containing vertices in the order they were visited.
                vector<Vertex> dfs(const Vertex &source) const
                {
                    vector<Vertex> order;
                    unordered_map<Vertex, bool> visited;
                    dfsHelper(source, visited, order);
                    return order;
                }

            private:
                unordered_map<Vertex, vector<pair<Vertex, Weight>>> adjList_;

                // Helper function to remove an edge from src to dest.
                bool removeEdgeHelper(const Vertex &src, const Vertex &dest)
                {
                    auto it = adjList_.find(src);

                    if (it == adjList_.end())
                        return false;

                    auto &edges = it->second;
                    auto origSize = edges.size();

                    edges.erase(remove_if(edges.begin(), edges.end(), [&dest](const pair<Vertex, Weight> &edge)
                                          { return edge.first == dest; }),
                                edges.end());

                    return edges.size() < origSize;
                }

                // Recursive helper for DFS.
                void dfsHelper(const Vertex &v, unordered_map<Vertex, bool> &visited, vector<Vertex> &order) const
                {
                    if (visited[v])
                        return;

                    visited[v] = true;
                    order.push_back(v);

                    for (const auto &neighbor : getNeighbors(v))
                    {
                        dfsHelper(neighbor.first, visited, order);
                    }
                }
            };
        } // namespace data_structures
    } // namespace core
} // namespace ml

#endif // ML_CORE_DATA_STRUCTURES_GRAPH_H