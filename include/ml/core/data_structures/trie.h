#ifndef ML_CORE_DATA_STRUCTURES_TRIE_H
#define ML_CORE_DATA_STRUCTURES_TRIE_H

#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace ml {
namespace core {
namespace data_structures {

/// @brief A robust Trie (prefix tree) for efficient string storage and retrieval.
class Trie {
public:
    /// @brief Constructs an empty Trie.
    Trie();

    /// @brief Destructor.
    ~Trie();

    /// @brief Inserts a word into the trie.
    /// @param word The word to insert.
    void insert(const std::string& word);

    /// @brief Checks if a word exists in the trie.
    /// @param word The word to search for.
    /// @return True if the word exists, false otherwise.
    bool search(const std::string& word) const;

    /// @brief Checks if any word in the trie starts with the given prefix.
    /// @param prefix The prefix to search for.
    /// @return True if there is at least one word with the prefix, false otherwise.
    bool startsWith(const std::string& prefix) const;

    /// @brief Removes a word from the trie.
    /// @param word The word to remove.
    /// @return True if the word was successfully removed, false if the word was not found.
    bool remove(const std::string& word);

    /// @brief Retrieves all words in the trie that start with the given prefix.
    /// @param prefix The prefix to search for.
    /// @return A vector containing all matching words.
    std::vector<std::string> getWordsWithPrefix(const std::string& prefix) const;

    /// @brief Counts the total number of words stored in the trie.
    /// @return The number of words.
    int countWords() const;

    /// @brief Clears all entries in the trie.
    void clear();

private:
    /// @brief Internal structure representing a node in the Trie.
    struct TrieNode {
        bool isEndOfWord; ///< True if the node marks the end of a valid word.
        std::unordered_map<char, std::unique_ptr<TrieNode>> children; ///< Child nodes indexed by character.

        TrieNode() : isEndOfWord(false) {}
    };

    /// @brief The root node of the Trie.
    std::unique_ptr<TrieNode> root_;

    /// @brief Helper function to search for a node corresponding to a prefix.
    /// @param prefix The prefix to search for.
    /// @return A pointer to the node representing the end of the prefix, or nullptr if not found.
    const TrieNode* searchPrefix(const std::string& prefix) const;

    /// @brief Recursively collects words from a given node.
    /// @param node The current node.
    /// @param prefix The accumulated prefix so far.
    /// @param words The vector to collect words into.
    void collectWords(const TrieNode* node, const std::string& prefix, std::vector<std::string>& words) const;

    /// @brief Recursively removes a word from the trie.
    /// @param node The current node.
    /// @param word The word to remove.
    /// @param depth The current depth in the trie.
    /// @return True if the current node should be deleted, false otherwise.
    bool removeHelper(TrieNode* node, const std::string& word, int depth);
};

} // namespace data_structures
} // namespace core
} // namespace ml

#endif // ML_CORE_DATA_STRUCTURES_TRIE_H
