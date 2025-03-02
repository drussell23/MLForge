#include "ml/core/data_structures/trie.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <algorithm>

using namespace std;

namespace ml
{
    namespace core
    {
        namespace data_structures
        {

            Trie::Trie() : root_(make_unique<TrieNode>()) {}

            Trie::~Trie()
            {
                clear();
            }

            void Trie::insert(const string &word)
            {
                TrieNode *current = root_.get();
                for (char ch : word)
                {
                    if (current->children.find(ch) == current->children.end())
                    {
                        current->children[ch] = make_unique<TrieNode>();
                    }
                    current = current->children[ch].get();
                }
                current->isEndOfWord = true;
            }

            bool Trie::search(const string &word) const
            {
                const TrieNode *node = searchPrefix(word);
                return (node != nullptr && node->isEndOfWord);
            }

            bool Trie::startsWith(const string &prefix) const
            {
                return (searchPrefix(prefix) != nullptr);
            }

            const Trie::TrieNode *Trie::searchPrefix(const string &prefix) const
            {
                const TrieNode *current = root_.get();
                for (char ch : prefix)
                {
                    auto it = current->children.find(ch);
                    if (it == current->children.end())
                    {
                        return nullptr;
                    }
                    current = it->second.get();
                }
                return current;
            }

            vector<string> Trie::getWordsWithPrefix(const string &prefix) const
            {
                vector<string> words;
                const TrieNode *node = searchPrefix(prefix);
                if (node != nullptr)
                {
                    collectWords(node, prefix, words);
                }
                return words;
            }

            void Trie::collectWords(const TrieNode *node, const string &prefix, vector<string> &words) const
            {
                if (node->isEndOfWord)
                {
                    words.push_back(prefix);
                }
                for (const auto &pair : node->children)
                {
                    char ch = pair.first;
                    collectWords(pair.second.get(), prefix + ch, words);
                }
            }

            bool Trie::remove(const string &word)
            {
                return removeHelper(root_.get(), word, 0);
            }

            bool Trie::removeHelper(TrieNode *node, const string &word, int depth)
            {
                if (depth == word.size())
                {
                    // Word is found. Mark the end as false.
                    if (!node->isEndOfWord)
                    {
                        return false; // Word not found.
                    }
                    node->isEndOfWord = false;
                    // If node has no children, it can be deleted.
                    return node->children.empty();
                }

                char ch = word[depth];
                auto it = node->children.find(ch);
                if (it == node->children.end())
                {
                    return false; // Word not found.
                }
                bool shouldDeleteChild = removeHelper(it->second.get(), word, depth + 1);
                if (shouldDeleteChild)
                {
                    node->children.erase(ch);
                    // Return true if current node is not end of another word and has no other children.
                    return !node->isEndOfWord && node->children.empty();
                }
                return false;
            }

            int Trie::countWords() const
            {
                vector<string> words = getWordsWithPrefix("");
                return static_cast<int>(words.size());
            }

            void Trie::clear()
            {
                root_.reset(new TrieNode());
            }

        } // namespace data_structures
    } // namespace core
} // namespace ml
