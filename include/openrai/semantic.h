#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <cstddef>

namespace openrai {

// Forward declarations
struct SemanticHit {
    std::string text;
    std::string tag;
    float similarity = 0.0f;
    std::chrono::system_clock::time_point timestamp;
};

// 1) Tokenizer -----------------------------------------------------------
class Tokenizer {
public:
    std::vector<std::string> tokenize(const std::string& text) const;
    std::string detokenize(const std::vector<std::string>& tokens) const;
    std::string get_value_from_tokens(const std::vector<std::string>& tokens, 
                                      const std::vector<SemanticHit>& context,
                                      size_t max_words = 50) const;
};

// 2) Embedding layer -----------------------------------------------------
class EmbeddingLayer {
public:
    explicit EmbeddingLayer(size_t dim = 32);
    size_t dimension() const { return dim_; }
    std::vector<float> embed(const std::vector<std::string>& tokens) const;

private:
    size_t dim_;
};

class VectorMemory {
public:
    explicit VectorMemory(size_t dim = 32, size_t capacity = 5000);

    void store(const std::string& text,
               const std::vector<float>& embedding,
               const std::string& tag = "reasoning");

    std::vector<SemanticHit> search(const std::vector<float>& query, size_t top_k = 5) const;

    std::vector<SemanticHit> retrieve_context(const std::string& text,
                                              const Tokenizer& tokenizer,
                                              const EmbeddingLayer& embedder,
                                              size_t top_k = 5) const;

    size_t size() const;

    bool save_to_disk(const std::string& path) const;
    bool load_from_disk(const std::string& path);
    void set_persistence_path(const std::string& path);
    const std::string& persistence_path() const { return persistence_path_; }

private:
    struct Item {
        std::string text;
        std::string tag;
        std::vector<float> embedding;
        std::chrono::system_clock::time_point timestamp;
    };

    size_t dim_;
    size_t capacity_;
    std::vector<Item> items_;
    std::string persistence_path_;

    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
};

// 4) Knowledge graph ----------------------------------------------------
class KnowledgeGraph {
public:
    void ingest(const std::string& text, const Tokenizer& tokenizer);
    std::vector<std::string> neighbors(const std::string& node, size_t top_k = 5) const;
    float support_score(const std::string& text, const Tokenizer& tokenizer) const;
    size_t edge_count() const;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, int>> edges_;
};

// 5) Goal system ---------------------------------------------------------
struct GoalState {
    std::string name;
    float score = 0.0f;
};

class GoalSystem {
public:
    GoalSystem();

    void update_from_text(const std::string& text);
    std::vector<GoalState> top_goals(size_t top_k = 3) const;
    float alignment_score(const std::string& text) const;

private:
    std::vector<GoalState> goals_;

    static bool contains_any(const std::string& text, const std::vector<std::string>& words);
};

// 6) Self-reflection / challenge loop -----------------------------------
struct ReflectionResult {
    std::string conclusion;
    float confidence = 0.0f;
    bool should_output = false;
    std::vector<std::string> counterpoints;
};

class SelfReflection {
public:
    ReflectionResult reflect(const std::string& conclusion,
                             const std::vector<SemanticHit>& context,
                             const KnowledgeGraph& graph,
                             const GoalSystem& goals) const;
};

} // namespace openrai
