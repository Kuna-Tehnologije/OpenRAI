#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <cstddef>
#include <tuple>

namespace openrai {

class GPUVectorIndex;

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

// Vector memory: items on the host (text + tag + embedding), similarity
// search on the GPU through GPUVectorIndex (CPU brute force without CUDA).
// Thread-safe: the memory expert searches from its own thread while the
// main loop stores.
class VectorMemory {
public:
    explicit VectorMemory(size_t dim = 32, size_t capacity = 100000, bool use_gpu_index = true);
    ~VectorMemory();

    void store(const std::string& text,
               const std::vector<float>& embedding,
               const std::string& tag = "reasoning");

    std::vector<SemanticHit> search(const std::vector<float>& query, size_t top_k = 5) const;
    // Same, but only items whose tag starts with `tag_prefix` (e.g. "experience:")
    std::vector<SemanticHit> search_tagged(const std::vector<float>& query, const std::string& tag_prefix,
                                           size_t top_k = 5) const;

    std::vector<SemanticHit> retrieve_context(const std::string& text,
                                              const Tokenizer& tokenizer,
                                              const EmbeddingLayer& embedder,
                                              size_t top_k = 5) const;

    size_t size() const;
    size_t capacity() const { return capacity_; }
    bool   gpu_index_active() const;
    double last_search_us() const;
    uint64_t searches() const;

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
    std::unique_ptr<GPUVectorIndex> index_;
    mutable std::shared_mutex mutex_;

    void rebuild_index_unlocked();
    std::vector<SemanticHit> search_unlocked(const std::vector<float>& query, size_t top_k,
                                             const std::string* tag_prefix) const;
    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
};

// 4) Knowledge graph ----------------------------------------------------
class KnowledgeGraph {
public:
    void ingest(const std::string& text, const Tokenizer& tokenizer);
    std::vector<std::string> neighbors(const std::string& node, size_t top_k = 5) const;
    float support_score(const std::string& text, const Tokenizer& tokenizer) const;
    size_t edge_count() const;

    // persistence
    std::vector<std::tuple<std::string, std::string, int>> export_edges() const;
    void import_edges(const std::vector<std::tuple<std::string, std::string, int>>& edges);

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

    // Relax every goal toward its baseline by `rate` (0..1) - called once
    // per tick so the primary's focus follows recent input instead of
    // saturating at 1.0 forever.
    void decay(float rate = 0.02f);
    float score_of(const std::string& name) const;
    void set_score(const std::string& name, float score);

private:
    std::vector<GoalState> goals_;
    std::vector<float>     baselines_;

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
