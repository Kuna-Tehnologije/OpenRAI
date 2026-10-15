#include <openrai/semantic.h>
#include <openrai/gpu_runtime.h>
#include <openrai/log.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <unordered_set>

namespace openrai {

// 1) Tokenizer -----------------------------------------------------------
std::vector<std::string> Tokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current;

    for (char ch : text) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (std::isalnum(c)) {
            current.push_back(static_cast<char>(std::tolower(c)));
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }

    if (!current.empty())
        tokens.push_back(current);

    return tokens;
}

std::string Tokenizer::detokenize(const std::vector<std::string>& tokens) const {
    if (tokens.empty()) return "";

    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) result += " ";
        result += tokens[i];
    }
    return result;
}

std::string Tokenizer::get_value_from_tokens(const std::vector<std::string>& tokens,
                                              const std::vector<SemanticHit>& context,
                                              size_t max_words) const {
    std::unordered_set<std::string> seen_words;
    std::vector<std::string> value_words;

    for (const auto& token : tokens) {
        if (seen_words.insert(token).second) {
            value_words.push_back(token);
        }
    }

    for (const auto& hit : context) {
        if (value_words.size() >= max_words) break;

        auto context_tokens = tokenize(hit.text);
        for (const auto& ct : context_tokens) {
            if (value_words.size() >= max_words) break;
            if (seen_words.insert(ct).second) {
                value_words.push_back(ct);
            }
        }
    }

    return detokenize(value_words);
}

// 2) Embedding layer -----------------------------------------------------
EmbeddingLayer::EmbeddingLayer(size_t dim)
    : dim_(dim == 0 ? 32 : dim) {}

std::vector<float> EmbeddingLayer::embed(const std::vector<std::string>& tokens) const {
    std::vector<float> vec(dim_, 0.0f);
    if (tokens.empty()) return vec;

    std::hash<std::string> hasher;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto& t = tokens[i];
        const size_t h = hasher(t);
        const size_t bucket = h % dim_;
        const float base = 1.0f + static_cast<float>(t.size()) * 0.08f;

        vec[bucket] += base;
        vec[(bucket + 7) % dim_] += base * 0.25f;
        vec[(bucket + 13) % dim_] += base * 0.15f;
    }

    float norm = std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0f));
    if (norm > 0.0f) {
        for (auto& v : vec) v /= norm;
    }

    return vec;
}

// 3) Vector memory -------------------------------------------------------
VectorMemory::VectorMemory(size_t dim, size_t capacity, bool use_gpu_index)
    : dim_(dim == 0 ? 32 : dim),
      capacity_(capacity == 0 ? 100000 : capacity),
      persistence_path_("data/vector_memory.db")
{
    if (use_gpu_index)
        index_ = std::make_unique<GPUVectorIndex>(dim_, std::min<size_t>(capacity_, 16384), 60, "vector-memory");
}

VectorMemory::~VectorMemory() = default;

void VectorMemory::store(const std::string& text,
                         const std::vector<float>& embedding,
                         const std::string& tag) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    // Exact duplicates of a recent item add nothing but noise to searches
    {
        const size_t look_back = std::min<size_t>(items_.size(), 64);
        for (size_t i = items_.size() - look_back; i < items_.size(); ++i)
            if (items_[i].text == text && items_[i].tag == tag) return;
    }
    Item item{ text, tag, embedding, std::chrono::system_clock::now() };
    if (item.embedding.size() != dim_) {
        item.embedding.resize(dim_, 0.0f);
    }

    if (items_.size() >= capacity_) {
        // Batch FIFO eviction (10 %) so the index rebuild is amortised
        const size_t drop = std::max<size_t>(1, capacity_ / 10);
        items_.erase(items_.begin(), items_.begin() + static_cast<std::ptrdiff_t>(std::min(drop, items_.size())));
        rebuild_index_unlocked();
    }

    if (index_) index_->add(item.embedding.data(), item.embedding.size());
    items_.push_back(std::move(item));
}

void VectorMemory::rebuild_index_unlocked() {
    if (!index_) return;
    std::vector<std::vector<float>> all;
    all.reserve(items_.size());
    for (const auto& it : items_) all.push_back(it.embedding);
    index_->rebuild(all);
}

float VectorMemory::cosine_similarity(const std::vector<float>& a,
                                      const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;

    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    if (na <= 0.0f || nb <= 0.0f) return 0.0f;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

std::vector<SemanticHit> VectorMemory::search_unlocked(const std::vector<float>& query, size_t top_k,
                                                       const std::string* tag_prefix) const {
    std::vector<SemanticHit> hits;
    if (items_.empty() || top_k == 0) return hits;

    if (index_ && index_->size() == items_.size()) {
        // GPU (or CPU-index) path: scores for every item, top-k selected.
        // With a tag filter we ask for more candidates and filter.
        const size_t want = tag_prefix ? std::min(items_.size(), std::max<size_t>(top_k * 8, 64)) : top_k;
        auto raw = index_->search(query.data(), query.size(), want);
        hits.reserve(top_k);
        for (const auto& h : raw) {
            if (h.index >= items_.size()) continue;
            const Item& it = items_[h.index];
            if (tag_prefix && it.tag.compare(0, tag_prefix->size(), *tag_prefix) != 0) continue;
            hits.push_back(SemanticHit{ it.text, it.tag, h.score, it.timestamp });
            if (hits.size() >= top_k) break;
        }
        return hits;
    }

    // Plain host path
    hits.reserve(items_.size());
    for (const auto& item : items_) {
        if (tag_prefix && item.tag.compare(0, tag_prefix->size(), *tag_prefix) != 0) continue;
        float sim = cosine_similarity(query, item.embedding);
        hits.push_back(SemanticHit{ item.text, item.tag, sim, item.timestamp });
    }
    std::sort(hits.begin(), hits.end(),
              [](const SemanticHit& a, const SemanticHit& b) { return a.similarity > b.similarity; });
    if (hits.size() > top_k) hits.resize(top_k);
    return hits;
}

std::vector<SemanticHit> VectorMemory::search(const std::vector<float>& query, size_t top_k) const {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    return search_unlocked(query, top_k, nullptr);
}

std::vector<SemanticHit> VectorMemory::search_tagged(const std::vector<float>& query,
                                                     const std::string& tag_prefix, size_t top_k) const {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    return search_unlocked(query, top_k, &tag_prefix);
}

std::vector<SemanticHit> VectorMemory::retrieve_context(const std::string& text,
                                                        const Tokenizer& tokenizer,
                                                        const EmbeddingLayer& embedder,
                                                        size_t top_k) const {
    auto tokens = tokenizer.tokenize(text);
    auto query = embedder.embed(tokens);
    return search(query, top_k);
}

size_t VectorMemory::size() const {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    return items_.size();
}

bool VectorMemory::gpu_index_active() const { return index_ && index_->is_cuda(); }
double VectorMemory::last_search_us() const { return index_ ? index_->last_search_us() : 0.0; }
uint64_t VectorMemory::searches() const { return index_ ? index_->searches() : 0; }

void VectorMemory::set_persistence_path(const std::string& path) {
    persistence_path_ = path;
}

bool VectorMemory::save_to_disk(const std::string& path) const {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        std::ofstream file(path, std::ios::trunc);
        if (!file.is_open()) return false;

        file << dim_ << ' ' << capacity_ << ' ' << items_.size() << '\n';
        for (const auto& item : items_) {
            const auto ts = std::chrono::duration_cast<std::chrono::seconds>(
                item.timestamp.time_since_epoch()).count();

            file << std::quoted(item.text) << ' '
                 << std::quoted(item.tag) << ' '
                 << ts << ' ';

            for (size_t i = 0; i < item.embedding.size(); ++i) {
                if (i) file << ',';
                file << item.embedding[i];
            }
            file << '\n';
        }

        return true;
    } catch (...) {
        return false;
    }
}

bool VectorMemory::load_from_disk(const std::string& path) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    try {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        size_t file_dim = 0;
        size_t file_capacity = 0;
        size_t count = 0;
        file >> file_dim >> file_capacity >> count;
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        if (file_dim != dim_ && file_dim != 0) {
            dim_ = file_dim;
            if (index_) index_ = std::make_unique<GPUVectorIndex>(dim_, std::min<size_t>(capacity_, 16384), 60, "vector-memory");
        }
        // capacity is a runtime setting now; the file value is informational

        items_.clear();
        items_.reserve(std::min(count, capacity_));

        for (size_t n = 0; n < count; ++n) {
            Item item;
            long long ts = 0;
            file >> std::quoted(item.text) >> std::quoted(item.tag) >> ts;

            std::string embedding_line;
            std::getline(file, embedding_line);
            if (embedding_line.empty() && !file.eof()) {
                std::getline(file, embedding_line);
            }

            item.embedding.assign(dim_, 0.0f);
            std::stringstream emb_stream(embedding_line);
            std::string val;
            size_t idx = 0;
            while (std::getline(emb_stream, val, ',') && idx < dim_) {
                if (!val.empty()) item.embedding[idx] = std::stof(val);
                ++idx;
            }
            item.timestamp = std::chrono::system_clock::time_point(std::chrono::seconds(ts));
            items_.push_back(std::move(item));
        }
        if (items_.size() > capacity_)
            items_.erase(items_.begin(), items_.begin() + static_cast<std::ptrdiff_t>(items_.size() - capacity_));

        rebuild_index_unlocked();
        return true;
    } catch (...) {
        rebuild_index_unlocked();
        return false;
    }
}

// 4) Knowledge graph ----------------------------------------------------
void KnowledgeGraph::ingest(const std::string& text, const Tokenizer& tokenizer) {
    auto tokens = tokenizer.tokenize(text);
    if (tokens.size() < 2) return;

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        edges_[tokens[i]][tokens[i + 1]]++;
    }
}

std::vector<std::string> KnowledgeGraph::neighbors(const std::string& node, size_t top_k) const {
    std::vector<std::pair<std::string, int>> scored;
    auto it = edges_.find(node);
    if (it == edges_.end()) return {};

    for (const auto& kv : it->second)
        scored.push_back(kv);

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<std::string> out;
    for (size_t i = 0; i < scored.size() && i < top_k; ++i)
        out.push_back(scored[i].first);
    return out;
}

float KnowledgeGraph::support_score(const std::string& text, const Tokenizer& tokenizer) const {
    auto tokens = tokenizer.tokenize(text);
    if (tokens.size() < 2) return 0.0f;

    float support = 0.0f;
    int pairs = 0;

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        auto node = edges_.find(tokens[i]);
        if (node != edges_.end()) {
            auto edge = node->second.find(tokens[i + 1]);
            if (edge != node->second.end()) {
                support += std::min(1.0f, 0.2f + 0.1f * static_cast<float>(edge->second));
                ++pairs;
            }
        }
    }

    if (pairs == 0) return 0.0f;
    return std::min(1.0f, support / static_cast<float>(pairs));
}

std::vector<std::tuple<std::string, std::string, int>> KnowledgeGraph::export_edges() const {
    std::vector<std::tuple<std::string, std::string, int>> out;
    for (const auto& node : edges_)
        for (const auto& edge : node.second)
            out.emplace_back(node.first, edge.first, edge.second);
    return out;
}

void KnowledgeGraph::import_edges(const std::vector<std::tuple<std::string, std::string, int>>& edges) {
    for (const auto& e : edges)
        edges_[std::get<0>(e)][std::get<1>(e)] += std::get<2>(e);
}

size_t KnowledgeGraph::edge_count() const {
    size_t total = 0;
    for (const auto& node : edges_)
        for (const auto& edge : node.second)
            total += static_cast<size_t>(edge.second);
    return total;
}

// 5) Goal system ---------------------------------------------------------
GoalSystem::GoalSystem()
    : goals_{
        {"stability", 0.35f},
        {"relevance", 0.35f},
        {"memory",    0.30f}
      },
      baselines_{0.35f, 0.35f, 0.30f} {}

void GoalSystem::decay(float rate) {
    rate = std::clamp(rate, 0.0f, 1.0f);
    for (size_t i = 0; i < goals_.size() && i < baselines_.size(); ++i)
        goals_[i].score += (baselines_[i] - goals_[i].score) * rate;
}

void GoalSystem::set_score(const std::string& name, float score) {
    for (auto& g : goals_)
        if (g.name == name) { g.score = std::clamp(score, 0.0f, 1.0f); return; }
}

float GoalSystem::score_of(const std::string& name) const {
    for (const auto& g : goals_)
        if (g.name == name) return g.score;
    return 0.0f;
}

bool GoalSystem::contains_any(const std::string& text, const std::vector<std::string>& words) {
    for (const auto& w : words)
        if (text.find(w) != std::string::npos)
            return true;
    return false;
}

void GoalSystem::update_from_text(const std::string& text) {
    if (contains_any(text, {"error", "fail", "overload", "temperature", "crash"}))
        goals_[0].score = std::min(1.0f, goals_[0].score + 0.08f);

    if (contains_any(text, {"result", "conclusion", "important", "relevance"}))
        goals_[1].score = std::min(1.0f, goals_[1].score + 0.06f);

    if (contains_any(text, {"memory", "context", "history", "retrieve"}))
        goals_[2].score = std::min(1.0f, goals_[2].score + 0.05f);
}

std::vector<GoalState> GoalSystem::top_goals(size_t top_k) const {
    auto copy = goals_;
    std::sort(copy.begin(), copy.end(),
              [](const GoalState& a, const GoalState& b) {
                  return a.score > b.score;
              });
    if (copy.size() > top_k) copy.resize(top_k);
    return copy;
}

float GoalSystem::alignment_score(const std::string& text) const {
    float score = 0.0f;
    if (contains_any(text, {"stability", "safe", "error", "temperature", "fail"}))
        score += goals_[0].score;
    if (contains_any(text, {"result", "conclusion", "important", "reason"}))
        score += goals_[1].score;
    if (contains_any(text, {"memory", "context", "retrieve", "history"}))
        score += goals_[2].score;

    return std::min(1.0f, score);
}

// 6) Self-reflection -----------------------------------------------------
ReflectionResult SelfReflection::reflect(const std::string& conclusion,
                                         const std::vector<SemanticHit>& context,
                                         const KnowledgeGraph& graph,
                                         const GoalSystem& goals) const {
    ReflectionResult result;
    result.conclusion = conclusion;

    const float avg_sim = [&]() {
        if (context.empty()) return 0.0f;
        float sum = 0.0f;
        for (const auto& hit : context) sum += std::max(0.0f, hit.similarity);
        return sum / static_cast<float>(context.size());
    }();

    float confidence = 0.38f;
    confidence += avg_sim * 0.35f;
    confidence += graph.support_score(conclusion, Tokenizer{}) * 0.15f;
    confidence += goals.alignment_score(conclusion) * 0.20f;

    if (context.empty()) {
        result.counterpoints.push_back("low memory support");
        confidence -= 0.08f;
    } else if (context.front().similarity < 0.20f) {
        result.counterpoints.push_back("weak semantic match");
        confidence -= 0.04f;
    }

    if (conclusion.find("maybe") != std::string::npos || conclusion.find("uncertain") != std::string::npos) {
        result.counterpoints.push_back("already uncertain wording");
        confidence -= 0.04f;
    }

    auto top_goals = goals.top_goals(1);
    if (!top_goals.empty() && top_goals.front().score < 0.40f) {
        result.counterpoints.push_back("goal signal still weak");
        confidence -= 0.04f;
    }

    confidence = std::clamp(confidence, 0.0f, 1.0f);
    result.confidence = confidence;
    result.should_output = confidence >= 0.58f;

    if (!result.should_output) {
        result.conclusion += " [needs more evidence]";
    }

    return result;
}

} // namespace openrai
