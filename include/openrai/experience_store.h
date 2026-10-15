#pragma once

// -----------------------------------------------------------------------
// experience_store.h
//   Persistent "learning from use" store (SQLite, bundled amalgamation).
//
//   Everything the engine learns while it runs survives a restart:
//     experiences  - question -> final answer, which expert produced it,
//                    confidence, how often it was recalled, user reward
//                    (/good, /bad) and the question embedding so the memory
//                    expert can recall it by similarity next time
//     node_stats   - trust / priority / wins / misses of every expert
//     goals        - goal scores of the primary
//     graph_edges  - knowledge-graph bigram counts
//     kv           - free-form key/value (config snapshots, counters)
//
//   All methods are thread-safe (the connection is opened FULLMUTEX and
//   every call additionally takes an internal mutex).
// -----------------------------------------------------------------------

#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

struct sqlite3;

namespace openrai {

struct NodeStats;
struct GoalState;

struct Experience {
    int64_t     id = 0;
    std::string question;
    std::string answer;             // final (possibly LM-phrased) answer
    std::string immediate;          // the expert's raw answer before phrasing
    std::string expert;
    float       confidence = 0.f;
    float       utility    = 0.f;
    bool        solved     = false;
    int64_t     timestamp  = 0;     // unix seconds
    int         uses       = 0;     // how often it was recalled
    float       reward     = 0.f;   // cumulative user feedback (-n..+n)
    std::vector<float> embedding;   // question embedding (32 floats)
};

class ExperienceStore {
public:
    ExperienceStore();
    ~ExperienceStore();

    ExperienceStore(const ExperienceStore&)            = delete;
    ExperienceStore& operator=(const ExperienceStore&) = delete;

    bool open(const std::string& path);
    void close();
    bool is_open() const { return db_ != nullptr; }
    const std::string& path() const { return path_; }

    // ---- experiences ----
    int64_t add(const Experience& e);                   // returns id (0 on failure)
    bool    get(int64_t id, Experience& out) const;
    bool    reward(int64_t id, float delta);            // user feedback
    bool    touch(int64_t id);                          // uses += 1
    std::vector<Experience> recent(size_t n) const;
    std::vector<Experience> all(bool with_embeddings = true) const;
    int64_t count() const;

    // ---- node statistics ----
    bool save_node_stats(const std::string& node, const NodeStats& st, int priority);
    bool load_node_stats(const std::string& node, NodeStats& st, int& priority) const;

    // ---- goals ----
    bool save_goals(const std::vector<GoalState>& goals);
    std::vector<GoalState> load_goals() const;

    // ---- knowledge graph ----
    using Edge = std::tuple<std::string, std::string, int>;
    bool save_graph(const std::vector<Edge>& edges);    // replaces the table
    std::vector<Edge> load_graph() const;

    // ---- key / value ----
    bool set_kv(const std::string& key, const std::string& value);
    bool get_kv(const std::string& key, std::string& value) const;

private:
    sqlite3*            db_ = nullptr;
    std::string         path_;
    mutable std::mutex  mutex_;

    bool exec(const char* sql);
    bool create_tables();
};

} // namespace openrai
