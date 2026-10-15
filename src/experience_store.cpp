#include <openrai/experience_store.h>
#include <openrai/expert_node.h>
#include <openrai/semantic.h>
#include <openrai/log.h>

#include <sqlite3.h>

#include <chrono>
#include <cstring>
#include <filesystem>

namespace openrai {

namespace {

struct Stmt {
    sqlite3_stmt* s = nullptr;
    Stmt(sqlite3* db, const char* sql) {
        if (sqlite3_prepare_v2(db, sql, -1, &s, nullptr) != SQLITE_OK) {
            LOG("[ExperienceStore] prepare failed: " << sqlite3_errmsg(db) << " sql=" << sql);
            s = nullptr;
        }
    }
    ~Stmt() { if (s) sqlite3_finalize(s); }
    bool ok() const { return s != nullptr; }
};

std::string col_text(sqlite3_stmt* s, int i) {
    const unsigned char* t = sqlite3_column_text(s, i);
    return t ? reinterpret_cast<const char*>(t) : "";
}

std::vector<float> col_blob_floats(sqlite3_stmt* s, int i) {
    const void* p = sqlite3_column_blob(s, i);
    const int   n = sqlite3_column_bytes(s, i);
    std::vector<float> v;
    if (p && n >= static_cast<int>(sizeof(float))) {
        v.resize(static_cast<size_t>(n) / sizeof(float));
        std::memcpy(v.data(), p, v.size() * sizeof(float));
    }
    return v;
}

void read_experience(sqlite3_stmt* s, Experience& e, bool with_embedding) {
    e.id         = sqlite3_column_int64(s, 0);
    e.question   = col_text(s, 1);
    e.answer     = col_text(s, 2);
    e.expert     = col_text(s, 3);
    e.confidence = static_cast<float>(sqlite3_column_double(s, 4));
    e.utility    = static_cast<float>(sqlite3_column_double(s, 5));
    e.solved     = sqlite3_column_int(s, 6) != 0;
    e.timestamp  = sqlite3_column_int64(s, 7);
    e.uses       = sqlite3_column_int(s, 8);
    e.reward     = static_cast<float>(sqlite3_column_double(s, 9));
    if (with_embedding) e.embedding = col_blob_floats(s, 10);
    e.immediate  = col_text(s, 11);
}

const char* kSelectExperience =
    "SELECT id, question, answer, expert, confidence, utility, solved, timestamp, uses, reward, embedding, immediate FROM experiences";

} // namespace

ExperienceStore::ExperienceStore() = default;
ExperienceStore::~ExperienceStore() { close(); }

bool ExperienceStore::open(const std::string& path) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (db_) { sqlite3_close(db_); db_ = nullptr; }
    path_ = path;
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    } catch (...) {}

    const int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX;
    if (sqlite3_open_v2(path.c_str(), &db_, flags, nullptr) != SQLITE_OK) {
        LOG("[ExperienceStore] cannot open " << path << ": " << (db_ ? sqlite3_errmsg(db_) : "?"));
        if (db_) { sqlite3_close(db_); db_ = nullptr; }
        return false;
    }
    exec("PRAGMA journal_mode=WAL;");
    exec("PRAGMA synchronous=NORMAL;");
    if (!create_tables()) {
        sqlite3_close(db_);
        db_ = nullptr;
        return false;
    }
    return true;
}

void ExperienceStore::close() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (db_) { sqlite3_close(db_); db_ = nullptr; }
}

bool ExperienceStore::exec(const char* sql) {
    if (!db_) return false;
    char* err = nullptr;
    if (sqlite3_exec(db_, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        LOG("[ExperienceStore] SQL error: " << (err ? err : "?"));
        sqlite3_free(err);
        return false;
    }
    return true;
}

bool ExperienceStore::create_tables() {
    const bool ok = exec(R"(
        CREATE TABLE IF NOT EXISTS experiences (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            question   TEXT NOT NULL,
            answer     TEXT NOT NULL,
            expert     TEXT,
            confidence REAL DEFAULT 0,
            utility    REAL DEFAULT 0,
            solved     INTEGER DEFAULT 0,
            timestamp  INTEGER NOT NULL,
            uses       INTEGER DEFAULT 0,
            reward     REAL DEFAULT 0,
            embedding  BLOB,
            immediate  TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_exp_ts ON experiences(timestamp);
        CREATE TABLE IF NOT EXISTS node_stats (
            node       TEXT PRIMARY KEY,
            trust      REAL, priority INTEGER,
            fired INTEGER, completed INTEGER, wins INTEGER,
            deadline_misses INTEGER, kills INTEGER, suspends INTEGER, restarts INTEGER,
            ema_latency_ms REAL,
            updated    INTEGER
        );
        CREATE TABLE IF NOT EXISTS goals (name TEXT PRIMARY KEY, score REAL);
        CREATE TABLE IF NOT EXISTS graph_edges (a TEXT, b TEXT, count INTEGER, PRIMARY KEY(a, b));
        CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
    )");
    if (!ok) return false;
    // migration: older stores lack the `immediate` column
    {
        Stmt st(db_, "SELECT immediate FROM experiences LIMIT 1");
        if (!st.ok()) exec("ALTER TABLE experiences ADD COLUMN immediate TEXT DEFAULT '';");
    }
    return true;
}

// ---- experiences ----------------------------------------------------------
int64_t ExperienceStore::add(const Experience& e) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return 0;
    Stmt st(db_, "INSERT INTO experiences (question, answer, expert, confidence, utility, solved, timestamp, uses, reward, embedding, immediate)"
                 " VALUES (?,?,?,?,?,?,?,?,?,?,?)");
    if (!st.ok()) return 0;
    const int64_t ts = e.timestamp ? e.timestamp
        : std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    sqlite3_bind_text  (st.s, 1, e.question.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text  (st.s, 2, e.answer.c_str(),   -1, SQLITE_TRANSIENT);
    sqlite3_bind_text  (st.s, 3, e.expert.c_str(),   -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(st.s, 4, e.confidence);
    sqlite3_bind_double(st.s, 5, e.utility);
    sqlite3_bind_int   (st.s, 6, e.solved ? 1 : 0);
    sqlite3_bind_int64 (st.s, 7, ts);
    sqlite3_bind_int   (st.s, 8, e.uses);
    sqlite3_bind_double(st.s, 9, e.reward);
    if (e.embedding.empty())
        sqlite3_bind_null(st.s, 10);
    else
        sqlite3_bind_blob(st.s, 10, e.embedding.data(),
                          static_cast<int>(e.embedding.size() * sizeof(float)), SQLITE_TRANSIENT);
    sqlite3_bind_text(st.s, 11, e.immediate.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(st.s) != SQLITE_DONE) {
        LOG("[ExperienceStore] insert failed: " << sqlite3_errmsg(db_));
        return 0;
    }
    return sqlite3_last_insert_rowid(db_);
}

bool ExperienceStore::get(int64_t id, Experience& out) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return false;
    std::string sql = std::string(kSelectExperience) + " WHERE id = ?";
    Stmt st(db_, sql.c_str());
    if (!st.ok()) return false;
    sqlite3_bind_int64(st.s, 1, id);
    if (sqlite3_step(st.s) != SQLITE_ROW) return false;
    read_experience(st.s, out, true);
    return true;
}

bool ExperienceStore::reward(int64_t id, float delta) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return false;
    Stmt st(db_, "UPDATE experiences SET reward = reward + ? WHERE id = ?");
    if (!st.ok()) return false;
    sqlite3_bind_double(st.s, 1, delta);
    sqlite3_bind_int64(st.s, 2, id);
    return sqlite3_step(st.s) == SQLITE_DONE && sqlite3_changes(db_) > 0;
}

bool ExperienceStore::touch(int64_t id) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return false;
    Stmt st(db_, "UPDATE experiences SET uses = uses + 1 WHERE id = ?");
    if (!st.ok()) return false;
    sqlite3_bind_int64(st.s, 1, id);
    return sqlite3_step(st.s) == SQLITE_DONE;
}

std::vector<Experience> ExperienceStore::recent(size_t n) const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<Experience> out;
    if (!db_) return out;
    std::string sql = std::string(kSelectExperience) + " ORDER BY id DESC LIMIT ?";
    Stmt st(db_, sql.c_str());
    if (!st.ok()) return out;
    sqlite3_bind_int(st.s, 1, static_cast<int>(n));
    while (sqlite3_step(st.s) == SQLITE_ROW) {
        Experience e;
        read_experience(st.s, e, false);
        out.push_back(std::move(e));
    }
    return out;
}

std::vector<Experience> ExperienceStore::all(bool with_embeddings) const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<Experience> out;
    if (!db_) return out;
    std::string sql = std::string(kSelectExperience) + " ORDER BY id ASC";
    Stmt st(db_, sql.c_str());
    if (!st.ok()) return out;
    while (sqlite3_step(st.s) == SQLITE_ROW) {
        Experience e;
        read_experience(st.s, e, with_embeddings);
        out.push_back(std::move(e));
    }
    return out;
}

int64_t ExperienceStore::count() const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return 0;
    Stmt st(db_, "SELECT COUNT(*) FROM experiences");
    if (!st.ok() || sqlite3_step(st.s) != SQLITE_ROW) return 0;
    return sqlite3_column_int64(st.s, 0);
}

// ---- node statistics --------------------------------------------------------
bool ExperienceStore::save_node_stats(const std::string& node, const NodeStats& s, int priority) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return false;
    Stmt st(db_, "INSERT OR REPLACE INTO node_stats (node, trust, priority, fired, completed, wins, deadline_misses,"
                 " kills, suspends, restarts, ema_latency_ms, updated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)");
    if (!st.ok()) return false;
    const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    sqlite3_bind_text  (st.s, 1, node.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(st.s, 2, s.trust);
    sqlite3_bind_int   (st.s, 3, priority);
    sqlite3_bind_int64 (st.s, 4, static_cast<sqlite3_int64>(s.fired));
    sqlite3_bind_int64 (st.s, 5, static_cast<sqlite3_int64>(s.completed));
    sqlite3_bind_int64 (st.s, 6, static_cast<sqlite3_int64>(s.wins));
    sqlite3_bind_int64 (st.s, 7, static_cast<sqlite3_int64>(s.deadline_misses));
    sqlite3_bind_int64 (st.s, 8, static_cast<sqlite3_int64>(s.kills));
    sqlite3_bind_int64 (st.s, 9, static_cast<sqlite3_int64>(s.suspends));
    sqlite3_bind_int64 (st.s, 10, static_cast<sqlite3_int64>(s.restarts));
    sqlite3_bind_double(st.s, 11, s.ema_latency_ms);
    sqlite3_bind_int64 (st.s, 12, now);
    return sqlite3_step(st.s) == SQLITE_DONE;
}

bool ExperienceStore::load_node_stats(const std::string& node, NodeStats& s, int& priority) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return false;
    Stmt st(db_, "SELECT trust, priority, fired, completed, wins, deadline_misses, kills, suspends, restarts,"
                 " ema_latency_ms FROM node_stats WHERE node = ?");
    if (!st.ok()) return false;
    sqlite3_bind_text(st.s, 1, node.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(st.s) != SQLITE_ROW) return false;
    s.trust           = static_cast<float>(sqlite3_column_double(st.s, 0));
    priority          = sqlite3_column_int(st.s, 1);
    s.fired           = static_cast<uint64_t>(sqlite3_column_int64(st.s, 2));
    s.completed       = static_cast<uint64_t>(sqlite3_column_int64(st.s, 3));
    s.wins            = static_cast<uint64_t>(sqlite3_column_int64(st.s, 4));
    s.deadline_misses = static_cast<uint64_t>(sqlite3_column_int64(st.s, 5));
    s.kills           = static_cast<uint64_t>(sqlite3_column_int64(st.s, 6));
    s.suspends        = static_cast<uint64_t>(sqlite3_column_int64(st.s, 7));
    s.restarts        = static_cast<uint64_t>(sqlite3_column_int64(st.s, 8));
    s.ema_latency_ms  = sqlite3_column_double(st.s, 9);
    return true;
}

// ---- goals ------------------------------------------------------------------------
bool ExperienceStore::save_goals(const std::vector<GoalState>& goals) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return false;
    Stmt st(db_, "INSERT OR REPLACE INTO goals (name, score) VALUES (?, ?)");
    if (!st.ok()) return false;
    for (const auto& g : goals) {
        sqlite3_reset(st.s);
        sqlite3_bind_text  (st.s, 1, g.name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(st.s, 2, g.score);
        if (sqlite3_step(st.s) != SQLITE_DONE) return false;
    }
    return true;
}

std::vector<GoalState> ExperienceStore::load_goals() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<GoalState> out;
    if (!db_) return out;
    Stmt st(db_, "SELECT name, score FROM goals");
    if (!st.ok()) return out;
    while (sqlite3_step(st.s) == SQLITE_ROW) {
        GoalState g;
        g.name  = col_text(st.s, 0);
        g.score = static_cast<float>(sqlite3_column_double(st.s, 1));
        out.push_back(g);
    }
    return out;
}

// ---- knowledge graph ----------------------------------------------------------------
bool ExperienceStore::save_graph(const std::vector<Edge>& edges) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return false;
    if (!exec("BEGIN; DELETE FROM graph_edges;")) return false;
    Stmt st(db_, "INSERT OR REPLACE INTO graph_edges (a, b, count) VALUES (?,?,?)");
    if (!st.ok()) { exec("ROLLBACK;"); return false; }
    for (const auto& e : edges) {
        sqlite3_reset(st.s);
        sqlite3_bind_text(st.s, 1, std::get<0>(e).c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(st.s, 2, std::get<1>(e).c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int (st.s, 3, std::get<2>(e));
        if (sqlite3_step(st.s) != SQLITE_DONE) { exec("ROLLBACK;"); return false; }
    }
    return exec("COMMIT;");
}

std::vector<ExperienceStore::Edge> ExperienceStore::load_graph() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<Edge> out;
    if (!db_) return out;
    Stmt st(db_, "SELECT a, b, count FROM graph_edges");
    if (!st.ok()) return out;
    while (sqlite3_step(st.s) == SQLITE_ROW)
        out.emplace_back(col_text(st.s, 0), col_text(st.s, 1), sqlite3_column_int(st.s, 2));
    return out;
}

// ---- key / value -----------------------------------------------------------------------
bool ExperienceStore::set_kv(const std::string& key, const std::string& value) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return false;
    Stmt st(db_, "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)");
    if (!st.ok()) return false;
    sqlite3_bind_text(st.s, 1, key.c_str(),   -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st.s, 2, value.c_str(), -1, SQLITE_TRANSIENT);
    return sqlite3_step(st.s) == SQLITE_DONE;
}

bool ExperienceStore::get_kv(const std::string& key, std::string& value) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!db_) return false;
    Stmt st(db_, "SELECT value FROM kv WHERE key = ?");
    if (!st.ok()) return false;
    sqlite3_bind_text(st.s, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(st.s) != SQLITE_ROW) return false;
    value = col_text(st.s, 0);
    return true;
}

} // namespace openrai
