#include "openrai/vector_db.h"
#include "openrai/log.h"
#include <sqlite3.h>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace openrai {

VectorDatabase::VectorDatabase() {}

VectorDatabase::~VectorDatabase() {
    close();
}

bool VectorDatabase::initialize(const std::string& db_path) {
    db_path_ = db_path;

    int rc = sqlite3_open(db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        LOG("[VectorDB] Failed to open database: " << sqlite3_errmsg(db_));
        return false;
    }

    if (!create_tables()) {
        LOG("[VectorDB] Failed to create tables");
        close();
        return false;
    }

    // Get count
    int64_t cnt = count();
    LOG("[VectorDB] Initialized: " << db_path << " (entries: " << cnt << ")");

    return true;
}

void VectorDatabase::close() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

bool VectorDatabase::create_tables() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS vector_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_value TEXT NOT NULL,
            embedding BLOB NOT NULL,
            tag TEXT,
            timestamp INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_tag ON vector_memory(tag);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON vector_memory(timestamp);
    )";

    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &err_msg);

    if (rc != SQLITE_OK) {
        LOG("[VectorDB] SQL error: " << err_msg);
        sqlite3_free(err_msg);
        return false;
    }

    return true;
}

std::vector<uint8_t> VectorDatabase::serialize_embedding(const std::vector<float>& embedding) {
    std::vector<uint8_t> blob(embedding.size() * sizeof(float));
    std::memcpy(blob.data(), embedding.data(), blob.size());
    return blob;
}

std::vector<float> VectorDatabase::deserialize_embedding(const uint8_t* data, size_t size) {
    std::vector<float> embedding(size / sizeof(float));
    std::memcpy(embedding.data(), data, size);
    return embedding;
}

bool VectorDatabase::store(const std::string& text_value,
                            const std::vector<float>& embedding,
                            const std::string& tag) {
    if (!db_) return false;

    const char* sql = "INSERT INTO vector_memory (text_value, embedding, tag, timestamp) VALUES (?, ?, ?, ?);";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        LOG("[VectorDB] Failed to prepare statement: " << sqlite3_errmsg(db_));
        return false;
    }

    // Bind parameters
    sqlite3_bind_text(stmt, 1, text_value.c_str(), -1, SQLITE_TRANSIENT);

    auto blob = serialize_embedding(embedding);
    sqlite3_bind_blob(stmt, 2, blob.data(), static_cast<int>(blob.size()), SQLITE_TRANSIENT);

    sqlite3_bind_text(stmt, 3, tag.c_str(), -1, SQLITE_TRANSIENT);

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    sqlite3_bind_int64(stmt, 4, timestamp);

    // Execute
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        LOG("[VectorDB] Failed to insert: " << sqlite3_errmsg(db_));
        return false;
    }

    return true;
}

float VectorDatabase::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
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

std::vector<VectorSearchResult> VectorDatabase::search(const std::vector<float>& query_embedding,
                                                        size_t top_k) {
    std::vector<VectorSearchResult> results;

    if (!db_) return results;

    const char* sql = "SELECT id, text_value, embedding, tag, timestamp FROM vector_memory;";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        LOG("[VectorDB] Failed to prepare search: " << sqlite3_errmsg(db_));
        return results;
    }

    // Iterate all rows and compute similarity
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t id = sqlite3_column_int64(stmt, 0);

        const char* text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        std::string text_value = text ? text : "";

        const uint8_t* blob_data = reinterpret_cast<const uint8_t*>(sqlite3_column_blob(stmt, 2));
        int blob_size = sqlite3_column_bytes(stmt, 2);
        auto embedding = deserialize_embedding(blob_data, blob_size);

        const char* tag_ptr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        std::string tag = tag_ptr ? tag_ptr : "";

        int64_t timestamp = sqlite3_column_int64(stmt, 4);

        // Compute similarity
        float sim = cosine_similarity(query_embedding, embedding);

        results.push_back(VectorSearchResult{id, text_value, tag, sim, timestamp});
    }

    sqlite3_finalize(stmt);

    // Sort by similarity descending
    std::sort(results.begin(), results.end(),
              [](const VectorSearchResult& a, const VectorSearchResult& b) {
                  return a.similarity > b.similarity;
              });

    // Return top_k
    if (results.size() > top_k) {
        results.resize(top_k);
    }

    return results;
}

int64_t VectorDatabase::count() const {
    if (!db_) return 0;

    const char* sql = "SELECT COUNT(*) FROM vector_memory;";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        return 0;
    }

    int64_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }

    sqlite3_finalize(stmt);
    return count;
}

bool VectorDatabase::clear() {
    if (!db_) return false;

    const char* sql = "DELETE FROM vector_memory;";

    char* err_msg = nullptr;
    int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &err_msg);

    if (rc != SQLITE_OK) {
        LOG("[VectorDB] Failed to clear: " << err_msg);
        sqlite3_free(err_msg);
        return false;
    }

    return true;
}

} // namespace openrai
