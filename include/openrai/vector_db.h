#pragma once

#include <string>
#include <vector>
#include <memory>

// Forward declare sqlite3
struct sqlite3;

namespace openrai {

// Vector database entry
struct VectorEntry {
    int64_t id;
    std::string text_value;
    std::vector<float> embedding;
    std::string tag;
    int64_t timestamp;
};

// Search result
struct VectorSearchResult {
    int64_t id;
    std::string text_value;
    std::string tag;
    float similarity;
    int64_t timestamp;
};

// SQLite-based vector database
class VectorDatabase {
public:
    VectorDatabase();
    ~VectorDatabase();

    // Initialize database
    bool initialize(const std::string& db_path);

    // Close database
    void close();

    // Store vector with text value (not node name!)
    bool store(const std::string& text_value,
               const std::vector<float>& embedding,
               const std::string& tag = "");

    // Search by vector similarity
    std::vector<VectorSearchResult> search(const std::vector<float>& query_embedding,
                                           size_t top_k = 5);

    // Get total count
    int64_t count() const;

    // Clear all entries
    bool clear();

private:
    sqlite3* db_ = nullptr;
    std::string db_path_;
    size_t embedding_dim_ = 32;

    // Helper: create tables if not exist
    bool create_tables();

    // Helper: serialize embedding to blob
    static std::vector<uint8_t> serialize_embedding(const std::vector<float>& embedding);

    // Helper: deserialize embedding from blob
    static std::vector<float> deserialize_embedding(const uint8_t* data, size_t size);

    // Helper: cosine similarity
    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
};

} // namespace openrai
