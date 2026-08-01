#pragma once

#include <vector>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include "core.h"

namespace openrai {

// Memory System - tri memorije
class MemorySystem {
public:
    MemorySystem();

    // Trenutna memorija (cache) - sprema prvih 10 outputa
    void add_current_output(const Output& output);
    std::vector<Output> get_current_outputs() const;

    // Kratkotrajna memorija - sprema 4 najrelevantnija outputa
    std::vector<Output> get_short_term_memory();
    void filter_to_short_term();

    // Dobavi zadnji output iz short-term memorije
    Output get_last_short_term_output() const;

    // Dugotrajna memorija (vektorska baza) - sprema važne reasoning misli
    void add_to_long_term(const MemoryEntry& entry);
    std::vector<MemoryEntry> retrieve_from_long_term(const std::string& query, size_t top_k = 5);
    size_t long_term_size() const;

    void clear_current();

private:
    // Trenutna memorija - max 110 outputa (100 nodes + injected slack)
    std::deque<Output> current_memory_;
    static constexpr size_t MAX_CURRENT    = 110;

    // Kratkotrajna memorija - top 10 od 100
    std::deque<Output> short_term_memory_;
    static constexpr size_t MAX_SHORT_TERM = 10;

    // Dugotrajna memorija (mock vektorska baza)
    std::deque<MemoryEntry> long_term_memory_;
    static constexpr size_t MAX_LONG_TERM  = 5000;

    mutable std::shared_mutex mutex_;

    // Helper za relevance sortiranje
    static bool compare_relevance(const Output& a, const Output& b);
};

} // namespace openrai
