#include <openrai/memory.h>
#include <openrai/log.h>
#include <algorithm>

namespace openrai {

MemorySystem::MemorySystem() { LOG("[Memory] Initialized"); }

void MemorySystem::add_current_output(const Output& output) {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    current_memory_.push_back(output);
    if (current_memory_.size() > MAX_CURRENT) current_memory_.pop_front();
}

std::vector<Output> MemorySystem::get_current_outputs() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return {current_memory_.begin(), current_memory_.end()};
}

std::vector<Output> MemorySystem::get_short_term_memory() {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    return {short_term_memory_.begin(), short_term_memory_.end()};
}

void MemorySystem::filter_to_short_term() {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    auto sorted = std::vector<Output>(current_memory_.begin(), current_memory_.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const Output& a, const Output& b) {
                  return a.relevance_score > b.relevance_score;
              });
    short_term_memory_.clear();
    for (size_t i = 0; i < std::min(MAX_SHORT_TERM, sorted.size()); ++i)
        short_term_memory_.push_back(sorted[i]);

    LOG("[Memory] Filtered " << sorted.size()
        << " -> " << short_term_memory_.size() << " short-term entries");
}

void MemorySystem::add_to_long_term(const MemoryEntry& entry) {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    long_term_memory_.push_back(entry);
    if (long_term_memory_.size() > MAX_LONG_TERM) long_term_memory_.pop_front();
}

std::vector<MemoryEntry> MemorySystem::retrieve_from_long_term(
        const std::string& query, size_t top_k) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<MemoryEntry> results;
    for (const auto& e : long_term_memory_)
        if (e.content.find(query) != std::string::npos)
            results.push_back(e);
    if (results.size() > top_k) results.resize(top_k);
    return results;
}

void MemorySystem::clear_current() {
    std::lock_guard<std::shared_mutex> lock(mutex_);
    current_memory_.clear();
}

size_t MemorySystem::long_term_size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return long_term_memory_.size();
}

bool MemorySystem::compare_relevance(const Output& a, const Output& b) {
    return a.relevance_score > b.relevance_score;
}

} // namespace openrai
