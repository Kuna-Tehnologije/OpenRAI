#include <openrai/reasoning.h>
#include <openrai/log.h>
#include <iomanip>
#include <cmath>

namespace openrai {

ReasoningModule::ReasoningModule(MemorySystem* memory_system)
    : memory_system_(memory_system) {
    LOG("[Reasoning] Module initialized");
}

void ReasoningModule::analyze(const std::vector<Output>& short_term_memory) {
    LOG("[Reasoning] Analyzing " << short_term_memory.size() << " NN-scored entries");

    auto long_term = memory_system_->retrieve_from_long_term("context", 3);
    LOG("[Reasoning] Long-term context hits: " << long_term.size());

    for (const auto& output : short_term_memory) {
        float importance = output.importance_score;
        bool  should_out = output.nn_should_output;

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3)
           << "[Reasoning]  " << output.data
           << "  imp=" << importance
           << "  out=" << (should_out ? "YES" : "no");
        LOG(ss.str());

        if (should_out)
            LOG("[OUTPUT] << " << output.data << "  (imp=" << importance << ")");

        MemoryEntry entry;
        entry.content    = output.data;
        entry.importance = importance;
        entry.created_at = std::chrono::system_clock::now();
        memory_system_->add_to_long_term(entry);
    }
}

ReasoningModule::ReasoningResult
ReasoningModule::evaluate_thought(const std::string& thought) {
    ReasoningResult result;
    result.thought          = thought;
    result.importance_score = calculate_importance(thought);
    result.should_output    = is_relevant_for_output(thought);
    return result;
}

float ReasoningModule::calculate_importance(const std::string& thought) {
    float score = 0.0f;
    if (thought.length() > 50)                           score += 0.3f;
    if (thought.find("important") != std::string::npos)  score += 0.5f;
    if (thought.find("error")     != std::string::npos)  score += 0.3f;
    return std::min(1.0f, score);
}

bool ReasoningModule::is_relevant_for_output(const std::string& thought) {
    return thought.find("important")  != std::string::npos ||
           thought.find("result")     != std::string::npos ||
           thought.find("conclusion") != std::string::npos;
}

} // namespace openrai
