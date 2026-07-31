#pragma once

#include <string>
#include <vector>
#include "memory.h"
#include "core.h"

namespace openrai {

// Reasoning Module - izvodi analizu i prioritizaciju
class ReasoningModule {
public:
    ReasoningModule(MemorySystem* memory_system);

    // Proces reasoning-a
    void analyze(const std::vector<Output>& short_term_memory);

    // Logika odluèivanja: relevantno za output ili samo za vektorsku bazu
    struct ReasoningResult {
        std::string thought;
        bool should_output = false;
        float importance_score = 0.0f;
    };

    ReasoningResult evaluate_thought(const std::string& thought);

private:
    MemorySystem* memory_system_;

    // Helper funkcije
    float calculate_importance(const std::string& thought);
    bool is_relevant_for_output(const std::string& thought);
};

} // namespace openrai
