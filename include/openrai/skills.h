#pragma once

// -----------------------------------------------------------------------
// skills.h
//   A Skill is a *tool* an expert node may use next to its models: a
//   calculator, the GPU memory index, an external language model.  Given
//   the input it may produce a concrete answer plus the trace ("inner
//   thinking") that led to it.
//
//   Tools do NOT decide which expert fires.  That decision belongs to the
//   expert models (ExpertModel competence + the trained MLP scorer): the
//   primary fires the math expert because the math expert's *model* finds
//   the input familiar, and only then does the expert reach for its
//   calculator.  A tool that cannot apply (no numbers to compute, nothing
//   in memory, no LM) simply returns handled=false.
// -----------------------------------------------------------------------

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace openrai {

class VectorMemory;
class ExperienceStore;
class Tokenizer;
class EmbeddingLayer;
class LMAdapter;

// What a skill gets to see.  `text` is always set; the rest is filled by
// the primary when it has it (memory hits, other experts' findings, goals).
struct SkillContext {
    std::string              text;
    std::vector<std::string> memory;     // relevant memory lines
    std::vector<std::string> findings;   // other experts' findings this tick
    std::vector<std::string> goals;      // "stability:0.81" ...
    uint64_t                 tick = 0;
};

struct SkillResult {
    bool        handled    = false;   // the skill recognised and solved the input
    std::string answer;               // final answer text
    std::string thinking;             // step-by-step trace
    float       confidence = 0.f;     // 0..1
    std::vector<std::string> steps;   // same trace, one entry per step
    std::vector<std::string> context; // memory lines the skill found (memory expert)
    int64_t     experience_id = 0;    // recalled experience, if any
    std::string recalled_immediate;   // the raw expert answer of that experience
};

struct Skill {
    std::string name;
    // use the tool on the input (handled=false when it does not apply)
    std::function<SkillResult(const SkillContext&)> run;
    // slow skills (LLM calls) take seconds: the primary fires them after
    // arbitration and collects their outcome in a later tick
    bool slow = false;
    bool valid() const { return static_cast<bool>(run); }
};

// ---- Math skill ---------------------------------------------------------------
// Understands arithmetic in plain text, e.g.
//   "what is 100*100 and add 3/4 or sum divided by 1.34%"
//   "calculate (12 + 8) * 3 minus 10 percent"
//   "2^10 plus square root of 144"
// Clauses separated by "and", "then", "or", "," are applied to the running
// result ("add X", "subtract X", "times X", "divided by X", "sum divided
// by X", ...).  '%' after a number means percent (x / 100).
Skill make_math_skill();

// Evaluate one arithmetic expression (no natural language).  Throws
// std::runtime_error on syntax errors.  `acc` is the value of the '#'
// placeholder ("the sum / result / it").
double evaluate_arithmetic(const std::string& expression, double acc = 0.0);

// Full natural-language solve (what make_math_skill().run does).
SkillResult solve_math_text(const std::string& text);

// ---- Memory skill ------------------------------------------------------------
// Searches the vector memory (GPU index) for the input, recalls a stored
// experience (question -> answer learned earlier) when one is similar
// enough, and always returns the closest memories as context for the
// other experts / the language model.
struct MemorySkillOptions {
    float  recall_threshold = 0.90f;   // cosine needed to reuse a stored answer
    size_t top_k            = 5;
};
Skill make_memory_skill(VectorMemory* memory, ExperienceStore* store,
                        const Tokenizer* tokenizer, const EmbeddingLayer* embedder,
                        MemorySkillOptions opt = {});

// ---- Language skill (slow) -------------------------------------------------------
// Composes the final natural-language answer from the expert findings and
// memory context through the external LM.
Skill make_language_skill(LMAdapter* lm);

} // namespace openrai
