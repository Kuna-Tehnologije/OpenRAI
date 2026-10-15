#pragma once

// -----------------------------------------------------------------------
// lm_adapter.h
//   Bridge to an external language model (ported from the
//   feature/language-model branch and made portable + configurable).
//
//   Backends
//     http  - any local OpenAI-compatible or Ollama server reached through
//             the system `curl` binary (no C++ HTTP dependency):
//               openai-chat : POST {url}/v1/chat/completions  (Jan, LM Studio,
//                             llama.cpp server, vLLM, Docker Model Runner)
//               openai      : POST {url}/v1/completions
//               ollama      : POST {url}/api/generate
//     mock  - deterministic stand-in for tests / offline runs
//     off   - disabled
//
//   The adapter never runs on the 7.82 Hz loop thread: the language expert
//   node calls it on its own thread and the primary treats that node as a
//   "slow" expert whose outcome arrives in a later tick.
// -----------------------------------------------------------------------

#include <memory>
#include <string>
#include <vector>

namespace openrai {

struct LMResponse {
    std::string text;
    bool        ok = false;
    std::string error;
    float       confidence = 0.f;
    int         tokens_used = 0;
    double      latency_ms = 0.0;
};

struct LMConfig {
    std::string backend     = "off";          // off | http | mock
    std::string url         = "http://127.0.0.1:1337";
    std::string api         = "openai-chat";  // openai-chat | openai | ollama
    std::string model;                        // server model id ("" = server default)
    std::string system_prompt =
        "You are OpenRAI, a real-time cognitive engine. Expert nodes have already "
        "analysed the input; use their findings and the memory context. Answer "
        "concisely and factually. If an expert computed a result, state it exactly.";
    int    max_tokens  = 256;
    float  temperature = 0.4f;
    int    timeout_s   = 30;                  // curl --max-time
    size_t max_prompt_chars = 6000;
};

class ILanguageModel {
public:
    virtual ~ILanguageModel() = default;
    virtual bool        load(const LMConfig& cfg) = 0;
    virtual LMResponse  generate(const std::string& system, const std::string& user) = 0;
    virtual bool        is_loaded() const = 0;
    virtual std::string model_info() const = 0;
};

class LMAdapter {
public:
    explicit LMAdapter(LMConfig cfg);
    ~LMAdapter();

    // Create the backend; for http also probes the server (GET {url}/v1/models
    // with a 2 s timeout).  Returns true when generate() can be used.
    bool initialize();
    bool is_ready() const { return ready_; }
    const LMConfig& config() const { return cfg_; }
    std::string model_info() const;

    LMResponse generate(const std::string& user_prompt);
    // Prompt with cognitive context: expert findings + memory hits + goals
    LMResponse generate_with_context(const std::string& question,
                                     const std::vector<std::string>& expert_findings,
                                     const std::vector<std::string>& memory_context,
                                     const std::vector<std::string>& goals);

    static std::string build_prompt(const std::string& question,
                                    const std::vector<std::string>& expert_findings,
                                    const std::vector<std::string>& memory_context,
                                    const std::vector<std::string>& goals,
                                    size_t max_chars);

private:
    LMConfig cfg_;
    std::unique_ptr<ILanguageModel> backend_;
    bool ready_ = false;
};

// ---- helpers shared with tests ---------------------------------------------------
std::string json_escape(const std::string& s);
// Extract the string value of the first occurrence of "key":"..." handling
// escapes.  Returns false when the key is missing.
bool json_extract_string(const std::string& json, const std::string& key, std::string& out);
// Run a command, capture stdout (+stderr).  Returns exit code (-1 if it could not run).
int run_command_capture(const std::string& cmd, std::string& output);

} // namespace openrai
