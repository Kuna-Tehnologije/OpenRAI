#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace openrai {

// Language Model response
struct LMResponse {
    std::string text;
    float confidence = 0.0f;
    std::vector<float> embedding;
    int tokens_used = 0;
};

// LM Configuration
struct LMConfig {
    std::string model_path;
    std::string model_type = "llama";  // "llama", "qwen", "onnx"
    int context_length = 4096;
    int max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int n_threads = 8;
    bool use_gpu = true;
};

// Abstract Language Model interface
class ILanguageModel {
public:
    virtual ~ILanguageModel() = default;

    virtual bool load(const LMConfig& config) = 0;
    virtual LMResponse generate(const std::string& prompt) = 0;
    virtual std::vector<float> embed(const std::string& text) = 0;
    virtual bool is_loaded() const = 0;
    virtual std::string model_info() const = 0;
};

// LM Adapter - bridges external LM with OpenRAI cognition
class LMAdapter {
public:
    explicit LMAdapter(const LMConfig& config);
    ~LMAdapter();

    // Initialize LM backend
    bool initialize();

    // Generate text from prompt
    LMResponse generate(const std::string& prompt);

    // Get embedding for cognitive processing
    std::vector<float> get_embedding(const std::string& text);

    // Enhanced generation with OpenRAI context
    LMResponse generate_with_context(
        const std::string& prompt,
        const std::vector<std::string>& memory_context,
        const std::vector<std::string>& graph_entities
    );

    // Check if model is ready
    bool is_ready() const;

    // Get model information
    std::string get_model_info() const;

private:
    LMConfig config_;
    std::unique_ptr<ILanguageModel> lm_backend_;
    bool initialized_ = false;

    std::string build_enhanced_prompt(
        const std::string& user_prompt,
        const std::vector<std::string>& context,
        const std::vector<std::string>& entities
    );
};

// Factory for creating LM backends
class LMFactory {
public:
    static std::unique_ptr<ILanguageModel> create(const std::string& model_type);
};

} // namespace openrai
