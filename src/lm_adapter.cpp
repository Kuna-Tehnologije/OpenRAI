#include "openrai/lm_adapter.h"
#include <sstream>
#include <iostream>

namespace openrai {

// =============================================================================
// Llama.cpp Backend (stub - will integrate actual llama.cpp library)
// =============================================================================
class LlamaCppBackend : public ILanguageModel {
public:
    bool load(const LMConfig& config) override {
        std::cout << "[LlamaCppBackend] Loading model: " << config.model_path << "\n";
        // TODO: Integrate llama.cpp library
        // model_ = llama_load_model_from_file(config.model_path.c_str(), params);
        return false; // Stub: not yet implemented
    }

    LMResponse generate(const std::string& prompt) override {
        LMResponse response;
        response.text = "[LlamaCppBackend] Not yet implemented - integrate llama.cpp";
        response.confidence = 0.0f;
        return response;
    }

    std::vector<float> embed(const std::string& text) override {
        // TODO: Use llama.cpp embedding API
        return std::vector<float>(768, 0.0f); // Stub
    }

    bool is_loaded() const override { return false; }

    std::string model_info() const override {
        return "Llama.cpp backend (not yet integrated)";
    }
};

// =============================================================================
// ONNX Runtime Backend (stub)
// =============================================================================
class ONNXBackend : public ILanguageModel {
public:
    bool load(const LMConfig& config) override {
        std::cout << "[ONNXBackend] Loading model: " << config.model_path << "\n";
        // TODO: Integrate ONNX Runtime
        return false;
    }

    LMResponse generate(const std::string& prompt) override {
        LMResponse response;
        response.text = "[ONNXBackend] Not yet implemented - integrate ONNX Runtime";
        response.confidence = 0.0f;
        return response;
    }

    std::vector<float> embed(const std::string& text) override {
        return std::vector<float>(768, 0.0f);
    }

    bool is_loaded() const override { return false; }

    std::string model_info() const override {
        return "ONNX Runtime backend (not yet integrated)";
    }
};

// =============================================================================
// Mock Backend (for testing without external dependencies)
// =============================================================================
class MockBackend : public ILanguageModel {
public:
    bool load(const LMConfig& config) override {
        std::cout << "[MockBackend] Mock LM loaded successfully\n";
        loaded_ = true;
        return true;
    }

    LMResponse generate(const std::string& prompt) override {
        LMResponse response;
        response.text = "Mock LM response: Processing prompt with " 
                       + std::to_string(prompt.size()) + " characters";
        response.confidence = 0.85f;
        response.tokens_used = static_cast<int>(prompt.size() / 4);

        // Mock embedding
        response.embedding.resize(768);
        for (size_t i = 0; i < 768; ++i) {
            response.embedding[i] = static_cast<float>(i) / 768.0f;
        }

        return response;
    }

    std::vector<float> embed(const std::string& text) override {
        std::vector<float> embedding(768);
        float hash_value = 0.0f;
        for (char c : text) {
            hash_value += static_cast<float>(c);
        }
        hash_value /= 255.0f;

        for (size_t i = 0; i < 768; ++i) {
            embedding[i] = (hash_value + static_cast<float>(i) / 768.0f) / 2.0f;
        }
        return embedding;
    }

    bool is_loaded() const override { return loaded_; }

    std::string model_info() const override {
        return "Mock LM backend (for testing)";
    }

private:
    bool loaded_ = false;
};

// =============================================================================
// LM Factory
// =============================================================================
std::unique_ptr<ILanguageModel> LMFactory::create(const std::string& model_type) {
    if (model_type == "llama") {
        return std::make_unique<LlamaCppBackend>();
    } else if (model_type == "onnx" || model_type == "qwen") {
        return std::make_unique<ONNXBackend>();
    } else if (model_type == "mock") {
        return std::make_unique<MockBackend>();
    }

    std::cerr << "[LMFactory] Unknown model type: " << model_type 
              << ", falling back to mock\n";
    return std::make_unique<MockBackend>();
}

// =============================================================================
// LM Adapter Implementation
// =============================================================================
LMAdapter::LMAdapter(const LMConfig& config) 
    : config_(config) {
}

LMAdapter::~LMAdapter() = default;

bool LMAdapter::initialize() {
    if (initialized_) {
        return true;
    }

    std::cout << "[LMAdapter] Initializing with model type: " 
              << config_.model_type << "\n";

    lm_backend_ = LMFactory::create(config_.model_type);
    if (!lm_backend_) {
        std::cerr << "[LMAdapter] Failed to create LM backend\n";
        return false;
    }

    bool loaded = lm_backend_->load(config_);
    if (!loaded) {
        std::cerr << "[LMAdapter] Failed to load LM model\n";
        std::cout << "[LMAdapter] Note: Real LM integration (llama.cpp/ONNX) "
                  << "not yet implemented. Using mock backend.\n";
        std::cout << "[LMAdapter] To integrate:\n";
        std::cout << "  1. Add llama.cpp as git submodule\n";
        std::cout << "  2. Link llama.cpp in CMakeLists.txt\n";
        std::cout << "  3. Implement LlamaCppBackend::load/generate\n";

        // Fall back to mock for now
        config_.model_type = "mock";
        lm_backend_ = LMFactory::create("mock");
        loaded = lm_backend_->load(config_);
    }

    initialized_ = loaded;

    if (initialized_) {
        std::cout << "[LMAdapter] Initialized: " << lm_backend_->model_info() << "\n";
    }

    return initialized_;
}

LMResponse LMAdapter::generate(const std::string& prompt) {
    if (!initialized_ || !lm_backend_) {
        LMResponse error;
        error.text = "[Error] LM not initialized";
        return error;
    }

    return lm_backend_->generate(prompt);
}

std::vector<float> LMAdapter::get_embedding(const std::string& text) {
    if (!initialized_ || !lm_backend_) {
        return std::vector<float>(768, 0.0f);
    }

    return lm_backend_->embed(text);
}

std::string LMAdapter::build_enhanced_prompt(
    const std::string& user_prompt,
    const std::vector<std::string>& context,
    const std::vector<std::string>& entities) 
{
    std::ostringstream oss;

    // System context
    oss << "You are OpenRAI, a cognitive AI with memory, knowledge graph, and goal alignment.\n\n";

    // Memory context
    if (!context.empty()) {
        oss << "=== MEMORY CONTEXT ===\n";
        for (size_t i = 0; i < context.size() && i < 5; ++i) {
            oss << "- " << context[i] << "\n";
        }
        oss << "\n";
    }

    // Knowledge graph entities
    if (!entities.empty()) {
        oss << "=== RELEVANT ENTITIES ===\n";
        for (size_t i = 0; i < entities.size() && i < 10; ++i) {
            oss << "- " << entities[i] << "\n";
        }
        oss << "\n";
    }

    // User prompt
    oss << "=== USER INPUT ===\n";
    oss << user_prompt << "\n\n";
    oss << "=== RESPONSE ===\n";

    return oss.str();
}

LMResponse LMAdapter::generate_with_context(
    const std::string& prompt,
    const std::vector<std::string>& memory_context,
    const std::vector<std::string>& graph_entities)
{
    std::string enhanced_prompt = build_enhanced_prompt(
        prompt, memory_context, graph_entities
    );

    return generate(enhanced_prompt);
}

bool LMAdapter::is_ready() const {
    return initialized_ && lm_backend_ && lm_backend_->is_loaded();
}

std::string LMAdapter::get_model_info() const {
    if (lm_backend_) {
        return lm_backend_->model_info();
    }
    return "No model loaded";
}

} // namespace openrai
