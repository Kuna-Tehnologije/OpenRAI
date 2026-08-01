#include "openrai/lm_adapter.h"
#include <sstream>
#include <iostream>
#include <fstream>

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
// Simple HTTP Backend (uses system curl - works on Windows 10+, no dependencies!)
// =============================================================================
class SimpleHTTPBackend : public ILanguageModel {
public:
    bool load(const LMConfig& config) override {
        api_url_ = config.model_path;
        if (api_url_.empty()) {
            api_url_ = "http://localhost:8000";
        }

        // Detect API style based on URL/port
        if (api_url_.find(":1337") != std::string::npos || 
            api_url_.find("/v1") != std::string::npos) {
            api_style_ = "openai";  // OpenAI-compatible API
            endpoint_ = "/v1/completions";
            std::cout << "[SimpleHTTPBackend] Using OpenAI-compatible API\n";
        } else {
            api_style_ = "ollama";  // Ollama/Docker Model Runner API
            endpoint_ = "/api/generate";
            std::cout << "[SimpleHTTPBackend] Using Ollama-compatible API\n";
        }

        std::cout << "[SimpleHTTPBackend] Connecting to LM at " << api_url_ << "\n";
        std::cout << "[SimpleHTTPBackend] Using system curl (Windows 10+ built-in)\n";

        loaded_ = true; // Assume it works, will fail on first generate if not

        std::cout << "[SimpleHTTPBackend] Ready to connect!\n";
        std::cout << "[SimpleHTTPBackend] API: " << api_url_ << endpoint_ << "\n";

        return loaded_;
    }

    LMResponse generate(const std::string& prompt) override {
        LMResponse response;

        if (!loaded_) {
            response.text = "[Error] HTTP backend not connected";
            return response;
        }

        // Escape string for JSON (backslash first, then quotes, then newlines)
        std::string escaped_prompt;
        escaped_prompt.reserve(prompt.size() * 2);

        for (char c : prompt) {
            switch (c) {
                case '\\': escaped_prompt += "\\\\"; break;
                case '\"': escaped_prompt += "\\\""; break;
                case '\n': escaped_prompt += "\\n"; break;
                case '\r': escaped_prompt += "\\r"; break;
                case '\t': escaped_prompt += "\\t"; break;
                default: escaped_prompt += c; break;
            }
        }

        // Limit prompt length
        if (escaped_prompt.size() > 400) {
            escaped_prompt = escaped_prompt.substr(0, 400) + "...";
        }

        // Write request to temporary file (more reliable than shell escaping)
        std::string temp_file = "openrai_request.json";
        {
            std::ofstream file(temp_file);
            if (!file) {
                response.text = "[Error] Failed to create temp request file";
                return response;
            }

            if (api_style_ == "openai") {
                // OpenAI-compatible format
                file << "{"
                     << "\"model\":\"qwen2_5-0_5b-instruct-q2_k\","
                     << "\"prompt\":\"" << escaped_prompt << "\","
                     << "\"max_tokens\":512,"
                     << "\"temperature\":0.7,"
                     << "\"stream\":false"
                     << "}";
            } else {
                // Ollama/Docker Model Runner format
                file << "{"
                     << "\"model\":\"docker.io/ai/smollm2:latest\","
                     << "\"prompt\":\"" << escaped_prompt << "\","
                     << "\"stream\":false"
                     << "}";
            }
        }

        // Build curl command using temp file
        std::ostringstream cmd;
        cmd << "curl.exe -s -X POST " << api_url_ << endpoint_ << " "
            << "-H \"Content-Type: application/json\" "
            << "--data-binary \"@" << temp_file << "\" 2>&1";

        std::cout << "[SimpleHTTPBackend] Calling API...\n";

        // Execute curl
        FILE* pipe = _popen(cmd.str().c_str(), "r");
        if (!pipe) {
            response.text = "[Error] Failed to execute curl - is it installed?";
            return response;
        }

        char buffer[256];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
            result += buffer;
        }
        int exit_code = _pclose(pipe);

        if (exit_code != 0) {
            response.text = "[Error] curl failed with code " + std::to_string(exit_code);
            std::cout << "[SimpleHTTPBackend] Error: " << result << "\n";
            return response;
        }

        // Parse response based on API style
        if (api_style_ == "openai") {
            // OpenAI format: {"choices":[{"text":"..."}]}
            size_t text_start = result.find("\"text\":\"");
            if (text_start != std::string::npos) {
                text_start += 8; // Skip "text":"
                size_t text_end = result.find("\"", text_start);
                if (text_end != std::string::npos) {
                    response.text = result.substr(text_start, text_end - text_start);
                    response.confidence = 0.9f;
                    response.tokens_used = static_cast<int>(response.text.size() / 4);

                    std::cout << "[SimpleHTTPBackend] ? Generated " << response.text.size() 
                             << " chars\n";
                } else {
                    response.text = "[Error] Failed to parse OpenAI response (no closing quote)";
                }
            } else {
                response.text = "[API Response] " + result.substr(0, 200);
                std::cout << "[SimpleHTTPBackend] Warning: Unexpected OpenAI response format\n";
                std::cout << "Full response: " << result << "\n";
            }
        } else {
            // Ollama format: {"response":"..."}
            size_t text_start = result.find("\"response\":\"");
            if (text_start != std::string::npos) {
                text_start += 12; // Skip "response":"
                size_t text_end = result.find("\"", text_start);
                if (text_end != std::string::npos) {
                    response.text = result.substr(text_start, text_end - text_start);
                    response.confidence = 0.9f;
                    response.tokens_used = static_cast<int>(response.text.size() / 4);

                    std::cout << "[SimpleHTTPBackend] ? Generated " << response.text.size() 
                             << " chars\n";
                } else {
                    response.text = "[Error] Failed to parse Ollama response (no closing quote)";
                }
            } else {
                response.text = "[API Response] " + result.substr(0, 200);
                std::cout << "[SimpleHTTPBackend] Warning: Unexpected Ollama response format\n";
                std::cout << "Full response: " << result << "\n";
            }
        }

        return response;
    }

    std::vector<float> embed(const std::string& text) override {
        return std::vector<float>(768, 0.5f);
    }

    bool is_loaded() const override { return loaded_; }

    std::string model_info() const override {
        return "Simple HTTP Backend (curl) @ " + api_url_;
    }

private:
    std::string api_url_;
    std::string api_style_;  // "openai" or "ollama"
    std::string endpoint_;   // "/v1/completions" or "/api/generate"
    bool loaded_ = false;
};

// =============================================================================
// LM Factory
// =============================================================================
std::unique_ptr<ILanguageModel> LMFactory::create(const std::string& model_type) {
    if (model_type == "http" || model_type == "docker") {
        return std::make_unique<SimpleHTTPBackend>();
    } else if (model_type == "llama") {
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
