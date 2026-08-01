# Language Model Integration Guide

## Quick Start

OpenRAI now includes a **Language Model Adapter** that bridges cognitive processing with natural language generation.

```
OpenRAI Cognition Layer (Memory + Graph + Goals + Reflection)
        +
External LM (Qwen/Llama/Phi)
        =
Hybrid AI with both intelligence and language
```

---

## Current Status

? **Implemented:**
- LM Adapter interface (`include/openrai/lm_adapter.h`)
- Mock backend (for testing without dependencies)
- Integration with OpenRAI core
- Cognitive context injection (memory + knowledge graph)

? **Pending:**
- Llama.cpp integration
- ONNX Runtime integration
- Real model loading

---

## Using the Mock Backend

The mock backend is **already working** - you can test the architecture now:

```cpp
#include <openrai/core.h>

int main() {
    openrai::OpenRAI rai;
    rai.start();

    // Generate response with cognitive context
    std::string response = rai.generate_response(
        "Summarize the current system state"
    );

    std::cout << response << "\n";

    rai.stop();
    return 0;
}
```

---

## Integrating Real Language Models

### Option 1: Llama.cpp (Recommended)

**Step 1: Add llama.cpp as submodule**
```bash
cd C:\dev\openRAI
git submodule add https://github.com/ggerganov/llama.cpp external/llama.cpp
cd external/llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

**Step 2: Download a model**
```bash
# Llama 3.2 3B (recommended for testing)
wget https://huggingface.co/TheBloke/Llama-3.2-3B-GGUF/resolve/main/llama-3.2-3b.Q4_K_M.gguf -O models/llama-3.2-3b.gguf

# Or Llama 3.3 7B (more powerful)
wget https://huggingface.co/TheBloke/Llama-3.3-7B-GGUF/resolve/main/llama-3.3-7b.Q4_K_M.gguf -O models/llama-3.3-7b.gguf
```

**Step 3: Update CMakeLists.txt**
```cmake
# Add llama.cpp
add_subdirectory(external/llama.cpp)
target_link_libraries(openrai PRIVATE llama)
```

**Step 4: Implement LlamaCppBackend in `src/lm_adapter.cpp`**
```cpp
#include "llama.h"

class LlamaCppBackend : public ILanguageModel {
private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;

public:
    bool load(const LMConfig& config) override {
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config.use_gpu ? 99 : 0;

        model_ = llama_load_model_from_file(
            config.model_path.c_str(), 
            model_params
        );

        if (!model_) return false;

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = config.context_length;
        ctx_params.n_threads = config.n_threads;

        ctx_ = llama_new_context_with_model(model_, ctx_params);
        return ctx_ != nullptr;
    }

    LMResponse generate(const std::string& prompt) override {
        // Tokenize prompt
        std::vector<llama_token> tokens = 
            llama_tokenize(ctx_, prompt, true);

        // Evaluate prompt
        llama_decode(ctx_, llama_batch_get_one(
            tokens.data(), tokens.size()
        ));

        // Generate tokens
        std::string response;
        for (int i = 0; i < config_.max_tokens; ++i) {
            llama_token token = llama_sample_token(ctx_, /* sampling params */);

            if (token == llama_token_eos(model_)) break;

            response += llama_token_to_piece(ctx_, token);
        }

        LMResponse result;
        result.text = response;
        result.confidence = 0.9f;
        return result;
    }

    // ... rest of implementation
};
```

**Step 5: Change config in `src/core.cpp`**
```cpp
LMConfig lm_config;
lm_config.model_type = "llama";  // Changed from "mock"
lm_config.model_path = "models/llama-3.2-3b.gguf";
lm_config.context_length = 4096;
lm_config.max_tokens = 512;
lm_config.use_gpu = true;
```

---

### Option 2: ONNX Runtime (for Qwen)

**Step 1: Install ONNX Runtime**
```bash
vcpkg install onnxruntime-gpu
```

**Step 2: Download Qwen model**
```bash
# Qwen 2.5 7B
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-ONNX models/qwen-2.5-7b
```

**Step 3: Update CMakeLists.txt**
```cmake
find_package(onnxruntime REQUIRED)
target_link_libraries(openrai PRIVATE onnxruntime::onnxruntime)
```

**Step 4: Implement ONNXBackend**
```cpp
#include <onnxruntime_cxx_api.h>

class ONNXBackend : public ILanguageModel {
private:
    Ort::Env env_;
    Ort::Session session_;

public:
    bool load(const LMConfig& config) override {
        Ort::SessionOptions options;
        if (config.use_gpu) {
            OrtCUDAProviderOptions cuda_options;
            options.AppendExecutionProvider_CUDA(cuda_options);
        }

        session_ = Ort::Session(
            env_, 
            config.model_path.c_str(), 
            options
        );

        return true;
    }

    // ... rest of implementation
};
```

---

## Model Recommendations

| Model | Size | RAM | Speed | Use Case |
|-------|------|-----|-------|----------|
| **Llama 3.2 3B** | 2.5 GB | 4 GB | Fast | Testing, development |
| **Qwen 2.5 7B** | 5 GB | 8 GB | Medium | Production, quality |
| **Llama 3.3 7B** | 5 GB | 8 GB | Medium | Production, balanced |
| **Phi-3-Mini** | 2 GB | 3 GB | Very fast | Edge, low-resource |

---

## Performance Notes

### With Mock Backend (current)
- Init: instant
- Generate: <1ms
- Memory: minimal
- Purpose: **Testing architecture**

### With Real LM (after integration)
- Init: 2-5 seconds (model loading)
- Generate: 50-200ms (depends on model size)
- Memory: 3-8 GB
- Purpose: **Production use**

---

## Architecture Benefits

Even with a small language model, the hybrid approach gives you:

1. **Persistent Memory** (OpenRAI)
   - Conversations remembered across sessions
   - Vector DB with 5000 embeddings

2. **Knowledge Graph** (OpenRAI)
   - Explicit entity/relation tracking
   - Semantic reasoning over facts

3. **Goal Alignment** (OpenRAI)
   - Runtime value steering
   - No need for RLHF training

4. **Self-Reflection** (OpenRAI)
   - Meta-cognitive analysis
   - Explainable decisions

5. **Natural Language** (External LM)
   - Human-like text generation
   - Language understanding

---

## Example: RAG with OpenRAI

```cpp
// Retrieval-Augmented Generation with cognitive context
std::string query = "What was the temperature issue?";

// 1. OpenRAI retrieves relevant memory
auto context = rai.retrieve_context(query, 5);

// 2. Build prompt with context
std::string prompt = build_rag_prompt(query, context);

// 3. LM generates response grounded in memory
std::string response = rai.generate_response(prompt);

// 4. Store response in memory for future queries
// (automatic in generate_response)
```

---

## Next Steps

1. **Phase 1** (current): ? Mock backend working
2. **Phase 2**: Integrate llama.cpp
3. **Phase 3**: Integrate ONNX Runtime
4. **Phase 4**: Production optimization
5. **Phase 5**: Advanced RAG features

---

## Questions?

- See `HYBRID_ARCHITECTURE.md` for detailed architecture
- See `examples/lm_example.cpp` for usage examples
- See `TRAINING_ESTIMATE_ENGLISH.md` for LM training context

**Status**: ?? Ready for real LM integration!
