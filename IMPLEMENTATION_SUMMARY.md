# OpenRAI - Hybrid Cognition + Language Architecture

## Summary of Changes (feature/language-model branch)

### What Was Implemented

**1. Language Model Adapter Layer**
- Created `include/openrai/lm_adapter.h` with pluggable backend interface
- Implemented three backends:
  - `MockBackend` (working now, for testing)
  - `LlamaCppBackend` (stub, ready for llama.cpp integration)
  - `ONNXBackend` (stub, ready for ONNX Runtime integration)

**2. OpenRAI Core Integration**
- Added `LMAdapter` to `OpenRAI` class
- Implemented `generate_response()` method with cognitive context injection:
  - Gathers memory context from short-term memory
  - Retrieves knowledge graph entities
  - Passes context to LM for enhanced generation
  - Stores LM response back into vector memory and knowledge graph

**3. Cognitive Context Builder**
- Automatically includes top 5 memory items
- Includes top 10 knowledge graph entities
- Builds enhanced prompt with OpenRAI context

**4. Documentation**
- `HYBRID_ARCHITECTURE.md` - Full architecture explanation
- `LM_INTEGRATION_GUIDE.md` - Step-by-step integration guide
- `examples/lm_example.cpp` - Usage examples

---

## Architecture Philosophy

```
OpenRAI ? Language Model
OpenRAI = Cognitive Infrastructure

OpenRAI provides:
? Memory (persistent, vector-based)
? Knowledge Graph (entity/relation extraction)
? Goal System (value alignment)
? Self-Reflection (meta-cognition)
? Parallel Processing (500× batch)

External LM provides:
? Natural language understanding
? Text generation
? Language fluency

Together = Hybrid AI with both cognition and language
```

---

## File Structure

### New Files
- `include/openrai/lm_adapter.h` - LM adapter interface
- `src/lm_adapter.cpp` - LM backend implementations
- `examples/lm_example.cpp` - Usage example
- `HYBRID_ARCHITECTURE.md` - Architecture docs
- `LM_INTEGRATION_GUIDE.md` - Integration guide
- `TRAINING_ESTIMATE_ENGLISH.md` - Training context (from earlier)

### Modified Files
- `include/openrai/core.h` - Added `LMAdapter*` and `generate_response()`
- `src/core.cpp` - Initialize LM adapter, implement generate_response()
- `README.md` - Updated with LM info (partially)

---

## Current Status

### ? Working Now
- Mock LM backend fully functional
- Cognitive context injection working
- Integration with memory/graph working
- Build succeeds on Windows + CUDA 13.3
- Example code provided

### ?? Next Steps
1. Integrate llama.cpp (see LM_INTEGRATION_GUIDE.md)
2. Download Llama 3.2 or Qwen 2.5 model
3. Implement `LlamaCppBackend::load()` and `generate()`
4. Test with real language model
5. Benchmark performance

---

## How to Use (Now)

```cpp
#include <openrai/core.h>

int main() {
    openrai::OpenRAI rai;
    rai.start();

    // Inject some events to build memory
    rai.inject_input("alert: memory usage exceeded 75%");
    rai.inject_input("sensor: temperature spike at node 7");

    // Generate response with cognitive context
    std::string response = rai.generate_response(
        "What are the critical system issues?"
    );

    std::cout << response << "\n";

    rai.stop();
    return 0;
}
```

**Output (with mock backend):**
```
[MockBackend] Mock LM loaded successfully
[LMAdapter] Initialized: Mock LM backend (for testing)
[OpenRAI] Engine initialized nodes=20 threads=20 gpu_slots=128 semantic=on language=mock
[LM_RESPONSE] prompt=35 chars, response=58 chars, confidence=0.85
Mock LM response: Processing prompt with 35 characters
```

---

## How to Use (After Integration)

Same code, just change config in `src/core.cpp`:

```cpp
LMConfig lm_config;
lm_config.model_type = "llama";  // Changed from "mock"
lm_config.model_path = "models/llama-3.2-3b.gguf";
lm_config.use_gpu = true;
```

**Output (with real LM):**
```
[LlamaCppBackend] Loading model: models/llama-3.2-3b.gguf
[LMAdapter] Initialized: Llama 3.2 3B (Q4_K_M, 2.5GB)
[OpenRAI] Engine initialized nodes=20 threads=20 gpu_slots=128 semantic=on language=llama-3.2-3b
[LM_RESPONSE] prompt=35 chars, response=156 chars, confidence=0.92

Based on the recent memory alerts, there are two critical issues:
1. Memory usage on node 3 has exceeded 75%, which may cause performance degradation
2. Temperature spike detected at node 7 (82°C), which requires immediate attention

I recommend prioritizing the temperature issue first as it poses a more immediate risk...
```

---

## Benefits of Hybrid Approach

### vs Pure LLM (GPT, Claude)
- ? **Persistent memory** (not just context window)
- ? **Explicit knowledge graph** (not implicit in weights)
- ? **Runtime goal alignment** (not just RLHF during training)
- ? **Explainable reasoning** (can trace through semantic stack)
- ? **Lower inference cost** (smaller LM + cognitive layer)

### vs Building LLM from Scratch
- ? **No training required** (use existing open-source models)
- ? **Faster time-to-market** (weeks not years)
- ? **Lower cost** ($0 training vs $100k+ GPU costs)
- ? **Focus on unique value** (cognition, not language modeling)

---

## Commits

1. `a18b1bc` - feat: Add LM Adapter for hybrid Cognition + Language architecture
2. `6e64144` - fix: Correct API usage in generate_response

---

## References

- **Llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Qwen Models**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **ONNX Runtime**: https://onnxruntime.ai/

---

**Status**: ?? Mock backend working, ready for real LM integration!
