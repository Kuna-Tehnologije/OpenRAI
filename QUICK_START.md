# OpenRAI Quick Reference

## One-Minute Overview

**OpenRAI** = AI Engine (not chatbot)

**Architecture**: 
- C++ core
- 10 parallel worker threads
- 3-layer memory (current ? short-term ? long-term)
- Reasoning module
- GPU job scheduler

**Main Loop** (3 seconds per iteration):
1. 10 threads generate outputs
2. Filter to top 4 by relevance
3. Reasoning analyzes + stores thoughts
4. Enqueue GPU job
5. Sleep 3s ? repeat

---

## Quick Build & Run

### Linux/macOS
```bash
chmod +x build.sh
./build.sh
```

### Windows
```bash
cl.exe /std:c++17 /EHsc /I include src\*.cpp /Fe:openrai.exe
openrai.exe
```

### All Platforms (CMake)
```bash
mkdir build && cd build
cmake .. && cmake --build .
./openrai
```

---

## File Structure

```
openRAI/
??? include/openrai/        # Headers
?   ??? core.h             # Main engine
?   ??? thread_pool.h      # 10 workers
?   ??? memory.h           # 3 layers
?   ??? gpu_queue.h        # GPU scheduler
?   ??? reasoning.h        # Analysis
?
??? src/                    # Implementation
?   ??? main.cpp
?   ??? core.cpp
?   ??? thread_pool.cpp
?   ??? memory.cpp
?   ??? gpu_queue.cpp
?   ??? reasoning.cpp
?
??? CMakeLists.txt         # Build config
??? README.md              # This file
??? ARCHITECTURE.md        # Design
??? MODULES.md            # API docs
??? ROADMAP.md            # Next phases
??? BUILD_GUIDE.md        # Compilation
```

---

## Key Classes

### ThreadPool (10 threads)
```cpp
ThreadPool pool(10);
pool.enqueue([](){ /* work */ });
pool.wait_all();
```

### MemorySystem (3 layers)
```cpp
memory->add_current_output(output);
memory->filter_to_short_term();
auto st = memory->get_short_term_memory();
memory->add_to_long_term(entry);
```

### ReasoningModule
```cpp
reasoning->analyze(short_term);
auto result = reasoning->evaluate_thought("thought");
// result.should_output, result.importance_score
```

### GPUJobQueue
```cpp
gpu_queue->enqueue_job(job);
```

### OpenRAI (Main)
```cpp
auto engine = std::make_unique<OpenRAI>();
engine->start();    // Run in background
// ... wait ...
engine->stop();     // Graceful shutdown
```

---

## Main Loop Phases

| Phase | What | Input | Output |
|-------|------|-------|--------|
| 1 | 10 threads | Nodes | 10 outputs |
| 2 | Filter | 10 outputs | 4 top outputs |
| 3 | Reason | 4 outputs | Thoughts + decision |
| 4 | GPU | Decision | Enqueued job |

---

## Memory Layers

| Layer | Size | Type | Purpose |
|-------|------|------|---------|
| Current | 10 | Cache | I/O buffer |
| Short-term | 4 | Sorted | Reasoning input |
| Long-term | 1000 | Vector DB | Persistent |

---

## Thread Safety

| Component | Lock Type | Protected |
|-----------|-----------|-----------|
| ThreadPool | mutex + CV | task queue |
| Memory | shared_mutex | all 3 layers |
| GPUQueue | mutex + semaphore | job queue |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Threads | 10 (active) |
| Thread-safe operations | ?s-scale |
| Main loop iteration | ~3 sec |
| Memory overhead | <10 MB |

---

## Common Tasks

### Add New Task to ThreadPool
```cpp
thread_pool_->enqueue([this, param]() {
    // Do work
    memory_system_->add_current_output(result);
});
```

### Retrieve from Long-Term
```cpp
auto results = memory_->retrieve_from_long_term(
    "query_string", 5  // top-5
);
```

### Evaluate for Output
```cpp
auto result = reasoning_->evaluate_thought("idea");
if (result.should_output) {
    std::cout << "[OUTPUT] " << result.thought << std::endl;
}
```

### Store Reasoning
```cpp
MemoryEntry entry;
entry.content = thought;
entry.importance = score;
entry.created_at = std::chrono::system_clock::now();
memory_->add_to_long_term(entry);
```

---

## Debugging

### Enable Logging
All components log to `std::cout`. Look for:
- `[ThreadPool]` - Thread operations
- `[Memory]` - Memory transitions
- `[Reasoning]` - Analysis results
- `[OUTPUT]` - User-facing output
- `[PHASE N]` - Loop phases

### Check Memory Growth
```bash
# Linux
watch -n 1 'ps aux | grep openrai'

# macOS
top -pid $(pgrep openrai)
```

### Trace Execution
Add print statements to see detailed flow:
```cpp
std::cout << "[DEBUG] Current memory size: " 
          << memory_->get_current_outputs().size() << std::endl;
```

---

## Known Issues

| Issue | Workaround |
|-------|-----------|
| No GPU execution | Phase 2 CUDA integration |
| Mock vector DB | Phase 2 FAISS integration |
| No persistence | Phase 2 SQLite integration |
| Single node only | Phase 4 distributed support |

---

## Environment Variables

| Var | Effect |
|-----|--------|
| `OPENRAI_THREADS` | Override thread count (future) |
| `OPENRAI_MEMORY_SIZE` | Override memory limits (future) |
| `OPENRAI_LOG_LEVEL` | Verbosity (future) |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Compilation error |
| 2 | Runtime error |

---

## Tips & Tricks

### Fast Compilation
```bash
# Use -O1 for faster builds during development
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Parallel Build
```bash
cmake --build . -j$(nproc)  # Linux/macOS
cmake --build . -j%NUMBER_OF_PROCESSORS%  # Windows
```

### Static Analysis
```bash
clang-tidy src/core.cpp -- -I include
```

### Memory Check
```bash
valgrind --leak-check=full ./openrai
```

---

## Next Steps

1. ? Build & run Phase 1 PoC
2. ?? Read ARCHITECTURE.md for design overview
3. ?? Read MODULES.md for API details
4. ??? Pick task from ROADMAP.md
5. ?? Start Phase 2 development

---

## Resources

- **C++17 Concurrency**: https://cppreference.com/w/cpp/thread
- **Task Scheduling**: Data parallelism patterns
- **Vector Databases**: FAISS, hnswlib, Annoy docs
- **CUDA**: NVIDIA developer documentation

---

## Support

For issues:
1. Check existing GitHub issues
2. Search documentation
3. Create new issue with:
   - OS / Compiler version
   - Build steps
   - Error message
   - Expected vs actual behavior

---

**Current Status**: Phase 1 ? PoC Complete  
**Next Phase**: Phase 2 - GPU & Persistence  
**Estimated Completion**: 3-4 weeks

Happy building! ??
