# OpenRAI FAQ

## General

### Q: What is OpenRAI?
**A:** OpenRAI is a standalone AI platform (not a chatbot) with a modern C++ architecture designed for:
- Parallel processing (10 worker threads)
- Intelligent memory management (3 layers)
- Semantic reasoning
- GPU acceleration (future)

### Q: Why C++?
**A:** 
- Performance critical for real-time processing
- Memory control for resource-constrained environments
- Concurrency primitives built-in
- Foundation for GPU integration

### Q: Is this production-ready?
**A:** No, this is **Phase 1 PoC** (Proof of Concept). See ROADMAP.md for production timeline.

### Q: Can I use this in my project?
**A:** Yes! MIT License. See LICENSE.txt for details.

---

## Architecture

### Q: Why 10 threads specifically?
**A:** 
- Empirically good for CPU-bound tasks
- Configurable in code (see MODULES.md)
- Allows 10 parallel outputs per iteration
- Scales well on modern 8-16 core CPUs

### Q: What does each memory layer do?

| Layer | Size | Purpose | Example |
|-------|------|---------|---------|
| Current | 10 | Fast I/O buffer | Raw thread outputs |
| Short-term | 4 | Reasoning input | Top 4 by relevance |
| Long-term | 1000+ | Persistent storage | All reasoning thoughts |

### Q: How is memory thread-safe?
**A:** Using `std::shared_mutex`:
- Multiple readers (fast retrieval)
- Exclusive writers (atomic updates)
- See src/memory.cpp for implementation

### Q: What is the reasoning module?
**A:** Component that:
1. Receives top 4 outputs
2. Evaluates importance (0.0-1.0)
3. Checks if output-worthy
4. Stores ALL thoughts in long-term
5. Returns decision

---

## Building & Running

### Q: Which compiler do I need?
**A:** Any C++17 compatible:
- GCC 7+
- Clang 5+
- MSVC 19.0+
- MinGW

### Q: Do I need external libraries?
**A:** For Phase 1 PoC: NO (pure C++17 stdlib)

For Phase 2+:
- FAISS or hnswlib (vector DB)
- CUDA toolkit (GPU)
- SQLite (persistence)

### Q: How do I build on Windows?
**A:** See BUILD_GUIDE.md. Quick version:

```bash
cl.exe /std:c++17 /EHsc /I include src\*.cpp /Fe:openrai.exe
```

### Q: Build errors?
**A:** Check:
1. Compiler version: `g++ --version`
2. C++17 support: Add `-std=c++17` flag
3. Include paths: `-I include`
4. All source files: `src/*.cpp`

See BUILD_GUIDE.md for detailed troubleshooting.

### Q: How long does it take to compile?
**A:** 
- Debug: ~2-5 seconds
- Release: ~5-10 seconds
- Clean rebuild: ~10-20 seconds

Depends on CPU/disk speed.

---

## Running

### Q: What does the output mean?

```
[ThreadPool] Started with 10 worker threads
```
? All 10 worker threads initialized and waiting for tasks.

```
[PHASE 1] Enqueuing 10 parallel processing tasks...
  [Thread 0] Generated output with relevance: 0.3
```
? Thread 0 processed a node, created output with score 0.3.

```
[OUTPUT] <some thought>
```
? Reasoning module determined this thought is important enough to output.

### Q: Why 3 seconds between iterations?
**A:** Configurable. See main_loop() in src/core.cpp:
```cpp
std::this_thread::sleep_for(std::chrono::seconds(3));
```

Change to any duration.

### Q: How many iterations run?
**A:** Depends on runtime. In main.cpp: runs 15 seconds = ~5 iterations.

### Q: Can I give it input?
**A:** Phase 1: No. process_input() is stub for future.

Phase 2: Will accept stdin/files/network.

---

## Memory

### Q: How much RAM does OpenRAI use?
**A:** ~10-50 MB depending on:
- Memory layer sizes (1000 long-term entries)
- Thread count (10)
- Output data sizes

### Q: What if memory runs out?
**A:** 
- Current memory: FIFO eviction (keep 10 newest)
- Short-term: Always 4 entries
- Long-term: Circular buffer (keep 1000 newest)

No crash, graceful overflow.

### Q: Can I adjust memory sizes?
**A:** Yes! In include/openrai/memory.h:
```cpp
static constexpr size_t MAX_CURRENT = 10;
static constexpr size_t MAX_SHORT_TERM = 4;
static constexpr size_t MAX_LONG_TERM = 1000;
```

Then rebuild.

### Q: How is the long-term memory implemented?
**A:** Phase 1: std::deque with string matching (mock)

Phase 2: Real vector database (FAISS/hnswlib) with semantic search.

---

## Reasoning

### Q: How does importance scoring work?

```cpp
float score = 0.0f;

if (thought.length() > 50)
    score += 0.3f;  // Length bonus

if (thought.find("important") != npos)
    score += 0.5f;  // Keyword bonus

if (thought.find("error") != npos)
    score += 0.3f;  // Error detection

return min(1.0f, score);  // 0.0 to 1.0
```

See src/reasoning.cpp for full logic.

### Q: What makes a thought "output-worthy"?
**A:** If it contains:
- "important"
- "result"
- "conclusion"

Configurable in `is_relevant_for_output()`.

### Q: Does OpenRAI learn?
**A:** Phase 1: No.

Phase 5: Yes, through adaptive scoring and reinforcement.

---

## Concurrency

### Q: How does the ThreadPool work?
**A:** 
1. Main creates 10 worker threads
2. Each waits on condition_variable
3. Main enqueues task
4. One worker wakes up, executes
5. Other workers compete for next task

Lock-free for most operations.

### Q: Is it thread-safe?
**A:** Yes, using:
- Mutexes (task queue)
- Shared mutexes (memory layers)
- Semaphores (GPU queue)
- Atomics (counters)

See ARCHITECTURE.md for thread safety matrix.

### Q: What if I run multiple OpenRAI instances?
**A:** Phase 1: They run independently (no coordination).

Phase 4: Distributed mode with consensus protocol.

---

## GPU

### Q: Does it use my GPU?
**A:** Phase 1: No (jobs enqueued but not executed).

Phase 2: Yes, with CUDA/ROCm.

### Q: How will GPU be integrated?
**A:** 
1. CUDA kernels for inference
2. GPU memory management
3. Async job execution
4. Result streaming back to CPU

See ROADMAP.md Phase 2.1.

### Q: Which GPUs are supported?
**A:** Phase 2 will support:
- NVIDIA (CUDA)
- AMD (ROCm)
- Intel (Level Zero - future)

---

## Persistence

### Q: Does it save state on shutdown?
**A:** Phase 1: No (all data lost).

Phase 2: Yes, to SQLite + vector index.

### Q: Where will data be stored?
**A:** Default: ./openrai_data/ directory

Configurable via config file (Phase 2).

### Q: Can I query the saved data?
**A:** Phase 1: N/A

Phase 2: CLI + Python API.

---

## Advanced

### Q: What's work-stealing scheduling?
**A:** Phase 3 feature:
- Each thread has local queue
- Idle threads "steal" from busy threads
- Better load balancing
- O(1) amortized per steal

### Q: How will distributed mode work?
**A:** Phase 4:
- Multiple OpenRAI nodes
- gRPC communication
- Raft consensus (distributed memory)
- Master-worker topology

### Q: Can I extend OpenRAI?
**A:** Yes!

**Easy**: Modify reasoning heuristics, memory sizes
**Medium**: Add new memory layers, change loop phases
**Hard**: Refactor threading model, add custom schedulers

See ARCHITECTURE.md for extension points.

---

## Troubleshooting

### Q: Compilation fails with "unknown preprocessor directive"
**A:** Your compiler doesn't support C++17. Update:
- GCC: 11+
- Clang: 14+
- MSVC: 2019+

### Q: Program crashes on startup
**A:** Likely: Thread creation failure. Check:
- Available file descriptors: `ulimit -n`
- System resources: `free -h` (Linux)
- Run with debugger: `gdb ./openrai`

### Q: High CPU usage
**A:** Normal for Phase 1 (spinning loop). Phase 2 will add adaptive throttling.

To reduce: Increase sleep duration in main_loop().

### Q: Memory keeps growing
**A:** Check long-term memory size. Limit in memory.h:
```cpp
static constexpr size_t MAX_LONG_TERM = 1000;  // Was 10000?
```

### Q: Why no output?
**A:** Check:
1. Console output enabled (not redirected)
2. Reasoning threshold too high
3. Keywords missing from thoughts

Add debug output in reasoning.cpp.

---

## Performance

### Q: How many thoughts per second?
**A:** Phase 1: ~3-10 depending on CPU
(10 thoughts per iteration, 3 sec per iteration = 3-10/sec)

Phase 2: 100+ with GPU.

### Q: What's the latency?
**A:** 
- Main loop iteration: ~3 seconds (configurable)
- Thread pool enqueue: < 1 microsecond
- Memory access: < 10 microseconds

### Q: How to profile?
**A:** 
```bash
# Linux
perf record ./openrai
perf report

# macOS
Instruments app (Xcode)

# Windows
Visual Studio Profiler
```

---

## Contributing

### Q: How do I contribute?
**A:** 
1. Fork repository
2. Pick task from ROADMAP.md
3. Create feature branch
4. Implement + test
5. Submit PR

### Q: What needs most help?
**A:** 
- Phase 2a: Vector DB integration
- Phase 2b: CUDA backend
- Phase 3: DAG scheduler
- Testing + documentation

---

## Legal

### Q: What license is OpenRAI under?
**A:** MIT License. See LICENSE.txt.

### Q: Can I use it commercially?
**A:** Yes, freely.

### Q: Must I give credit?
**A:** No requirement, but appreciated.

---

## Miscellaneous

### Q: Where is OpenRAI going?
**A:** See ROADMAP.md:
- Phase 2: GPU + Persistence
- Phase 3: Advanced Scheduling
- Phase 4: Distributed Computing
- Phase 5: Advanced Reasoning

### Q: How can I stay updated?
**A:** 
- Watch GitHub repository
- Join discussions
- Check releases page

### Q: Who maintains this?
**A:** Kuna Tehnologije j.d.o.o. (Neven Joviæ)

### Q: Any plans for Python bindings?
**A:** Phase 3+, Python extension module.

### Q: Can OpenRAI read/write files?
**A:** Phase 1: No.

Phase 2: Yes, input from files/sockets.

### Q: Will it run on embedded systems?
**A:** Possibly with:
- Smaller thread count (2-4)
- Reduced memory (100 long-term instead of 1000)
- No GPU

Compile and test on target.

---

## Still Have Questions?

1. **Search**: Check ARCHITECTURE.md, MODULES.md, ROADMAP.md
2. **Google**: Generic C++17 / concurrency questions
3. **GitHub Issues**: For bugs and feature requests
4. **Discussions**: For general questions

Happy coding! ??
