# OpenRAI PoC - Architecture Document

## Overview

OpenRAI je **AI Engine** sa svjesnom paralelnom obradom i inteligentnom memorijom.

**Identity**: raw / real / reimagined AI  
**Type**: Autonomous AI Platform (not an assistant)  
**Core**: C++17 with GPU support (CUDA/ROCm)  
**Architecture**: Modular, concurrent, event-driven

---

## Core Architecture

### 1. Thread Pool (10 Workers)

**File**: `src/thread_pool.cpp`, `include/openrai/thread_pool.h`

```
ThreadPool (10 threads)
?? Task Queue (thread-safe)
?? Condition Variables
?? Active Task Counter
?? Worker Loop
    ?? Wait for task
    ?? Execute
    ?? Notify
```

**Key Features**:
- 10 always-active worker threads
- Lock-free task queue design (potential optimization)
- Concurrent task execution
- Active task tracking

**Usage Pattern**:
```cpp
thread_pool->enqueue([&]() {
    // Parallel work
});
thread_pool->wait_all();
```

---

### 2. Memory System (3 Layers)

**File**: `src/memory.cpp`, `include/openrai/memory.h`

```
Memory Hierarchy
?
?? Layer 1: Current Memory (CACHE)
?  ?? Capacity: 10 outputs
?  ?? Type: FIFO queue
?  ?? Purpose: Fast I/O buffer
?
?? Layer 2: Short-Term Memory (WORKING)
?  ?? Capacity: 4 outputs
?  ?? Type: Sorted by relevance
?  ?? Purpose: Input to Reasoning
?
?? Layer 3: Long-Term Memory (VECTOR DB)
   ?? Capacity: 1000 entries
   ?? Type: Semantic storage
   ?? Purpose: Persistent reasoning records
```

**Processing Flow**:
1. ThreadPool generates 10 outputs
2. All stored in Current Memory
3. Top 4 by relevance ? Short-Term
4. Reasoning analyzes Short-Term
5. ALL reasoning thoughts ? Long-Term

**Synchronization**:
- `std::shared_mutex` for thread-safe reads
- Lock guards for writes

---

### 3. Reasoning Module

**File**: `src/reasoning.cpp`, `include/openrai/reasoning.h`

```cpp
ReasoningModule
?? analyze(short_term_memory)
?  ?? Retrieve old context from long-term
?  ?? Evaluate each thought
?  ?? Store all in long-term
?
?? evaluate_thought()
   ?? Calculate importance (0.0-1.0)
   ?? Check output relevance
   ?? Return: ReasoningResult
      ?? thought: string
      ?? should_output: bool
      ?? importance_score: float
```

**Decision Logic**:
```
If important_score > threshold AND contains_keywords:
    ? Output to console
Else:
    ? Still store in long-term
```

---

### 4. GPU Job Queue

**File**: `src/gpu_queue.cpp`, `include/openrai/gpu_queue.h`

```
GPUJobQueue
?? Queue: FIFO task buffer
?? Semaphore: Concurrent slot limiter
?? Max Concurrent Jobs: 10 (configurable)

Job Types:
?? "inference" - Model predictions
?? "embedding" - Vector generation
?? "matrix_op" - Tensor operations
```

**Concurrency Control**:
- Binary semaphore for slot management
- Acquire on enqueue, release on dequeue
- Prevents GPU overload

---

## Main Loop (OPENRAI LOOP MODEL)

**File**: `src/core.cpp` - `OpenRAI::main_loop()`

```
????????????????????????????????????????????????????????
?           OPENRAI MAIN LOOP - ITERATION N            ?
????????????????????????????????????????????????????????

?? PHASE 1: Parallel Processing (T0) ?????????????????
?                                                     ?
?  ThreadPool.enqueue(10 tasks):                      ?
?  ?? Thread 0: Process Node 0 ? Output_0            ?
?  ?? Thread 1: Process Node 1 ? Output_1            ?
?  ?? ...                                            ?
?  ?? Thread 9: Process Node 9 ? Output_9            ?
?                                                     ?
?  Result: 10 outputs in Current Memory              ?
?          (each with relevance_score)                ?
???????????????????????????????????????????????????????
                          ?
?? PHASE 2: Memory Filtering (T1) ???????????????????
?                                                     ?
?  MemorySystem.filter_to_short_term():              ?
?  ?? Sort 10 outputs by relevance_score             ?
?  ?? Keep top 4                                     ?
?  ?? Short-Term Memory = {O_best, O_2, O_3, O_4}   ?
?                                                     ?
?  Parallel read-lock protected                       ?
???????????????????????????????????????????????????????
                          ?
?? PHASE 3: Reasoning Analysis (T2-T3) ??????????????
?                                                     ?
?  ReasoningModule.analyze(short_term):              ?
?                                                     ?
?  For each of 4 short-term entries:                 ?
?  ?? Retrieve context from long-term               ?
?  ?? Evaluate relevance to final output             ?
?  ?? Calculate importance_score                     ?
?  ?                                                 ?
?  ?? Decision Tree:                                 ?
?     ?? If important AND keywords:                  ?
?     ?  ?? [OUTPUT] ? console                       ?
?     ?? Regardless:                                 ?
?        ?? Store in long-term memory                ?
?                                                     ?
?  Thread-safe long-term memory writes               ?
???????????????????????????????????????????????????????
                          ?
?? PHASE 4: GPU Job Enqueue (T4) ????????????????????
?                                                     ?
?  GPUJobQueue.enqueue_job():                        ?
?  ?? Job type: inference                           ?
?  ?? Data: iteration-specific input                ?
?  ?? Priority: 1                                    ?
?  ?? Semaphore-acquire (wait for slot)             ?
?                                                     ?
?  Non-blocking if slots available                   ?
???????????????????????????????????????????????????????
                          ?
           ??????????????????????????????
           ?  Loop Sleep: 3 seconds     ?
           ??????????????????????????????
                          ?
              RETURN TO PHASE 1 (NEXT ITERATION)
```

---

## Data Structures

### Output
```cpp
struct Output {
    std::string data;                    // Node output
    float relevance_score = 0.0f;       // 0.0-1.0
    std::chrono::system_clock::time_point timestamp;
    int thread_id = 0;
};
```

### MemoryEntry
```cpp
struct MemoryEntry {
    std::string content;                // Thought/reasoning
    float importance = 0.0f;            // 0.0-1.0
    std::vector<float> embedding;       // Semantic vector (future)
    std::chrono::system_clock::time_point created_at;
};
```

### GPUJob
```cpp
struct GPUJob {
    std::string type;                   // Job type
    std::string data;                   // Input data
    int priority = 0;
};
```

---

## Concurrency Model

### Thread Safety

| Component | Mechanism | Type |
|-----------|-----------|------|
| ThreadPool | Mutex + CV | Task queue |
| Memory (Current) | shared_mutex | Read/write lock |
| Memory (Short-term) | shared_mutex | Read/write lock |
| Memory (Long-term) | shared_mutex | Semantic storage |
| GPUQueue | Mutex + Semaphore | Slot management |

### Lock Hierarchy
```
ThreadPool::enqueue()
    ?
MemorySystem::add_current_output()
    ? (after thread completion)
MemorySystem::filter_to_short_term()
    ?
ReasoningModule::analyze()
    ?
GPUJobQueue::enqueue_job()
```

---

## Execution Timeline (Example: 15 seconds, 5 iterations)

```
t=0s   [Engine Start]
       ?
t=1s   ?? Iteration 1 started
       ?  ?? Phase 1: 10 threads (0.5s)
       ?  ?? Phase 2: Filter (0.1s)
       ?  ?? Phase 3: Reasoning (0.4s)
       ?  ?? Phase 4: GPU enqueue (0.0s)
       ?
t=4s   ?? Iteration 2 started
       ?
t=7s   ?? Iteration 3 started
       ?
t=10s  ?? Iteration 4 started
       ?
t=13s  ?? Iteration 5 started
       ?
t=15s  ?? [Engine Stop]
```

---

## Integration Points (Future)

### Vector Database (Phase 2)
- Replace mock long-term memory with FAISS/hnswlib
- Semantic search on embeddings
- Persistent storage

### GPU Backend (Phase 2)
- CUDA kernel for inference
- Tensor operations
- Batch processing

### Task DAG Scheduler (Phase 3)
- Replace simple enqueue with DAG
- Dependency resolution
- Resource allocation

---

## Performance Characteristics

| Metric | Target | Implementation |
|--------|--------|-----------------|
| Thread Count | 10 | Fixed |
| Current Memory | 10 entries | FIFO queue |
| Short-Term Memory | 4 entries | Sorted vector |
| Long-Term Capacity | 1000 entries | Deque |
| GPU Concurrent Jobs | 10 | Semaphore |
| Iteration Time | 3s | Configurable |
| Lock Type | Fine-grained | shared_mutex |

---

## Build & Run

### Prerequisites
- C++17 compatible compiler (MSVC, GCC, Clang)
- Standard library with threading support
- pthread on Unix-like systems

### Compilation

**Windows (MSVC)**:
```bash
cl.exe /std:c++17 /EHsc /I include src\*.cpp /Fe:openrai.exe
```

**Linux/Mac (GCC/Clang)**:
```bash
g++ -std=c++17 -pthread -I include src/*.cpp -o openrai
./openrai
```

**CMake** (recommended):
```bash
mkdir build
cd build
cmake ..
cmake --build .
./openrai
```

### Expected Output
```
????????????????????????????????????????
?         OpenRAI - PoC Engine         ?
?   raw / real / reimagined AI         ?
????????????????????????????????????????

[ThreadPool] Started with 10 worker threads
[Memory] Initialized
[GPUQueue] Initialized with 10 concurrent slots
[Reasoning] Module initialized
[OpenRAI] Engine initialized
[OpenRAI] Starting main loop...

=== OPENRAI MAIN LOOP STARTED ===
[PHASE 1] Enqueuing 10 parallel processing tasks...
  [Thread 0] Generated output with relevance: 0.3
  [Thread 1] Generated output with relevance: 0.55
  ...
[PHASE 2] Filtering 10 current outputs ? 4 short-term entries...
[PHASE 3] Reasoning module analyzing short-term memory...
[OUTPUT] <reasoning conclusions>
...
```

---

## Next Phases (Roadmap)

### Phase 2: GPU & Persistence
- [ ] CUDA/ROCm GPU kernel integration
- [ ] FAISS or hnswlib for semantic search
- [ ] SQLite/RocksDB for persistent storage
- [ ] Protocol Buffers for serialization

### Phase 3: Advanced Scheduling
- [ ] Directed Acyclic Graph (DAG) task scheduler
- [ ] Work-stealing algorithm
- [ ] Priority queues per component
- [ ] Load balancing

### Phase 4: Distributed
- [ ] gRPC communication
- [ ] Multi-machine coordination
- [ ] Distributed memory consensus
- [ ] Federated learning support

---

## Code Organization

```
openRAI/
??? include/openrai/
?   ??? core.h              # Main engine
?   ??? thread_pool.h       # Worker threads
?   ??? memory.h            # 3-layer memory
?   ??? gpu_queue.h         # GPU job scheduler
?   ??? reasoning.h         # Reasoning module
??? src/
?   ??? main.cpp            # Entry point
?   ??? core.cpp            # Main loop
?   ??? thread_pool.cpp     # ThreadPool impl
?   ??? memory.cpp          # Memory impl
?   ??? gpu_queue.cpp       # GPU queue impl
?   ??? reasoning.cpp       # Reasoning impl
??? CMakeLists.txt          # Build config
??? Makefile                # Alternative build
??? build.bat               # Windows build
??? README.md               # Quick start
```

---

## Notes

- **Not a chatbot**: OpenRAI is an engine, not a conversational AI
- **Raw & Real**: Minimalist architecture, no corporate fluff
- **Reimagined**: Modern C++ concurrent design with proven patterns
- **Proof of Concept**: Phase 1 is minimal viable implementation
- **Production Ready Foundation**: Can evolve to distributed system
