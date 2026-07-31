# OpenRAI Development Roadmap

## Current Status: Phase 1 ? Complete

**Proof of Concept** with minimal viable implementation:
- ? 10 parallel worker threads
- ? 3-layer memory system
- ? Reasoning module with importance scoring
- ? GPU job queue with semaphore control
- ? Main loop orchestration
- ? Console output/logging

---

## Phase 2: GPU & Persistence (3-4 weeks)

### 2.1 Vector Database Integration

**Goal**: Replace mock long-term memory with real semantic search.

**Tasks**:
- [ ] Evaluate FAISS vs hnswlib vs Annoy
  - FAISS: Production-ready, optimized for large-scale
  - hnswlib: Lightweight, good for PoC ? Production
  - Annoy: Simple, Spotify-backed

- [ ] Implement semantic embeddings
  - Use sentence-transformers or similar
  - Generate embeddings for reasoning thoughts

- [ ] Replace `retrieve_from_long_term(query)`
  - Instead of string matching ? vector similarity
  - Top-K by cosine distance

- [ ] Implement persistence
  - Save vector DB to disk (SQLite + FAISS index)
  - Load on startup
  - Incremental updates

**Files to create/modify**:
```
include/openrai/vector_db.h
src/vector_db.cpp
src/embeddings.cpp  (or external library)
```

**Example API**:
```cpp
class VectorDatabase {
public:
    void add_embedding(const std::string& id, 
                      const std::vector<float>& embedding);

    std::vector<std::pair<std::string, float>> 
    search_k(const std::vector<float>& query, size_t k);

    void save(const std::string& path);
    void load(const std::string& path);
};
```

### 2.2 CUDA Integration

**Goal**: Enable GPU inference and tensor operations.

**Tasks**:
- [ ] Add CUDA toolkit detection to CMake
  - `find_package(CUDA)`
  - Compiler flags for NVCC

- [ ] Create CUDA kernels
  - `kernels/inference.cu` - Model execution
  - `kernels/embedding.cu` - Vector generation

- [ ] GPU memory management
  - Device buffer allocation/deallocation
  - Host ? Device transfers
  - Stream management

- [ ] Replace mock GPU job processing
  - Actual kernel execution
  - Asynchronous handling
  - Result callbacks

**Files to create**:
```
cuda/
??? CMakeLists.txt
??? kernels/
?   ??? inference.cu
?   ??? embedding.cu
?   ??? matrix_ops.cu
??? gpu_runtime.cpp
```

**Integration**:
```cpp
// In GPUJobQueue
GPUJob job = {type: "inference", data: input};
gpu_queue->enqueue_job(job);

// In GPU worker thread
GPURuntime::execute(job);
// Runs CUDA kernel, returns results
```

### 2.3 Persistent Storage

**Goal**: Survive restarts without losing memory.

**Tasks**:
- [ ] SQLite integration
  - Store reasoning entries
  - Metadata (timestamps, importance)

- [ ] Vector index persistence
  - Save FAISS/hnswlib index
  - Load on startup

- [ ] Checkpoint system
  - Periodic snapshots (every N iterations)
  - Rollback capability

- [ ] Configuration files
  - JSON/YAML for engine settings
  - Thread count, memory sizes, etc.

**Files to create**:
```
include/openrai/storage.h
src/storage.cpp
config/
??? default.json
??? settings.schema.json
```

---

## Phase 3: Advanced Scheduling (2-3 weeks)

### 3.1 DAG Task Scheduler

**Goal**: Replace simple ThreadPool with intelligent task dependency graph.

**Motivation**:
- Some tasks depend on others
- Optimize resource allocation
- Better load balancing

**Architecture**:
```
TaskDAG
?? Nodes (tasks)
?? Edges (dependencies)
?? Topological sort
?? Work-stealing executor
```

**API**:
```cpp
class TaskDAG {
public:
    TaskNodeId add_task(Task task);
    void add_dependency(TaskNodeId from, TaskNodeId to);
    void execute();  // Execute respecting dependencies
};
```

### 3.2 Work-Stealing Scheduler

**Goal**: Distribute work more evenly across threads.

**Implementation**:
- Each worker thread has local queue
- Idle threads "steal" work from busy threads
- O(1) amortized per operation

**Benefit**: Better CPU utilization for variable-length tasks

### 3.3 Priority Queues

**Goal**: Support task prioritization.

**Design**:
- Per-module priority queues
- Reasoning tasks get higher priority than GPU prep
- Adaptive priority based on memory pressure

---

## Phase 4: Distributed Computing (4-6 weeks)

### 4.1 Inter-Process Communication

**Goal**: Enable OpenRAI instances to communicate.

**Stack**:
- gRPC for RPC
- Protocol Buffers for serialization
- Load balancing across instances

**Architecture**:
```
OpenRAI Node 1 ?? OpenRAI Node 2 ?? OpenRAI Node 3
     ?                  ?                ?
  Shared Vector Database (Optional)
```

### 4.2 Distributed Memory

**Goal**: Replicate long-term memory across nodes.

**Approach**:
- Consensus protocol (Raft or similar)
- Eventual consistency model
- Conflict resolution

### 4.3 Federation

**Goal**: Multiple OpenRAI instances collaborate on complex tasks.

**Concept**:
- Master node orchestrates
- Worker nodes execute reasoning in parallel
- Aggregate results from multiple perspectives

---

## Phase 5: Advanced Reasoning (Ongoing)

### 5.1 Semantic Understanding

**Goal**: Move beyond keyword matching.

**Tasks**:
- [ ] NLP preprocessing
  - Tokenization
  - POS tagging
  - Named entity recognition

- [ ] Semantic similarity
  - Word embeddings (GloVe, FastText)
  - Sentence embeddings (BERT, Sentence Transformers)
  - Document embeddings

- [ ] Knowledge graphs
  - Entity relationships
  - Reasoning chains
  - Link prediction

### 5.2 Learning from Feedback

**Goal**: Improve reasoning over time.

**Implementation**:
- Track output correctness/relevance
- Fine-tune importance scoring
- Adapt decision thresholds

### 5.3 Explainability

**Goal**: Understand why OpenRAI made decisions.

**Output**:
- Decision trace
- Contributing factors
- Confidence levels

---

## Milestone Timeline

| Phase | Timeline | Status | Key Deliverables |
|-------|----------|--------|------------------|
| 1: PoC | Week 1 | ? Complete | Core loop, threading, memory |
| 2a: Vector DB | Week 2-3 | ? Next | Semantic search |
| 2b: CUDA | Week 3-4 | ? Next | GPU acceleration |
| 2c: Storage | Week 4-5 | ? Next | Persistence |
| 3: Scheduling | Week 5-7 | ?? Planned | DAG, work-stealing |
| 4: Distributed | Week 8-12 | ?? Planned | Multi-node, consensus |
| 5: Advanced | Ongoing | ?? Planned | NLP, learning, explainability |

---

## Testing Strategy

### Unit Tests
```bash
# Using Google Test or Catch2
cd build
ctest
```

**Test areas**:
- ThreadPool queue behavior
- Memory layer transitions
- Reasoning scoring
- GPU job ordering

### Integration Tests
```bash
# Test full loop
./run_integration_tests.sh
```

**Scenarios**:
- Normal operation (15 iterations)
- High load (50 iterations)
- Shutdown gracefully
- Persistence load/save

### Benchmarks
```bash
./benchmark_suite.sh
```

**Metrics**:
- Throughput (outputs/sec)
- Latency (loop iteration time)
- Memory growth over time
- GPU utilization

---

## Code Quality

### Style Guide
- Google C++ Style Guide (with modifications)
- `clang-format` configuration included
- `clang-tidy` for static analysis

### Documentation
- README for quick start
- ARCHITECTURE.md for high-level design
- MODULES.md for API details
- Inline comments for complex logic

### Version Control
- Commit per feature
- PR review before merge
- Tag releases (v0.1.0, v0.2.0, etc.)

---

## Known Limitations (Phase 1)

1. **Mock Vector Database**: Uses string matching, not semantic search
2. **No GPU**: Jobs enqueued but not executed
3. **No Persistence**: Memory lost on shutdown
4. **Simple Reasoning**: Keyword-based heuristics
5. **Single Node**: No distributed capability
6. **No External Dependencies**: Pure C++17 stdlib (future: FAISS, CUDA, etc.)

---

## Success Criteria

### Phase 1 (Complete)
- ? Compiles without warnings
- ? Runs 5 iterations without crashes
- ? All 3 memory layers functional
- ? Reasoning module produces output

### Phase 2
- [ ] Vector DB queries < 10ms for 1000 entries
- [ ] GPU inference 10x faster than CPU
- [ ] Persistent storage loads within 1 second
- [ ] No data loss across restarts

### Phase 3
- [ ] DAG execution in topological order
- [ ] Work-stealing improves utilization by 20%+
- [ ] Priority queue respects ordering

### Phase 4
- [ ] 3-node cluster runs without split-brain
- [ ] Message latency < 100ms
- [ ] Consensus achieved in < 1 round

---

## Getting Involved

To contribute:

1. Pick a task from a phase
2. Create feature branch: `git checkout -b feature/vector-db`
3. Implement + test
4. Submit PR with description
5. Code review + merge

### Current Focus

**Recommended next task**: Vector Database Integration (Phase 2.1)
- High impact (enables semantic reasoning)
- Self-contained (minimal refactoring)
- Foundation for Phase 5 (learning)

---

## Questions?

See documentation files:
- `README.md` - Quick start
- `ARCHITECTURE.md` - Design overview
- `MODULES.md` - API reference
- `BUILD_GUIDE.md` - Compilation help
