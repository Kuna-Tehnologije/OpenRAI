# OpenRAI Modules Documentation

## Overview

OpenRAI consists of 5 core modules:

1. **ThreadPool** - Parallel task execution
2. **MemorySystem** - 3-layer memory hierarchy
3. **GPUJobQueue** - GPU task scheduling
4. **ReasoningModule** - Thought analysis and evaluation
5. **OpenRAI (Core)** - Main engine and loop orchestration

---

## 1. ThreadPool Module

**Location**: `include/openrai/thread_pool.h`, `src/thread_pool.cpp`

### Purpose
Manages 10 continuously active worker threads for parallel task execution.

### Public API

```cpp
class ThreadPool {
public:
    // Initialize with N worker threads (default: 10)
    explicit ThreadPool(size_t num_threads = 10);

    ~ThreadPool();

    // Enqueue a task for execution
    void enqueue(Task task);

    // Block until all queued tasks complete
    void wait_all();

    // Get current active task count
    size_t active_tasks() const;
};
```

### Usage Example

```cpp
auto pool = std::make_unique<ThreadPool>(10);

// Enqueue 10 parallel tasks
for (int i = 0; i < 10; ++i) {
    pool->enqueue([i]() {
        std::cout << "Task " << i << " executing\n";
    });
}

// Wait for all to complete
pool->wait_all();
```

### Internal Implementation

```
ThreadPool::ThreadPool(10)
?? Creates 10 std::thread objects
?? Each thread runs worker() loop
?? All threads wait on condition_variable

ThreadPool::enqueue(task)
?? Lock queue_mutex_
?? Push task to queue_
?? Unlock and notify_one()
?? Worker thread wakes up

ThreadPool::worker() (runs in each thread)
?? Loop:
?  ?? Wait on condition_variable
?  ?? Check if shutdown or task available
?  ?? Dequeue task
?  ?? Increment active_tasks_
?  ?? Execute task (unlocked)
?  ?? Decrement active_tasks_
?  ?? Notify (for wait_all)
```

### Thread Safety

- **Queue**: `std::mutex` + `std::condition_variable`
- **Active tasks counter**: `std::atomic<size_t>`
- **Shutdown flag**: `std::atomic<bool>`

### Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| enqueue() | O(1) amortized | Lock time ? microseconds |
| wait_all() | Blocking | Spins on condition variable |
| active_tasks() | O(1) | Atomic read |

---

## 2. MemorySystem Module

**Location**: `include/openrai/memory.h`, `src/memory.cpp`

### Purpose
Implements 3-layer memory hierarchy for data flow from parallel processing ? reasoning ? persistent storage.

### Layer 1: Current Memory

```cpp
void add_current_output(const Output& output);
std::vector<Output> get_current_outputs() const;
void clear_current();
```

- **Capacity**: 10 entries (FIFO)
- **Type**: `std::deque<Output>`
- **Use Case**: Fast input buffer from ThreadPool
- **Access**: Read/write protected by `shared_mutex`

### Layer 2: Short-Term Memory

```cpp
std::vector<Output> get_short_term_memory();
void filter_to_short_term();
```

- **Capacity**: 4 entries (sorted by relevance)
- **Type**: `std::deque<Output>`
- **Use Case**: Input to reasoning module
- **Access**: Filtered from current memory
- **Sorting**: Descending relevance_score

### Layer 3: Long-Term Memory (Vector Database Mock)

```cpp
void add_to_long_term(const MemoryEntry& entry);
std::vector<MemoryEntry> retrieve_from_long_term(
    const std::string& query, 
    size_t top_k = 5
);
```

- **Capacity**: 1000 entries (configurable)
- **Type**: `std::deque<MemoryEntry>`
- **Use Case**: Persistent reasoning records
- **Retrieval**: Simple string matching (future: semantic search)

### Data Structures

```cpp
struct Output {
    std::string data;
    float relevance_score = 0.0f;  // 0.0 to 1.0
    std::chrono::system_clock::time_point timestamp;
    int thread_id = 0;
};

struct MemoryEntry {
    std::string content;
    float importance = 0.0f;
    std::vector<float> embedding;  // Future use
    std::chrono::system_clock::time_point created_at;
};
```

### Thread Safety

All operations protected by:
```cpp
mutable std::shared_mutex mutex_;

// Shared lock for reads
std::shared_lock<std::shared_mutex> lock(mutex_);

// Exclusive lock for writes
std::lock_guard<std::shared_mutex> lock(mutex_);
```

### Usage Flow

```
add_current_output(output_0)
    ? (repeat 10 times)
add_current_output(output_9)
    ?
filter_to_short_term()  // Sort by relevance, keep top 4
    ?
get_short_term_memory()  // Return sorted 4 entries
    ? (for each entry)
reasoning_module.analyze()
    ?
add_to_long_term(entry)  // Store reasoning result
    ?
retrieve_from_long_term(query)  // Retrieve context for next iteration
```

---

## 3. GPU Job Queue Module

**Location**: `include/openrai/gpu_queue.h`, `src/gpu_queue.cpp`

### Purpose
Manages GPU workload distribution with semaphore-based concurrency control.

### Public API

```cpp
class GPUJobQueue {
public:
    explicit GPUJobQueue(size_t max_concurrent_jobs = 10);

    void enqueue_job(const GPUJob& job);
    bool try_dequeue_job(GPUJob& job);
    size_t queue_size() const;
};
```

### Supported Job Types

```cpp
struct GPUJob {
    std::string type;      // "inference", "embedding", "matrix_op"
    std::string data;      // Job-specific input
    int priority = 0;      // Priority level
};
```

### Implementation Details

```
GPUJobQueue::enqueue_job(job)
?? Acquire semaphore (max_concurrent_jobs slots)
?  ?? If no slots available: BLOCK
?? Lock mutex
?? Push job to queue
?? Unlock

GPUJobQueue::try_dequeue_job(job)
?? Lock mutex
?? If queue empty: return false
?? Pop job from queue
?? Release semaphore (frees 1 slot)
?? Return true
```

### Thread Safety

- **Queue**: `std::mutex`
- **Slot management**: `std::binary_semaphore`
- **Max concurrent**: Configured at construction

### Usage Pattern

```cpp
auto gpu_queue = std::make_unique<GPUJobQueue>(10);

// Enqueue (may block if > 10 concurrent)
GPUJob job;
job.type = "inference";
job.data = "model_input";
gpu_queue->enqueue_job(job);

// Dequeue (in GPU worker thread)
GPUJob job_out;
if (gpu_queue->try_dequeue_job(job_out)) {
    // Process job_out on GPU
}
```

### Performance

- **Enqueue latency**: < 1ms (usually)
- **Dequeue latency**: < 1ms (usually)
- **Concurrent jobs**: Bounded by semaphore (default 10)

---

## 4. Reasoning Module

**Location**: `include/openrai/reasoning.h`, `src/reasoning.cpp`

### Purpose
Analyzes short-term memory entries, evaluates relevance, and decides whether to output or store in long-term memory.

### Public API

```cpp
class ReasoningModule {
public:
    ReasoningModule(MemorySystem* memory_system);

    void analyze(const std::vector<Output>& short_term_memory);

    struct ReasoningResult {
        std::string thought;
        bool should_output = false;
        float importance_score = 0.0f;
    };

    ReasoningResult evaluate_thought(const std::string& thought);
};
```

### Analysis Flow

```
ReasoningModule::analyze(short_term_memory: 4 entries)
?? Retrieve long-term context (top 3)
?
?? For each of 4 short-term entries:
?  ?? Call evaluate_thought(output.data)
?  ?? Get ReasoningResult with:
?  ?  ?? importance_score (0.0-1.0)
?  ?  ?? should_output (bool)
?  ?  ?? final_thought (string)
?  ?
?  ?? If should_output:
?  ?  ?? Print [OUTPUT] to console
?  ?
?  ?? Regardless:
?     ?? Add MemoryEntry to long-term
?        ?? content = thought
?        ?? importance = importance_score
?        ?? created_at = now()
```

### Evaluation Logic

#### calculate_importance(thought)

```cpp
float score = 0.0f;

if (thought.length() > 50)
    score += 0.3f;  // Length bonus

if (thought.find("important") != npos)
    score += 0.5f;  // Keyword bonus

if (thought.find("error") != npos)
    score += 0.3f;  // Error detection

return min(1.0f, score);  // Clamp to [0.0, 1.0]
```

#### is_relevant_for_output(thought)

```cpp
return thought.find("important") != npos ||
       thought.find("result") != npos ||
       thought.find("conclusion") != npos;
```

### Decision Tree

```
ReasoningResult = evaluate_thought(thought)
?
?? importance_score = calculate_importance(thought)
?
?? should_output = is_relevant_for_output(thought)
?
?? If should_output == true:
?  ?? [OUTPUT] Console: thought
?
?? Create MemoryEntry(thought, importance_score)
   ?? Store in long-term memory
```

### Usage

```cpp
auto reasoning = std::make_unique<ReasoningModule>(
    memory_system.get()
);

// Analyze short-term memory
auto short_term = memory_system->get_short_term_memory();
reasoning->analyze(short_term);

// Or evaluate single thought
auto result = reasoning->evaluate_thought(
    "important: system processed 1000 requests"
);
// result.should_output = true
// result.importance_score ? 0.8
```

---

## 5. OpenRAI Core Engine

**Location**: `include/openrai/core.h`, `src/core.cpp`

### Purpose
Main orchestrator that runs the OPENRAI main loop with all components.

### Public API

```cpp
class OpenRAI {
public:
    OpenRAI();
    ~OpenRAI();

    void start();                              // Start main loop
    void stop();                               // Stop main loop
    void process_input(const std::string& input);  // Future use
};
```

### Constructor

```cpp
OpenRAI::OpenRAI()
    : thread_pool_(make_unique<ThreadPool>(10)),
      memory_system_(make_unique<MemorySystem>()),
      gpu_queue_(make_unique<GPUJobQueue>(10)),
      reasoning_module_(make_unique<ReasoningModule>(
          memory_system_.get())) { }
```

- Initializes all 5 components
- ThreadPool with 10 workers
- Memory with 3 layers
- GPU queue with 10 concurrent slots
- Reasoning module connected to memory

### Main Loop (main_loop())

```
While is_running:
?
?? PHASE 1: Clear current memory
?           Enqueue 10 tasks to ThreadPool
?           Each task generates 1 Output
?           Wait for all tasks
?
?? PHASE 2: Call filter_to_short_term()
?           Sort current (10) by relevance
?           Keep top 4 in short-term
?
?? PHASE 3: Call reasoning->analyze(short_term)
?           For each of 4 short-term entries:
?           ?? Evaluate relevance
?           ?? Possibly output to console
?           ?? Always store in long-term
?
?? PHASE 4: Enqueue 1 GPU job
?           Type: "inference"
?           Data: iteration-specific
?
?? PHASE 5: Sleep 3 seconds
            Return to PHASE 1
```

### Component Relationships

```
main()
  ?
  ?? OpenRAI::start()
  ?   ?? Detach thread: main_loop()
  ?
  ?? Sleep 15 seconds
  ?
  ?? OpenRAI::stop()
      ?? Set is_running_ = false

main_loop() (runs in detached thread)
  ?
  ?? ThreadPool::enqueue() × 10
  ?   ?? Each generates Output
  ?
  ?? ThreadPool::wait_all()
  ?   ?? Wait for completion
  ?
  ?? MemorySystem::filter_to_short_term()
  ?   ?? Sort & filter
  ?
  ?? ReasoningModule::analyze(short_term)
  ?   ?
  ?   ?? MemorySystem::retrieve_from_long_term()
  ?   ?
  ?   ?? MemorySystem::add_to_long_term()
  ?       (for each evaluation)
  ?
  ?? GPUJobQueue::enqueue_job()
  ?
  ?? sleep 3s
```

### Synchronization

```
is_running_: std::atomic<bool>
?? Set by start() ? true
?? Set by stop()  ? false
?? Checked in main_loop() condition
```

---

## Integration Example

```cpp
// main.cpp
int main() {
    auto engine = std::make_unique<openrai::OpenRAI>();
    engine->start();  // Start main loop in background

    std::this_thread::sleep_for(
        std::chrono::seconds(15)  // Run for 15 seconds
    );

    engine->stop();  // Signal loop to stop

    // Destructor joins all threads/cleans up
    return 0;
}
```

### Execution Sequence

```
t=0s:  OpenRAI constructor
       ?? ThreadPool created (10 threads idle)
       ?? MemorySystem created (3 layers empty)
       ?? GPUJobQueue created (10 slots free)
       ?? ReasoningModule created

t=0s:  engine->start()
       ?? Detach main_loop thread

t=0s:  main thread sleeps 15 seconds

t=1s:  Iteration 1
       ?? 10 threads generate outputs
       ?? Filter to top 4
       ?? Reason & store
       ?? Enqueue GPU job
       ?? Sleep 3s

t=4s:  Iteration 2 (same)

t=7s:  Iteration 3 (same)

t=10s: Iteration 4 (same)

t=13s: Iteration 5 (same)

t=15s: engine->stop()
       ?? Set is_running_ = false

t=15s+: main_loop detects shutdown
        ?? Exits gracefully

t=15s+: Destructor cleanup
        ?? ThreadPool joins all threads
        ?? Memory freed
        ?? GPU queue cleared
        ?? Reasoning module destroyed
```

---

## Configuration Tuning

### Thread Pool Size

```cpp
// In OpenRAI constructor:
thread_pool_(std::make_unique<ThreadPool>(16))  // Instead of 10
```

### Memory Capacities

```cpp
// In memory.h:
static constexpr size_t MAX_CURRENT = 10;      // Current
static constexpr size_t MAX_SHORT_TERM = 4;    // Short-term
static constexpr size_t MAX_LONG_TERM = 1000;  // Long-term
```

### GPU Queue Slots

```cpp
// In OpenRAI constructor:
gpu_queue_(std::make_unique<GPUJobQueue>(20))  // Instead of 10
```

### Loop Iteration Time

```cpp
// In main_loop():
std::this_thread::sleep_for(
    std::chrono::seconds(5)  // Instead of 3
);
```

---

## Future Extensions

### Phase 2: Semantic Search
- Replace mock string matching with FAISS
- Use embeddings in MemoryEntry
- Similarity-based retrieval

### Phase 2: GPU Integration
- Replace mock GPU queue with real CUDA
- Kernel execution tracking
- Result streaming back to memory

### Phase 3: DAG Scheduler
- Replace ThreadPool with DAG-aware scheduler
- Task dependency resolution
- Better resource utilization

---

## Summary Table

| Module | Threads | Memory | Capacity | Purpose |
|--------|---------|--------|----------|---------|
| ThreadPool | 10 | Task queue | Unlimited | Parallel execution |
| Memory (Current) | 1 | 10 entries | Small | Fast I/O buffer |
| Memory (Short-term) | 1 | 4 entries | Small | Reasoning input |
| Memory (Long-term) | 1 | 1000 entries | Large | Persistent storage |
| GPU Queue | 1 | Job queue | 10 slots | GPU scheduling |
| Reasoning | 1 | Analysis state | Per-entry | Evaluation logic |

Total: ~15 active threads, ~10KB base memory overhead
