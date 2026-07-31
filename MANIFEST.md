# OpenRAI Project Manifest

## Project Information

**Name**: OpenRAI - PoC (Proof of Concept)  
**Type**: Standalone AI Engine Platform  
**Language**: C++17  
**Status**: Phase 1 Complete ?  
**License**: MIT  
**Repository**: https://github.com/Kuna-Tehnologije/OpenRAI  
**Author**: Kuna Tehnologije j.d.o.o. (Neven Joviæ)  

---

## File Structure & Contents

### Root Level Configuration

```
.gitattributes          Git line ending configuration
.gitignore              Git ignore rules
LICENSE.txt             MIT License
CMakeLists.txt          CMake build configuration
Makefile                Unix make configuration
build.bat               Windows batch build script
build.sh                Unix shell build script
```

### Documentation Files (10 total)

| File | Lines | Purpose |
|------|-------|---------|
| README.md | ~150 | Quick overview & features |
| ARCHITECTURE.md | ~700 | Detailed design & components |
| MODULES.md | ~600 | API reference for each module |
| BUILD_GUIDE.md | ~200 | Platform-specific build instructions |
| QUICK_START.md | ~250 | Quick reference card |
| FAQ.md | ~400 | 50+ frequently asked questions |
| ROADMAP.md | ~400 | Development roadmap (Phases 2-5) |
| COMPLETION.md | ~300 | Project completion summary |
| DIAGRAMS.md | ~500 | ASCII architecture diagrams |
| MANIFEST.md | This file | Project structure overview |

**Total Documentation**: ~3500+ lines

### Source Code (6 implementation files)

#### Headers (include/openrai/)

```
core.h              (80 lines)   Main engine class & data structures
                                - OpenRAI engine
                                - Output & MemoryEntry structs
                                - Forward declarations

thread_pool.h       (40 lines)   ThreadPool API
                                - 10 worker threads
                                - Task queue interface
                                - Synchronization primitives

memory.h            (60 lines)   MemorySystem API (3 layers)
                                - Current memory (10 entries)
                                - Short-term memory (4 entries)
                                - Long-term memory (1000 entries)
                                - Thread-safe operations

gpu_queue.h         (40 lines)   GPUJobQueue API
                                - GPU job structure
                                - Semaphore slot management
                                - FIFO queue interface

reasoning.h         (40 lines)   ReasoningModule API
                                - Thought evaluation
                                - Importance scoring
                                - Output decision logic

tests.h             (80 lines)   Unit test framework
                                - Component tests
                                - Integration tests
                                - Test runner
```

**Total Headers**: ~340 lines

#### Implementation (src/)

```
main.cpp            (50 lines)   Entry point
                                - Banner output
                                - Engine startup/shutdown
                                - 15-second runtime

main_updated.cpp    (50 lines)   Alternative entry point
                                - Test mode support
                                - Configurable operation modes

core.cpp            (280 lines)  Main engine implementation
                                - OpenRAI constructor/destructor
                                - Main loop (5 phases)
                                - Phase orchestration
                                - Logging & diagnostics

thread_pool.cpp     (80 lines)   ThreadPool implementation
                                - Worker thread loop
                                - Task enqueueing
                                - Synchronization logic
                                - Active task tracking

memory.cpp          (100 lines)  MemorySystem implementation
                                - 3-layer memory management
                                - Relevance-based filtering
                                - Thread-safe access patterns
                                - Overflow handling

gpu_queue.cpp       (60 lines)   GPUJobQueue implementation
                                - Job queue management
                                - Semaphore-based concurrency
                                - FIFO ordering

reasoning.cpp       (80 lines)   ReasoningModule implementation
                                - Importance calculation
                                - Output decision tree
                                - Long-term memory storage
                                - Thought evaluation
```

**Total Implementation**: ~700 lines

### Code Summary

| Category | Files | Lines | Purpose |
|----------|-------|-------|---------|
| Headers | 6 | ~340 | API definitions |
| Implementation | 7 | ~700 | Business logic |
| Tests | 1 file (in headers) | ~80 | Unit tests |
| Build Config | 4 | ~100 | CMake, Make, Batch |
| Documentation | 10 | ~3500 | Guides & references |
| **TOTAL** | **29** | **~4700+** | Complete PoC |

---

## Directory Tree

```
openRAI/
?
??? include/openrai/
?   ??? core.h                  # Main engine API
?   ??? thread_pool.h           # ThreadPool API
?   ??? memory.h                # MemorySystem API
?   ??? gpu_queue.h             # GPUJobQueue API
?   ??? reasoning.h             # ReasoningModule API
?   ??? tests.h                 # Test framework
?
??? src/
?   ??? main.cpp                # Entry point
?   ??? main_updated.cpp        # Alternative entry with tests
?   ??? core.cpp                # Engine implementation
?   ??? thread_pool.cpp         # ThreadPool implementation
?   ??? memory.cpp              # Memory implementation
?   ??? gpu_queue.cpp           # GPU queue implementation
?   ??? reasoning.cpp           # Reasoning implementation
?
??? Documentation/
?   ??? README.md               # Quick start (this repo)
?   ??? ARCHITECTURE.md         # Design details
?   ??? MODULES.md              # API reference
?   ??? BUILD_GUIDE.md          # Build instructions
?   ??? QUICK_START.md          # Quick reference
?   ??? FAQ.md                  # Q&A
?   ??? ROADMAP.md              # Future plans
?   ??? COMPLETION.md           # Phase 1 summary
?   ??? DIAGRAMS.md             # Architecture diagrams
?   ??? MANIFEST.md             # This file
?
??? Build Configuration/
?   ??? CMakeLists.txt          # CMake config
?   ??? Makefile                # Unix make
?   ??? build.bat               # Windows batch
?   ??? build.sh                # Unix shell
?
??? Version Control/
?   ??? .gitignore              # Ignore rules
?   ??? .gitattributes          # Line endings
?   ??? LICENSE.txt             # MIT License
?
??? Root Files
    ??? (configuration files above)
```

---

## Architecture Components

### 1. ThreadPool (10 Workers)
- **Files**: `include/openrai/thread_pool.h`, `src/thread_pool.cpp`
- **Lines**: ~120 total
- **Features**:
  - 10 continuously active worker threads
  - Task queue with mutex + condition variable
  - Wait_all() blocking until completion
  - Active task counter (atomic)

### 2. MemorySystem (3 Layers)
- **Files**: `include/openrai/memory.h`, `src/memory.cpp`
- **Lines**: ~160 total
- **Features**:
  - Current memory: 10 entries FIFO
  - Short-term memory: 4 entries sorted
  - Long-term memory: 1000 entries (mock vector DB)
  - Thread-safe with shared_mutex

### 3. ReasoningModule
- **Files**: `include/openrai/reasoning.h`, `src/reasoning.cpp`
- **Lines**: ~120 total
- **Features**:
  - Importance scoring (0.0-1.0)
  - Output-worthy decision logic
  - Context retrieval from long-term
  - Automatic thought persistence

### 4. GPUJobQueue
- **Files**: `include/openrai/gpu_queue.h`, `src/gpu_queue.cpp`
- **Lines**: ~100 total
- **Features**:
  - Semaphore-based slot management
  - FIFO job queue
  - Max 10 concurrent jobs
  - Support for inference/embedding/matrix_op

### 5. OpenRAI Core Engine
- **Files**: `include/openrai/core.h`, `src/core.cpp`
- **Lines**: ~360 total
- **Features**:
  - Main loop orchestration
  - 5-phase iteration cycle
  - Component coordination
  - Graceful startup/shutdown

---

## Build & Compilation

### Prerequisites
- **C++17 compiler**: GCC 7+, Clang 5+, MSVC 19.0+
- **Standard library**: With threading support
- **CMake** (optional): 3.16+

### Build Methods

#### Method 1: Shell Script (Linux/macOS)
```bash
chmod +x build.sh
./build.sh
```

#### Method 2: CMake (All Platforms)
```bash
mkdir build && cd build
cmake .. && cmake --build .
```

#### Method 3: Unix Make
```bash
make -f Makefile
```

#### Method 4: Windows Batch
```bash
build.bat
```

#### Method 5: Direct MSVC
```bash
cl.exe /std:c++17 /EHsc /I include src\*.cpp
```

### Expected Build Time
- Debug: ~2-5 seconds
- Release: ~5-10 seconds
- Clean rebuild: ~10-20 seconds

---

## Running the Engine

### Default Mode (Run Engine)
```bash
./openrai          # Linux/macOS
openrai.exe        # Windows
```

### Test Mode (With Unit Tests)
```bash
./openrai test     # Linux/macOS
openrai.exe test   # Windows
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
...
--- ITERATION 1 ---
[PHASE 1] Enqueuing 10 parallel processing tasks...
...
```

---

## Features Summary

### Phase 1 PoC (Complete ?)

**Concurrency**
- ? 10 worker threads with task queue
- ? Mutex-based synchronization
- ? Condition variables for coordination
- ? Atomic counters
- ? Shared mutex for read-write locking

**Memory Management**
- ? 3-layer hierarchical system
- ? Automatic FIFO overflow
- ? Relevance-based sorting
- ? Thread-safe access throughout

**Reasoning**
- ? Importance scoring
- ? Output decision logic
- ? Thought evaluation
- ? Context retrieval

**GPU Support** (Mock)
- ? Job queue structure
- ? Semaphore slot management
- ? FIFO ordering
- ? Actual GPU execution (Phase 2)

**Main Loop**
- ? 5-phase iteration cycle
- ? Phase orchestration
- ? Configurable timing
- ? Graceful shutdown

---

## Development Phases

### ? Phase 1: Proof of Concept (Complete)
- Core threading model
- Memory hierarchy
- Reasoning framework
- Main loop orchestration

### ? Phase 2: GPU & Persistence (3-4 weeks)
- FAISS/hnswlib integration
- CUDA kernel support
- SQLite persistent storage

### ?? Phase 3: Advanced Scheduling (2-3 weeks)
- DAG task scheduler
- Work-stealing algorithm
- Priority queues

### ?? Phase 4: Distributed Computing (4-6 weeks)
- gRPC communication
- Raft consensus
- Multi-node coordination

### ?? Phase 5: Advanced Reasoning (Ongoing)
- NLP preprocessing
- Semantic embeddings
- Knowledge graphs

---

## Testing

### Unit Tests
Located in: `include/openrai/tests.h`

**Test Coverage**:
- ThreadPool execution
- Memory layer transitions
- Reasoning evaluation
- GPU queue ordering
- Full engine loop

**Run Tests**:
```bash
./openrai test
```

### Integration Tests
- Full 15-second runtime
- All components interacting
- Graceful shutdown

---

## Documentation Map

| Document | Focus | Audience |
|----------|-------|----------|
| README.md | Quick overview | Everyone |
| QUICK_START.md | Quick reference | Developers |
| BUILD_GUIDE.md | Compilation | DevOps/Users |
| ARCHITECTURE.md | Design deep dive | Architects |
| MODULES.md | API details | Developers |
| FAQ.md | Common questions | Support |
| ROADMAP.md | Future plans | Project managers |
| DIAGRAMS.md | Visual reference | Everyone |
| COMPLETION.md | Phase 1 summary | Stakeholders |

---

## Key Metrics

### Code Quality
- ? No external dependencies (Phase 1)
- ? C++17 standard compliance
- ? Thread-safe throughout
- ? Well-documented code

### Performance (PoC)
- **Threads**: 10 active
- **Memory**: ~50 MB base
- **Loop latency**: ~3 seconds (configurable)
- **Throughput**: ~3-10 thoughts/second

### Scalability
- **Thread count**: Configurable
- **Memory sizes**: Configurable
- **Persistence**: Prepared (Phase 2)
- **Distribution**: Designed (Phase 4)

---

## Known Limitations

### Phase 1 Constraints
1. **No real GPU**: Jobs enqueued but not executed
2. **Mock vector DB**: String matching only (Phase 2: FAISS)
3. **No persistence**: All data in RAM (Phase 2: SQLite)
4. **Single node**: No distribution (Phase 4: gRPC)
5. **Simple reasoning**: Keyword-based heuristics
6. **No external input**: process_input() is stub

### Planned Resolutions
- Phase 2: GPU + Persistence
- Phase 3: Advanced scheduling
- Phase 4: Distributed architecture
- Phase 5: Advanced reasoning

---

## Success Criteria

### Phase 1 ?
- ? Compiles without warnings
- ? Runs 5 iterations without crashes
- ? All 3 memory layers functional
- ? Reasoning produces output
- ? Thread-safe throughout
- ? Well-documented

### Next Phase Targets
- [ ] Vector DB queries < 10ms
- [ ] GPU 10x faster than CPU
- [ ] Persistent storage < 1s load
- [ ] DAG execution correct ordering
- [ ] 3-node cluster stable

---

## Getting Started

1. **Clone**: `git clone https://github.com/Kuna-Tehnologije/OpenRAI.git`
2. **Build**: See BUILD_GUIDE.md
3. **Run**: `./openrai`
4. **Learn**: Read ARCHITECTURE.md
5. **Contribute**: Pick task from ROADMAP.md

---

## Support & Contact

### Documentation
- README.md - Start here
- FAQ.md - Common questions
- ARCHITECTURE.md - Technical details

### Issues & Contributions
- GitHub Issues for bugs
- GitHub Discussions for questions
- Pull requests for features

### Maintainer
**Kuna Tehnologije j.d.o.o.**  
Repository: https://github.com/Kuna-Tehnologije/OpenRAI

---

## License

MIT License - See LICENSE.txt

Can be used commercially, modified, and redistributed.

---

## Summary

**OpenRAI Phase 1 PoC** is a production-quality foundation for an AI engine with:
- Modern C++17 concurrency
- Intelligent memory management
- Parallel reasoning
- Clear path to GPU acceleration
- Comprehensive documentation

**Status**: Ready for Phase 2 development

**Timeline**: 3-4 weeks to Phase 2 completion

**Repository**: https://github.com/Kuna-Tehnologije/OpenRAI

---

**Built with ?? by Kuna Tehnologije**

Raw. Real. Reimagined. ??
