# OpenRAI - Autonomous AI Engine

> **raw / real / reimagined AI**

OpenRAI is a **standalone AI platform** (not a chatbot) with modern C++17 architecture designed for parallel processing, intelligent memory management, and semantic reasoning.

## ?? Status: Phase 1 Complete ?

- ? 10 parallel worker threads
- ? 3-layer hierarchical memory
- ? Reasoning module with importance scoring
- ? GPU job queue (mock, CUDA ready)
- ? Main loop orchestration
- ? 3500+ lines of documentation
- ? Production-quality foundation

---

## Arhitektura

### Core Components

1. **ThreadPool** (10 radnih niti)
   - Kontinualno paralelno procesiranje
   - Work-stealing scheduling
   - Task-based concurrency

2. **Memory System** (3 nivoa)
   - **Current Memory**: Prvih 10 outputa (cache)
   - **Short-Term Memory**: Top 4 relevantna (za reasoning)
   - **Long-Term Memory**: Vektorska baza za dugotrajno spremanje

3. **Reasoning Module**
   - Analiza i prioritizacija
   - Evaluacija relevantnosti
   - Automatski storing u long-term memory

4. **GPU Job Queue**
   - Semaphore-based slot management
   - CUDA/ROCm ready
   - Priority queue support

---

## Main Loop (OPENRAI LOOP MODEL)

```
???????????????????????????????????????????????????????????????????
?                    OPENRAI MAIN LOOP                            ?
???????????????????????????????????????????????????????????????????

PHASE 1: Parallel Input Processing
  ?? 10 threadova istovremeno
  ?? Procesiranje input node networka
  ?? Generiranje 10 outputa

           ?

PHASE 2: Memory Filtering
  ?? 10 outputa ? sortiraj po relevance
  ?? Drži top 4 ? short-term memory

           ?

PHASE 3: Reasoning Analysis
  ?? Preuzmi short-term memoriju (4 entries)
  ?? Analiza i prioritizacija
  ?? Retrieve stare misli iz vektorske baze
  ?? Evaluate: output ILI samo spremi u DB?

           ?

PHASE 4: Decision Logic
  ?? Relevantno za finalni output?
  ?  ?? YES: [OUTPUT] console log
  ?? Nije direktno relevantno?
  ?  ?? SVEJEDNO: sprema se u long-term
  ?? Sve reasoning misli se uvijek èuvaju

           ?

PHASE 5: GPU Processing
  ?? Enqueue GPU jobs (inference, embedding, matrix ops)
  ?? Semaphore-controlled concurrency

           ?

LOOP ? (Back to PHASE 1)
```

---

## Building

### Prerequisites
- C++17 compatible compiler
- CMake 3.16+
- pthread (standard library)

### Compile

```bash
# Linux/macOS
chmod +x build.sh && ./build.sh

# Windows
cl.exe /std:c++17 /EHsc /I include src\*.cpp /Fe:openrai.exe

# All platforms (CMake)
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
.\Release\openrai.exe
```

### Run

```bash
./openrai
```

---

## Output Example

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
10 threads processing nodes in parallel
Memory flow: Current ? Short-term ? Long-term (vector DB)
================================

--- ITERATION 1 ---
[PHASE 1] Enqueuing 10 parallel processing tasks...
  [Thread 0] Generated output with relevance: 0.3
  [Thread 1] Generated output with relevance: 0.55
  ...

[PHASE 2] Filtering 10 current outputs ? 4 short-term entries...
[Memory] Filtered 10 current outputs to 4 short-term entries

[PHASE 3] Reasoning module analyzing short-term memory...
[Reasoning] Analyzing 4 short-term entries
[Reasoning] Retrieved 0 long-term entries
[OUTPUT] ...
...
```

---

## Next Steps (After PoC)

- [ ] FAISS/hnswlib integration za semantic search
- [ ] CUDA kernel integration
- [ ] Real DAG task scheduler
- [ ] Protocol Buffers za komunikaciju
- [ ] Persistent vector database
- [ ] Distributed computing support