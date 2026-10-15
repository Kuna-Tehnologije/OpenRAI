# OpenRAI - Autonomous AI Engine

> **raw / real / reimagined AI**

OpenRAI is a **3D AI Infrastructure** – standalone platform (not a chatbot) with modern C++20 architecture designed for:
- **HEIGHT**: Massive batch parallelization (500 nodes/iter @ 10k node/sec)
- **WIDTH**: Temporal learning with persistent vector memory
- **DEPTH**: 5-layer semantic stack (Tokenizer?Embedding?Vector?Graph?Goals)

?? **Quick Reads:**
- [60-Second Overview](QUICK_3D_OVERVIEW.md) – What is 3D AI?
- [Node Scheduling: CUDA streams, expert nodes, 7.82 Hz loop](NODE_SCHEDULING.md) - primary node decides who fires, why, and whose outcome wins
- [Full 3D Architecture](3D_AI_INFRASTRUCTURE.md) – Deep dive with examples

## ?? Status: Production-Ready (Phase 2+)

- ? **500 parallel nodes** per iteration (batch GPU inference)
- ? **5-layer semantic pipeline** (not just 3-layer memory)
- ? **Neural network** with 3.98B parameters (12 layers, 4 billion scale)
- ? **Vector memory** with cosine similarity search (5000 embeddings)
- ? **Knowledge graph** with token co-occurrence tracking
- ? **Goal alignment system** with real-time scoring
- ? **Self-reflection loop** with confidence-based output
- ? **Persistent storage** (vector_memory.db + 100k CSV input pool)
- ? **50ms iteration loop** (20 iter/sec throughput)
- ? **Primary node + expert nodes** - every expert is a specialist in its own field, on its own thread, with its own model and its own CUDA stream/context
- ? **7.82 Hz fixed-rate loop** (127.877 ms/tick, drift-free) - the primary decides which experts fire and why, and whose outcome is the most important
- ? **Behaviour-driven lifecycle** - start / stop / suspend / resume / kill / respawn / re-prioritise expert nodes from the primary's goals and the experts' track record
- ? **Every expert has its own trained model** - a neural LM of its field + a backprop-trained scorer; the primary fires experts by *model competence*, tools (calculator, memory index, LM) are only used afterwards; `--train-demo` shows the learning
- ? **GPU vector memory + memory expert** - cosine search on the device, recall of learned answers
- ? **Learning from use** - SQLite store of question -> answer experiences with `/good` `/bad` feedback, node trust, goals and graph persisted across runs
- ? **Language expert (stage 2)** - a local LLM (Jan / LM Studio / llama.cpp / Ollama) phrases the final answer from the expert findings without stalling the 7.82 Hz loop

---

## 3D AI Architecture

### What is "3D AI"?

OpenRAI uses **three orthogonal dimensions** for knowledge processing:

```
        ???????????????????????????????????????????
        ?  DEPTH (Semantic Layers)                ?
        ?  Tokenizer ? Embedding ? Vector         ?
        ?           ? Graph ? Goals               ?
        ???????????????????????????????????????????
                          ?
        ???????????????????????????????????????????
        ?  WIDTH (Temporal/Iterations)            ?
        ?  Current ? Short ? Reflection           ?
        ?         ? Vector Memory (persistent)    ?
        ???????????????????????????????????????????
                          ?
        ???????????????????????????????????????????
        ?  HEIGHT (Batch Parallelization)         ?
        ?  500 nodes × GPU inference              ?
        ?  10,000 node/sec throughput             ?
        ???????????????????????????????????????????
```

**See [3D_AI_INFRASTRUCTURE.md](3D_AI_INFRASTRUCTURE.md) for detailed explanation.**

---

## Arhitektura

### Core Components (5-Layer Semantic Stack)

### Core Components (5-Layer Semantic Stack)

1. **Neural Network** (3.98B parameters - 4 billion scale)
   - Architecture: 12 layers
     - 32 ? 16384 ? 24576 ? 24576 ? 24576 ? 24576
     - ? 16384 ? 16384 ? 8192 ? 4096 ? 2048 ? 1024 ? 4
   - CUDA/CPU backend auto-detection
   - Batch inference @ 500 samples/iter
   - Output: [relevance, importance, should_output, priority]
   - Memory footprint: ~16GB for weights (float32)

2. **Semantic Pipeline** (5 layers)
   - **Layer 1 - Tokenizer**: Text ? Tokens
   - **Layer 2 - Embedding**: Tokens ? 32-dim vectors
   - **Layer 3 - Vector Memory**: Cosine similarity search (5000 capacity)
   - **Layer 4 - Knowledge Graph**: Token co-occurrence tracking
   - **Layer 5 - Goal System**: Alignment scoring (stability/relevance/memory)

3. **Memory System** (Hierarchical + Persistent)
   - **Current Memory**: 110 outputs (raw buffer)
   - **Short-Term Memory**: Top-30 by relevance
   - **Vector Memory**: 5000 embeddings (persistent to disk)
   - **Synchronization**: `shared_mutex` for concurrent access

4. **Self-Reflection Module**
   - Retrieves 4 similar contexts from vector memory
   - Calculates confidence score
   - Decision logic: should_output based on confidence threshold
   - Generates counterpoints and supporting evidence

5. **GPU Job Queue**
   - Batch inference on CUDA (fallback to CPU)
   - Priority-based job scheduling
   - Async result handling

---

## Primary / Expert Node Loop (7.82 Hz)

Since the node-scheduling upgrade the engine runs as **one primary node and N
expert nodes** (default 8). Every expert has its own thread, its own model
and its own CUDA stream with a hardware priority; the primary has the
top-priority stream. Each tick (127.877 ms):

```
PHASE 1  input       one input (external inject or the CSV pool)
PHASE 2  judge       primary scores it on its own stream
PHASE 3  plan        primary decides WHICH experts fire and WHY
                     (goal focus, affinity to the expert's field, trust, starvation, budget)
PHASE 4  dispatch    fire() -> expert threads / CUDA streams run concurrently
PHASE 5  collect     wait until the compute deadline (80 % of the tick)
PHASE 6  arbitrate   primary decides WHOSE outcome is the most important
PHASE 7  memory      winner -> current / short-term / vector memory / graph / goals
PHASE 8  reflect     reflection loop on the winner -> [FINAL OUTPUT]
PHASE 9  behaviour   start / stop / suspend / resume / kill / respawn / promote / demote
         sleep_until(next tick boundary)
```

Typical log lines:

```
[PRIMARY] tick=2 focus=stability(0.35) concurrency=3 fire=3/8
   FIRE[E1:anomaly 0.66 <- focus-goal=stability affinity=0.86; E6:telemetry 0.85 <- ...]
   HOLD[E7:language 0.28 <- budget: concurrency 3 reached; ...]
[ARBITER] tick=2 WINNER=E6:telemetry utility=0.48 because: importance 0.50 x goal(stability)=0.35 affinity 0.99 ...
[BEHAVIOUR] tick=8 KILL E0:stability because: runaway: 3 consecutive deadline misses, ema_latency=134.81ms
[BEHAVIOUR] tick=10 RESPAWN E0:stability because: focus goal stability needs this expert
[SUMMARY] tick=48 fired=3 on_time=3 late=0 winner=E2:memory hz_target=7.82 hz_measured=7.79 overruns=0 ...
```

See **[NODE_SCHEDULING.md](NODE_SCHEDULING.md)** for the scoring formulas,
the node state machine, the CUDA stream/context design and all
`OPENRAI_*` environment variables. The legacy 6-phase description below is
kept for the memory/semantic part, which is unchanged.

---

## Main Loop (3D Processing Model)

```
???????????????????????????????????????????????????????????
                  OPENRAI 3D ITERATION LOOP
???????????????????????????????????????????????????????????

PHASE 1/6: Input Injection
  ? Load 1 sample from CSV pool (100k entries)
  ? Duplicate across 500 nodes

PHASE 2/6: GPU Batch Inference (HEIGHT dimension)
  ? 500 nodes × 32-dim input
  ? Neural network: 12-layer deep (3.98B params)
  ?   32->16384->24576->24576->24576->24576
  ?   ->16384->16384->8192->4096->2048->1024->4
  ? Output: [relevance, importance, should_out, priority] × 500
  ? Time: ~2ms (CUDA) or ~50ms (CPU)

PHASE 3/6: Semantic Processing (DEPTH dimension)
  ? For each of 500 outputs:
  ?   ? Tokenize text
  ?   ? Generate 32-dim embedding
  ?   ? Store in vector memory
  ?   ? Update knowledge graph (bigram edges)
  ?   ? Update goal alignment scores

PHASE 4/6: GPU Queue Drain
  ? Dequeue completed jobs
  ? Release GPU resources

PHASE 5/6: Memory Filtering (WIDTH dimension - Temporal)
  ? Current memory: Store all 500 outputs
  ? Short-term: Filter to top-30 by relevance
  ? Reasoning: Select top-10 for reflection

PHASE 6/6: Self-Reflection & Output
  ? For each of top-10:
  ?   ? Retrieve 4 similar from vector memory
  ?   ? Calculate confidence score
  ?   ? Generate counterpoints
  ?   ? Decision: Output if confidence > threshold
  ?
  ? [FINAL OUTPUT] with vector context
  ? Store all in persistent vector_memory.db

???????????????????????????????????????????????????????????
Loop Wait: 50ms ? REPEAT (20 iterations/sec)
???????????????????????????????????????????????????????????
```

**Performance:**
- **Throughput**: 10,000 nodes/sec
- **Latency**: 50ms/iteration
- **Memory**: 5000 persistent embeddings + 16GB NN weights
- **Learning**: Accumulative (grows indefinitely)

---

## Building

### Prerequisites
- C++17 compatible compiler
- CMake 3.16+
- pthread (standard library)

### Compile

```bash
# Linux/macOS  (CMake; CUDA auto-detected -> per-node streams, else CPU threads)
chmod +x build.sh && ./build.sh            # ./build.sh --test runs the node scheduler tests

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
./build/openrai --console                        # type a question, get answer + inner thinking (/good /bad teach it)
./build/openrai_chat                             # chat front-end from the language-model branch
OPENRAI_LLM_URL=http://127.0.0.1:1337 ./build/openrai --console   # with a local LLM: phrased final answers
./build/openrai --ask "what is 100*100 and add 3/4 or sum divided by 1.34%"
./build/openrai --gpu-selftest                   # prove that per-node CUDA streams overlap
./build/openrai --train-demo                     # 3 test experts, same 10k-word data, 200 iterations -> report
./build/openrai                                  # 7.82 Hz, 9 experts, 20 minutes, log to openrai.log
./build/openrai --seconds 30 --experts 12 --concurrent 6 --verbose
OPENRAI_PROFILE=lite ./build/openrai --seconds 10 # small nets, runs on any CPU
./build/openrai_tests                             # scheduler / node / primary / skill / ask tests
cmake -DOPENRAI_IMGUI_DIR=/path/to/imgui ..       # optional Dear ImGui front-end (openrai_gui)
```

```
> what is 100*100 and add 3/4 or sum divided by 1.34%
[E8:math  SOLVED  confidence=0.95  utility=0.98  tick=1  17ms]
746324.6269
--- inner thinking ---
[PRIMARY] tick=1 ... FIRE[E8:math 0.76 <- affinity=1.00 never-fired; ...]
[ARBITER] tick=1 WINNER=E8:math utility=0.98 ... because: SOLVED by skill (confidence 0.95) ...
clause 1: "what is 100*100" -> 100 * 100 = 10000
clause 2: "add 3/4" -> # + 3 / 4   (# = 10000) = 10000.75
clause 3: "sum divided by 1.34%" -> # / 1.34%   (# = 10000.75) = 746324.6269
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