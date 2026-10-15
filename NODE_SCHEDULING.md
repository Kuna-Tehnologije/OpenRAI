# OpenRAI Node Scheduling - CUDA streams, expert nodes and the 7.82 Hz loop

## 0. What it actually does - in plain words

OpenRAI is **not** a language model. It is a real-time *committee* of small
programs ("nodes") that runs at a fixed heartbeat of 7.82 Hz:

1. Every 127.877 ms one piece of text comes in (a question you typed, or a
   line from the input pool).
2. The **primary node** looks at it with its own network and decides, from
   its current goals (stability / relevance / memory), *which* experts are
   worth waking up for this input and *why*. It writes that reasoning down.
3. The chosen **expert nodes** run **at the same time**, each on its own
   thread and its own CUDA stream. An expert brings its own *knowledge*: a
   model trained on its field says how familiar the input is and what it
   expects next, its trained MLP scorer (on the stream) rates relevance and
   importance, and if the expert owns a *tool* (calculator, memory index,
   external LM) it may use it to produce a concrete answer.
4. At the deadline (80 % of the tick) the primary collects what came back,
   ranks the outcomes by utility and picks **the winner**. Late experts are
   penalised; chronically late ones are killed and later respawned; useless
   ones are suspended; good ones get higher stream priority.
5. The winner's outcome goes through memory (vector memory, knowledge graph,
   goals) and the reflection loop and becomes the tick's output. If you
   asked the question through `ask()` / the console / the GUI, that is the
   answer you get back, together with the whole trace ("inner thinking").

What is real and what is not:

* Real: the scheduling, the concurrency (threads + CUDA streams with
  priorities), the start/stop/kill/respawn behaviour, the arbitration, the
  memory, **the per-expert models and their training** (see 6c), the
  learning store, and the tools (calculator, GPU memory index, external LM).
* Honest limits: the expert models are *small* (tens of thousands of
  parameters, trained for 200 iterations on ~10 k words of synthetic domain
  text). They are enough to tell domains apart and to continue domain text;
  they are not a language model. Free-text answers are phrased by the
  external LM (stage 2) from the experts' findings; without an LM you get
  the experts' findings and the closest memories. Replace the synthetic
  corpora in `data/corpus/` with real text and raise `OPENRAI_TRAIN_ITERS`
  to make the experts know more.

---

This document describes the multithreaded expert architecture:
one **primary node** (the executive) and N **expert nodes**, each an expert
in its own field, each on its own thread and its own CUDA stream, all driven
by a fixed-rate **7.82 Hz** loop in which the primary decides *who fires, why,
and whose outcome matters most*.

```
                         7.82 Hz tick (127.877 ms)
   +---------------------------------------------------------------------+
   |  PRIMARY (executive)  own model, own top-priority CUDA stream        |
   |    judge -> plan (who fires & why) -> ... -> arbitrate -> behaviour |
   +----+----------+----------+----------+----------+----------+---------+
        | fire     | fire     | hold     | fire     | suspended| killed
        v          v          |          v          |          |
   [E0:stability] [E1:anomaly] [E2:memory] [E3:throughput] [E4:planning] [E5:...]
    thread 0       thread 1    thread 2    thread 3        thread 4      -
    stream p=66    stream p=77 stream p=85 stream p=61     (parked)      (context destroyed)
    own model      own model   own model   own model
        |              |                       |
        +------ outcomes by compute deadline (80 % of the tick) ------+
                                  |
                          ARBITER: whose outcome is most important?
                                  |
                  memory -> reflection -> [FINAL OUTPUT]
```

---

## 1. CUDA streams and contexts (`gpu_runtime.h`)

### `GPUStreamContext` - one per node

| Member | Purpose |
|---|---|
| `cudaStream_t` | created with `cudaStreamCreateWithPriority`; the node priority (0..100) is mapped onto the device's stream-priority range, so the primary (100) always wins scheduling on the SMs |
| `cudaEvent_t done` | recorded after every launch; `poll()` = `cudaEventQuery`, `synchronize()` = `cudaEventSynchronize` |
| pinned host buffers | `cudaHostAlloc` for async H2D / D2H copies |
| device workspace | input, ping/pong activations, output - sized once for `max_batch`, **no `cudaMalloc` in the hot loop** |
| `cancel()` | makes further launches no-ops; the node thread stops polling -> `kill()` is immediate |

**Why streams inside one context and not one CUDA context per node?**
Kernels from *different* CUDA contexts are time-sliced by the driver and can
never overlap. Streams inside the device's primary context can run
concurrently and honour priorities. So every OpenRAI node gets its own
stream + event + workspace (its "context" in OpenRAI terms) inside the one
primary CUDA context; multi-GPU placement is selected with
`OPENRAI_CUDA_DEVICE`.

### `GPUNeuralNet` - weights, shared or per node

* `GPUNeuralNet(NetArch, seed)` - any fully connected architecture; the
  original 12-layer network is `NetArch::full_4b()` (note: it has 3.06 B
  parameters, the earlier docs said 3.98 B).
* Weights are uploaded layer by layer and the host copy is released, so the
  big network needs ~2.4 GB of transient host RAM instead of 16 GB.
* `run_batch_async(inputs, N, ctx)` enqueues H2D + all layers + D2H on the
  node's stream and returns immediately; `fetch_results()` reads the pinned
  output once `ctx.poll()` is true.
* The legacy `run_batch()` still exists (uses an internal default context).
* Weights are read-only, so several nodes may share one `GPUNeuralNet`
  (`OPENRAI_SHARED_EXPERT_WEIGHTS=1`); by default every expert has its own
  model with its own seed.

### Kernel fix (`nn_kernels.cu`)

The previous kernel launched `block(out_dim)`; every hidden layer is wider
than the 1024-thread limit, so those launches were invalid. The new
warp-per-neuron kernel (coalesced `float4` row reads + shuffle reduction)
works for any width and takes an explicit `cudaStream_t`.

### CPU build

Without a CUDA toolkit the same API is implemented in
`gpu_runtime_cpu.cpp`: a context is a workspace, `run_batch_async` computes
inline on the node's own thread (nodes still run in parallel), and
cancellation is checked between layers.

---

## 2. Expert nodes (`expert_node.h`)

Each `ExpertNode` owns a thread, a `GPUStreamContext`, a model and a domain:

| Expert | Goal served | Knowledge (corpus `data/corpus/<name>.txt`) | Tool |
|---|---|---|---|
| stability | stability | crashes, failover, watchdog, recovery | - |
| anomaly | stability | anomaly scores, spikes, outliers, drift | - |
| memory | memory | recall, context, consolidation | GPU vector index + learned experiences (consulted for every question) |
| throughput | relevance | latency, queues, batches, bandwidth | - |
| planning | relevance | goals, schedules, strategies | - |
| reflection | memory | evidence, conclusions, self-checks | - |
| telemetry | stability | sensors, temperatures, probes | - |
| language | relevance | user requests, reports, chat | external LM (stage 2) |
| math | relevance | arithmetic statements | calculator |
| generalistN | rotating | generic system text (`OPENRAI_EXPERTS` > 9) | - |

The keyword lists in `make_expert_domain()` are only a bootstrap used when
an expert has no trained model (`OPENRAI_TRAIN_ITERS=0`).

State machine:

```
 Stopped --start()--> Idle --fire()--> Running --result--> Idle
    ^                  |  ^                      |
    |            suspend() resume()            kill()
    |                  v  |                      v
    +----stop()---- Suspended                 Killed --start()/respawn--> Idle
```

* `start()`  - creates the stream context, spawns the thread
* `stop()`   - graceful: in-flight work completes, thread joined, context released
* `suspend()`/`resume()` - parks the node (context stays warm) / un-parks it
* `kill()`   - hard: `ctx.cancel()`, thread joined, stream + workspace destroyed
* `set_priority(p)` - re-creates the stream with the new CUDA priority
* `fire(FireRequest)` - only accepted in `Idle`; `try_collect()` returns the
  `NodeOutcome` (results, latency, on_time flag, the reason it was fired)

---

## 3. Primary node (`primary_node.h`)

The primary has its own model on the highest-priority stream and does four
things per tick. Every decision carries a reason that appears in the log.

### judge
Scores the tick's input: `[relevance, importance, should_out, priority]`.
`priority` sets the *breadth* of the search: how many experts may fire
(between `min_concurrent` and the adaptive cap).

### plan - who fires and why
Score per idle expert:

```
score = 0.35 * affinity(input)        how much this is "my field"
      + 0.25 * goal weight             how much the primary currently cares
      + 0.25 * trust                   earned from past outcomes
      + 0.15 * starvation              keeps rarely used experts warm
      + 0.10 if the expert serves the focus goal
      - 0.30 if its EMA latency exceeds the compute budget
```
Top `concurrency` experts above `fire_threshold` fire (at least one always
fires). Experts that are Running / Suspended / Killed are "not eligible" and
the log says why.

```
[PRIMARY] tick=2 focus=stability(0.35) judge[...] concurrency=3 fire=3/8
  FIRE[E1:anomaly 0.66 <- focus-goal=stability affinity=0.86;
       E6:telemetry 0.85 <- focus-goal=stability affinity=0.99 never-fired; ...]
  HOLD[E7:language 0.28 <- budget: concurrency 3 reached; ...]
```

### arbitrate - whose outcome is most important
For every outcome returned by the deadline:

```
utility = (0.40*importance + 0.20*relevance + 0.20*affinity + 0.20*goal_w)
        * (0.6 + 0.4*trust)  (+0.05 if the expert says should_out)
```
The highest utility wins; it becomes the tick's output (memory, reflection,
`[FINAL OUTPUT]`). Trust moves toward the utility (EMA), the winner gets a
bonus, late experts lose trust and get a deadline miss. If nobody answered
in time the primary uses its own judgement (logged as `PRIMARY`).

```
[ARBITER] tick=2 WINNER=E6:telemetry utility=0.48 ... because: importance 0.50
  x goal(stability)=0.35 affinity 0.99 fired-for "focus-goal=stability ..."
  beat E1:anomaly by 0.02 | ranking=E6:telemetry:0.48,E1:anomaly:0.46,...
```

### review - behaviour policy (start / stop / kill / prioritise)

| Situation | Action | Reason logged |
|---|---|---|
| Running past the deadline `kill_after_misses` times in a row | **KILL** (cancel stream, join thread, destroy context) | `runaway: 3 consecutive deadline misses` |
| Killed, cooldown over or its goal became the focus | **RESPAWN** | `cooldown over` / `focus goal X needs this expert` |
| Idle with trust < `suspend_trust` for `suspend_patience` ticks (not focus) | **SUSPEND** | `low utility: trust 0.12 < 0.18 for 6 ticks` |
| Suspended and probation over / goal became focus | **RESUME** | `probation over` / `goal demands it` |
| Stopped and the focus goal needs it | **START** | `focus goal X needs this expert` |
| trust / goal weight changed the desired priority by >= 15 | **PROMOTE / DEMOTE** (stream re-created with new CUDA priority) | `trust 0.60 goal_w 0.23 -> priority 59` |
| tick overran | adaptive concurrency cap -1 (raised again after 8 clean ticks) | `overrun -> concurrency cap lowered to 3` |

---

## 4. The 7.82 Hz loop (`scheduler.h`, `core.cpp`)

`TickScheduler` is drift-free: the next boundary is `previous + period`
(127.877 ms), overruns are counted and, if the loop falls more than one period
behind, the schedule is re-anchored instead of bursting.

```
PHASE 1  input        one input (external inject or pool)
PHASE 2  judge        primary model on its own stream
PHASE 3  plan         who fires and why
PHASE 4  dispatch     fire() on each chosen expert (own thread + stream)
PHASE 5  collect      wait on the completion signal until start + 80 % of the tick
PHASE 6  arbitrate    whose outcome is most important
PHASE 7  memory       winner + others -> current / short-term / vector / graph / goals
PHASE 8  reflect      reflection on the winner -> [FINAL OUTPUT]
PHASE 9  behaviour    start / stop / suspend / resume / kill / respawn / priority
         sleep_until(next boundary)
```

Measured on a 4-core CPU box (lite profile): `hz_measured=7.78..7.82`,
`avg_busy_ms~30`, `overruns=0`.

---

## 5. Configuration

| Env var | Default | Meaning |
|---|---|---|
| `OPENRAI_PROFILE` | `full` with CUDA, else `lite` | `full` = 3.06 B primary net, `lite` = 7 M primary net |
| `OPENRAI_TICK_HZ` | `7.82` | loop frequency |
| `OPENRAI_EXPERTS` | `9` | number of expert nodes (8 domains + math) |
| `OPENRAI_MAX_CONCURRENT` | `4` | max experts fired per tick |
| `OPENRAI_EXPERT_HIDDEN` | `256` | hidden width of `expert_lite` (trained on the CPU at start) |
| `OPENRAI_PRIMARY_ARCH` / `OPENRAI_EXPERT_ARCH` | - | custom `"32,2048,1024,4"` |
| `OPENRAI_SHARED_EXPERT_WEIGHTS` | `0` | experts share one weight set |
| `OPENRAI_KILL_AFTER_MISSES` | `3` | consecutive deadline misses before kill |
| `OPENRAI_SUSPEND_TRUST` / `OPENRAI_SUSPEND_PATIENCE` | `0.18` / `6` | suspend policy |
| `OPENRAI_PROBATION_TICKS` / `OPENRAI_RESPAWN_TICKS` | `24` / `16` | resume / respawn delays |
| `OPENRAI_RUN_SECONDS` | `1200` | run time of `main` |
| `OPENRAI_VERBOSE` | `0` | per-node per-tick logging |
| `OPENRAI_CUDA_DEVICE` | `0` | CUDA device ordinal |
| `OPENRAI_LLM` | `auto` | `auto` / `off` / `http` / `mock` - language expert backend |
| `OPENRAI_LLM_URL` / `OPENRAI_LLM_API` / `OPENRAI_LLM_MODEL` | `http://127.0.0.1:1337` / `openai-chat` / first listed | local LLM server (Jan, LM Studio, llama.cpp, vLLM, Ollama with `ollama`) |
| `OPENRAI_LLM_TIMEOUT` / `OPENRAI_LLM_MAX_TOKENS` | `30` / `256` | curl `--max-time`, completion length |
| `OPENRAI_MEMORY_CAPACITY` / `OPENRAI_GPU_MEMORY_INDEX` | `100000` / `1` | vector memory size, GPU search |
| `OPENRAI_STORE` / `OPENRAI_MEMORY_PATH` | `data/openrai_store.db` / `data/vector_memory.db` | learning store, memory file |
| `OPENRAI_TRAIN_ITERS` / `OPENRAI_TRAIN_BATCH` | `200` / `64` | expert model + scorer training budget at start |
| `OPENRAI_CORPUS_DIR` / `OPENRAI_MODEL_DIR` | `data/corpus` / `data/models` | per-domain corpora, cached models |
| `OPENRAI_CORPUS_WORDS` | `10000` | words per generated synthetic corpus |
| `OPENRAI_RETRAIN` / `OPENRAI_TRAIN_MLP` | `0` / `1` | ignore cached models; also train the MLP scorer |

CLI shortcuts: `./openrai --seconds 30 --experts 12 --concurrent 6 --hz 7.82 --profile lite --inject "alert: ..."`.

If the `full` primary network does not fit in device memory (it needs
~12.3 GB), the engine downgrades to `primary_lite` with a warning instead of
destroying the tick rate by running 3 B parameters on the CPU.

---

## 6. Text in -> answer out

```cpp
openrai::OpenRAI engine(cfg);
engine.start();
std::future<openrai::Answer> f = engine.ask("what is 100*100 and add 3/4 or sum divided by 1.34%");
openrai::Answer a = f.get();      // answered by the next tick (~130 ms)
a.answer      // "746324.6269"
a.expert      // "E8:math"
a.solved      // true  (a skill computed it)
a.confidence  // 0.95
a.thinking    // "[PRIMARY] ... [ARBITER] ... clause 1: ... answer: 746324.6269"
```

Command line:

```
./openrai --ask "calculate (12 + 8) * 3 minus 10 percent"
./openrai --console          # interactive; /thinking /status /nodes /log on|off /quit
```

Example session:

```
> what is 100*100 and add 3/4 or sum divided by 1.34%
[E8:math  SOLVED  confidence=0.95  utility=0.98  tick=1  17ms]
746324.6269
--- inner thinking ---
[PRIMARY] tick=1 focus=stability(0.35) ... FIRE[E8:math 0.76 <- affinity=1.00 never-fired; ...]
[ARBITER] tick=1 WINNER=E8:math utility=0.98 ... because: SOLVED by skill (confidence 0.95, answer "746324.6269") ...
clause 1: "what is 100*100" -> 100 * 100 = 10000
clause 2: "add 3/4" -> # + 3 / 4   (# = 10000) = 10000.75
clause 3: "sum divided by 1.34%" -> # / 1.34%   (# = 10000.75) = 746324.6269
answer: 746324.6269
```

### Tools (`skills.h`)

A `Skill` is a tool: `{ name, run(SkillContext) -> SkillResult, slow }`.
It runs on the expert's thread while the expert's network runs on its
stream. `SkillResult{ handled, answer, thinking, confidence, steps }` is
attached to the `NodeOutcome`; the arbiter gives a solved outcome utility
`>= 0.55 + 0.45 * confidence`, so a real answer always beats a mere score.
**Tools never decide which expert fires** - the models do (6c). A tool that
does not apply returns `handled=false`.

The **math skill** (`E8:math`) understands: `+ - * / ^ % ( )`, `plus`,
`minus`, `times`, `divided by`, `percent`, `square root of`, `squared`,
`x of y` (20% of 50), relative percent (`100 minus 10%` = 90), thousands
separators, and follow-up clauses that operate on the running result
(`and add 3/4`, `then multiply it by 3`, `or sum divided by 1.34%`). Prose
that merely contains numbers ("node 4 value 80C") is declined so the other
experts keep it.

To add an expert with a real capability: write a `Skill`, add a domain to
`make_expert_domain()`, and attach it in the `OpenRAI` constructor
(`if (domain_name == "...") node->set_skill(...)`).

### GUI (Dear ImGui, optional)

`ui/imgui_frontend.cpp` is a text box + answer list + live node panel.
It needs a Dear ImGui checkout, GLFW and OpenGL:

```
git clone https://github.com/ocornut/imgui
cmake -DOPENRAI_IMGUI_DIR=$PWD/imgui ..
cmake --build . --target openrai_gui
```

It is not part of the default build (no ImGui in this repository) and was
written against the public ImGui/GLFW API without being compiled here.

---

## 6b. Memory on the GPU, the memory node, and learning from use

### GPU vector memory (`GPUVectorIndex`, `VectorMemory`)
Every item of the vector memory (input texts, the winner's view of each
tick, learned questions) is a 32-float embedding. `VectorMemory` keeps the
texts on the host and the embeddings in a `GPUVectorIndex`:

* vectors are L2-normalised and appended to a device matrix on the index's
  own CUDA stream (grows by doubling, 100 k items = 12.8 MB);
* `search()` / `search_batch()` upload the query, run `cosine_scores_kernel`
  (one thread per stored vector, B queries per launch), copy the scores
  back and pick the top-k on the host;
* without CUDA the same class does the brute force on the CPU.

Capacity is `OPENRAI_MEMORY_CAPACITY` (default 100 000, FIFO eviction in
10 % batches with an index rebuild). Exact duplicates of recent items are
not stored. `./openrai --gpu-selftest` benchmarks the index over 100 k
vectors and checks it against a CPU reference.

### The memory expert (`E2:memory`)
Has a skill (`make_memory_skill`) that runs on its own thread:
1. searches the learned experiences (`experience:<id>` tags) - if a stored
   question is similar enough (`recall_threshold` 0.90) and its reward is
   not negative, it **recalls the stored answer** (handled, confidence from
   similarity and reward);
2. searches the general memory and returns the closest lines as context
   for the other experts and the language model.

### Learning store (`ExperienceStore`, SQLite, `data/openrai_store.db`)
What survives a restart:

| table | content |
|---|---|
| `experiences` | question, final answer, raw expert answer, expert, confidence, uses, **reward**, question embedding |
| `node_stats` | trust, priority, fired/wins/misses/kills of every expert |
| `goals` | goal scores of the primary |
| `graph_edges` | knowledge-graph bigrams |

The loop of learning from use:
```
ask()  -> experts -> arbiter -> (language expert) -> deliver()
                                                   -> experience stored + question indexed
ask() again -> E2:memory recalls it (no LM call, ~1 tick)
/bad       -> reward -1: never recalled again, expert trust -0.08; next ask re-solves
/good      -> reward +1: recall confidence up, expert trust +0.08
restart    -> node trust/priority, goals, graph, memory and experiences are reloaded
```
Only real answers are learned (solved by a skill or phrased by the LM);
"could not solve" notices are not.

### The language expert (`E7:language`, stage 2)
When a local LLM is reachable (`OPENRAI_LLM=auto|http`, probe of
`{url}/v1/models` at start) the language domain gets a **slow skill**:
* the primary never fires it in the normal plan ("stage 2 expert");
* after arbitration of a *user question* it is fired with the expert
  findings (`E8:math (math): SOLVED -> 42 (confidence 0.95)`), the memory
  expert's context and the goals;
* it runs for as many ticks as the LLM needs (`curl --max-time`); the loop
  keeps its 7.82 Hz; the outcome is collected as a *deferred outcome* and
  the `ask()` future completes with the phrased answer (`Answer::language`,
  `Answer::immediate` keeps the raw expert answer);
* while it is busy, further questions get the expert answer directly;
* on timeout/failure the expert answer stands and the node loses trust.

Recalled answers skip the LLM entirely, so a repeated question costs one
tick instead of seconds.

### Chat front-ends
* `./openrai --console` - `/good /bad /memory TEXT /learned /status /nodes`
* `./openrai_chat` - console chat ported from the language-model branch
* `openrai_chat_gui` - native Win32 window (Windows build only)
* `openrai_chat_imgui` - ImGui/DX11 (Windows + vcpkg imgui)
* `openrai_gui` - ImGui/GLFW (`-DOPENRAI_IMGUI_DIR=...`)

See `MERGE_NOTES.md` for what was taken from the branch and why.

---

## 6c. Expert knowledge: every expert has its own trained model

```
                      corpus data/corpus/<domain>.txt  (~10 000 words)
                                   |
          +------------------------+-------------------------+
          v                                                  v
   ExpertModel (neural n-gram LM)                 MLP scorer (GPUNeuralNet, trainable)
   emb 32 -> ctx 3 -> hidden 128 -> softmax V     32 -> 256 -> 256 -> 128 -> 64 -> 4
   Adam, 200 iterations x batch 64, ~1-1.5 s      backprop, 200 iterations, ~3 s
   saved as data/models/<domain>.model            saved as data/models/<domain>.net,
                                                  then uploaded to the expert's stream
          |                                                  |
   score(text)   : mean log p(token|context)       relevance = P(this is my field)
   generate(...) : the expert's own continuation   importance, should_out, priority
```

How the knowledge is used every tick:

* **Who fires** (`PrimaryNode::plan`): every expert's model scores the input;
  the scores are compared *relative to the best expert*
  (`affinity = exp(logp - best_logp) * (0.5 + 0.5 * coverage)`), so the
  expert whose model finds the text most familiar gets 1.0 and the others
  fall off exponentially. The log says why:
  `E8:math 0.93 <- model-competence=0.94(logp -4.99)`.
* **What the expert says** (`ExpertNode::execute`): `competence` and a
  generated `finding` ("model expects: ...") go into the outcome, the
  inner-thinking trace and the stage-2 LM prompt.
* **Arbitration**: the trained scorer's relevance/importance (on the CUDA
  stream) and the competence-based affinity feed the utility.
* **Tools** are used only after the model brought the expert in: the math
  expert reaches for its calculator, the memory expert for the GPU index.

Startup: corpora are generated once (synthetic templates per domain, replace
them with real text), models train in parallel (one thread per expert,
~9 s for 9 experts on a 4-core CPU) and are cached in `data/models/`; the
next start loads them in milliseconds. `--retrain` / `OPENRAI_RETRAIN=1`
trains again, `OPENRAI_TRAIN_ITERS` sets the budget, `OPENRAI_CORPUS_DIR`
points at your own corpora.

### Confirming it at scale: `--train-big-demo`

```
./openrai --train-big-demo                 # ~500k-line math corpus, ~500k-word
                                           # language corpus, 10000 iterations each
./openrai --train-big-demo --big-math-lines 500000 --big-lang-words 500000 \
          --big-iters 10000 --big-batch 128
```
This is the same ExpertModel as above, just at the scale and iteration count
asked for, and it answers held-out questions **by `ExpertModel::generate()`
alone - the calculator/parser is never called**. Measured in this sandbox
(CPU, 4 cores, ~3 minutes total):

```
MATH MODEL (32-word vocabulary, context 14, hidden 256)
  corpus: 500000 lines / 4040820 words
  loss 3.46 -> 1.01 (10000 iterations x batch 128, 135 s)
  "what is 24 plus 44 equals"      -> model says "6 8"  (expected 68)  [ok]
  "what is 32 minus 12 equals"     -> model says "2 0"  (expected 20)  [ok]
  "what is 10 times 6 equals"      -> model says "6 0"  (expected 60)  [ok]
  "what is 25 divided by 5 equals" -> model says "5"    (expected 5)   [ok]
  accuracy: 278/300 = 92.7 % on questions never seen verbatim during training

LANGUAGE MODEL (54-word vocabulary, context 4, hidden 128)
  corpus: 500005 words, loss 3.99 -> 0.99 (10000 iterations, 44 s)
  generate: "gpu temperature 4" -> "5 c power draw 2 2 5 watts"
```
Operand ranges are deliberately small (addition/subtraction 0-49,
multiplication/division 0-12): a 67k-parameter context-window model
cannot run an addition *algorithm*, but it can learn this-sized mapping
outright when the corpus repeats it often enough - which is exactly what
500k lines over a few thousand distinct problems gives it. This is a
feasibility confirmation, not production-grade mental arithmetic; wider
operand ranges, a bigger model and more data (practical on a GPU) are the
next step. The same run is `test_big_training_demo` in `openrai_tests`,
gated behind `OPENRAI_RUN_BIG_TEST=1` (takes ~3 minutes, so it is not part
of the default test run).

### Verifying that the experts learn: `--train-demo`

```
./openrai --train-demo --train-nodes 3 --train-iters 200
```
creates N test experts that all get the *same* ~10 000-word corpus, trains
each on its own thread and reports: loss curve (3.98 -> ~1.0), held-out
in-domain perplexity (~3) vs. foreign-text perplexity (~300), generated
continuations, and the MLP scorer's domain accuracy (~98 %). Exit code 0 =
PASS (loss down >= 25 %, in-domain >= 2 nats more likely than foreign). Use
`--train-corpus FILE` for your own text. The same check runs in
`openrai_tests` (`test_expert_models`).

---

## 7. How to verify that CUDA multithreading really works

**1. Built-in self-test**

```
./openrai --gpu-selftest
```
creates N stream contexts with different priorities, runs the same work
(a) serially on one stream, (b) on N streams from one thread, (c) from N
threads each with its own stream, checks that every stream returns the same
numbers, and prints a verdict:

```
  stream 0: handle=0x55d1c9a0 node_priority=100 cuda_priority=-5 backend=cuda
  stream 1: handle=0x55d1d2f0 node_priority=88  cuda_priority=-4 backend=cuda
  ...
  serial      : 412.10 ms  (180 forward passes on ONE stream)
  concurrent  : 131.77 ms  (same work on 9 streams, one thread)  speedup x3.13
  threaded    : 128.02 ms  (9 threads, each its own stream)      speedup x3.22
  verdict     : CUDA streams OVERLAP: 9 streams finish x3.13 faster ...
```
The exit code is 0 when overlap was measured on a CUDA build (1 otherwise).
On a CPU build you get `backend=cpu`, `handle=0x0` and only the threaded
speed-up - that is what this sandbox shows.

**2. Engine log** (`openrai.log`)

Every expert start prints its stream: `[Node E3:throughput] START ...
stream=cuda handle=0x7f3a... cuda_prio=-3 thread=48213`. Distinct handles,
distinct priorities, distinct thread ids. `[GPUNeuralNet] CUDA ACTIVE` must
appear; `CPU FALLBACK` means the weights did not fit or no device was found.

**3. Nsight Systems timeline** (the definitive proof)

```
nsys profile --trace=cuda,nvtx -o openrai ./openrai --seconds 20
nsys stats --report cuda_gpu_kern_sum openrai.nsys-rep
```
Open the report in the Nsight Systems GUI: each `dense_layer_warp_kernel`
is drawn on the row of the stream it ran on; kernels of different expert
nodes overlap in time inside one tick and the primary's stream (highest
priority) pre-empts the others.

**4. Quick sanity** - `nvidia-smi dmon -s u` while the engine runs: GPU
utilisation should show bursts every ~128 ms; with `--concurrent 1` the
bursts get longer (experts serialised), with `--concurrent 6` shorter.

**5. Per-tick numbers** - `[SUMMARY] ... fired=4 on_time=4` together with the
nodes' `ema_ms` values: if four experts each report ~20 ms and the tick's
busy time is ~25 ms, they ran concurrently; serial execution would give
~80 ms.

**6. Memory search on the GPU** - `[CUDA] Vector index created owner=vector-memory ...`
at start, `vector_memory=...(gpu,search_us=...)` in every `[SUMMARY]`, and the
`memory index: GPU cosine search over 100000 vectors` line of `--gpu-selftest`.

---

## 8. Tests

```
./build.sh --test          # or: cd build && ctest
```
`tests/node_tests.cpp` covers the scheduler (rate + overrun), async stream
API vs. sync, the node state machine including kill-while-running and
respawn, the primary's plan / arbitrate / review policy, and a 2-second
engine run at 7.82 Hz.
