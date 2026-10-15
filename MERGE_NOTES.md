# Merge notes: `feature/language-model` -> node scheduler (master)

**Verdict: do not git-merge the branch wholesale.** Its node system is
superseded by the primary/expert scheduler; the parts worth keeping were
ported by hand into this tree and adapted to the 7.82 Hz loop. Treat this
tree as the new master and keep the branch for reference.

## What was taken from the branch (and how it changed)

| Branch component | Status here | Notes |
|---|---|---|
| `lm_adapter.h/.cpp` (HTTP to local LLM via curl) | **ported** -> `include/openrai/lm_adapter.h`, `src/lm_adapter.cpp` | portable (`popen`/`_popen`, `curl`/`curl.exe`), `--max-time`, OpenAI **chat** API by default (+ `openai`, `ollama`), no hard-coded model name (first model the server lists), real JSON string parsing (escapes, `\uXXXX`), server probe at start; becomes the **`E7:language` slow expert** |
| `vector_db.h/.cpp` (SQLite vectors) | **replaced** by `experience_store.h/.cpp` | SQLite is now the *learning store* (experiences, node stats, goals, graph); similarity search moved to the GPU (`GPUVectorIndex`) instead of SQL blobs scanned on the CPU |
| `src/vendor/sqlite` amalgamation | **kept** (3.47.2) | compiled as C into `openrai_core`; the duplicate `sqlite-amalgamation-3470200/` folder and `sqlite.zip` were not copied |
| `node.h/.cpp` (`Node`, `NodeManager`, specialties, per-node SQLite) | **superseded** | `ExpertNode` + `PrimaryNode` do the same job with real threads/streams, deadlines, kill/respawn and reasons. The canned "[Reasoning] Analyzed: ..." summaries are replaced by expert *findings* (skill answers or NN scores) fed to the LM |
| `core.cpp` orchestration (`process_with_orchestration`, `generate_response`, output callback) | **re-implemented** | `inject_input()` no longer blocks on curl; the LM runs as a stage-2 expert across ticks. `set_output_callback`, `generate_response`, `get_latest_output`, `get_latest_confidence`, `get_iteration_count` are provided for compatibility so the branch GUIs work unchanged |
| `chat_gui.h`, `chat_gui.cpp` (console), `main_chat.cpp` | **ported** (`openrai_chat`) | console version now does one `generate_response()` = one `ask()` through the loop |
| `chat_gui_win32.cpp` (native Win32 window) | **ported verbatim** (`openrai_chat_gui`, Windows only) | compatible through the callback API; not compiled here (no Windows) |
| `chat_gui_imgui.cpp` (ImGui + DX11 template) | **ported** (`openrai_chat_imgui`, built only if vcpkg `imgui` is found) | `find_package(imgui)` is optional now - the branch made it `REQUIRED`, which broke every build without vcpkg |
| `weight_loader.h/.cpp`, FP16 weights, `set_device` | **not ported** | the `NetArch`-based runtime replaced the fixed 12-layer weight code; FP16 storage would need a half-precision kernel. Worth doing later for the 3 B primary net (6 GB instead of 12 GB) |
| `data/nodes/*.db`, `openrai_math_1M.csv`, `test_*.json` | not copied | runtime artefacts |
| 30+ markdown status files | not copied | superseded by `NODE_SCHEDULING.md` |

## If you still want a git merge

```
git checkout master
git merge --no-commit --no-ff feature/language-model
# keep master's version of everything under src/ include/ CMakeLists.txt tests/
git checkout --ours -- src include CMakeLists.txt tests ui
# keep the branch's docs you care about, then
git commit
```
The only branch-side files that do not exist on master are its markdown
guides, `examples/`, `scripts/setup_imgui.ps1` and `vcpkg.json`; everything
else conflicts and master wins.

## Verification done here (CPU-only sandbox)

* `openrai_tests`: 335 checks, including store persistence, GPU index vs
  brute force, LM JSON parsing, mock-LM learning loop (phrase -> recall ->
  punish -> re-phrase -> recall after restart), ThreadSanitizer clean.
* End-to-end against a fake OpenAI-compatible server (`/v1/models`,
  `/v1/chat/completions`) with 0.3 s and 4 s latencies: the 7.82 Hz loop
  never stalls, the answer arrives when the LM returns.
* Not verified here: CUDA build of `GPUVectorIndex` (syntax-checked against
  a stub header), Win32/ImGui chat windows.
