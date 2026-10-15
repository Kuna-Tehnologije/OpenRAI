#pragma once

// -----------------------------------------------------------------------
// config.h
//   Runtime configuration of the OpenRAI engine.  Every value has a sane
//   default and can be overridden with an environment variable so the same
//   binary can run the full 4B-parameter profile on a GPU workstation and
//   the "lite" profile on a CPU-only laptop / CI box.
//
//   Environment variables
//   ---------------------
//   OPENRAI_PROFILE          full | lite   (default: full when CUDA is
//                                           compiled in, lite otherwise)
//   OPENRAI_TICK_HZ          loop frequency, default 7.82
//   OPENRAI_EXPERTS          number of expert nodes, default 9 (8 + math)
//   OPENRAI_MAX_CONCURRENT   max experts fired per tick, default 4
//   OPENRAI_EXPERT_HIDDEN    hidden width of expert_lite nets, default 256
//   OPENRAI_PRIMARY_ARCH     "32,2048,...,4" custom primary architecture
//   OPENRAI_EXPERT_ARCH      "32,1024,...,4" custom expert architecture
//   OPENRAI_RUN_SECONDS      how long main() runs, default 1200
//   OPENRAI_VERBOSE          1 = per-node per-tick logging
//   OPENRAI_SHARED_EXPERT_WEIGHTS 1 = all experts share one weight set
//                                    (saves memory; experts still differ
//                                     by their domain lens + own stream)
//   OPENRAI_LLM              auto | off | http | mock   (default auto:
//                            use the http server if it answers a probe)
//   OPENRAI_LLM_URL          default http://127.0.0.1:1337
//   OPENRAI_LLM_API          openai-chat | openai | ollama
//   OPENRAI_LLM_MODEL        model id ("" = first model the server lists)
//   OPENRAI_LLM_TIMEOUT      seconds, default 30
//   OPENRAI_LLM_MAX_TOKENS   default 256
//   OPENRAI_MEMORY_CAPACITY  vector memory items, default 100000
//   OPENRAI_GPU_MEMORY_INDEX 1 = similarity search on the GPU (default)
//   OPENRAI_TRAIN_ITERS      training iterations per expert model (default 200)
//   OPENRAI_CORPUS_DIR       per-domain corpora, default data/corpus
//   OPENRAI_CORPUS_WORDS     words generated per synthetic corpus (10000)
//   OPENRAI_MODEL_DIR        trained models, default data/models
//   OPENRAI_RETRAIN          1 = ignore saved models and train again
//   OPENRAI_TRAIN_MLP        1 = also train the per-expert MLP scorer (default)
//   OPENRAI_STORE            SQLite learning store, default data/openrai_store.db
//   OPENRAI_MEMORY_PATH      vector memory file, default data/vector_memory.db
// -----------------------------------------------------------------------

#include <openrai/gpu_runtime.h>

#include <chrono>
#include <cstddef>
#include <string>

namespace openrai {

struct EngineConfig {
    // ---- Loop timing ----------------------------------------------------
    double tick_hz          = 7.82;   // primary loop frequency
    double compute_fraction = 0.80;   // share of the tick reserved for expert
                                      // inference; the rest is arbitration,
                                      // memory and reflection
    // ---- Nodes ------------------------------------------------------------
    size_t expert_count     = 9;      // catalogue: 8 domains + math
    size_t max_concurrent   = 4;      // upper bound of experts fired per tick
    size_t min_concurrent   = 1;
    size_t node_max_batch   = 8;      // inputs a node may receive per tick
    bool   shared_expert_weights = false;

    // ---- Behaviour policy (primary node) ----------------------------------
    float  fire_threshold       = 0.30f; // min plan score to fire a node
    float  suspend_trust        = 0.18f; // trust below this -> suspend
    int    suspend_patience     = 6;     // consecutive low-trust ticks
    int    probation_ticks      = 24;    // suspended nodes retried after this
    int    kill_after_misses    = 3;     // consecutive deadline misses -> kill
    int    respawn_after_ticks  = 16;    // killed nodes respawned after this
    int    starvation_ticks     = 12;    // ticks without firing -> boost
    float  trust_alpha          = 0.25f; // EMA factor for trust updates

    // ---- Models -----------------------------------------------------------
    std::string profile = "full";
    NetArch primary_arch;
    NetArch expert_arch;

    // ---- Language model (slow expert) --------------------------------------
    std::string llm_backend = "auto";
    std::string llm_url     = "http://127.0.0.1:1337";
    std::string llm_api     = "openai-chat";
    std::string llm_model;
    int    llm_timeout_s    = 30;
    int    llm_max_tokens   = 256;
    int    slow_deadline_ticks = 240;   // ~30 s at 7.82 Hz

    // ---- Expert knowledge (per-expert models) ----------------------------------
    size_t      train_iterations = 200;
    size_t      train_batch      = 64;
    size_t      corpus_words     = 10000;
    std::string corpus_dir       = "data/corpus";
    std::string model_dir        = "data/models";
    bool        retrain          = false;
    bool        train_mlp        = true;
    size_t      model_vocab      = 2048;
    size_t      model_hidden     = 128;

    // ---- Memory & learning ----------------------------------------------------
    size_t      memory_capacity  = 100000;
    bool        gpu_memory_index = true;
    std::string store_path       = "data/openrai_store.db";
    std::string memory_path      = "data/vector_memory.db";
    bool        persist_learning = true;

    // ---- Misc -------------------------------------------------------------
    int  run_seconds = 1200;
    bool verbose     = false;

    // Derived helpers
    std::chrono::nanoseconds tick_period() const {
        return std::chrono::nanoseconds(
            static_cast<long long>(1e9 / (tick_hz > 0.0 ? tick_hz : 7.82)));
    }
    std::chrono::nanoseconds compute_budget() const {
        return std::chrono::nanoseconds(
            static_cast<long long>(tick_period().count() * compute_fraction));
    }

    // Build from defaults + environment.  `cuda_compiled` selects the
    // default profile.
    static EngineConfig from_env(bool cuda_compiled);
    std::string describe() const;
};

} // namespace openrai
