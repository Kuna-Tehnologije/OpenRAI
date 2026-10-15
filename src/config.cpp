#include <openrai/config.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

namespace openrai {

static bool env_str(const char* name, std::string& out) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    out = v;
    return true;
}

static double env_double(const char* name, double def) {
    std::string s;
    if (!env_str(name, s)) return def;
    try { return std::stod(s); } catch (...) { return def; }
}

static long env_long(const char* name, long def) {
    std::string s;
    if (!env_str(name, s)) return def;
    try { return std::stol(s); } catch (...) { return def; }
}

static bool env_flag(const char* name, bool def) {
    std::string s;
    if (!env_str(name, s)) return def;
    return s == "1" || s == "true" || s == "on" || s == "yes";
}

EngineConfig EngineConfig::from_env(bool cuda_compiled) {
    EngineConfig c;

    c.profile = cuda_compiled ? "full" : "lite";
    env_str("OPENRAI_PROFILE", c.profile);
    std::transform(c.profile.begin(), c.profile.end(), c.profile.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (c.profile != "full" && c.profile != "lite") c.profile = cuda_compiled ? "full" : "lite";

    c.tick_hz          = env_double("OPENRAI_TICK_HZ", 7.82);
    if (c.tick_hz <= 0.0 || c.tick_hz > 1000.0) c.tick_hz = 7.82;
    c.compute_fraction = std::clamp(env_double("OPENRAI_COMPUTE_FRACTION", 0.80), 0.2, 0.95);

    c.expert_count   = static_cast<size_t>(std::clamp(env_long("OPENRAI_EXPERTS", 9), 1L, 64L));
    c.max_concurrent = static_cast<size_t>(std::clamp(env_long("OPENRAI_MAX_CONCURRENT", 4), 1L, 64L));
    c.max_concurrent = std::min(c.max_concurrent, c.expert_count);
    c.min_concurrent = std::min<size_t>(1, c.max_concurrent);
    c.node_max_batch = static_cast<size_t>(std::clamp(env_long("OPENRAI_NODE_BATCH", 8), 1L, 512L));
    c.shared_expert_weights = env_flag("OPENRAI_SHARED_EXPERT_WEIGHTS", false);

    c.run_seconds = static_cast<int>(std::clamp(env_long("OPENRAI_RUN_SECONDS", 1200), 1L, 86400L * 30));
    c.verbose     = env_flag("OPENRAI_VERBOSE", false);

    c.kill_after_misses   = static_cast<int>(std::clamp(env_long("OPENRAI_KILL_AFTER_MISSES", 3), 1L, 100L));
    c.suspend_patience    = static_cast<int>(std::clamp(env_long("OPENRAI_SUSPEND_PATIENCE", 6), 1L, 1000L));
    c.probation_ticks     = static_cast<int>(std::clamp(env_long("OPENRAI_PROBATION_TICKS", 24), 1L, 100000L));
    c.respawn_after_ticks = static_cast<int>(std::clamp(env_long("OPENRAI_RESPAWN_TICKS", 16), 1L, 100000L));
    c.starvation_ticks    = static_cast<int>(std::clamp(env_long("OPENRAI_STARVATION_TICKS", 12), 1L, 100000L));
    c.fire_threshold      = static_cast<float>(std::clamp(env_double("OPENRAI_FIRE_THRESHOLD", 0.30), 0.0, 1.0));
    c.suspend_trust       = static_cast<float>(std::clamp(env_double("OPENRAI_SUSPEND_TRUST", 0.18), 0.0, 1.0));

    // ---- language model ----
    env_str("OPENRAI_LLM", c.llm_backend);
    std::transform(c.llm_backend.begin(), c.llm_backend.end(), c.llm_backend.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (c.llm_backend != "auto" && c.llm_backend != "off" && c.llm_backend != "http" && c.llm_backend != "mock")
        c.llm_backend = "auto";
    env_str("OPENRAI_LLM_URL", c.llm_url);
    env_str("OPENRAI_LLM_API", c.llm_api);
    env_str("OPENRAI_LLM_MODEL", c.llm_model);
    c.llm_timeout_s  = static_cast<int>(std::clamp(env_long("OPENRAI_LLM_TIMEOUT", 30), 2L, 600L));
    c.llm_max_tokens = static_cast<int>(std::clamp(env_long("OPENRAI_LLM_MAX_TOKENS", 256), 16L, 8192L));
    c.slow_deadline_ticks = static_cast<int>(std::clamp(
        static_cast<long>((c.llm_timeout_s + 2) * c.tick_hz), 8L, 100000L));

    // ---- expert knowledge ----
    c.train_iterations = static_cast<size_t>(std::clamp(env_long("OPENRAI_TRAIN_ITERS", 200), 0L, 1000000L));
    c.train_batch      = static_cast<size_t>(std::clamp(env_long("OPENRAI_TRAIN_BATCH", 64), 1L, 4096L));
    c.corpus_words     = static_cast<size_t>(std::clamp(env_long("OPENRAI_CORPUS_WORDS", 10000), 100L, 100000000L));
    env_str("OPENRAI_CORPUS_DIR", c.corpus_dir);
    env_str("OPENRAI_MODEL_DIR", c.model_dir);
    c.retrain   = env_flag("OPENRAI_RETRAIN", false);
    c.train_mlp = env_flag("OPENRAI_TRAIN_MLP", true);
    c.model_vocab  = static_cast<size_t>(std::clamp(env_long("OPENRAI_MODEL_VOCAB", 2048), 16L, 65536L));
    c.model_hidden = static_cast<size_t>(std::clamp(env_long("OPENRAI_MODEL_HIDDEN", 128), 8L, 4096L));

    // ---- memory & learning ----
    c.memory_capacity  = static_cast<size_t>(std::clamp(env_long("OPENRAI_MEMORY_CAPACITY", 100000), 256L, 50000000L));
    c.gpu_memory_index = env_flag("OPENRAI_GPU_MEMORY_INDEX", true);
    env_str("OPENRAI_STORE", c.store_path);
    env_str("OPENRAI_MEMORY_PATH", c.memory_path);
    c.persist_learning = env_flag("OPENRAI_PERSIST", true);

    // ---- model architectures ----
    // 256-wide expert scorers train on the CPU in a few seconds; raise it on
    // a GPU box if you want (inference runs on the streams either way).
    const size_t hidden = static_cast<size_t>(std::clamp(env_long("OPENRAI_EXPERT_HIDDEN", 256), 8L, 65536L));
    if (c.profile == "full") {
        c.primary_arch = NetArch::full_4b();
    } else {
        c.primary_arch = NetArch::primary_lite();
    }
    c.expert_arch = NetArch::expert_lite(hidden);

    std::string s;
    if (env_str("OPENRAI_PRIMARY_ARCH", s)) {
        NetArch a = NetArch::parse(s, "primary_custom");
        if (a.valid() && a.input_dim() == GPUNeuralNet::INPUT_DIM && a.output_dim() == GPUNeuralNet::OUTPUT_DIM)
            c.primary_arch = a;
    }
    if (env_str("OPENRAI_EXPERT_ARCH", s)) {
        NetArch a = NetArch::parse(s, "expert_custom");
        if (a.valid() && a.input_dim() == GPUNeuralNet::INPUT_DIM && a.output_dim() == GPUNeuralNet::OUTPUT_DIM)
            c.expert_arch = a;
    }

    return c;
}

std::string EngineConfig::describe() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3)
       << "profile=" << profile
       << " tick_hz=" << tick_hz
       << " period_ms=" << (tick_period().count() / 1e6)
       << " compute_budget_ms=" << (compute_budget().count() / 1e6)
       << " experts=" << expert_count
       << " max_concurrent=" << max_concurrent
       << " node_batch=" << node_max_batch
       << " primary=" << primary_arch.name << "[" << primary_arch.describe() << "]"
       << " expert=" << expert_arch.name << "[" << expert_arch.describe() << "]"
       << " shared_expert_weights=" << (shared_expert_weights ? "yes" : "no")
       << " kill_after_misses=" << kill_after_misses
       << " suspend_trust=" << suspend_trust
       << " llm=" << llm_backend << "[" << llm_url << " " << llm_api << (llm_model.empty() ? "" : " " + llm_model) << "]"
       << " memory_capacity=" << memory_capacity
       << " gpu_memory_index=" << (gpu_memory_index ? "yes" : "no")
       << " train_iters=" << train_iterations
       << " corpus_dir=" << corpus_dir
       << " model_dir=" << model_dir
       << " store=" << store_path
       << " run_seconds=" << run_seconds;
    return ss.str();
}

} // namespace openrai
