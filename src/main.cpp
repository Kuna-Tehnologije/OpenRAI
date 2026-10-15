#include <openrai/core.h>
#include <openrai/config.h>
#include <openrai/gpu_runtime.h>
#include <openrai/training_demo.h>
#include <openrai/log.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#ifdef OPENRAI_CUDA_ENABLED
static constexpr bool kCudaCompiled = true;
#else
static constexpr bool kCudaCompiled = false;
#endif

static void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --console          interactive text in / answer out (engine logs go to openrai.log only)\n"
        << "  --ask TEXT         answer one question and exit\n"
        << "  --gpu-selftest     prove that per-node CUDA streams overlap, then exit\n"
        << "  --train-demo       train N test experts on the same ~10k-word data and report, then exit\n"
        << "  --train-iters N    iterations for --train-demo and for expert training at start (env OPENRAI_TRAIN_ITERS)\n"
        << "  --train-nodes N    test experts in --train-demo (default 3)\n"
        << "  --train-corpus F   use your own text file (one sentence per line) in --train-demo\n"
        << "  --train-domain D   synthetic corpus domain for --train-demo (default telemetry)\n"
        << "  --retrain          ignore cached models in data/models and train again (env OPENRAI_RETRAIN)\n"
        << "  --train-big-demo   confirm learning at scale: ~500k-line math + ~500k-word language model,\n"
        << "                     10000 iterations, answers held-out math questions by generation only\n"
        << "  --big-math-lines N / --big-lang-words N / --big-iters N / --big-batch N   override the above\n"
        << "  --seconds N        run time in seconds          (env OPENRAI_RUN_SECONDS)\n"
        << "  --hz F             primary loop frequency       (env OPENRAI_TICK_HZ, default 7.82)\n"
        << "  --experts N        number of expert nodes       (env OPENRAI_EXPERTS)\n"
        << "  --concurrent N     max experts fired per tick   (env OPENRAI_MAX_CONCURRENT)\n"
        << "  --profile P        full | lite                  (env OPENRAI_PROFILE)\n"
        << "  --inject TEXT      inject TEXT as first input\n"
        << "  --quiet            no engine log on the console (file log only)\n"
        << "  --verbose          per-node per-tick logging    (env OPENRAI_VERBOSE)\n"
        << "  --help\n";
}

static void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static void print_answer(const openrai::Answer& a, bool show_thinking) {
    std::cout << std::fixed << std::setprecision(2)
              << "[" << a.expert
              << (a.solved ? "  SOLVED" : "")
              << (a.recalled ? "  RECALLED#" + std::to_string(a.experience_id) : "")
              << (a.language ? "  LM" : "")
              << "  confidence=" << a.confidence
              << "  utility=" << a.utility
              << "  tick=" << a.tick
              << "  " << std::setprecision(0) << a.latency_ms << "ms]\n"
              << a.answer << "\n";
    if (show_thinking) {
        std::cout << "--- inner thinking ---\n" << a.thinking << "\n";
    }
    std::cout << std::flush;
}

static int run_console(openrai::OpenRAI& engine) {
    bool show_thinking = true;
    std::cout << "OpenRAI console - type a question, Enter to ask.\n"
              << "  /thinking      toggle the inner-thinking trace (" << (show_thinking ? "on" : "off") << ")\n"
              << "  /good  /bad    reward or punish the last answer (the engine learns from it)\n"
              << "  /memory TEXT   similarity search in the vector memory (GPU index when available)\n"
              << "  /learned       recently learned question -> answer pairs\n"
              << "  /status        engine statistics\n"
              << "  /nodes         expert node states\n"
              << "  /log on|off    engine log echo on this console\n"
              << "  /quit          exit\n\n";
    std::string line;
    while (true) {
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        if (line == "/quit" || line == "/exit" || line == "/q") break;
        if (line == "/thinking") { show_thinking = !show_thinking; std::cout << "thinking " << (show_thinking ? "on" : "off") << "\n"; continue; }
        if (line == "/log on")  { openrai::Logger::get().set_console(true);  continue; }
        if (line == "/log off") { openrai::Logger::get().set_console(false); continue; }
        if (line == "/status") {
            auto st = engine.stats();
            std::cout << std::fixed << std::setprecision(2)
                      << "ticks=" << st.ticks << " hz=" << st.measured_hz
                      << " avg_busy_ms=" << st.avg_busy_ms << " overruns=" << st.overruns
                      << " expert_wins=" << st.wins << " primary_fallbacks=" << st.primary_fallbacks
                      << " final_outputs=" << st.final_outputs
                      << " backend=" << st.backend << " cuda_active=" << (st.cuda_active ? "yes" : "no")
                      << " memory_index=" << (st.gpu_memory_index ? "GPU" : "CPU") << " memory_items=" << st.vector_memory
                      << " experiences=" << st.experiences << " recalls=" << st.recalls
                      << " lm=" << st.llm << " lm_answers=" << st.language_answers << "\n";
            continue;
        }
        if (line == "/good" || line == "/bad") {
            const int64_t id = engine.last_experience_id();
            if (id <= 0) { std::cout << "nothing to rate yet\n"; continue; }
            const bool ok = engine.feedback(id, line == "/good" ? 1.f : -1.f);
            std::cout << (ok ? "noted: experience #" + std::to_string(id) + (line == "/good" ? " rewarded" : " punished")
                           : "could not update experience") << "\n";
            continue;
        }
        if (line.rfind("/memory ", 0) == 0) {
            for (const auto& h : engine.search_memory(line.substr(8), 8))
                std::cout << std::fixed << std::setprecision(2) << "  [" << h.second << "] " << h.first << "\n";
            continue;
        }
        if (line == "/learned") {
            auto v = engine.recent_experiences(10);
            if (v.empty()) std::cout << "  (nothing learned yet)\n";
            for (const auto& l : v) std::cout << "  " << l << "\n";
            continue;
        }
        if (line == "/nodes") {
            auto st = engine.stats();
            for (const auto& l : st.node_lines) std::cout << "  " << l << "\n";
            continue;
        }
        auto fut = engine.ask(line);
        if (fut.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
            std::cout << "(no answer within 10 s - is the engine running?)\n";
            continue;
        }
        print_answer(fut.get(), show_thinking);
        std::cout << "\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    // CLI overrides are applied through the environment so that
    // EngineConfig::from_env() stays the single source of truth.
    std::string inject, ask;
    bool console = false, selftest = false, quiet = false, train_demo = false, big_demo = false;
    openrai::TrainingDemoOptions demo;
    openrai::BigTrainOptions big;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* env) { if (i + 1 < argc) set_env(env, argv[++i]); };
        if      (a == "--seconds")      next("OPENRAI_RUN_SECONDS");
        else if (a == "--hz")           next("OPENRAI_TICK_HZ");
        else if (a == "--experts")      next("OPENRAI_EXPERTS");
        else if (a == "--concurrent")   next("OPENRAI_MAX_CONCURRENT");
        else if (a == "--profile")      next("OPENRAI_PROFILE");
        else if (a == "--inject")       { if (i + 1 < argc) inject = argv[++i]; }
        else if (a == "--ask")          { if (i + 1 < argc) ask = argv[++i]; }
        else if (a == "--console")      console = true;
        else if (a == "--gpu-selftest") selftest = true;
        else if (a == "--train-demo")   train_demo = true;
        else if (a == "--train-iters")  { if (i + 1 < argc) { set_env("OPENRAI_TRAIN_ITERS", argv[i + 1]); demo.iterations = static_cast<size_t>(std::atol(argv[++i])); } }
        else if (a == "--train-nodes")  { if (i + 1 < argc) demo.nodes = static_cast<size_t>(std::atol(argv[++i])); }
        else if (a == "--train-corpus") { if (i + 1 < argc) demo.corpus_file = argv[++i]; }
        else if (a == "--train-domain") { if (i + 1 < argc) demo.domain = argv[++i]; }
        else if (a == "--retrain")      set_env("OPENRAI_RETRAIN", "1");
        else if (a == "--train-big-demo") big_demo = true;
        else if (a == "--big-math-lines") { if (i + 1 < argc) big.math_lines = static_cast<size_t>(std::atol(argv[++i])); }
        else if (a == "--big-lang-words") { if (i + 1 < argc) big.language_words = static_cast<size_t>(std::atol(argv[++i])); }
        else if (a == "--big-iters")      { if (i + 1 < argc) big.iterations = static_cast<size_t>(std::atol(argv[++i])); }
        else if (a == "--big-batch")      { if (i + 1 < argc) big.batch = static_cast<size_t>(std::atol(argv[++i])); }
        else if (a == "--quiet")        quiet = true;
        else if (a == "--verbose")      set_env("OPENRAI_VERBOSE", "1");
        else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else { std::cerr << "unknown option " << a << "\n"; usage(argv[0]); return 2; }
    }

    const openrai::EngineConfig cfg = openrai::EngineConfig::from_env(kCudaCompiled);

    if (big_demo) {
        openrai::Logger::get().set_console(false);
        std::cout << "Large-scale confirmation: math_lines=" << big.math_lines
                  << " lang_words=" << big.language_words << " iterations=" << big.iterations
                  << " batch=" << big.batch << " ...\n";
        auto rep = openrai::run_big_training_demo(big);
        std::cout << rep.describe();
        LOG("[TRAIN-BIG-DEMO] " << (rep.passed ? "PASS" : "FAIL") << " " << rep.seconds << "s"
            << " math_acc=" << rep.eval_correct << "/" << rep.eval_total);
        return rep.passed ? 0 : 1;
    }

    if (train_demo) {
        openrai::Logger::get().set_console(false);
        std::cout << "Training " << demo.nodes << " test experts, " << demo.iterations << " iterations each ...\n";
        auto rep = openrai::run_training_demo(demo);
        std::cout << rep.describe();
        LOG("[TRAIN-DEMO] " << (rep.passed ? "PASS" : "FAIL") << " " << rep.seconds << "s");
        return rep.passed ? 0 : 1;
    }

    if (selftest) {
        openrai::Logger::get().set_console(!quiet);
        std::cout << "Running GPU concurrency self-test (" << (kCudaCompiled ? "CUDA build" : "CPU build") << ")...\n";
        auto rep = openrai::run_gpu_concurrency_selftest(cfg.expert_arch, std::max<size_t>(2, cfg.expert_count),
                                                         cfg.node_max_batch, 20);
        std::cout << rep.describe();
        LOG("[SELFTEST] " << rep.verdict);
        return rep.cuda && rep.speedup_concurrent < 1.25 ? 1 : 0;
    }

    if (console || !ask.empty() || quiet) openrai::Logger::get().set_console(false);

    std::cout << "=========================================\n"
              << "  OpenRAI  |  raw / real / reimagined AI\n"
              << "  primary node + " << cfg.expert_count << " expert nodes\n"
              << "  loop " << cfg.tick_hz << " Hz  (" << (cfg.tick_period().count() / 1e6) << " ms/tick)\n"
              << "  profile=" << cfg.profile << "  backend=" << (kCudaCompiled ? "CUDA streams" : "CPU threads") << "\n"
              << "  log -> openrai.log\n"
              << "=========================================\n\n";

    auto engine = std::make_unique<openrai::OpenRAI>(cfg);
    if (!inject.empty()) engine->inject_input(inject);
    engine->start();

    if (!ask.empty()) {
        auto fut = engine->ask(ask);
        if (fut.wait_for(std::chrono::seconds(10)) == std::future_status::ready) print_answer(fut.get(), true);
        else std::cout << "(no answer within 10 s)\n";
        engine->stop();
        return 0;
    }

    if (console) {
        run_console(*engine);
        engine->stop();
        return 0;
    }

    LOG("[Main] Engine running for " << cfg.run_seconds << " seconds at " << cfg.tick_hz << " Hz ...");
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(cfg.run_seconds);
    while (std::chrono::steady_clock::now() < until && engine->is_running())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    engine->stop();
    const auto st = engine->stats();
    engine.reset();

    std::cout << "\n=========================================\n"
              << "  OpenRAI Shutdown Complete\n"
              << "  ticks=" << st.ticks
              << "  measured_hz=" << st.measured_hz
              << "  avg_busy_ms=" << st.avg_busy_ms
              << "  overruns=" << st.overruns
              << "  expert_wins=" << st.wins
              << "  final_outputs=" << st.final_outputs << "\n"
              << "  Full log saved to: openrai.log\n"
              << "=========================================\n";

    return 0;
}
