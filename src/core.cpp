#include <openrai/core.h>
#include <openrai/thread_pool.h>
#include <openrai/memory.h>
#include <openrai/gpu_queue.h>
#include <openrai/reasoning.h>
#include <openrai/gpu_runtime.h>
#include <openrai/semantic.h>
#include <openrai/expert_node.h>
#include <openrai/primary_node.h>
#include <openrai/scheduler.h>
#include <openrai/skills.h>
#include <openrai/experience_store.h>
#include <openrai/lm_adapter.h>
#include <openrai/expert_model.h>
#include <openrai/corpus.h>
#include <openrai/log.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>

namespace openrai {

static constexpr int NODE_COUNT = 20;          // input-pool generator only
static constexpr int TOP_OUTPUT = 2;
static constexpr const char* STEP_INPUT_POOL_PATH = "data/step_input_pool.csv";
static constexpr size_t STEP_INPUT_POOL_SIZE = 100000;

static std::string generate_step_input(size_t index) {
    const int node = static_cast<int>(index % NODE_COUNT);
    const int port = 9000 + static_cast<int>(index % 1000);
    const int memory = 55 + static_cast<int>(index % 41);
    const int latency = 10 + static_cast<int>(index % 120);
    const int batch = 64 + static_cast<int>((index * 7) % 2048);
    const float threshold = 0.30f + static_cast<float>(index % 60) / 100.0f;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    switch (index % 12) {
    case 0:
        ss << "user: summarize current system state for window " << (index % 128);
        break;
    case 1:
        ss << "sensor: temperature spike detected at node " << node << " value " << (65 + (index % 30)) << "C";
        break;
    case 2:
        ss << "alert: memory usage exceeded " << memory << " percent on node " << node;
        break;
    case 3:
        ss << "query: summarize last " << (5 + (index % 50)) << " reasoning outputs";
        break;
    case 4:
        ss << "event: new data stream connected on port " << port;
        break;
    case 5:
        ss << "command: increase thread priority on node " << node;
        break;
    case 6:
        ss << "trigger: anomaly score above threshold " << threshold;
        break;
    case 7:
        ss << "probe: latency measurement on gpu queue exceeded " << latency << "ms";
        break;
    case 8:
        ss << "feed: new labelled sample batch received size " << batch;
        break;
    case 9:
        ss << "goal: prioritize stability over throughput for cluster " << static_cast<char>('A' + (index % 6));
        break;
    case 10:
        ss << "vector: similarity search over prior failures for node " << node;
        break;
    default:
        ss << "reflect: compare current output with earlier memory segment " << (index % 256);
        break;
    }

    return ss.str();
}

static std::vector<std::string> load_step_input_pool_from_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return {};

    std::vector<std::string> pool;
    std::string line;
    bool is_first_line = true;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (is_first_line && line == "input") {
            is_first_line = false;
            continue;
        }
        is_first_line = false;

        std::stringstream row(line);
        std::string value;
        if (row >> std::quoted(value))
            pool.push_back(value);
        else
            pool.push_back(line);
    }

    return pool;
}

static bool write_step_input_pool_csv(const std::string& path, size_t count) {
    try {
        std::filesystem::path csv_path(path);
        if (csv_path.has_parent_path()) {
            std::filesystem::create_directories(csv_path.parent_path());
        }

        std::ofstream file(path, std::ios::trunc);
        if (!file.is_open()) return false;

        file << "input\n";
        for (size_t i = 0; i < count; ++i) {
            file << std::quoted(generate_step_input(i)) << '\n';
        }

        return true;
    } catch (...) {
        return false;
    }
}

static std::vector<std::string> load_or_create_step_input_pool() {
    auto pool = load_step_input_pool_from_csv(STEP_INPUT_POOL_PATH);
    if (pool.size() >= STEP_INPUT_POOL_SIZE) return pool;

    if (write_step_input_pool_csv(STEP_INPUT_POOL_PATH, STEP_INPUT_POOL_SIZE)) {
        pool = load_step_input_pool_from_csv(STEP_INPUT_POOL_PATH);
        if (!pool.empty()) return pool;
    }

    pool.clear();
    pool.reserve(256);
    for (size_t i = 0; i < 256; ++i) {
        pool.push_back(generate_step_input(i));
    }
    return pool;
}

static const std::vector<std::string> STEP_INPUT_POOL = load_or_create_step_input_pool();

#ifdef OPENRAI_CUDA_ENABLED
static constexpr bool kCudaCompiled = true;
#else
static constexpr bool kCudaCompiled = false;
#endif

OpenRAI::OpenRAI() : OpenRAI(EngineConfig::from_env(kCudaCompiled)) {}

OpenRAI::OpenRAI(const EngineConfig& cfg)
    : config_(cfg),
      thread_pool_(std::make_unique<ThreadPool>(std::max<size_t>(2, std::min<size_t>(cfg.expert_count, 16)))),
      memory_system_(std::make_unique<MemorySystem>()),
      gpu_queue_(std::make_unique<GPUJobQueue>(128)),
      reasoning_module_(std::make_unique<ReasoningModule>(memory_system_.get())),
      tokenizer_(std::make_unique<Tokenizer>()),
      embedding_layer_(std::make_unique<EmbeddingLayer>(32)),
      vector_memory_(std::make_unique<VectorMemory>(32, cfg.memory_capacity, cfg.gpu_memory_index)),
      knowledge_graph_(std::make_unique<KnowledgeGraph>()),
      goal_system_(std::make_unique<GoalSystem>()),
      self_reflection_(std::make_unique<SelfReflection>()),
      node_signal_(std::make_unique<NodeSignal>())
{
    LOG("[OpenRAI] Config: " << config_.describe());

    vector_memory_->set_persistence_path(config_.memory_path);
    if (vector_memory_->load_from_disk(config_.memory_path)) {
        LOG("[VectorMemory] Loaded persistent memory from " << config_.memory_path
            << " size=" << vector_memory_->size());
    } else {
        LOG("[VectorMemory] No persistent memory found, starting fresh");
    }

    LOG("[InputPool] CSV=" << STEP_INPUT_POOL_PATH << " count=" << STEP_INPUT_POOL.size());

    // ---- Primary node: its own model on its own top-priority stream -------
    {
        // The full 4B network needs ~16 GB of device memory.  If the GPU
        // cannot hold it, running it on the CPU would take seconds per
        // tick and wreck the 7.82 Hz loop - downgrade loudly instead.
        size_t free_b = 0, total_b = 0;
        const bool have_gpu = gpu_memory_info(free_b, total_b);
        const size_t need = config_.primary_arch.weight_bytes() + (64u << 20);
        if (config_.primary_arch.param_count() > 100000000ull &&
            (!have_gpu || need > free_b)) {
            LOG("[OpenRAI] WARNING primary net " << config_.primary_arch.name
                << " needs " << (need >> 20) << "MB of device memory but "
                << (have_gpu ? std::to_string(free_b >> 20) + "MB are free" : "no CUDA device is available")
                << " -> downgrading primary to primary_lite to keep the "
                << config_.tick_hz << " Hz loop (set OPENRAI_PRIMARY_ARCH to override)");
            config_.primary_arch = NetArch::primary_lite();
            config_.profile      = "lite";
        }
    }
    primary_net_ = std::make_shared<GPUNeuralNet>(config_.primary_arch, 42u);
    primary_net_->print_summary();
    primary_ = std::make_unique<PrimaryNode>(config_, primary_net_, goal_system_.get());

    // ---- Expert nodes: own thread + own stream + own model each -----------
    std::shared_ptr<GPUNeuralNet> shared_expert_net;
    if (config_.shared_expert_weights)
        shared_expert_net = std::make_shared<GPUNeuralNet>(config_.expert_arch, 7u, true);

    nodes_.reserve(config_.expert_count);
    for (size_t i = 0; i < config_.expert_count; ++i) {
        ExpertDomain domain = make_expert_domain(i);
        std::shared_ptr<GPUNeuralNet> net = shared_expert_net
            ? shared_expert_net
            : std::make_shared<GPUNeuralNet>(config_.expert_arch, domain.seed, true);
        const std::string domain_name = domain.name;
        nodes_.push_back(std::make_unique<ExpertNode>(
            static_cast<int>(i), std::move(domain), net,
            config_.node_max_batch, 50, node_signal_.get(), config_.verbose));
        if (domain_name == "math") nodes_.back()->set_skill(make_math_skill());
    }

    // ---- Expert knowledge: every expert trains/loads its own models ------
    train_expert_knowledge();

    // ---- Learning store (SQLite) -----------------------------------------
    store_ = std::make_unique<ExperienceStore>();
    if (config_.persist_learning && store_->open(config_.store_path)) {
        LOG("[Store] opened " << config_.store_path << " experiences=" << store_->count());
        load_learning_state();
    } else {
        LOG("[Store] learning store disabled");
        store_.reset();
    }

    // ---- Memory expert: GPU vector search + recall of learned answers ----
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i]->domain().name == "memory") {
            memory_node_ = static_cast<int>(i);
            nodes_[i]->set_skill(make_memory_skill(vector_memory_.get(), store_.get(),
                                                   tokenizer_.get(), embedding_layer_.get()));
        }
    }

    // ---- Language expert (slow): external LM phrases the final answer -----
    {
        LMConfig lc;
        lc.backend     = config_.llm_backend == "auto" ? "http" : config_.llm_backend;
        lc.url         = config_.llm_url;
        lc.api         = config_.llm_api;
        lc.model       = config_.llm_model;
        lc.timeout_s   = config_.llm_timeout_s;
        lc.max_tokens  = config_.llm_max_tokens;
        lm_ = std::make_unique<LMAdapter>(lc);
        if (lm_->initialize()) {
            for (size_t i = 0; i < nodes_.size(); ++i) {
                if (nodes_[i]->domain().name == "language") {
                    language_node_ = static_cast<int>(i);
                    nodes_[i]->set_skill(make_language_skill(lm_.get()));
                }
            }
            LOG("[LM] language expert " << (language_node_ >= 0 ? nodes_[language_node_]->name() : "(none)")
                << " -> " << lm_->model_info());
        } else {
            LOG("[LM] no language model - answers come straight from the expert nodes");
        }
    }

    scheduler_ = std::make_unique<TickScheduler>(config_.tick_period(), config_.compute_budget());

    LOG("[OpenRAI] Engine initialized"
        << " experts=" << nodes_.size()
        << " memory_index=" << (vector_memory_->gpu_index_active() ? "GPU" : "CPU")
        << " memory_items=" << vector_memory_->size()
        << " primary_net=" << primary_net_->arch().name
        << " backend=" << gpu_backend_name()
        << " cuda_active=" << (primary_net_->is_cuda_active() ? "yes" : "no")
        << " tick_hz=" << config_.tick_hz
        << " gpu_slots=128"
        << " semantic=on");
}

OpenRAI::~OpenRAI() {
    stop();
    save_learning_state();
    // Stop every expert (joins threads, releases streams) before the
    // networks they reference go away.
    for (auto& n : nodes_) n->stop(stats_.ticks);
    nodes_.clear();
    primary_.reset();

    if (vector_memory_) {
        if (vector_memory_->save_to_disk(config_.memory_path)) {
            LOG("[VectorMemory] Saved persistent memory to " << config_.memory_path
                << " size=" << vector_memory_->size());
        } else {
            LOG("[VectorMemory] Failed to save persistent memory");
        }
    }
    LOG("[OpenRAI] Engine destroyed");
}

void OpenRAI::start() {
    if (is_running_.exchange(true)) return;
    LOG("[OpenRAI] Starting main loop @ " << config_.tick_hz << " Hz ...");
    loop_thread_ = std::thread([this] { main_loop(); });
}

void OpenRAI::stop() {
    is_running_ = false;
    if (loop_thread_.joinable() && loop_thread_.get_id() != std::this_thread::get_id())
        loop_thread_.join();
    fail_pending("engine stopped");
}

void OpenRAI::fail_pending(const std::string& why) {
    std::lock_guard<std::mutex> lk(pending_mutex_);
    while (!pending_inputs_.empty()) {
        auto p = std::move(pending_inputs_.front());
        pending_inputs_.pop();
        if (p.promise) {
            Answer a;
            a.question = p.text;
            a.answer   = "(no answer: " + why + ")";
            a.thinking = why;
            a.expert   = "-";
            p.promise->set_value(std::move(a));
        }
    }
}

void OpenRAI::inject_input(const std::string& input) {
    {
        std::lock_guard<std::mutex> lk(pending_mutex_);
        pending_inputs_.push(PendingInput{ input, nullptr, std::chrono::steady_clock::now() });
    }
    LOG("[INJECT] \"" << input << "\"");
}

std::future<Answer> OpenRAI::ask(const std::string& question) {
    auto promise = std::make_shared<std::promise<Answer>>();
    std::future<Answer> fut = promise->get_future();
    {
        std::lock_guard<std::mutex> lk(pending_mutex_);
        pending_inputs_.push(PendingInput{ question, promise, std::chrono::steady_clock::now() });
    }
    LOG("[ASK] \"" << question << "\"");
    if (!is_running_) fail_pending("engine is not running");
    return fut;
}

OpenRAI::Stats OpenRAI::stats() const {
    std::lock_guard<std::mutex> lk(stats_mutex_);
    return stats_;
}

// ---- learning -----------------------------------------------------------------
bool OpenRAI::feedback(int64_t experience_id, float delta) {
    if (!store_ || experience_id <= 0) return false;
    Experience e;
    if (!store_->get(experience_id, e)) return false;
    if (!store_->reward(experience_id, delta)) return false;
    // The expert that produced it learns too
    for (auto& n : nodes_) {
        if (n->name() == e.expert) {
            n->update_stats([&](NodeStats& st) {
                st.trust = std::clamp(st.trust + 0.08f * delta, 0.f, 1.f);
            });
        }
    }
    LOG("[FEEDBACK] experience #" << experience_id << " " << (delta >= 0 ? "+" : "") << delta
        << " (expert " << e.expert << ")");
    return true;
}

int64_t OpenRAI::last_experience_id() const { return last_experience_id_.load(); }

std::vector<std::pair<std::string, float>> OpenRAI::search_memory(const std::string& text, size_t k) const {
    std::vector<std::pair<std::string, float>> out;
    auto hits = vector_memory_->retrieve_context(text, *tokenizer_, *embedding_layer_, k);
    for (const auto& h : hits) out.emplace_back(h.tag + " | " + h.text, h.similarity);
    return out;
}

std::vector<std::string> OpenRAI::recent_experiences(size_t n) const {
    std::vector<std::string> out;
    if (!store_) return out;
    for (const auto& e : store_->recent(n)) {
        std::ostringstream ss;
        ss << "#" << e.id << " [" << e.expert << " conf=" << std::fixed << std::setprecision(2) << e.confidence
           << " uses=" << e.uses << " reward=" << e.reward << "] \"" << e.question << "\" -> \"" << e.answer << "\"";
        out.push_back(ss.str());
    }
    return out;
}

void OpenRAI::load_learning_state() {
    if (!store_) return;
    size_t restored = 0;
    for (auto& n : nodes_) {
        NodeStats st;
        int prio = 50;
        if (store_->load_node_stats(n->name(), st, prio)) {
            n->update_stats([&](NodeStats& x) {
                x.trust = st.trust;
                x.fired = st.fired; x.completed = st.completed; x.wins = st.wins;
                x.deadline_misses = st.deadline_misses; x.kills = st.kills;
                x.suspends = st.suspends; x.restarts = st.restarts;
                x.ema_latency_ms = st.ema_latency_ms;
            });
            n->set_priority(prio, 0);
            ++restored;
        }
    }
    auto goals = store_->load_goals();
    for (const auto& g : goals) goal_system_->set_score(g.name, g.score);
    auto edges = store_->load_graph();
    knowledge_graph_->import_edges(edges);
    LOG("[Store] restored node_stats=" << restored << " goals=" << goals.size()
        << " graph_edges=" << edges.size());
}

void OpenRAI::save_learning_state() {
    if (!store_) return;
    for (auto& n : nodes_) store_->save_node_stats(n->name(), n->stats(), n->priority());
    store_->save_goals(goal_system_->top_goals(16));
    store_->save_graph(knowledge_graph_->export_edges());
    if (vector_memory_) vector_memory_->save_to_disk(config_.memory_path);
    store_->set_kv("last_save_tick", std::to_string(stats_.ticks));
}

// ---- expert knowledge -------------------------------------------------------------
// Each expert gets (a) an ExpertModel trained on its domain corpus and (b) its
// MLP scorer trained as "is this my field?" classifier.  Models are cached in
// model_dir; every expert trains on its own thread.
void OpenRAI::train_expert_knowledge() {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t N = nodes_.size();

    // corpora first (needed as negatives for the classifiers)
    std::vector<std::vector<CorpusLine>> corpora(N);
    for (size_t i = 0; i < N; ++i)
        corpora[i] = load_or_generate_corpus(config_.corpus_dir, nodes_[i]->domain().name,
                                             config_.corpus_words, nodes_[i]->domain().seed);

    std::vector<std::string> lines(N);
    std::vector<std::thread> workers;
    for (size_t i = 0; i < N; ++i) {
        workers.emplace_back([&, i] {
            ExpertNode& node = *nodes_[i];
            const std::string& dom = node.domain().name;
            std::ostringstream line;
            line << node.name() << ": corpus " << corpora[i].size() << " lines / "
                 << corpus_word_count(corpora[i]) << " words";

            // (a) expert model
            ExpertModel::Config mc;
            mc.vocab  = config_.model_vocab;
            mc.hidden = config_.model_hidden;
            mc.seed   = node.domain().seed;
            auto model = std::make_shared<ExpertModel>(dom, mc);
            const std::string mpath = config_.model_dir + "/" + dom + ".model";
            if (!config_.retrain && model->load(mpath)) {
                line << " | model loaded (" << model->train_iterations() << " it, vocab " << model->vocab_size() << ")";
            } else if (config_.train_iterations > 0) {
                model->build_vocab(corpus_texts(corpora[i]));
                auto rep = model->train(corpus_texts(corpora[i]), config_.train_iterations, config_.train_batch);
                model->save(mpath);
                line << " | model trained: " << rep.describe();
            } else {
                line << " | model untrained (OPENRAI_TRAIN_ITERS=0)";
            }
            node.set_model(model);

            // (b) MLP scorer as domain classifier (output 0 = my field,
            //     1 = importance label, 2 = my field & important, 3 = importance)
            if (config_.train_mlp && node.net().is_trainable()) {
                auto& net = const_cast<GPUNeuralNet&>(node.net());
                const std::string npath = config_.model_dir + "/" + dom + ".net";
                if (!config_.retrain && net.load_weights(npath)) {
                    line << " | scorer loaded (" << net.train_steps() << " steps)";
                } else if (config_.train_iterations > 0) {
                    std::mt19937 rng(node.domain().seed * 31u);
                    const size_t B = std::min<size_t>(config_.train_batch, 64);
                    std::vector<float> in(B * GPUNeuralNet::INPUT_DIM), tg(B * GPUNeuralNet::OUTPUT_DIM);
                    float first = 0.f, last = 0.f;
                    for (size_t it = 0; it < config_.train_iterations; ++it) {
                        for (size_t b = 0; b < B; ++b) {
                            const bool positive = (b % 2 == 0);
                            const size_t src = positive ? i : (rng() % N);
                            const auto& c = corpora[src];
                            const CorpusLine& cl = c[rng() % c.size()];
                            encode_input_v2(cl.text, in.data() + b * GPUNeuralNet::INPUT_DIM, GPUNeuralNet::INPUT_DIM);
                            const float mine = (src == i) ? 1.f : 0.f;
                            float* t = tg.data() + b * GPUNeuralNet::OUTPUT_DIM;
                            t[0] = mine; t[1] = cl.importance; t[2] = mine * (cl.importance >= 0.5f ? 1.f : 0.f); t[3] = cl.importance;
                        }
                        const float loss = net.train_batch(in.data(), tg.data(), B, 0.02f);
                        if (it == 0) first = loss;
                        last = loss;
                    }
                    net.sync_weights_to_device();
                    net.save_weights(npath);
                    line << " | scorer trained: loss " << std::fixed << std::setprecision(3) << first << " -> " << last;
                }
            }
            lines[i] = line.str();
        });
    }
    for (auto& w : workers) w.join();

    knowledge_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    model_lines_ = lines;
    for (const auto& l : lines) LOG("[KNOWLEDGE] " << l);
    LOG("[KNOWLEDGE] " << N << " experts ready in " << std::fixed << std::setprecision(2) << knowledge_seconds_ << "s"
        << " (corpus_dir=" << config_.corpus_dir << " model_dir=" << config_.model_dir << ")");
}

// ---- compatibility API (chat GUIs of the language-model branch) ----------------
void OpenRAI::set_output_callback(OutputCallback cb) {
    std::lock_guard<std::mutex> lk(stats_mutex_);
    output_callback_ = std::move(cb);
}

std::string OpenRAI::generate_response(const std::string& prompt) {
    auto fut = ask(prompt);
    if (fut.wait_for(std::chrono::seconds(config_.llm_timeout_s + 5)) != std::future_status::ready)
        return "(no answer within the timeout)";
    return fut.get().answer;
}

int OpenRAI::get_iteration_count() const {
    std::lock_guard<std::mutex> lk(stats_mutex_);
    return static_cast<int>(stats_.ticks);
}

std::string OpenRAI::get_latest_output() const {
    std::lock_guard<std::mutex> lk(stats_mutex_);
    return latest_output_;
}

float OpenRAI::get_latest_confidence() const { return latest_confidence_.load(); }

// Final delivery of an answer for an external input: learn it, notify.
void OpenRAI::deliver(const PendingInput& in, Answer answer) {
    answer.latency_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - in.submitted).count();

    // Learn: store the experience and index the question so the memory
    // expert can recall the answer next time.
    // Only real answers are learned: something solved by an expert or
    // phrased by the language model - never the "could not solve" notice.
    if (store_ && !answer.answer.empty() && !answer.recalled && (answer.solved || answer.language)) {
        Experience e;
        e.question   = answer.question;
        e.answer     = answer.answer;
        e.expert     = answer.expert;
        e.confidence = answer.confidence;
        e.utility    = answer.utility;
        e.solved     = answer.solved;
        e.immediate  = answer.immediate.empty() ? answer.answer : answer.immediate;
        e.embedding  = embedding_layer_->embed(tokenizer_->tokenize(answer.question));
        const int64_t id = store_->add(e);
        if (id > 0) {
            answer.experience_id = id;
            vector_memory_->store(answer.question, e.embedding, "experience:" + std::to_string(id));
            LOG("[LEARN] experience #" << id << " stored: \"" << answer.question << "\" -> \""
                << answer.answer.substr(0, 80) << (answer.answer.size() > 80 ? "..." : "") << "\" (" << answer.expert << ")");
        }
    }
    if (answer.experience_id > 0) last_experience_id_ = answer.experience_id;

    OutputCallback cb;
    {
        std::lock_guard<std::mutex> lk(stats_mutex_);
        latest_output_ = answer.answer;
        latest_confidence_ = answer.confidence;
        cb = output_callback_;
    }
    if (cb) {
        try { cb(answer.answer); } catch (...) {}
    }
    if (in.promise) in.promise->set_value(std::move(answer));
}

// -----------------------------------------------------------------------
// main_loop - the 7.82 Hz heartbeat
//
//   PHASE 1  input          one injected input (external or pool)
//   PHASE 2  judge          primary scores it on its own stream
//   PHASE 3  plan           primary decides which experts fire and why
//   PHASE 4  dispatch       fire experts on their own threads/streams
//   PHASE 5  collect        wait for outcomes until the compute deadline
//   PHASE 6  arbitrate      primary picks whose outcome matters most
//   PHASE 7  memory         winner -> current/short-term/vector/graph/goals
//   PHASE 8  reflect        reflection loop on the winner -> FINAL OUTPUT
//   PHASE 9  behaviour      start/stop/suspend/kill/respawn/prioritise
//   sleep until the next tick boundary
// -----------------------------------------------------------------------
void OpenRAI::main_loop() {
    LOG("=== OPENRAI MAIN LOOP STARTED ==="
        << " tick_hz=" << config_.tick_hz
        << " period_ms=" << std::fixed << std::setprecision(3) << (config_.tick_period().count() / 1e6)
        << " compute_budget_ms=" << (config_.compute_budget().count() / 1e6)
        << " experts=" << nodes_.size()
        << " primary=" << primary_net_->arch().name
        << " backend=" << gpu_backend_name()
        << " cuda=" << (primary_net_->is_cuda_active() ? "ON" : "OFF"));

    // Cold start of every expert: thread + stream context
    for (auto& n : nodes_) n->start(0);

    scheduler_->start();

    uint64_t total_outputs = 0;
    uint64_t total_wins    = 0;
    uint64_t primary_fallbacks = 0;
    size_t   step_input_index = 0;
    std::vector<NodeOutcome> outcomes;
    std::vector<BehaviourDecision> pending_behaviour;
    std::chrono::nanoseconds last_overrun{0};

    while (is_running_) {
        const auto tick = scheduler_->begin_tick();
        const uint64_t t = tick.index;
        outcomes.clear();

        LOG("\n===== TICK " << t << " =====");

        // ---- PHASE 1: input ------------------------------------------------
        std::string input;
        bool external = false;
        PendingInput current;
        {
            std::lock_guard<std::mutex> lk(pending_mutex_);
            if (!pending_inputs_.empty()) {
                current = std::move(pending_inputs_.front());
                pending_inputs_.pop();
                input = current.text;
                external = true;
            }
        }
        if (!external && !STEP_INPUT_POOL.empty()) {
            input = STEP_INPUT_POOL[step_input_index % STEP_INPUT_POOL.size()];
            ++step_input_index;
        }
        LOG("[PHASE 1/9] input" << (external ? " (external)" : " (pool)") << ": \"" << input << "\"");

        // ---- PHASE 2: judge --------------------------------------------------
        const auto j0 = std::chrono::steady_clock::now();
        const NNResult judgement = primary_->judge(input);
        const auto judge_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - j0).count();
        LOG("[PHASE 2/9] primary judgement"
            << std::fixed << std::setprecision(3)
            << " rel=" << judgement.relevance
            << " imp=" << judgement.importance
            << " out=" << (judgement.should_out ? "yes" : "no")
            << " prio=" << judgement.priority
            << " time=" << judge_us << "us"
            << " stream=" << (primary_->context() && primary_->context()->is_cuda() ? "cuda" : "cpu"));

        // ---- PHASE 3: plan ---------------------------------------------------
        FiringPlan plan = primary_->plan(t, input, judgement, nodes_, tick.compute_deadline, external);
        LOG("[PHASE 3/9] " << plan.summary());

        // ---- PHASE 4: dispatch ---------------------------------------------------
        // Nodes that answered late in an earlier tick may still hold a stale
        // outcome; drop it before firing anything so a fresh result can
        // never be confused with an old one.  Slow experts (language model)
        // are different: their outcome is *expected* to arrive in a later
        // tick - it is kept as a deferred outcome.
        std::vector<NodeOutcome> deferred;
        for (auto& n : nodes_) {
            NodeOutcome o;
            if (!n->try_collect(o)) continue;
            if (n->is_slow()) {
                deferred.push_back(std::move(o));
                continue;
            }
            LOG("[PHASE 4/9] discarded stale outcome from " << o.node_name
                << " (tick " << o.tick << ", " << o.latency.count() / 1000 << "ms"
                << (o.cancelled ? ", cancelled" : "") << ")");
        }

        std::vector<ExpertNode*> fired;
        for (const auto& d : plan.decisions) {
            if (!d.fire) continue;
            ExpertNode* node = nodes_[static_cast<size_t>(d.node_id)].get();
            FireRequest req;
            req.tick     = t;
            req.inputs   = { input };
            req.affinity = d.affinity;
            req.deadline = tick.compute_deadline;
            req.reason   = d.reason;
            if (node->fire(std::move(req))) {
                fired.push_back(node);
                GPUJob job;
                job.type       = "expert:" + node->domain().name;
                job.input_data = input;
                job.priority   = node->priority();
                gpu_queue_->enqueue_job(job);
            }
        }
        LOG("[PHASE 4/9] dispatched " << fired.size() << " experts on their own streams"
            << " (deadline in " << std::chrono::duration_cast<std::chrono::milliseconds>(
                   tick.compute_deadline - std::chrono::steady_clock::now()).count() << "ms)");

        // ---- PHASE 5: collect until the deadline ------------------------------
        {
            std::vector<bool> got(fired.size(), false);
            size_t remaining = fired.size();
            while (remaining > 0) {
                const uint64_t seen = node_signal_->completions.load();
                for (size_t i = 0; i < fired.size(); ++i) {
                    if (got[i]) continue;
                    NodeOutcome o;
                    if (fired[i]->try_collect(o)) {
                        if (o.tick != t) {
                            LOG("[PHASE 5/9] discarded stale outcome from " << o.node_name
                                << " (tick " << o.tick << ")");
                            continue;
                        }
                        got[i] = true;
                        --remaining;
                        outcomes.push_back(std::move(o));
                    }
                }
                if (remaining == 0) break;
                const auto now = std::chrono::steady_clock::now();
                if (now >= tick.compute_deadline) break;
                std::unique_lock<std::mutex> lk(node_signal_->mutex);
                node_signal_->cv.wait_until(lk, tick.compute_deadline,
                    [&] { return node_signal_->completions.load() != seen; });
            }
            for (size_t i = 0; i < fired.size(); ++i) {
                if (got[i]) continue;
                primary_->note_deadline_miss(*fired[i], t);
                LOG("[PHASE 5/9] " << fired[i]->name() << " missed the deadline (still "
                    << node_state_name(fired[i]->state()) << ")");
            }
            LOG("[PHASE 5/9] collected " << outcomes.size() << "/" << fired.size()
                << " outcomes, " << (fired.size() - outcomes.size()) << " late");
        }

        // ---- PHASE 6: arbitrate ------------------------------------------------
        Arbitration arb = primary_->arbitrate(t, outcomes, plan, nodes_);
        LOG("[PHASE 6/9] " << arb.summary());
        if (arb.has_winner) {
            ++total_wins;
        } else {
            // Liveness: no expert delivered in time, so the primary's own
            // judgement becomes the outcome of this tick (clearly labelled).
            NodeOutcome self;
            self.node_id     = -1;
            self.node_name   = "PRIMARY";
            self.domain      = "executive";
            self.goal        = plan.focus_goal;
            self.tick        = t;
            self.inputs      = { input };
            self.results     = { judgement };
            self.best        = judgement;
            self.best_index  = 0;
            self.affinity    = 0.f;
            self.utility     = 0.5f * judgement.importance + 0.25f * judgement.relevance;
            self.latency     = std::chrono::microseconds(judge_us);
            self.fire_reason = "fallback: no expert answered by the deadline";
            arb.has_winner   = true;
            arb.winner       = self;
            arb.ranked.insert(arb.ranked.begin(), self);
            arb.reason      += " -> PRIMARY answers itself (utility " + std::to_string(self.utility).substr(0, 4) + ")";
            ++primary_fallbacks;
            LOG("[PHASE 6/9] no expert outcome in time -> PRIMARY uses its own judgement");
        }

        // ---- PHASE 6b: deferred outcomes of slow experts -------------------------
        for (auto& o : deferred) {
            auto it = std::find_if(pending_answers_.begin(), pending_answers_.end(),
                                   [&](const PendingAnswer& p) { return p.node_id == o.node_id && p.tick == o.tick; });
            if (it == pending_answers_.end()) {
                LOG("[PHASE 6b] deferred outcome from " << o.node_name << " (tick " << o.tick << ") has no waiting question");
                continue;
            }
            Answer final = it->immediate;
            ExpertNode* ln = nodes_[static_cast<size_t>(o.node_id)].get();
            if (o.skill.handled && !o.cancelled) {
                final.immediate  = it->immediate.answer;
                final.answer     = o.skill.answer;
                final.language   = true;
                final.expert     = o.node_name + " <- " + it->immediate.expert;
                final.confidence = std::max(final.confidence, o.skill.confidence);
                final.thinking  += "\n[LANGUAGE] " + o.node_name + " (" + std::to_string(o.latency.count() / 1000) + " ms)\n" + o.skill.thinking;
                ++language_answers_;
                ln->update_stats([&](NodeStats& st) {
                    st.consecutive_misses = 0;
                    ++st.wins;
                    st.trust = std::min(1.f, st.trust + 0.05f);
                });
                LOG("[PHASE 6b] " << o.node_name << " phrased the final answer for tick " << o.tick
                    << " in " << o.latency.count() / 1000 << " ms");
            } else {
                final.thinking += "\n[LANGUAGE] " + o.node_name + " failed: " + o.skill.thinking + " -> keeping the expert answer";
                ln->update_stats([&](NodeStats& st) { st.trust = std::max(0.f, st.trust - 0.05f); });
                LOG("[PHASE 6b] " << o.node_name << " failed (" << o.skill.thinking.substr(0, 120) << ") - expert answer stands");
            }
            LOG("[FINAL OUTPUT] " << final.answer);
            deliver(it->in, std::move(final));
            pending_answers_.erase(it);
        }
        {
            const auto now = std::chrono::steady_clock::now();
            for (auto it = pending_answers_.begin(); it != pending_answers_.end();) {
                if (now < it->timeout) { ++it; continue; }
                ExpertNode* ln = nodes_[static_cast<size_t>(it->node_id)].get();
                primary_->note_deadline_miss(*ln, t);
                Answer final = it->immediate;
                final.thinking += "\n[LANGUAGE] timed out - expert answer stands";
                LOG("[PHASE 6b] " << ln->name() << " timed out for tick " << it->tick << " - expert answer delivered");
                deliver(it->in, std::move(final));
                it = pending_answers_.erase(it);
            }
        }

        // ---- PHASE 7: memory -----------------------------------------------------
        memory_system_->clear_current();
        {
            // Embeddings are pure -> compute them in parallel on the pool,
            // then store sequentially (vector memory is single-writer).
            struct Pending { std::string text; std::string tag; std::vector<float> emb; };
            std::vector<Pending> pend;
            pend.push_back({ input, external ? "question" : "input", {} });
            if (arb.has_winner && arb.winner.node_id >= 0) {
                // Only the winner's view is remembered (the others would just
                // duplicate the input text with a different prefix).
                std::string line = arb.winner.node_name + " :: " + arb.winner.inputs[arb.winner.best_index];
                if (arb.winner.skill.handled) line += " => " + arb.winner.skill.answer;
                pend.push_back({ line, "winner:" + arb.winner.domain, {} });
            }
            for (auto& p : pend) {
                Pending* pp = &p;
                thread_pool_->enqueue([pp, this] {
                    pp->emb = embedding_layer_->embed(tokenizer_->tokenize(pp->text));
                });
            }
            thread_pool_->wait_all();
            for (auto& p : pend) vector_memory_->store(p.text, p.emb, p.tag);

            knowledge_graph_->ingest(input, *tokenizer_);
            goal_system_->decay(0.03f);
            goal_system_->update_from_text(input);
            if (arb.has_winner) goal_system_->update_from_text(arb.winner.domain);

            for (const auto& o : arb.ranked) {
                Output out;
                out.data             = o.inputs[o.best_index];
                out.thread_id        = o.node_id;
                out.source           = o.node_name;
                out.timestamp        = std::chrono::system_clock::now();
                out.relevance_score  = o.best.relevance;
                out.importance_score = o.best.importance;
                out.nn_should_output = o.best.should_out;
                out.utility_score    = o.utility;
                memory_system_->add_current_output(out);
            }
            memory_system_->filter_to_short_term();
        }
        LOG("[PHASE 7/9] memory updated: current=" << memory_system_->get_current_outputs().size()
            << " short_term=" << memory_system_->get_short_term_memory().size()
            << " vector=" << vector_memory_->size()
            << " graph_edges=" << knowledge_graph_->edge_count());

        // ---- PHASE 8: reasoning + reflection on the winner -------------------------
        uint64_t tick_outputs = 0;
        Answer answer;
        answer.question   = input;
        answer.tick       = t;
        answer.expert     = arb.has_winner ? arb.winner.node_name : "-";
        answer.utility    = arb.has_winner ? arb.winner.utility : 0.f;
        if (arb.has_winner) {
            auto short_term = memory_system_->get_short_term_memory();
            std::sort(short_term.begin(), short_term.end(),
                      [](const Output& a, const Output& b) { return a.utility_score > b.utility_score; });
            for (size_t k = 0; k < std::min<size_t>(TOP_OUTPUT, short_term.size()); ++k)
                short_term[k].nn_should_output = true;
            reasoning_module_->analyze(short_term);

            const std::string& candidate = arb.winner.inputs[arb.winner.best_index];
            auto context    = vector_memory_->retrieve_context(candidate, *tokenizer_, *embedding_layer_, 4);
            auto reflection = self_reflection_->reflect(candidate, context, *knowledge_graph_, *goal_system_);

            std::ostringstream challenge;
            challenge << std::fixed << std::setprecision(3)
                      << "[REFLECTION] " << arb.winner.node_name << " -> " << candidate
                      << "  conf=" << reflection.confidence
                      << "  final=" << (reflection.should_output ? "YES" : "no");
            if (!reflection.counterpoints.empty()) {
                challenge << "  counterpoints=";
                for (size_t i = 0; i < reflection.counterpoints.size(); ++i) {
                    if (i) challenge << ",";
                    challenge << reflection.counterpoints[i];
                }
            }
            LOG("[PHASE 8/9] " << challenge.str());

            auto tokens = tokenizer_->tokenize(candidate);
            const std::string actual_value = tokenizer_->get_value_from_tokens(tokens, context, 50);

            if (arb.winner.skill.handled) {
                // A deterministic expert solved it: its answer IS the output.
                answer.answer     = arb.winner.skill.answer;
                answer.confidence = arb.winner.skill.confidence;
                answer.solved     = true;
                if (arb.winner.skill.experience_id > 0) {
                    answer.recalled      = true;
                    answer.experience_id = arb.winner.skill.experience_id;
                    ++recalls_;
                } else {
                    // The memory expert recalled an experience whose raw expert
                    // answer equals what we just computed: it is the same
                    // question - reuse the learned final (phrased) answer
                    // instead of asking the language model again.
                    for (const auto& o : arb.ranked) {
                        if (o.skill.experience_id > 0 && o.skill.recalled_immediate == answer.answer) {
                            answer.immediate     = answer.answer;
                            answer.answer        = o.skill.answer;
                            answer.recalled      = true;
                            answer.experience_id = o.skill.experience_id;
                            ++recalls_;
                            LOG("[RECALL] same question as experience #" << o.skill.experience_id
                                << " - reusing its final answer");
                            break;
                        }
                    }
                }
                LOG("[SOLVED] " << arb.winner.node_name << " answer=" << answer.answer
                    << " confidence=" << std::fixed << std::setprecision(2) << answer.confidence);
                for (const auto& step : arb.winner.skill.steps) LOG("[THINKING] " << step);
                LOG("[FINAL OUTPUT] " << answer.answer << " | expert=" << arb.winner.node_name
                    << " | confidence=" << answer.confidence << " | utility=" << arb.winner.utility);
                ++tick_outputs;
            } else {
                answer.confidence = reflection.confidence;
                std::ostringstream a;
                a << std::fixed << std::setprecision(2)
                  << "(no expert could solve this input; " << arb.winner.node_name
                  << " rated it relevance " << arb.winner.best.relevance
                  << " / importance " << arb.winner.best.importance;
                if (!context.empty() && context.front().text != candidate)
                    a << "; closest memory: \"" << context.front().text << "\" sim=" << context.front().similarity;
                a << (lm_ && lm_->is_ready() ? ")" : "; connect a language model for a phrased answer)");
                answer.answer = a.str();

                if (reflection.should_output) {
                    std::ostringstream final_text;
                    final_text << std::fixed << std::setprecision(3)
                               << "Conclusion: " << reflection.conclusion
                               << " | expert=" << arb.winner.node_name
                               << " | utility=" << arb.winner.utility
                               << " | confidence=" << reflection.confidence
                               << " | vector_hits=" << context.size()
                               << " | short_term=" << short_term.size();
                    if (!context.empty()) {
                        final_text << " | best_vector_match=\"" << context.front().text << "\""
                                   << " (sim=" << context.front().similarity << ")";
                    }
                    final_text << " | tokens=" << tokens.size();
                    LOG("[FINAL OUTPUT] " << final_text.str());
                    LOG("[VALUE] " << actual_value);
                    ++tick_outputs;
                }
            }

            std::ostringstream th;
            th << plan.summary() << "\n" << arb.summary() << "\n" << challenge.str();
            for (const auto& o : arb.ranked) {
                if (!o.competence.valid && o.finding.empty()) continue;
                th << "\n[" << o.node_name << "] familiarity logp=" << std::fixed << std::setprecision(2)
                   << o.competence.mean_logprob << " ppl=" << std::setprecision(1) << o.competence.perplexity
                   << " coverage=" << std::setprecision(2) << o.competence.coverage
                   << " | model expects: \"" << o.finding << "\"";
            }
            if (!context.empty())
                th << "\nclosest memory: \"" << context.front().text << "\" (sim="
                   << std::fixed << std::setprecision(3) << context.front().similarity << ")";
            if (arb.winner.skill.handled) th << "\n" << arb.winner.skill.thinking;
            else if (!arb.winner.skill.thinking.empty()) th << "\nskill: " << arb.winner.skill.thinking;
            answer.thinking = th.str();
        } else {
            LOG("[PHASE 8/9] no winner this tick - nothing to reflect on");
            answer.answer   = "(no expert produced an outcome this tick)";
            answer.thinking = plan.summary() + "\n" + arb.summary();
        }
        if (external) {
            // Stage 2 for user questions: let the language expert phrase the
            // final answer from the expert findings + memory, if available.
            ExpertNode* lang = language_node_ >= 0 ? nodes_[static_cast<size_t>(language_node_)].get() : nullptr;
            const bool can_stage = lang && lang->is_idle() && lm_ && lm_->is_ready() && !answer.recalled;
            if (can_stage) {
                FireRequest req;
                req.tick     = t;
                req.inputs   = { input };
                req.affinity = 1.f;
                req.external = true;
                req.deadline = std::chrono::steady_clock::now() + scheduler_->period() * config_.slow_deadline_ticks;
                req.reason   = "stage 2: phrase the final answer";
                for (const auto& o : arb.ranked) {
                    std::ostringstream f;
                    f << o.node_name << " (" << o.domain << "): ";
                    if (o.skill.handled) f << "SOLVED -> " << o.skill.answer << " (confidence " << std::fixed << std::setprecision(2) << o.skill.confidence << ")";
                    else f << "relevance " << std::fixed << std::setprecision(2) << o.best.relevance
                           << ", importance " << o.best.importance << ", utility " << o.utility;
                    if (o.competence.valid)
                        f << ", familiarity logp " << std::fixed << std::setprecision(2) << o.competence.mean_logprob;
                    if (!o.finding.empty()) f << ", expects: \"" << o.finding << "\"";
                    req.findings.push_back(f.str());
                    if (o.node_id == memory_node_) req.memory = o.skill.context;
                }
                if (req.memory.empty()) {
                    auto ctx = vector_memory_->retrieve_context(input, *tokenizer_, *embedding_layer_, 4);
                    for (const auto& h : ctx) if (h.similarity > 0.15f && h.text != input) req.memory.push_back(h.text);
                }
                for (const auto& g : goal_system_->top_goals(3)) {
                    std::ostringstream gs; gs << g.name << ":" << std::fixed << std::setprecision(2) << g.score;
                    req.goals.push_back(gs.str());
                }
                if (lang->fire(std::move(req))) {
                    PendingAnswer pa;
                    pa.in        = current;
                    pa.immediate = answer;
                    pa.node_id   = language_node_;
                    pa.tick      = t;
                    pa.timeout   = std::chrono::steady_clock::now() + std::chrono::seconds(config_.llm_timeout_s + 3);
                    pending_answers_.push_back(std::move(pa));
                    LOG("[PHASE 8/9] " << lang->name() << " fired (stage 2) - final answer will follow; immediate answer: \""
                        << answer.answer.substr(0, 80) << "\"");
                } else {
                    deliver(current, answer);
                }
            } else {
                if (lang && lm_ && lm_->is_ready() && !lang->is_idle() && !answer.recalled)
                    answer.thinking += "\n[LANGUAGE] " + lang->name() + " busy (" + node_state_name(lang->state()) + ") - expert answer delivered directly";
                deliver(current, answer);
            }
        }
        total_outputs += tick_outputs;

        // Drain the GPU job ledger
        {
            GPUJob completed;
            int drained = 0;
            while (gpu_queue_->try_dequeue_job(completed)) ++drained;
            if (config_.verbose) LOG("[GPUQueue] Drained " << drained << " jobs");
        }

        // ---- PHASE 9: behaviour policy ----------------------------------------------
        // Decisions are computed now but applied at the very end of the tick
        // (after the summary) so that a kill/stop never competes with the
        // inference window of this tick.
        const bool overran = last_overrun.count() > 0 || std::chrono::steady_clock::now() > tick.end;
        pending_behaviour = primary_->review(t, nodes_, arb, overran);
        LOG("[PHASE 9/9] behaviour decisions=" << pending_behaviour.size()
            << " adaptive_concurrency=" << primary_->adaptive_concurrency());
        primary_->apply(pending_behaviour, nodes_, t);

        // ---- summary -----------------------------------------------------------------
        {
            auto top_goals = goal_system_->top_goals(3);
            std::ostringstream line;
            line << std::fixed << std::setprecision(2)
                 << "[SUMMARY] tick=" << t
                 << " fired=" << fired.size()
                 << " on_time=" << arb.ranked.size()
                 << " late=" << arb.late.size()
                 << " winner=" << (arb.has_winner ? arb.winner.node_name : "-")
                 << " expert_wins=" << total_wins
                 << " primary_fallbacks=" << primary_fallbacks
                 << " final_outputs=" << total_outputs
                 << " vector_memory=" << vector_memory_->size()
                 << "(" << (vector_memory_->gpu_index_active() ? "gpu" : "cpu") << ",search_us="
                 << std::setprecision(0) << vector_memory_->last_search_us() << std::setprecision(2) << ")"
                 << " experiences=" << (store_ ? store_->count() : 0)
                 << " recalls=" << recalls_
                 << " lm_answers=" << language_answers_
                 << " pending_lm=" << pending_answers_.size()
                 << " hz_target=" << config_.tick_hz
                 << " hz_measured=" << scheduler_->measured_hz()
                 << " overruns=" << scheduler_->overruns()
                 << " goals=";
            for (size_t i = 0; i < top_goals.size(); ++i) {
                if (i) line << "|";
                line << top_goals[i].name << ":" << top_goals[i].score;
            }
            line << " nodes=";
            for (size_t i = 0; i < nodes_.size(); ++i) {
                if (i) line << " ";
                line << nodes_[i]->describe();
            }
            LOG(line.str());
        }

        {
            std::lock_guard<std::mutex> lk(stats_mutex_);
            stats_.ticks         = t;
            stats_.overruns      = scheduler_->overruns();
            stats_.final_outputs = total_outputs;
            stats_.wins          = total_wins;
            stats_.measured_hz   = scheduler_->measured_hz();
            stats_.avg_busy_ms   = scheduler_->avg_busy_ms();
            stats_.primary_fallbacks = primary_fallbacks;
            stats_.backend       = gpu_backend_name();
            stats_.cuda_active   = primary_net_->is_cuda_active();
            stats_.node_lines.clear();
            for (const auto& n : nodes_) stats_.node_lines.push_back(n->describe());
            stats_.gpu_memory_index = vector_memory_->gpu_index_active();
            stats_.vector_memory    = vector_memory_->size();
            stats_.experiences      = store_ ? store_->count() : 0;
            stats_.recalls          = recalls_;
            stats_.language_answers = language_answers_;
            stats_.llm              = (lm_ && lm_->is_ready()) ? lm_->model_info() : "off";
            stats_.model_lines      = model_lines_;
            stats_.knowledge_seconds = knowledge_seconds_;
        }
        if (t % 470 == 0) save_learning_state();   // ~ once a minute

        last_overrun = scheduler_->end_tick();
        if (last_overrun.count() > 0) {
            LOG("[LOOP] tick " << t << " overran by " << (last_overrun.count() / 1e6) << "ms"
                << " (busy=" << (scheduler_->last_busy().count() / 1e6) << "ms"
                << " period=" << (scheduler_->period().count() / 1e6) << "ms)");
        }
    }

    for (auto& pa : pending_answers_) {
        Answer final = pa.immediate;
        final.thinking += "\n[LANGUAGE] engine stopped before the language expert answered";
        deliver(pa.in, std::move(final));
    }
    pending_answers_.clear();

    LOG("=== OPENRAI MAIN LOOP STOPPED ==="
        << " ticks=" << scheduler_->ticks()
        << " measured_hz=" << std::fixed << std::setprecision(3) << scheduler_->measured_hz()
        << " avg_busy_ms=" << scheduler_->avg_busy_ms()
        << " worst_busy_ms=" << (scheduler_->worst_busy().count() / 1e6)
        << " overruns=" << scheduler_->overruns()
        << " resyncs=" << scheduler_->resyncs());
}

} // namespace openrai
