#include <openrai/expert_node.h>
#include <openrai/log.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>

namespace openrai {

// -----------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------
const char* node_state_name(NodeState s) {
    switch (s) {
        case NodeState::Stopped:   return "STOPPED";
        case NodeState::Idle:      return "IDLE";
        case NodeState::Running:   return "RUNNING";
        case NodeState::Suspended: return "SUSPENDED";
        case NodeState::Killed:    return "KILLED";
    }
    return "?";
}

ExpertDomain make_expert_domain(size_t index) {
    static const std::vector<ExpertDomain> catalogue = {
        { "stability",  "stability",
          { "error", "fail", "overload", "crash", "stability", "stable", "safe", "recover", "restart", "exceeded" }, 101 },
        { "anomaly",    "stability",
          { "anomaly", "threshold", "spike", "trigger", "detect", "alert", "outlier", "above", "unusual" }, 202 },
        { "memory",     "memory",
          { "memory", "context", "history", "retrieve", "vector", "similarity", "recall", "earlier", "segment", "prior" }, 303, true },
        { "throughput", "relevance",
          { "latency", "gpu", "queue", "batch", "throughput", "stream", "port", "thread", "ms", "connected" }, 404 },
        { "planning",   "relevance",
          { "goal", "prioritize", "command", "plan", "increase", "decrease", "cluster", "schedule", "priority", "over" }, 505 },
        { "reflection", "memory",
          { "reflect", "compare", "reasoning", "conclusion", "output", "evidence", "judge", "outputs", "last" }, 606 },
        { "telemetry",  "stability",
          { "sensor", "temperature", "value", "probe", "measurement", "usage", "percent", "detected", "node" }, 707 },
        { "language",   "relevance",
          { "user", "query", "summarize", "state", "window", "sample", "labelled", "feed", "data", "received" }, 808 },
        { "math",       "relevance",
          { "calculate", "compute", "sum", "plus", "minus", "times", "divided", "multipl", "percent", "%",
            "*", "+", "^", "sqrt", "square", "root", "what is", "how much", "equals" }, 909 },
    };

    if (index < catalogue.size()) return catalogue[index];

    // Generalists beyond the catalogue: rotate goals, broad vocabulary
    static const char* goals[] = { "stability", "relevance", "memory" };
    ExpertDomain d;
    d.name     = "generalist" + std::to_string(index - catalogue.size() + 1);
    d.goal     = goals[index % 3];
    d.keywords = { "input", "system", "node", "data", "state", "current", "new", "for", "on" };
    d.seed     = 900u + static_cast<unsigned>(index) * 17u;
    return d;
}

static std::string to_lower(const std::string& s) {
    std::string out(s.size(), ' ');
    std::transform(s.begin(), s.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// -----------------------------------------------------------------------
// ExpertNode
// -----------------------------------------------------------------------
ExpertNode::ExpertNode(int id,
                       ExpertDomain domain,
                       std::shared_ptr<GPUNeuralNet> net,
                       size_t max_batch,
                       int priority,
                       NodeSignal* signal,
                       bool verbose)
    : id_(id),
      name_("E" + std::to_string(id) + ":" + domain.name),
      domain_(std::move(domain)),
      net_(std::move(net)),
      max_batch_(std::max<size_t>(1, max_batch)),
      priority_(std::clamp(priority, 0, 100)),
      signal_(signal),
      verbose_(verbose)
{
    // Domain lens: deterministic per-domain modulation of the 32 input
    // features so that even experts sharing one weight set perceive the
    // same input differently.
    std::mt19937 rng(domain_.seed * 7919u + 13u);
    std::uniform_real_distribution<float> d(0.70f, 1.30f);
    lens_.resize(GPUNeuralNet::INPUT_DIM);
    for (auto& v : lens_) v = d(rng);

    input_buffer_.assign(max_batch_ * GPUNeuralNet::INPUT_DIM, 0.f);
}

ExpertNode::~ExpertNode() {
    quit_ = true;
    if (ctx_) ctx_->cancel();
    cv_.notify_all();
    join_thread();
    release_context();
}

NodeState ExpertNode::state() const { return state_.load(); }

bool ExpertNode::is_alive() const {
    const NodeState s = state();
    return s == NodeState::Idle || s == NodeState::Running || s == NodeState::Suspended;
}

void ExpertNode::set_state(NodeState s, uint64_t tick) {
    state_ = s;
    stats_.state_since_tick = tick;
}

void ExpertNode::create_context() {
    ctx_ = std::make_unique<GPUStreamContext>(name_, priority_.load(), max_batch_, net_->arch());
}

void ExpertNode::release_context() {
    ctx_.reset();
}

void ExpertNode::join_thread() {
    if (thread_.joinable()) thread_.join();
}

// ---- lifecycle ----------------------------------------------------------
bool ExpertNode::start(uint64_t tick) {
    std::unique_lock<std::mutex> lk(mutex_);
    const NodeState s = state();
    if (s == NodeState::Idle || s == NodeState::Running) return true;
    if (s == NodeState::Suspended) { lk.unlock(); return resume(tick); }

    // Stopped / Killed -> cold start
    lk.unlock();
    join_thread();                 // make sure an old thread is gone
    release_context();
    create_context();
    quit_ = false;
    ctx_->reset_cancel();
    {
        std::lock_guard<std::mutex> g(mutex_);
        pending_.reset();
        outcome_.reset();
        set_state(NodeState::Idle, tick);
        if (s == NodeState::Killed) ++stats_.restarts;
        stats_.consecutive_misses = 0;
        stats_.low_trust_streak   = 0;
    }
    thread_ = std::thread([this] { worker_loop(); });

    LOG("[Node " << name_ << "] START tick=" << tick
        << " priority=" << priority_.load()
        << " stream=" << (ctx_->is_cuda() ? "cuda" : "cpu")
        << " handle=0x" << std::hex << ctx_->native_handle() << std::dec
        << " cuda_prio=" << ctx_->cuda_priority()
        << " thread=" << std::hash<std::thread::id>{}(thread_.get_id()) % 100000
        << " net=" << net_->arch().name
        << " seed=" << domain_.seed);
    return true;
}

void ExpertNode::stop(uint64_t tick) {
    if (state() == NodeState::Stopped) return;
    const NodeState before = state();
    quit_ = true;
    cv_.notify_all();
    join_thread();                // graceful: in-flight work completes
    release_context();
    {
        std::lock_guard<std::mutex> g(mutex_);
        pending_.reset();
        set_state(NodeState::Stopped, tick);
    }
    LOG("[Node " << name_ << "] STOP tick=" << tick << " from=" << node_state_name(before)
        << " (thread joined, context released)");
}

bool ExpertNode::suspend(uint64_t tick) {
    std::lock_guard<std::mutex> g(mutex_);
    if (state() != NodeState::Idle) return false;
    set_state(NodeState::Suspended, tick);
    ++stats_.suspends;
    return true;
}

bool ExpertNode::resume(uint64_t tick) {
    std::lock_guard<std::mutex> g(mutex_);
    if (state() != NodeState::Suspended) return false;
    set_state(NodeState::Idle, tick);
    stats_.low_trust_streak = 0;
    return true;
}

void ExpertNode::kill(uint64_t tick) {
    const NodeState before = state();
    if (before == NodeState::Killed) return;
    quit_ = true;
    if (ctx_) ctx_->cancel();     // abort: worker stops polling, launches become no-ops
    cv_.notify_all();
    join_thread();
    release_context();            // destroys stream / frees workspace
    {
        std::lock_guard<std::mutex> g(mutex_);
        pending_.reset();
        outcome_.reset();
        set_state(NodeState::Killed, tick);
        ++stats_.kills;
    }
    LOG("[Node " << name_ << "] KILL tick=" << tick << " from=" << node_state_name(before)
        << " (in-flight work cancelled, thread joined, context destroyed)");
}

// ---- work ---------------------------------------------------------------
bool ExpertNode::fire(FireRequest req) {
    {
        std::lock_guard<std::mutex> g(mutex_);
        if (state() != NodeState::Idle || !ctx_) return false;
        if (req.inputs.empty()) return false;
        if (req.inputs.size() > max_batch_) req.inputs.resize(max_batch_);
        outcome_.reset();              // a stale, uncollected result is void
        pending_ = std::move(req);
        set_state(NodeState::Running, pending_->tick);
        ++stats_.fired;
        stats_.last_fired_tick = pending_->tick;
    }
    cv_.notify_all();
    return true;
}

bool ExpertNode::try_collect(NodeOutcome& out) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!outcome_) return false;
    out = std::move(*outcome_);
    outcome_.reset();
    return true;
}

bool ExpertNode::has_outcome() const {
    std::lock_guard<std::mutex> g(mutex_);
    return outcome_.has_value();
}

bool ExpertNode::set_priority(int p, uint64_t tick) {
    p = std::clamp(p, 0, 100);
    if (p == priority_.load()) return false;
    std::lock_guard<std::mutex> g(mutex_);
    priority_ = p;
    if (state() == NodeState::Idle || state() == NodeState::Suspended) {
        // Worker is parked -> safe to swap the stream context
        release_context();
        create_context();
        (void)tick;
        return true;
    }
    return false;   // applied on next start()
}

// ---- worker -------------------------------------------------------------
void ExpertNode::worker_loop() {
    while (true) {
        FireRequest req;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_.wait(lk, [this] { return quit_.load() || pending_.has_value(); });
            if (quit_.load() && !pending_) break;
            if (quit_.load() && pending_ && ctx_ && ctx_->cancelled()) {
                // kill(): drop the request without executing it
                pending_.reset();
                break;
            }
            req = std::move(*pending_);
            pending_.reset();
        }

        execute(req);

        if (quit_.load()) break;
    }
}

void ExpertNode::execute(FireRequest& req) {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t N = std::min(req.inputs.size(), max_batch_);
    const size_t in_dim = GPUNeuralNet::INPUT_DIM;

    NodeOutcome out;
    out.node_id     = id_;
    out.node_name   = name_;
    out.domain      = domain_.name;
    out.goal        = domain_.goal;
    out.tick        = req.tick;
    out.inputs      = req.inputs;
    out.affinity    = req.affinity;
    out.fire_reason = req.reason;

    // Encode through the domain lens
    for (size_t n = 0; n < N; ++n) {
        float* row = input_buffer_.data() + n * in_dim;
        encode_input_v2(req.inputs[n], row, in_dim);
        for (size_t i = 0; i < in_dim; ++i) row[i] *= lens_[i];
    }

    bool launched = false;
    bool cancelled = false;
    if (ctx_ && !ctx_->cancelled()) {
        launched = net_->run_batch_async(input_buffer_.data(), N, *ctx_);
    }

    // The expert's own model speaks while the stream works on the network:
    // how familiar is this input, and what does the expert expect next?
    if (model_ && model_->trained() && !req.inputs.empty() && !(ctx_ && ctx_->cancelled())) {
        out.competence = model_->score(req.inputs[0]);
        out.finding    = model_->generate(req.inputs[0], 8, 0.f);
        if (out.finding.empty()) {
            // the model expects the sentence to end here - report what it
            // considered likely instead
            std::string alt = "(complete; next likely:";
            for (const auto& p : model_->predict_next(req.inputs[0], 3)) alt += " " + p.first;
            out.finding = alt + ")";
        }
    }

    // Tools run on this node's thread as well (true overlap with the CUDA
    // stream; on CPU the network already finished).
    if (skill_.valid() && !req.inputs.empty() && !(ctx_ && ctx_->cancelled())) {
        try {
            SkillContext sc;
            sc.text     = req.inputs[0];
            sc.memory   = req.memory;
            sc.findings = req.findings;
            sc.goals    = req.goals;
            sc.tick     = req.tick;
            out.skill = skill_.run(sc);
        } catch (const std::exception& e) {
            out.skill = SkillResult{};
            out.skill.thinking = std::string("skill error: ") + e.what();
        }
    }

    if (launched) {
        // Poll the stream: responsive to kill() / quit, no allocation.
        int spins = 0;
        while (!ctx_->poll()) {
            if (quit_.load() || ctx_->cancelled()) { cancelled = true; break; }
            if (++spins < 64) std::this_thread::yield();
            else std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        if (!cancelled) net_->fetch_results(*ctx_, N, out.results);
    } else {
        cancelled = ctx_ ? ctx_->cancelled() : true;
    }

    const auto t1 = std::chrono::steady_clock::now();
    out.latency   = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    out.on_time   = t1 <= req.deadline;
    out.cancelled = cancelled;
    out.launched  = launched;

    if (!out.results.empty()) {
        size_t best = 0;
        for (size_t i = 1; i < out.results.size(); ++i)
            if (out.results[i].importance > out.results[best].importance) best = i;
        out.best_index = best;
        out.best       = out.results[best];
    }

    {
        std::lock_guard<std::mutex> g(mutex_);
        const double ms = out.latency.count() / 1000.0;
        stats_.last_latency_ms = ms;
        stats_.ema_latency_ms  = stats_.ema_latency_ms <= 0.0 ? ms : (0.8 * stats_.ema_latency_ms + 0.2 * ms);
        if (!cancelled) ++stats_.completed;
        outcome_ = std::move(out);
        if (state() == NodeState::Running) set_state(NodeState::Idle, req.tick);
    }

    if (verbose_) {
        LOG("[Node " << name_ << "] tick=" << req.tick
            << " N=" << N
            << " latency_us=" << (t1 - t0).count() / 1000
            << " on_time=" << (t1 <= req.deadline ? "yes" : "NO")
            << (cancelled ? " CANCELLED" : "")
            << " reason=\"" << req.reason << "\"");
    }

    if (signal_) signal_->notify();
}

// ---- introspection --------------------------------------------------------
NodeStats ExpertNode::stats() const {
    std::lock_guard<std::mutex> g(mutex_);
    return stats_;
}

void ExpertNode::set_skill(Skill skill) {
    skill_ = std::move(skill);
}

void ExpertNode::set_model(std::shared_ptr<ExpertModel> model) {
    model_ = std::move(model);
}

ModelScore ExpertNode::competence(const std::string& text) const {
    if (!model_ || !model_->trained()) return ModelScore{};
    return model_->score(text);
}

float ExpertNode::keyword_affinity(const std::string& text) const {
    const std::string low = to_lower(text);
    int hits = 0;
    for (const auto& kw : domain_.keywords)
        if (low.find(kw) != std::string::npos) ++hits;
    // 0 hits -> 0, 1 -> 0.63, 2 -> 0.86, 3 -> 0.95
    return 1.f - std::exp(-static_cast<float>(hits));
}

std::string ExpertNode::describe() const {
    const NodeStats st = stats();
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2)
       << name_ << "{state=" << node_state_name(state())
       << " prio=" << priority_.load()
       << " model=" << (model_ ? (model_->trained() ? std::to_string(model_->train_iterations()) + "it" : "untrained") : "none")
       << " trust=" << st.trust
       << " fired=" << st.fired
       << " wins=" << st.wins
       << " misses=" << st.deadline_misses
       << " kills=" << st.kills
       << " ema_ms=" << std::setprecision(1) << st.ema_latency_ms
       << (ctx_ ? (ctx_->is_cuda() ? " cuda" : " cpu") : " cold")
       << "}";
    return ss.str();
}

} // namespace openrai
