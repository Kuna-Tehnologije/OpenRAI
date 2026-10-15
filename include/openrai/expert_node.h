#pragma once

// -----------------------------------------------------------------------
// expert_node.h
//   ExpertNode - one autonomous "expert in its own field".
//
//   Every expert node owns:
//     * its own worker thread            (multithreaded approach)
//     * its own GPUStreamContext         (CUDA stream + priority + workspace)
//     * its own knowledge: an ExpertModel (neural LM trained on the
//       expert's corpus - decides how familiar an input is and produces the
//       expert's own finding) and a trained MLP scorer run on the stream
//     * optional tools (calculator, memory index, external LM)
//     * a domain: name, goal it serves, corpus
//
//   Lifecycle (driven by the PrimaryNode):
//
//        start()            fire()             result
//   Stopped ----> Idle ----------> Running ----------> Idle
//      ^           | suspend()        |                 |
//      |           v                  | kill()          |
//      |       Suspended              v                 |
//      |           | resume()      Killed --respawn-->  start()
//      +-- stop() -+
//
//   Idle       : thread parked, context alive, accepts fire()
//   Running    : executing a FireRequest on its stream
//   Suspended  : parked by the primary (low utility); context kept warm
//   Stopped    : thread joined, context released (cold)
//   Killed     : in-flight work cancelled, thread joined, context destroyed
// -----------------------------------------------------------------------

#include <openrai/gpu_runtime.h>
#include <openrai/skills.h>
#include <openrai/expert_model.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace openrai {

enum class NodeState { Stopped, Idle, Running, Suspended, Killed };
const char* node_state_name(NodeState s);

// What field is this node an expert in?
struct ExpertDomain {
    std::string name;                    // "stability", "anomaly", ...
    std::string goal;                    // GoalSystem goal it serves
    std::vector<std::string> keywords;   // bootstrap vocabulary - only used
                                         // until the expert model is trained
    unsigned seed = 1;                   // weight seed -> own model
    bool consult_for_questions = false;  // always fired for user questions
                                         // (memory lookup), outside the budget
};

// Built-in catalogue of expert domains (index wraps for > catalogue size)
ExpertDomain make_expert_domain(size_t index);

// Request sent by the primary node to an expert
struct FireRequest {
    uint64_t tick = 0;
    std::vector<std::string> inputs;          // 1..max_batch texts
    float affinity = 0.f;                     // how much this is "my field"
    std::chrono::steady_clock::time_point deadline;
    std::string reason;                       // why the primary fired me
    // context for skills (filled by the primary when available)
    std::vector<std::string> memory;
    std::vector<std::string> findings;
    std::vector<std::string> goals;
    bool external = false;                    // a user question (ask())
};

// Result returned by an expert
struct NodeOutcome {
    int         node_id = -1;
    std::string node_name;
    std::string domain;
    std::string goal;
    uint64_t    tick = 0;
    std::vector<std::string> inputs;
    std::vector<NNResult>    results;        // one per input
    NNResult    best;                        // max-importance result
    size_t      best_index = 0;
    float       affinity = 0.f;
    float       utility  = 0.f;              // filled by the arbiter
    std::chrono::microseconds latency{0};
    bool        on_time   = true;
    bool        cancelled = false;
    bool        launched  = true;            // false if run_batch_async refused
    std::string fire_reason;
    SkillResult skill;                       // concrete answer if the expert used a tool
    ModelScore  competence;                  // how familiar the input is to the expert's model
    std::string finding;                     // the expert model's own continuation of the input
};

// Shared completion signal: nodes notify, the scheduler waits with deadline
struct NodeSignal {
    std::mutex              mutex;
    std::condition_variable cv;
    std::atomic<uint64_t>   completions{0};
    void notify() { ++completions; std::lock_guard<std::mutex> lk(mutex); cv.notify_all(); }
};

struct NodeStats {
    uint64_t fired = 0;
    uint64_t completed = 0;
    uint64_t wins = 0;
    uint64_t deadline_misses = 0;
    uint64_t consecutive_misses = 0;
    uint64_t kills = 0;
    uint64_t suspends = 0;
    uint64_t restarts = 0;
    uint64_t last_fired_tick = 0;
    uint64_t last_win_tick = 0;
    uint64_t state_since_tick = 0;
    double   ema_latency_ms = 0.0;
    double   last_latency_ms = 0.0;
    float    trust = 0.5f;          // maintained by the primary (0..1)
    float    last_utility = 0.f;
    int      low_trust_streak = 0;
};

class ExpertNode {
public:
    ExpertNode(int id,
               ExpertDomain domain,
               std::shared_ptr<GPUNeuralNet> net,
               size_t max_batch,
               int priority,
               NodeSignal* signal,
               bool verbose = false);
    ~ExpertNode();

    ExpertNode(const ExpertNode&)            = delete;
    ExpertNode& operator=(const ExpertNode&) = delete;

    // ---- lifecycle --------------------------------------------------------
    bool start(uint64_t tick);      // Stopped/Killed -> Idle (thread + context)
    void stop(uint64_t tick);       // graceful: finish in-flight work, join, release context
    bool suspend(uint64_t tick);    // Idle -> Suspended (context kept warm)
    bool resume(uint64_t tick);     // Suspended -> Idle
    void kill(uint64_t tick);       // hard: cancel stream work, join, destroy context

    // ---- work -------------------------------------------------------------
    bool fire(FireRequest req);                 // Idle -> Running
    bool try_collect(NodeOutcome& out);         // pop finished outcome
    bool has_outcome() const;

    // ---- introspection ------------------------------------------------------
    NodeState   state()  const;
    bool        is_idle() const { return state() == NodeState::Idle; }
    bool        is_alive() const;               // thread running (Idle/Running/Suspended)
    int         id()     const { return id_; }
    const std::string&  name()   const { return name_; }
    const ExpertDomain& domain() const { return domain_; }
    int         priority() const { return priority_.load(); }
    // Re-prioritise.  If the node is Idle its stream is re-created with the
    // matching CUDA priority immediately, otherwise on the next start().
    bool        set_priority(int p, uint64_t tick);
    // Thread-safe snapshot / update of the statistics block
    NodeStats stats() const;
    template <typename F> void update_stats(F&& f) {
        std::lock_guard<std::mutex> g(mutex_);
        f(stats_);
    }
    const GPUStreamContext* context() const { return ctx_.get(); }
    const GPUNeuralNet&     net()     const { return *net_; }

    // ---- knowledge ----
    // The expert's own model of its field.  Thread-safe (the primary scores
    // inputs on the loop thread, the node generates on its own thread).
    void set_model(std::shared_ptr<ExpertModel> model);
    std::shared_ptr<ExpertModel> model() const { return model_; }
    bool has_trained_model() const { return model_ && model_->trained(); }
    // Likelihood of the text under the expert's model (valid=false without one)
    ModelScore competence(const std::string& text) const;

    // ---- tools ----
    void set_skill(Skill skill);
    const Skill* skill() const { return skill_.valid() ? &skill_ : nullptr; }
    // Slow experts (LLM) are fired after arbitration and answer in a later tick
    bool is_slow() const { return skill_.valid() && skill_.slow; }

    // Bootstrap affinity from the domain's keyword vocabulary.  Used only
    // while the expert has no trained model; the primary prefers
    // competence() whenever it is available.
    float keyword_affinity(const std::string& text) const;
    std::string describe() const;

private:
    int                           id_;
    std::string                   name_;
    ExpertDomain                  domain_;
    std::shared_ptr<GPUNeuralNet> net_;
    size_t                        max_batch_;
    std::atomic<int>              priority_;
    NodeSignal*                   signal_;
    bool                          verbose_;

    std::vector<float>            lens_;      // per-domain input modulation
    Skill                         skill_;
    std::shared_ptr<ExpertModel>  model_;

    std::unique_ptr<GPUStreamContext> ctx_;
    std::thread                   thread_;

    mutable std::mutex            mutex_;
    std::condition_variable       cv_;
    std::atomic<NodeState>        state_{NodeState::Stopped};
    std::atomic<bool>             quit_{false};
    std::optional<FireRequest>    pending_;
    std::optional<NodeOutcome>    outcome_;
    std::vector<float>            input_buffer_;
    NodeStats                     stats_;

    void worker_loop();
    void execute(FireRequest& req);
    void create_context();
    void release_context();
    void join_thread();
    void set_state(NodeState s, uint64_t tick);
};

} // namespace openrai
