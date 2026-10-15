#pragma once

// -----------------------------------------------------------------------
// primary_node.h
//   PrimaryNode - the executive of OpenRAI.
//
//   The primary node does not "answer"; it decides.  Every tick it:
//     1. judge()     - scores the incoming input with its own model on its
//                      own (highest-priority) CUDA stream
//     2. plan()      - decides WHICH experts fire and WHY, given its goals,
//                      each expert's affinity to the input, the expert's
//                      trust (earned from past outcomes), starvation and
//                      the compute budget of the 7.82 Hz tick
//     3. arbitrate() - after the deadline, decides WHOSE outcome matters
//                      most and becomes the engine output of this tick
//     4. review()    - behaviour policy: start / stop / suspend / resume /
//                      kill / respawn / promote / demote expert nodes
//
//   Everything it decides carries a human readable reason so the log
//   explains the behaviour of the whole system.
// -----------------------------------------------------------------------

#include <openrai/config.h>
#include <openrai/expert_node.h>
#include <openrai/gpu_runtime.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace openrai {

class GoalSystem;
class Tokenizer;

using NodeList = std::vector<std::unique_ptr<ExpertNode>>;

struct FiringDecision {
    int         node_id = -1;
    std::string node_name;
    bool        fire  = false;
    float       score = 0.f;
    float       affinity = 0.f;
    std::string reason;
};

struct FiringPlan {
    uint64_t    tick = 0;
    std::string input;
    std::string focus_goal;          // dominant goal of the primary right now
    float       focus_score = 0.f;
    NNResult    judgement;           // primary model's view of the input
    size_t      concurrency = 0;     // how many experts may fire this tick
    std::chrono::steady_clock::time_point deadline;
    std::vector<FiringDecision> decisions;

    size_t fired_count() const;
    std::string summary() const;
};

struct Arbitration {
    uint64_t    tick = 0;
    bool        has_winner = false;
    NodeOutcome winner;
    std::vector<NodeOutcome> ranked;      // on-time outcomes, best first
    std::vector<std::string> late;        // node names that missed the deadline
    std::string reason;
    std::string summary() const;
};

enum class BehaviourAction { Start, Stop, Suspend, Resume, Kill, Respawn, Promote, Demote };
const char* behaviour_action_name(BehaviourAction a);

struct BehaviourDecision {
    int             node_id = -1;
    std::string     node_name;
    BehaviourAction action = BehaviourAction::Start;
    int             new_priority = -1;
    std::string     reason;
};

class PrimaryNode {
public:
    PrimaryNode(const EngineConfig& cfg,
                std::shared_ptr<GPUNeuralNet> judgement_net,
                GoalSystem* goals);
    ~PrimaryNode();

    // 1. Own model, own high-priority stream
    NNResult judge(const std::string& input);

    // 2. Who fires and why
    FiringPlan plan(uint64_t tick,
                    const std::string& input,
                    const NNResult& judgement,
                    const NodeList& nodes,
                    std::chrono::steady_clock::time_point deadline,
                    bool external = false);   // true for user questions (ask())

    // 3. Whose outcome is the most important
    Arbitration arbitrate(uint64_t tick,
                          std::vector<NodeOutcome>& outcomes,
                          const FiringPlan& plan,
                          NodeList& nodes);

    // Called by the scheduler for nodes that were fired but had not
    // returned by the deadline.
    void note_deadline_miss(ExpertNode& node, uint64_t tick);

    // 4. Behaviour policy
    std::vector<BehaviourDecision> review(uint64_t tick,
                                          NodeList& nodes,
                                          const Arbitration& arb,
                                          bool tick_overran);
    void apply(const std::vector<BehaviourDecision>& decisions,
               NodeList& nodes, uint64_t tick);

    // ---- introspection ----
    size_t adaptive_concurrency() const { return adaptive_cap_; }
    const GPUStreamContext* context() const { return ctx_.get(); }
    const GPUNeuralNet&     net()     const { return *net_; }
    float goal_weight(const std::string& goal) const;   // normalised 0..1
    std::string focus_goal() const;

private:
    EngineConfig                  cfg_;
    std::shared_ptr<GPUNeuralNet> net_;
    GoalSystem*                   goals_;
    std::unique_ptr<GPUStreamContext> ctx_;
    std::vector<float>            input_buffer_;

    size_t   adaptive_cap_;
    int      on_time_streak_ = 0;
    uint64_t overruns_ = 0;
};

} // namespace openrai
