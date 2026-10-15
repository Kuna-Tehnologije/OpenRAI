#pragma once

#include <openrai/config.h>

#include <queue>
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <future>
#include <string>
#include <mutex>
#include <thread>

namespace openrai {

// Forward declarations
class MemorySystem;
class ThreadPool;
class GPUJobQueue;
class ReasoningModule;
class GPUNeuralNet;
class Tokenizer;
class EmbeddingLayer;
class VectorMemory;
class KnowledgeGraph;
class GoalSystem;
class SelfReflection;
class ExpertNode;
class PrimaryNode;
class TickScheduler;
struct NodeSignal;
class ExperienceStore;
class LMAdapter;
struct NodeOutcome;

// Core output structure
struct Output {
    std::string data;
    float relevance_score   = 0.0f;
    float importance_score  = 0.0f;
    bool  nn_should_output  = false;
    std::chrono::system_clock::time_point timestamp;
    int thread_id = 0;
    std::string source;          // node that produced it ("E2:memory", "PRIMARY")
    float utility_score = 0.0f;  // arbiter utility (0 when not arbitrated)
};

// Answer to a text question submitted through OpenRAI::ask()
struct Answer {
    std::string question;
    std::string answer;        // final text
    std::string thinking;      // inner trace: plan, arbitration, expert steps
    std::string expert;        // node that produced the winning outcome
    float       confidence = 0.f;
    float       utility    = 0.f;
    bool        solved     = false;   // a deterministic skill produced the answer
    uint64_t    tick       = 0;
    double      latency_ms = 0.0;     // submit -> answer
    int64_t     experience_id = 0;    // id in the learning store (for feedback)
    bool        recalled   = false;   // answered from a learned experience
    bool        language   = false;   // phrased by the language model
    std::string immediate;            // the expert answer before the LM rephrased it
};

// Memory types
struct MemoryEntry {
    std::string content;
    float importance = 0.0f;
    std::vector<float> embedding;
    std::chrono::system_clock::time_point created_at;
};

// Main OpenRAI engine
class OpenRAI {
public:
    OpenRAI();
    explicit OpenRAI(const EngineConfig& cfg);
    ~OpenRAI();

    void start();
    void stop();
    bool is_running() const { return is_running_.load(); }

    // Inject an arbitrary input string; it will be picked up
    // by the engine before the next loop iteration.
    void inject_input(const std::string& input);

    // Text in -> answer out.  The question is processed by the next tick
    // (primary judges, experts fire, arbiter picks the winner) and the
    // future is fulfilled with the winner's answer and its inner thinking.
    // When a language model is connected the language expert phrases the
    // final answer; the future then completes when the LM returns.
    std::future<Answer> ask(const std::string& question);

    // Learning from use: reward (+1) or punish (-1) an answer.  Updates the
    // stored experience and the trust of the expert that produced it.
    bool feedback(int64_t experience_id, float delta);
    int64_t last_experience_id() const;

    // Direct memory access (thread-safe): GPU/CPU similarity search and the
    // most recent learned experiences.
    std::vector<std::pair<std::string, float>> search_memory(const std::string& text, size_t k = 5) const;
    std::vector<std::string> recent_experiences(size_t n = 10) const;

    // ---- compatibility API for the chat GUIs of the language-model branch ----
    using OutputCallback = std::function<void(const std::string&)>;
    void set_output_callback(OutputCallback cb);         // final answer of every external input
    std::string generate_response(const std::string& prompt);   // blocking ask()
    int         get_iteration_count() const;
    std::string get_latest_output() const;
    float       get_latest_confidence() const;

    const EngineConfig& config() const { return config_; }

    // Run-time statistics (safe to call from another thread)
    struct Stats {
        uint64_t ticks = 0;
        uint64_t overruns = 0;
        uint64_t final_outputs = 0;
        uint64_t wins = 0;
        double   measured_hz = 0.0;
        double   avg_busy_ms = 0.0;
        uint64_t primary_fallbacks = 0;
        std::vector<std::string> node_lines;   // one describe() per expert
        std::string backend;                   // "CUDA" / "CPU"
        bool     cuda_active = false;
        bool     gpu_memory_index = false;
        size_t   vector_memory = 0;
        int64_t  experiences = 0;
        uint64_t recalls = 0;
        uint64_t language_answers = 0;
        std::string llm;                       // model info or "off"
        std::vector<std::string> model_lines;  // one per expert: training report
        double   knowledge_seconds = 0.0;      // time spent training/loading at start
    };
    Stats stats() const;

private:
    EngineConfig                     config_;

    std::unique_ptr<ThreadPool>      thread_pool_;
    std::unique_ptr<MemorySystem>    memory_system_;
    std::unique_ptr<GPUJobQueue>     gpu_queue_;
    std::unique_ptr<ReasoningModule> reasoning_module_;

    // Semantic pipeline
    std::unique_ptr<Tokenizer>       tokenizer_;
    std::unique_ptr<EmbeddingLayer>  embedding_layer_;
    std::unique_ptr<VectorMemory>    vector_memory_;
    std::unique_ptr<KnowledgeGraph>  knowledge_graph_;
    std::unique_ptr<GoalSystem>      goal_system_;
    std::unique_ptr<SelfReflection>  self_reflection_;

    // Multithreaded expert system (7.82 Hz loop)
    std::unique_ptr<NodeSignal>              node_signal_;
    std::vector<std::unique_ptr<ExpertNode>> nodes_;
    std::unique_ptr<PrimaryNode>             primary_;
    std::unique_ptr<TickScheduler>           scheduler_;
    std::shared_ptr<GPUNeuralNet>            primary_net_;

    // Learning store + language model
    std::unique_ptr<ExperienceStore>         store_;
    std::unique_ptr<LMAdapter>               lm_;
    int                                      language_node_ = -1;   // index of the slow expert
    int                                      memory_node_   = -1;
    std::atomic<int64_t>                     last_experience_id_{0};
    uint64_t                                 recalls_ = 0;
    uint64_t                                 language_answers_ = 0;
    OutputCallback                           output_callback_;
    std::vector<std::string>                 model_lines_;
    double                                   knowledge_seconds_ = 0.0;
    std::string                              latest_output_;
    std::atomic<float>                       latest_confidence_{0.f};

    std::atomic<bool>        is_running_{false};
    std::thread              loop_thread_;
    mutable std::mutex       stats_mutex_;
    Stats                    stats_;

    // Thread-safe queue for external inputs (with optional answer promise)
    struct PendingInput {
        std::string text;
        std::shared_ptr<std::promise<Answer>> promise;
        std::chrono::steady_clock::time_point submitted;
    };
    std::queue<PendingInput> pending_inputs_;
    std::mutex               pending_mutex_;
    void fail_pending(const std::string& why);

    // Questions waiting for the slow language expert
    struct PendingAnswer {
        PendingInput in;
        Answer       immediate;
        int          node_id = -1;
        uint64_t     tick = 0;
        std::chrono::steady_clock::time_point timeout;
    };
    std::vector<PendingAnswer> pending_answers_;

    void deliver(const PendingInput& in, Answer answer);   // store experience, callbacks, promise
    void save_learning_state();
    void load_learning_state();
    void train_expert_knowledge();                         // models + MLP scorers (parallel)

    void main_loop();
};

} // namespace openrai
