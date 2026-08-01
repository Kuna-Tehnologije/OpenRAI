#pragma once

#include <queue>
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include <string>
#include <mutex>
#include <functional>

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
class LMAdapter;

// Core output structure
struct Output {
    std::string data;
    float relevance_score   = 0.0f;
    float importance_score  = 0.0f;
    bool  nn_should_output  = false;
    std::chrono::system_clock::time_point timestamp;
    int thread_id = 0;
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
    ~OpenRAI();

    void start();
    void stop();

    // Inject an arbitrary input string; it will be picked up
    // by the engine before the next loop iteration.
    void inject_input(const std::string& input);

    // Generate language response with cognitive context
    std::string generate_response(const std::string& prompt);

    // Set callback for sending outputs to GUI
    using OutputCallback = std::function<void(const std::string&)>;
    void set_output_callback(OutputCallback callback) { output_callback_ = callback; }

    // Get stats for UI
    int get_iteration_count() const { return iteration_count_.load(); }
    std::string get_latest_output() const;
    float get_latest_confidence() const { return latest_confidence_.load(); }

private:
    std::unique_ptr<ThreadPool>      thread_pool_;
    std::unique_ptr<MemorySystem>    memory_system_;
    std::unique_ptr<GPUJobQueue>     gpu_queue_;
    std::unique_ptr<ReasoningModule> reasoning_module_;
    std::unique_ptr<GPUNeuralNet>    gpu_nn_;           // Phase 2

    // Semantic pipeline
    std::unique_ptr<Tokenizer>       tokenizer_;
    std::unique_ptr<EmbeddingLayer>  embedding_layer_;
    std::unique_ptr<VectorMemory>    vector_memory_;
    std::unique_ptr<KnowledgeGraph>  knowledge_graph_;
    std::unique_ptr<GoalSystem>      goal_system_;
    std::unique_ptr<SelfReflection>  self_reflection_;

    // Language model adapter (Cognition + Language hybrid)
    std::unique_ptr<LMAdapter>       lm_adapter_;

    std::atomic<bool>        is_running_{false};

    // Thread-safe queue for external inputs
    std::queue<std::string>  pending_inputs_;
    std::mutex               pending_mutex_;

    // Callback for sending outputs to GUI
    OutputCallback           output_callback_;

    // Waiting for user response tracking
    std::atomic<bool>        waiting_for_user_response_{false};
    std::chrono::steady_clock::time_point last_question_time_;
    std::mutex               response_mutex_;

    // Stats for UI
    std::atomic<int>         iteration_count_{0};
    std::string              latest_output_;
    std::atomic<float>       latest_confidence_{0.0f};
    mutable std::mutex       stats_mutex_;

    void main_loop();
};

} // namespace openrai
