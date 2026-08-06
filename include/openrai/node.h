// -----------------------------------------------------------------------
// node.h
//   Per-node abstraction for distributed cognitive architecture.
//   Each node represents a specialized processing unit with its own
//   neural network, vector database, and context.
//
//   Priority 3: MEDIUM (6-10 hours estimated)
// -----------------------------------------------------------------------

#pragma once

#include <openrai/gpu_runtime.h>
#include <openrai/vector_db.h>

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <mutex>

namespace openrai {

// -----------------------------------------------------------------------
// NodeSpecialty - Different types of cognitive nodes
// -----------------------------------------------------------------------
enum class NodeSpecialty {
    Reasoning,    // Logical inference, causal analysis
    Planning,     // Goal decomposition, action sequencing
    Memory,       // Long-term storage, recall optimization
    Retrieval,    // Context search, knowledge lookup
    Coding,       // Code generation, syntax analysis
    Analysis,     // Pattern recognition, data interpretation
    Integration,  // Multi-modal fusion, synthesis
    Formatter,    // Output structuring, presentation
    Debug,        // Error detection, diagnostic reasoning
    Autonomy      // Self-directed goal selection
};

// Convert specialty to string
const char* specialty_to_string(NodeSpecialty s);

// -----------------------------------------------------------------------
// NodeInput - Input data for node processing
// -----------------------------------------------------------------------
struct NodeInput {
    std::string text;
    std::vector<float> embedding;
    std::string context;
    float priority = 0.5f;
};

// -----------------------------------------------------------------------
// NodeOutput - Result from node processing
// -----------------------------------------------------------------------
struct NodeOutput {
    std::string summary;
    std::vector<float> embedding;
    float confidence = 0.0f;
    bool should_propagate = false;
};

// -----------------------------------------------------------------------
// Node - Specialized processing unit
// -----------------------------------------------------------------------
class Node {
public:
    Node(const std::string& name, NodeSpecialty specialty, int device_id = 0);
    ~Node();

    // Process input and produce output
    NodeOutput process(const NodeInput& input);

    // Search context from this node's vector DB
    std::vector<VectorSearchResult> search_context(const std::vector<float>& query, size_t k = 5);

    // Store information in this node's vector DB
    void store_memory(const std::string& text_value, const std::vector<float>& embedding, const std::string& tag = "");

    // Configuration
    void set_device(int gpu_id);
    void set_use_fp16(bool enable);
    void set_config_path(const std::string& path);

    // Accessors
    const std::string& name() const { return name_; }
    NodeSpecialty specialty() const { return specialty_; }
    int device() const { return device_id_; }
    bool is_fp16_enabled() const;

    // Stats
    size_t processed_count() const { return processed_count_; }
    size_t memory_count() const;

private:
    std::string name_;
    NodeSpecialty specialty_;
    int device_id_;
    std::string config_path_;

    // Neural network for this node (can be shared pointer or unique)
    std::unique_ptr<GPUNeuralNet> neural_net_;

    // Vector database for this node's memory
    std::unique_ptr<VectorDatabase> vector_db_;

    // Statistics
    size_t processed_count_ = 0;
    mutable std::mutex mutex_;

    // Internal processing helpers
    void initialize_neural_net();
    void initialize_vector_db();
    std::string generate_summary(const NodeInput& input, const NNResult& nn_result);
};

// -----------------------------------------------------------------------
// NodeTaskResult - Result from async node task with timeout handling
// -----------------------------------------------------------------------
struct NodeTaskResult {
    NodeOutput output;
    bool completed = false;
    bool timed_out = false;
    std::string error_message;
};

// -----------------------------------------------------------------------
// NodeManager - Registry and dispatcher for nodes
// -----------------------------------------------------------------------
class NodeManager {
public:
    NodeManager();
    ~NodeManager();

    // Register a new node
    void register_node(std::unique_ptr<Node> node);

    // Get node by name
    Node* get_node(const std::string& name);

    // Process input on a specific node
    NodeOutput process_on_node(const std::string& node_name, const NodeInput& input);

    // Parallel processing: dispatch to multiple nodes concurrently
    std::vector<NodeOutput> process_parallel(const std::vector<std::string>& node_names, const NodeInput& input);

    // Async task scheduling with timeout (Priority 6)
    void enqueue_node_task(
        const std::string& node_name,
        const NodeInput& input,
        std::function<void(const NodeTaskResult&)> callback,
        std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)
    );

    // Set thread pool for async operations
    void set_thread_pool(std::shared_ptr<class ThreadPool> pool);

    // Get all registered node names
    std::vector<std::string> get_node_names() const;

    // Stats
    size_t node_count() const;

private:
    std::vector<std::unique_ptr<Node>> nodes_;
    mutable std::mutex mutex_;
    std::shared_ptr<class ThreadPool> thread_pool_;

    Node* find_node_unlocked(const std::string& name);
};

} // namespace openrai
