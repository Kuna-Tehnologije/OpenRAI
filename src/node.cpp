// -----------------------------------------------------------------------
// node.cpp
//   Implementation of per-node abstraction and NodeManager.
// -----------------------------------------------------------------------

#include <openrai/node.h>
#include <openrai/thread_pool.h>
#include <openrai/log.h>

#include <algorithm>
#include <sstream>
#include <thread>
#include <future>
#include <chrono>
#include <filesystem>

namespace openrai {

// -----------------------------------------------------------------------
// Specialty to string conversion
// -----------------------------------------------------------------------
const char* specialty_to_string(NodeSpecialty s) {
    switch (s) {
        case NodeSpecialty::Reasoning:    return "Reasoning";
        case NodeSpecialty::Planning:     return "Planning";
        case NodeSpecialty::Memory:       return "Memory";
        case NodeSpecialty::Retrieval:    return "Retrieval";
        case NodeSpecialty::Coding:       return "Coding";
        case NodeSpecialty::Analysis:     return "Analysis";
        case NodeSpecialty::Integration:  return "Integration";
        case NodeSpecialty::Formatter:    return "Formatter";
        case NodeSpecialty::Debug:        return "Debug";
        case NodeSpecialty::Autonomy:     return "Autonomy";
    }
    return "Unknown";
}

// -----------------------------------------------------------------------
// Node implementation
// -----------------------------------------------------------------------

Node::Node(const std::string& name, NodeSpecialty specialty, int device_id)
    : name_(name)
    , specialty_(specialty)
    , device_id_(device_id)
{
    initialize_neural_net();
    initialize_vector_db();
    LOG("[Node:" << name_ << "] Created  specialty=" << specialty_to_string(specialty_)
        << "  device=" << device_id_);
}

Node::~Node() {
    LOG("[Node:" << name_ << "] Destroyed  processed=" << processed_count_);
}

void Node::initialize_neural_net() {
    neural_net_ = std::make_unique<GPUNeuralNet>();
    neural_net_->set_device(device_id_);
    // Default to FP16 for memory efficiency
    neural_net_->set_use_fp16(true);
}

void Node::initialize_vector_db() {
    // Create directory if it doesn't exist
    std::filesystem::path dir_path = "data/nodes";
    if (!std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
        LOG("[Node] Created directory: " << dir_path);
    }

    std::string db_path = "data/nodes/" + name_ + ".db";
    vector_db_ = std::make_unique<VectorDatabase>();
    if (!vector_db_->initialize(db_path)) {
        LOG("[Node:" << name_ << "] WARNING: Failed to initialize vector DB at " << db_path);
    }
}

NodeOutput Node::process(const NodeInput& input) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++processed_count_;

    NodeOutput output;

    // Encode input text
    float encoded[GPUNeuralNet::INPUT_DIM];
    encode_input_v2(input.text, encoded, GPUNeuralNet::INPUT_DIM);

    // Run through neural network
    NNResult nn_result = neural_net_->run_single(encoded);

    // Generate summary based on specialty and NN output
    output.summary = generate_summary(input, nn_result);

    // Copy or generate embedding
    if (!input.embedding.empty()) {
        output.embedding = input.embedding;
    } else {
        output.embedding.assign(encoded, encoded + GPUNeuralNet::INPUT_DIM);
    }

    // Set confidence and propagation flag
    output.confidence = nn_result.relevance;
    output.should_propagate = nn_result.should_out;

    // Store in vector DB if confidence is high
    if (output.confidence > 0.7f) {
        store_memory(output.summary, output.embedding, specialty_to_string(specialty_));
    }

    return output;
}

std::string Node::generate_summary(const NodeInput& input, const NNResult& nn_result) {
    std::ostringstream oss;

    // Specialty-specific summary formatting
    switch (specialty_) {
        case NodeSpecialty::Reasoning:
            oss << "[Reasoning] Analyzed: " << input.text.substr(0, 50);
            if (nn_result.relevance > 0.8f)
                oss << " ? Strong logical connection";
            break;

        case NodeSpecialty::Planning:
            oss << "[Planning] Goal: " << input.text.substr(0, 50);
            if (nn_result.priority > 0.7f)
                oss << " ? High priority action";
            break;

        case NodeSpecialty::Memory:
            oss << "[Memory] Stored: " << input.text.substr(0, 50);
            break;

        case NodeSpecialty::Retrieval:
            oss << "[Retrieval] Query: " << input.text.substr(0, 50);
            oss << " ? Found " << search_context(input.embedding, 5).size() << " matches";
            break;

        case NodeSpecialty::Coding:
            oss << "[Coding] Code analysis: " << input.text.substr(0, 50);
            break;

        case NodeSpecialty::Analysis:
            oss << "[Analysis] Pattern detected in: " << input.text.substr(0, 50);
            break;

        case NodeSpecialty::Integration:
            oss << "[Integration] Synthesizing: " << input.text.substr(0, 50);
            break;

        case NodeSpecialty::Formatter:
            oss << "[Formatter] Formatted output for: " << input.text.substr(0, 50);
            break;

        case NodeSpecialty::Debug:
            oss << "[Debug] Diagnostic: " << input.text.substr(0, 50);
            if (nn_result.relevance < 0.5f)
                oss << " ? Potential issue detected";
            break;

        case NodeSpecialty::Autonomy:
            oss << "[Autonomy] Self-directed action: " << input.text.substr(0, 50);
            break;
    }

    return oss.str();
}

std::vector<VectorSearchResult> Node::search_context(const std::vector<float>& query, size_t k) {
    std::lock_guard<std::mutex> lock(mutex_);
    return vector_db_->search(query, k);
}

void Node::store_memory(const std::string& text_value, const std::vector<float>& embedding, const std::string& tag) {
    std::lock_guard<std::mutex> lock(mutex_);
    vector_db_->store(text_value, embedding, tag);
}

void Node::set_device(int gpu_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    device_id_ = gpu_id;
    if (neural_net_) {
        neural_net_->set_device(gpu_id);
    }
}

void Node::set_use_fp16(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (neural_net_) {
        neural_net_->set_use_fp16(enable);
    }
}

void Node::set_config_path(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_path_ = path;
}

bool Node::is_fp16_enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return neural_net_ ? neural_net_->is_fp16_enabled() : false;
}

size_t Node::memory_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vector_db_ ? vector_db_->count() : 0;
}

// -----------------------------------------------------------------------
// NodeManager implementation
// -----------------------------------------------------------------------

NodeManager::NodeManager() {
    LOG("[NodeManager] Initialized");
}

NodeManager::~NodeManager() {
    LOG("[NodeManager] Destroyed  total_nodes=" << nodes_.size());
}

void NodeManager::register_node(std::unique_ptr<Node> node) {
    std::lock_guard<std::mutex> lock(mutex_);
    LOG("[NodeManager] Registering node: " << node->name() 
        << " (" << specialty_to_string(node->specialty()) << ")");
    nodes_.push_back(std::move(node));
}

Node* NodeManager::get_node(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return find_node_unlocked(name);
}

Node* NodeManager::find_node_unlocked(const std::string& name) {
    auto it = std::find_if(nodes_.begin(), nodes_.end(),
        [&name](const std::unique_ptr<Node>& n) { return n->name() == name; });
    return it != nodes_.end() ? it->get() : nullptr;
}

NodeOutput NodeManager::process_on_node(const std::string& node_name, const NodeInput& input) {
    Node* node = get_node(node_name);
    if (!node) {
        LOG("[NodeManager] Node not found: " << node_name);
        NodeOutput empty;
        empty.summary = "ERROR: Node not found";
        return empty;
    }
    return node->process(input);
}

std::vector<NodeOutput> NodeManager::process_parallel(
    const std::vector<std::string>& node_names,
    const NodeInput& input)
{
    std::vector<std::future<NodeOutput>> futures;
    futures.reserve(node_names.size());

    // Launch parallel processing
    for (const auto& name : node_names) {
        futures.push_back(std::async(std::launch::async, [this, name, input]() {
            try {
                return process_on_node(name, input);
            } catch (const std::exception& e) {
                LOG("[NodeManager] Exception in node " << name << ": " << e.what());
                NodeOutput error;
                error.summary = "[ERROR] Exception: " + std::string(e.what());
                error.confidence = 0.0f;
                return error;
            } catch (...) {
                LOG("[NodeManager] Unknown exception in node " << name);
                NodeOutput error;
                error.summary = "[ERROR] Unknown exception";
                error.confidence = 0.0f;
                return error;
            }
        }));
    }

    // Collect results
    std::vector<NodeOutput> results;
    results.reserve(futures.size());
    for (auto& fut : futures) {
        try {
            results.push_back(fut.get());
        } catch (const std::exception& e) {
            LOG("[NodeManager] Exception collecting result: " << e.what());
            NodeOutput error;
            error.summary = "[ERROR] " + std::string(e.what());
            error.confidence = 0.0f;
            results.push_back(error);
        }
    }

    return results;
}

std::vector<std::string> NodeManager::get_node_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(nodes_.size());
    for (const auto& node : nodes_) {
        names.push_back(node->name());
    }
    return names;
}

size_t NodeManager::node_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return nodes_.size();
}

void NodeManager::set_thread_pool(std::shared_ptr<ThreadPool> pool) {
    std::lock_guard<std::mutex> lock(mutex_);
    thread_pool_ = pool;
    LOG("[NodeManager] ThreadPool assigned for async task scheduling");
}

void NodeManager::enqueue_node_task(
    const std::string& node_name,
    const NodeInput& input,
    std::function<void(const NodeTaskResult&)> callback,
    std::chrono::milliseconds timeout)
{
    if (!thread_pool_) {
        LOG("[NodeManager] ERROR: ThreadPool not set for async operations");
        NodeTaskResult result;
        result.completed = false;
        result.timed_out = false;
        result.error_message = "ThreadPool not initialized";
        callback(result);
        return;
    }

    // Enqueue task to thread pool
    thread_pool_->enqueue([this, node_name, input, callback, timeout]() {
        NodeTaskResult result;

        auto start_time = std::chrono::steady_clock::now();

        // Check if node exists
        Node* node = get_node(node_name);
        if (!node) {
            result.completed = false;
            result.timed_out = false;
            result.error_message = "Node not found: " + node_name;
            callback(result);
            return;
        }

        // Launch async task with timeout
        auto future = std::async(std::launch::async, [node, input]() {
            return node->process(input);
        });

        // Wait for result with timeout
        auto status = future.wait_for(timeout);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);

        if (status == std::future_status::ready) {
            // Task completed within timeout
            result.output = future.get();
            result.completed = true;
            result.timed_out = false;
            LOG("[NodeManager] Task completed on node '" << node_name 
                << "' in " << elapsed.count() << "ms");
        } else {
            // Task timed out
            result.completed = false;
            result.timed_out = true;
            result.error_message = "Task timed out after " + std::to_string(timeout.count()) + "ms";
            LOG("[NodeManager] Task TIMED OUT on node '" << node_name 
                << "' after " << timeout.count() << "ms");
        }

        // Invoke callback with result
        callback(result);
    });
}

} // namespace openrai
