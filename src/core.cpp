#include <openrai/core.h>
#include <openrai/thread_pool.h>
#include <openrai/memory.h>
#include <openrai/gpu_queue.h>
#include <openrai/reasoning.h>
#include <openrai/gpu_runtime.h>
#include <openrai/semantic.h>
#include <openrai/lm_adapter.h>
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

static constexpr int NODE_COUNT = 100;
static constexpr int TOP_K = 10;
static constexpr int TOP_OUTPUT = 4;
static constexpr int REASONING_NODE_COUNT = 20;
static constexpr int LOOP_WAIT_MS = 33;
static constexpr const char* VECTOR_MEMORY_PATH = "data/vector_memory.db";
static constexpr const char* STEP_INPUT_POOL_PATH = "data/openrai_math_1M.csv";

// Load step inputs from CSV file only (manual entry)
static std::vector<std::string> load_step_input_pool_from_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG("[InputPool] CSV file not found: " << path);
        return {};
    }

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

    LOG("[InputPool] Loaded " << pool.size() << " inputs from CSV");
    return pool;
}

static const std::vector<std::string> STEP_INPUT_POOL = load_step_input_pool_from_csv(STEP_INPUT_POOL_PATH);

OpenRAI::OpenRAI()
    : thread_pool_(std::make_unique<ThreadPool>(NODE_COUNT)),
      memory_system_(std::make_unique<MemorySystem>()),
      gpu_queue_(std::make_unique<GPUJobQueue>(128)),
      reasoning_module_(std::make_unique<ReasoningModule>(memory_system_.get())),
      gpu_nn_(std::make_unique<GPUNeuralNet>()),
      tokenizer_(std::make_unique<Tokenizer>()),
      embedding_layer_(std::make_unique<EmbeddingLayer>(32)),
      vector_memory_(std::make_unique<VectorMemory>(32, 5000)),
      knowledge_graph_(std::make_unique<KnowledgeGraph>()),
      goal_system_(std::make_unique<GoalSystem>()),
      self_reflection_(std::make_unique<SelfReflection>())
{
    vector_memory_->set_persistence_path(VECTOR_MEMORY_PATH);
    if (vector_memory_->load_from_disk(VECTOR_MEMORY_PATH)) {
        LOG("[VectorMemory] Loaded persistent memory from " << VECTOR_MEMORY_PATH
            << " size=" << vector_memory_->size());
    } else {
        LOG("[VectorMemory] No persistent memory found, starting fresh");
    }

    LOG("[InputPool] CSV=" << STEP_INPUT_POOL_PATH << " count=" << STEP_INPUT_POOL.size());

    // Initialize LM Adapter (Cognition + Language Layer)
    LMConfig lm_config;
    lm_config.model_type = "http";  // Local LLM server
    lm_config.model_path = "http://127.0.0.1:1337";  // Your local server
    lm_config.context_length = 4096;
    lm_config.max_tokens = 512;
    lm_config.temperature = 0.7f;
    lm_config.use_gpu = true;

    lm_adapter_ = std::make_unique<LMAdapter>(lm_config);
    if (lm_adapter_->initialize()) {
        LOG("[LMAdapter] Language layer initialized: " << lm_adapter_->get_model_info());
    } else {
        LOG("[LMAdapter] Warning: Language layer not available");
    }

    LOG("[OpenRAI] Engine initialized"
        << " nodes=" << NODE_COUNT
        << " threads=" << NODE_COUNT
        << " gpu_slots=128"
        << " semantic=on"
        << " language=" << (lm_adapter_->is_ready() ? "on" : "mock"));
    gpu_nn_->print_summary();
}

OpenRAI::~OpenRAI() {
    if (vector_memory_) {
        if (vector_memory_->save_to_disk(VECTOR_MEMORY_PATH)) {
            LOG("[VectorMemory] Saved persistent memory to " << VECTOR_MEMORY_PATH
                << " size=" << vector_memory_->size());
        } else {
            LOG("[VectorMemory] Failed to save persistent memory");
        }
    }
    stop();
    LOG("[OpenRAI] Engine destroyed");
}

void OpenRAI::start() {
    if (is_running_.exchange(true)) return;
    LOG("[OpenRAI] Starting main loop...");
    std::thread([this] { main_loop(); }).detach();
}

void OpenRAI::stop() {
    is_running_ = false;
}

void OpenRAI::inject_input(const std::string& input) {
    {
        std::lock_guard<std::mutex> lk(pending_mutex_);
        pending_inputs_.push(input);
    }

    // Korisnik je odgovorio - resetuj waiting flag
    {
        std::lock_guard<std::mutex> lk(response_mutex_);
        waiting_for_user_response_ = false;
    }

    LOG("[INJECT] \"" << input << "\"");
}

std::string OpenRAI::generate_response(const std::string& prompt) {
    if (!lm_adapter_ || !lm_adapter_->is_ready()) {
        return "[Error] Language model not available";
    }

    // Gather cognitive context from OpenRAI memory
    std::vector<std::string> memory_context;
    auto short_term = memory_system_->get_short_term_memory();
    for (const auto& output : short_term) {
        if (memory_context.size() >= 5) break;
        memory_context.push_back(output.data);
    }

    // Gather knowledge graph entities (using neighbors to get entities)
    std::vector<std::string> graph_entities;
    if (!prompt.empty()) {
        auto neighbors = knowledge_graph_->neighbors(prompt.substr(0, 20), 10);
        for (const auto& neighbor : neighbors) {
            graph_entities.push_back(neighbor);
        }
    }

    // Generate with cognitive context
    auto response = lm_adapter_->generate_with_context(
        prompt, memory_context, graph_entities
    );

    // Store response in memory for future context
    std::vector<std::string> response_tokens = tokenizer_->tokenize(response.text);
    std::vector<float> embedding = embedding_layer_->embed(response_tokens);
    vector_memory_->store(response.text, embedding, "lm_response");

    // Update knowledge graph by ingesting both prompt and response
    knowledge_graph_->ingest(prompt, *tokenizer_);
    knowledge_graph_->ingest(response.text, *tokenizer_);

    LOG("[LM_RESPONSE] prompt=" << prompt.size() << " chars, "
        << "response=" << response.text.size() << " chars, "
        << "confidence=" << response.confidence);

    return response.text;
}

void OpenRAI::main_loop() {
    LOG("=== OPENRAI MAIN LOOP STARTED ==="
        << " nodes=" << NODE_COUNT
        << " nn=32->16k->24k->24k->24k->24k->16k->16k->8k->4k->2k->1k->4 (12-layer, ~4B params)"
        << " cuda=" << (gpu_nn_->is_cuda_active() ? "ON" : "OFF")
        << " semantic=Tokenizer->Embedding->VectorMemory->Graph->Goals->Reflection"
        << " wait=" << LOOP_WAIT_MS << "ms");

    int iteration = 0;
    int total_outputs = 0;
    int total_injected = 0;
    size_t step_input_index = 0;

    // Rate limiting: 1 output svakih 5 sekundi
    auto last_output_time = std::chrono::steady_clock::now();
    const auto output_cooldown = std::chrono::seconds(5);

    const size_t input_dim = GPUNeuralNet::INPUT_DIM;
    std::vector<float> batch_inputs;
    std::vector<std::string> batch_labels;

    while (is_running_) {
        ++iteration;
        auto t0 = std::chrono::steady_clock::now();
        LOG("\n===== ITERATION " << iteration << " =====");

        LOG("[PHASE 1/6] Inject one input for this round");
        if (!STEP_INPUT_POOL.empty()) {
            const auto& next_input = STEP_INPUT_POOL[step_input_index % STEP_INPUT_POOL.size()];
            step_input_index++;
            inject_input(next_input);
        }

        std::vector<std::string> injected;
        {
            std::lock_guard<std::mutex> lk(pending_mutex_);
            if (!pending_inputs_.empty()) {
                injected.push_back(pending_inputs_.front());
                pending_inputs_.pop();
            }
        }
        LOG("[PHASE 1/6] Dataset ready: injected=" << injected.size()
            << " nodes=" << NODE_COUNT
            << " total_inputs=" << (NODE_COUNT + injected.size()));

        batch_inputs.clear();
        batch_labels.clear();
        std::vector<std::string> batch_original_texts;

        for (const auto& raw : injected) {
            batch_inputs.resize(batch_inputs.size() + input_dim, 0.f);
            encode_input_v2(raw,
                            batch_inputs.data() + (batch_labels.size() * input_dim),
                            input_dim);
            batch_labels.push_back("[INJECTED] " + raw);
            batch_original_texts.push_back(raw);
        }

        for (int i = 0; i < NODE_COUNT; ++i) {
            std::string name = "Node_" + std::to_string(i) + "_i" + std::to_string(iteration);
            const std::string node_text = injected.empty() ? name : injected.front();
            batch_inputs.resize(batch_inputs.size() + input_dim, 0.f);
            encode_input_v2(node_text,
                            batch_inputs.data() + (batch_labels.size() * input_dim),
                            input_dim);
            batch_labels.push_back(name);
            batch_original_texts.push_back(node_text);
        }

        const size_t total_samples = batch_labels.size();

        LOG("[PHASE 2/6] GPU batch inference - encode inputs and score relevance/importance");
        std::vector<NNResult> nn_results;
        {
            auto gpu_t0 = std::chrono::steady_clock::now();
            gpu_nn_->run_batch(batch_inputs.data(), total_samples, nn_results);
            auto gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - gpu_t0).count();
            LOG("[GPU] Batch inference complete"
                << " samples=" << total_samples
                << " time=" << gpu_us << "us"
                << " backend=" << (gpu_nn_->is_cuda_active() ? "CUDA" : "CPU"));
        }

        LOG("[PHASE 3/6] Write outputs to current memory, vector memory, knowledge graph, and goals");
        memory_system_->clear_current();

        for (size_t s = 0; s < total_samples; ++s) {
            const auto& r = nn_results[s];
            const bool is_injected = s < injected.size();
            const std::string& original_text = batch_original_texts[s];

            Output output;
            output.data             = original_text;
            output.thread_id        = static_cast<int>(s);
            output.timestamp        = std::chrono::system_clock::now();
            output.relevance_score  = r.relevance;
            output.importance_score = r.importance;
            output.nn_should_output = r.should_out;

            GPUJob job;
            job.type          = is_injected ? "injected" : "inference";
            job.input_data    = original_text;
            job.priority      = is_injected ? 99 : static_cast<int>(s);
            job.output_tensor = { r.relevance, r.importance, r.should_out ? 1.f : 0.f, r.priority };
            gpu_queue_->enqueue_job(job);

            memory_system_->add_current_output(output);

            auto tokens = tokenizer_->tokenize(original_text);
            auto embedding = embedding_layer_->embed(tokens);
            vector_memory_->store(original_text, embedding, is_injected ? "injected" : "node");
            knowledge_graph_->ingest(original_text, *tokenizer_);
            goal_system_->update_from_text(original_text);
        }

        total_injected += static_cast<int>(injected.size());

        LOG("[PHASE 4/6] Drain GPU queue - simulate backend completion");
        {
            GPUJob completed;
            int drained = 0;
            while (gpu_queue_->try_dequeue_job(completed)) ++drained;
            LOG("[GPUQueue] Drained " << drained << " jobs");
        }

        LOG("[PHASE 5/6] Short-term filtering and primary reasoning");
        LOG("[P2] Filter " << total_samples << " -> top-" << TOP_K);
        memory_system_->filter_to_short_term();

        LOG("[P3] Primary reasoning");
        auto short_term = memory_system_->get_short_term_memory();
        if (!short_term.empty()) {
            auto ranked = short_term;
            std::sort(ranked.begin(), ranked.end(),
                      [](const Output& a, const Output& b) {
                          return a.importance_score > b.importance_score;
                      });
            for (size_t k = 0; k < std::min<size_t>(TOP_OUTPUT, ranked.size()); ++k)
                ranked[k].nn_should_output = true;

            reasoning_module_->analyze(ranked);

            LOG("[PHASE 6/6] Challenge / reflection loop nodes=" << REASONING_NODE_COUNT
                << " - judge if the conclusion should become final output");
            int iter_outputs = 0;
            for (size_t k = 0; k < std::min<size_t>(TOP_OUTPUT, ranked.size()); ++k) {
                const auto& candidate = ranked[k];
                auto context = vector_memory_->retrieve_context(candidate.data, *tokenizer_, *embedding_layer_, 4);
                auto reflection = self_reflection_->reflect(candidate.data, context, *knowledge_graph_, *goal_system_);

                std::ostringstream challenge;
                challenge << std::fixed << std::setprecision(3)
                          << "[REFLECTION] " << candidate.data
                          << "  conf=" << reflection.confidence
                          << "  final=" << (reflection.should_output ? "YES" : "no");
                if (!reflection.counterpoints.empty()) {
                    challenge << "  counterpoints=";
                    for (size_t i = 0; i < reflection.counterpoints.size(); ++i) {
                        if (i) challenge << ",";
                        challenge << reflection.counterpoints[i];
                    }
                }
                LOG(challenge.str());

                if (reflection.should_output) {
                    auto tokens = tokenizer_->tokenize(candidate.data);
                    std::string actual_value = tokenizer_->get_value_from_tokens(tokens, context, 50);

                    std::ostringstream final_text;
                    final_text << "Conclusion: " << reflection.conclusion
                               << " | confidence=" << std::fixed << std::setprecision(3) << reflection.confidence
                               << " | vector_hits=" << context.size()
                               << " | current_memory=" << memory_system_->get_current_outputs().size()
                               << " | short_term=" << short_term.size();

                    if (!context.empty()) {
                        final_text << " | best_vector_match=\"" << context.front().text << "\""
                                   << " (sim=" << context.front().similarity << ")";
                    }

                    final_text << " | tokens=" << tokens.size();

                    LOG("[FINAL OUTPUT] " << final_text.str());
                    LOG("[VALUE] " << actual_value);

                    // Output samo ako je confidence >= 0.9
                    if (reflection.confidence >= 0.9f) {
                        auto now = std::chrono::steady_clock::now();

                        // Provjeri da li èekamo odgovor od korisnika
                        bool waiting = false;
                        {
                            std::lock_guard<std::mutex> lk(response_mutex_);
                            if (waiting_for_user_response_) {
                                auto time_since_question = std::chrono::duration_cast<std::chrono::seconds>(
                                    now - last_question_time_).count();

                                if (time_since_question < 30) {
                                    // Još èekamo odgovor (< 30s)
                                    waiting = true;
                                    LOG("[GUI OUTPUT] Waiting for user response (" << time_since_question << "s / 30s)");
                                } else {
                                    // Timeout - korisnik nije odgovorio u 30s, nastavi normalno
                                    waiting_for_user_response_ = false;
                                    LOG("[GUI OUTPUT] User response timeout (30s) - resuming outputs");
                                }
                            }
                        }

                        if (!waiting) {
                            // Rate limiting: 1 output svakih 5 sekundi
                            auto time_since_last = std::chrono::duration_cast<std::chrono::seconds>(now - last_output_time).count();

                            if (time_since_last >= 5) {
                                if (output_callback_ && !context.empty()) {
                                    // Koristi best match iz vector memory, ne node data
                                    std::string memory_value = context.front().text;

                                    // PRVO: Generiši LLM response na osnovu memory value-a
                                    LOG("[LM] Generating response for: \"" << memory_value << "\"");
                                    std::string llm_response = generate_response(memory_value);

                                    // Pošalji LLM response u GUI
                                    output_callback_(llm_response);
                                    last_output_time = now;

                                    // Postavi waiting flag - èekamo odgovor korisnika
                                    {
                                        std::lock_guard<std::mutex> lk(response_mutex_);
                                        waiting_for_user_response_ = true;
                                        last_question_time_ = now;
                                    }

                                    LOG("[GUI OUTPUT] Sent LLM response to chat: \"" << llm_response << "\"");
                                    LOG("[GUI OUTPUT] Waiting for user response (timeout: 30s)");
                                }
                            } else {
                                LOG("[GUI OUTPUT] Skipped (cooldown: " << (5 - time_since_last) << "s remaining)");
                            }
                        }
                    } else {
                        LOG("[GUI OUTPUT] Skipped (confidence too low: " << std::fixed << std::setprecision(3) << reflection.confidence << " < 0.9)");
                    }

                    ++iter_outputs;
                }
            }

            total_outputs += iter_outputs;
        }

        auto iter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();

        auto top_goals = goal_system_->top_goals(3);
        std::ostringstream goal_line;
        goal_line << "[SUMMARY] iter=" << iteration
                  << " samples=" << total_samples
                  << " final_outputs=" << total_outputs
                  << " injected=" << total_injected
                  << " vector_memory=" << vector_memory_->size()
                  << " graph_edges=" << knowledge_graph_->edge_count()
                  << " iter_ms=" << iter_ms
                  << " cuda=" << (gpu_nn_->is_cuda_active() ? "ON" : "OFF")
                  << " goals=";
        for (size_t i = 0; i < top_goals.size(); ++i) {
            if (i) goal_line << "|";
            goal_line << top_goals[i].name << ":" << std::fixed << std::setprecision(2) << top_goals[i].score;
        }
        LOG(goal_line.str());

        LOG("[LOOP] Waiting " << LOOP_WAIT_MS << "ms...");
        std::this_thread::sleep_for(std::chrono::milliseconds(LOOP_WAIT_MS));
    }

    LOG("=== OPENRAI MAIN LOOP STOPPED ===");
}

} // namespace openrai
