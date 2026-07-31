#include <openrai/core.h>
#include <openrai/thread_pool.h>
#include <openrai/memory.h>
#include <openrai/gpu_queue.h>
#include <openrai/reasoning.h>
#include <openrai/gpu_runtime.h>
#include <openrai/log.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>

namespace openrai {

static constexpr int  NODE_COUNT    = 100;
static constexpr int  TOP_K         = 10;   // short-term slots
static constexpr int  TOP_OUTPUT    = 3;    // outputs per iteration
static constexpr int  LOOP_WAIT_SEC = 2;

OpenRAI::OpenRAI()
    : thread_pool_     (std::make_unique<ThreadPool>(NODE_COUNT)),
      memory_system_   (std::make_unique<MemorySystem>()),
      gpu_queue_       (std::make_unique<GPUJobQueue>(128)),
      reasoning_module_(std::make_unique<ReasoningModule>(memory_system_.get())),
      gpu_nn_          (std::make_unique<GPUNeuralNet>())
{
    LOG("[OpenRAI] Engine initialized"
        "  nodes=" << NODE_COUNT <<
        "  threads=" << NODE_COUNT <<
        "  gpu_slots=128");
    gpu_nn_->print_summary();
}

OpenRAI::~OpenRAI() {
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
    { std::lock_guard<std::mutex> lk(pending_mutex_); pending_inputs_.push(input); }
    LOG("[INJECT] \"" << input << "\"");
}

// -----------------------------------------------------------------------
// main_loop
// -----------------------------------------------------------------------
void OpenRAI::main_loop() {
    LOG("=== OPENRAI MAIN LOOP STARTED ==="
        "  nodes=" << NODE_COUNT <<
        "  nn=32->256->512->256->128->4"
        "  cuda=" << (gpu_nn_->is_cuda_active() ? "ON" : "OFF") <<
        "  wait=" << LOOP_WAIT_SEC << "s");

    int iteration      = 0;
    int total_outputs  = 0;
    int total_injected = 0;

    // Pre-allocate encoding buffer: (NODE_COUNT + extras) x INPUT_DIM
    const size_t input_dim = GPUNeuralNet::INPUT_DIM;
    std::vector<float> batch_inputs;
    std::vector<std::string> batch_labels;

    while (is_running_) {
        ++iteration;
        auto t0 = std::chrono::steady_clock::now();
        LOG("\n===== ITERATION " << iteration << " =====");

        // ---- PRE-ITERATION: injected inputs ----------------------------
        std::vector<std::string> injected;
        {
            std::lock_guard<std::mutex> lk(pending_mutex_);
            while (!pending_inputs_.empty()) {
                injected.push_back(pending_inputs_.front());
                pending_inputs_.pop();
            }
        }

        // ---- Build batch: injected + 100 nodes -------------------------
        batch_inputs.clear();
        batch_labels.clear();

        // Injected entries first (high priority)
        for (const auto& raw : injected) {
            batch_inputs.resize(batch_inputs.size() + input_dim, 0.f);
            encode_input_v2(raw,
                batch_inputs.data() + (batch_labels.size() * input_dim),
                input_dim);
            batch_labels.push_back("[INJECTED] " + raw);
        }

        // Node entries
        for (int i = 0; i < NODE_COUNT; ++i) {
            std::string name = "Node_" + std::to_string(i)
                             + "_i"    + std::to_string(iteration);
            batch_inputs.resize(batch_inputs.size() + input_dim, 0.f);
            encode_input_v2(name,
                batch_inputs.data() + (batch_labels.size() * input_dim),
                input_dim);
            batch_labels.push_back(name);
        }

        const size_t total_samples = batch_labels.size();

        // ---- GPU batch inference (all samples at once) -----------------
        std::vector<NNResult> nn_results;
        {
            auto gpu_t0 = std::chrono::steady_clock::now();
            gpu_nn_->run_batch(batch_inputs.data(), total_samples, nn_results);
            auto gpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - gpu_t0).count();
            LOG("[GPU] Batch inference complete"
                "  samples=" << total_samples <<
                "  time=" << gpu_ms << "us"
                "  backend=" << (gpu_nn_->is_cuda_active() ? "CUDA" : "CPU"));
        }

        // ---- Write Outputs + enqueue GPU jobs --------------------------
        memory_system_->clear_current();

        for (size_t s = 0; s < total_samples; ++s) {
            const auto& r = nn_results[s];

            Output output;
            output.data             = batch_labels[s];
            output.thread_id        = static_cast<int>(s);
            output.timestamp        = std::chrono::system_clock::now();
            output.relevance_score  = r.relevance;
            output.importance_score = r.importance;
            output.nn_should_output = r.should_out;

            // Enqueue GPU job with NN result already filled
            GPUJob job;
            job.type          = s < injected.size() ? "injected" : "inference";
            job.input_data    = batch_labels[s];
            job.priority      = s < injected.size() ? 99 : static_cast<int>(s);
            job.output_tensor = { r.relevance, r.importance,
                                  r.should_out ? 1.f : 0.f, r.priority };
            gpu_queue_->enqueue_job(job);

            // Log every sample
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(3);
            if (s < injected.size())
                ss << "[PRE][INJ] \"" << injected[s] << "\"";
            else
                ss << "[P1][T" << std::setw(3) << (s - injected.size())
                   << "] " << batch_labels[s];
            ss << "  rel=" << r.relevance
               << "  imp=" << r.importance
               << "  out=" << (r.should_out ? "YES" : "no") << (r.should_out ? " VALUE: " + output.data : "");
            LOG(ss.str());

            memory_system_->add_current_output(output);
        }

        total_injected += static_cast<int>(injected.size());

        // Drain GPU queue
        { GPUJob c; int d = 0; while (gpu_queue_->try_dequeue_job(c)) ++d;
          LOG("[GPUQueue] Drained " << d << " jobs"); }

        // ---- PHASE 2: filter -> top-10 ---------------------------------
        LOG("[P2] Filter " << total_samples << " -> top-" << TOP_K);
        memory_system_->filter_to_short_term();

        // ---- PHASE 3: reasoning + adaptive output ----------------------
        LOG("[P3] Reasoning");
        auto short_term = memory_system_->get_short_term_memory();

        int iter_outputs = 0;
        if (!short_term.empty()) {
            // Sort by importance and force top-3 to output
            auto ranked = short_term;
            std::sort(ranked.begin(), ranked.end(),
                      [](const Output& a, const Output& b){
                          return a.importance_score > b.importance_score; });
            for (size_t k = 0; k < std::min<size_t>(TOP_OUTPUT, ranked.size()); ++k)
                ranked[k].nn_should_output = true;

            reasoning_module_->analyze(ranked);
            iter_outputs = static_cast<int>(
                std::min<size_t>(TOP_OUTPUT, ranked.size()));
        }

        total_outputs += iter_outputs;

        // ---- Summary ---------------------------------------------------
        auto iter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();

        LOG("[SUMMARY] iter="        << iteration
            << "  samples="          << total_samples
            << "  iter_outputs="     << iter_outputs
            << "  total_outputs="    << total_outputs
            << "  total_injected="   << total_injected
            << "  long_term="        << memory_system_->long_term_size()
            << "  iter_ms="          << iter_ms
            << "  cuda="             << (gpu_nn_->is_cuda_active() ? "ON" : "OFF"));

        LOG("[LOOP] Waiting " << LOOP_WAIT_SEC << "s...\n");
        std::this_thread::sleep_for(std::chrono::seconds(LOOP_WAIT_SEC));
    }

    LOG("=== OPENRAI MAIN LOOP STOPPED ===");
}

} // namespace openrai
