#pragma once

// -----------------------------------------------------------------------
// gpu_runtime.h
//   Public C++ interface to the CUDA neural-network backend.
//   Included from regular .cpp files - must stay free of CUDA types.
//
//   When OPENRAI_CUDA_ENABLED is NOT defined the implementation falls
//   back to a CPU implementation with the *same* API so the node
//   scheduler, streams/contexts and the 7.82 Hz loop behave identically
//   (only slower).
//
//   Concurrency model
//   -----------------
//   * One GPUNeuralNet owns one set of weights on the device.  Weights are
//     read-only at inference time, so any number of nodes may share one
//     network concurrently.
//   * One GPUStreamContext belongs to exactly one OpenRAI node.  It owns:
//       - a cudaStream_t created with a priority derived from the node
//         priority (cudaStreamCreateWithPriority),
//       - a "done" cudaEvent_t used for non-blocking polling,
//       - pinned host buffers + device workspace sized for the node's
//         max batch, so a tick never touches cudaMalloc.
//     All contexts live inside the device's primary CUDA context.  That is
//     deliberate: kernels from *different* CUDA contexts are time-sliced,
//     while streams inside one context can really overlap on the SMs.
//   * run_batch_async() enqueues H2D copy + all layer kernels + D2H copy on
//     the node's stream and returns immediately.  The node thread then
//     polls ctx.poll() (or blocks in ctx.synchronize()) and finally calls
//     fetch_results().
// -----------------------------------------------------------------------

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <cstddef>
#include <cstdint>

namespace openrai {

// One inference result per node
struct NNResult {
    float relevance   = 0.f;   // output[0]
    float importance  = 0.f;   // output[1]
    bool  should_out  = false; // output[2] > 0.5
    float priority    = 0.f;   // output[3]
};

// -----------------------------------------------------------------------
// NetArch
//   Fully-connected architecture description: dims[0] is the input width,
//   dims.back() the output width, everything in between is a LeakyReLU
//   hidden layer.  The last layer uses Sigmoid.
// -----------------------------------------------------------------------
struct NetArch {
    std::string         name = "custom";
    std::vector<size_t> dims;          // e.g. {32, 16384, ..., 4}

    size_t input_dim()   const { return dims.empty() ? 0 : dims.front(); }
    size_t output_dim()  const { return dims.empty() ? 0 : dims.back(); }
    size_t layer_count() const { return dims.size() < 2 ? 0 : dims.size() - 1; }
    size_t max_dim()     const;
    size_t param_count() const;
    size_t weight_bytes() const { return param_count() * sizeof(float); }
    bool   valid()       const;
    std::string describe() const;   // "32->16384->...->4"

    // The original OpenRAI 12-layer / ~3.98B parameter network.
    static NetArch full_4b();
    // Small per-expert network (default hidden width 1024, 4 hidden layers).
    static NetArch expert_lite(size_t hidden = 1024);
    // Small judgement network for the primary node in CPU / lite mode.
    static NetArch primary_lite();
    // Parse "32,1024,512,4" style strings (used by OPENRAI_* env vars).
    static NetArch parse(const std::string& csv, const std::string& name = "custom");
};

// -----------------------------------------------------------------------
// GPUStreamContext
//   Per-node execution context (see header comment).  Opaque - the
//   implementation lives in gpu_runtime.cu / gpu_runtime_cpu.cpp.
// -----------------------------------------------------------------------
class GPUStreamContext {
public:
    // owner     : node name, for logging
    // priority  : node priority 0..100 (100 = most important).  Mapped onto
    //             the CUDA stream-priority range of the device.
    // max_batch : largest N that will be passed to run_batch_async()
    // arch      : network that will be executed in this context (sizes the
    //             activation workspace)
    GPUStreamContext(std::string owner, int priority, size_t max_batch, const NetArch& arch);
    ~GPUStreamContext();

    GPUStreamContext(const GPUStreamContext&)            = delete;
    GPUStreamContext& operator=(const GPUStreamContext&) = delete;

    bool   is_cuda()       const;   // true when backed by a real cudaStream_t
    bool   poll()          const;   // true when all enqueued work finished
    void   synchronize();           // block until all enqueued work finished
    void   cancel();                // abort: later launches become no-ops
    bool   cancelled()     const;
    void   reset_cancel();

    int    priority()      const;   // node priority (0..100)
    int    cuda_priority() const;   // effective cudaStream priority (0 on CPU)
    int    device()        const;   // CUDA device ordinal (-1 on CPU)
    size_t max_batch()     const;
    const std::string& owner() const;
    uint64_t launches()    const;   // number of run_batch_async() calls
    uintptr_t native_handle() const; // cudaStream_t value (0 on CPU) - for logs / profilers

    // Implementation access (only used by GPUNeuralNet)
    struct Impl;
    Impl* impl() const { return impl_.get(); }

private:
    std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------
// GPUNeuralNet
//   Manages one set of weights on the device and runs batched inference.
//
//   Default architecture (ULTRA-DEEP & WIDE - 3D AI Infrastructure):
//     12 layers total
//     input  32
//     hidden 16384  LeakyReLU
//     hidden 24576  LeakyReLU   (x4)
//     hidden 16384  LeakyReLU   (x2)
//     hidden 8192 / 4096 / 2048 / 1024  LeakyReLU
//     output    4   Sigmoid
//   Total params: ~3.98B  (4 billion scale)
//
//   Any other NetArch can be passed - expert nodes use small per-domain
//   networks so that 8-16 of them fit next to the primary network.
// -----------------------------------------------------------------------
class GPUNeuralNet {
public:
    static constexpr size_t INPUT_DIM  = 32;
    static constexpr size_t OUTPUT_DIM = 4;

    // trainable=true keeps a host copy of the weights so train_batch() can
    // update them (sync_weights_to_device() pushes them to the GPU).
    explicit GPUNeuralNet(NetArch arch = NetArch::full_4b(), unsigned seed = 42, bool trainable = false);
    ~GPUNeuralNet();

    GPUNeuralNet(const GPUNeuralNet&)            = delete;
    GPUNeuralNet& operator=(const GPUNeuralNet&) = delete;

    // ---- Synchronous API (legacy, uses an internal default context) ----
    void run_batch(const float* inputs, size_t N,
                   std::vector<NNResult>& results) const;
    NNResult run_single(const float* input) const;

    // ---- Asynchronous, per-node-stream API ----
    // Enqueue a forward pass of N rows (row = arch().input_dim() floats)
    // on ctx's stream.  N must be <= ctx.max_batch().  Returns false if the
    // context is cancelled or the request is invalid (nothing enqueued).
    bool run_batch_async(const float* inputs, size_t N, GPUStreamContext& ctx) const;

    // Copy the results of the last run_batch_async() out of the context.
    // Only valid once ctx.poll() == true (or after ctx.synchronize()).
    void fetch_results(const GPUStreamContext& ctx, size_t N,
                       std::vector<NNResult>& results) const;

    // ---- Training (CPU backprop on the host weights) ----
    // inputs : N x input_dim, targets : N x output_dim in [0,1]
    // Binary cross-entropy on the sigmoid outputs, SGD with momentum.
    // Returns the mean loss of the batch.  Call sync_weights_to_device()
    // when done so the CUDA streams use the learned weights.
    float  train_batch(const float* inputs, const float* targets, size_t N, float lr = 0.01f);
    void   sync_weights_to_device();
    bool   is_trainable() const { return trainable_; }
    size_t train_steps()  const { return train_steps_; }
    bool   save_weights(const std::string& path) const;
    bool   load_weights(const std::string& path);   // then sync_weights_to_device()

    const NetArch& arch()   const { return arch_; }
    size_t param_count()    const { return arch_.param_count(); }
    bool   is_cuda_active() const { return cuda_active_; }
    void   print_summary()  const;

private:
    NetArch arch_;
    bool    cuda_active_ = false;
    int     device_      = -1;
    bool    trainable_   = false;
    size_t  train_steps_ = 0;
    std::vector<std::vector<float>> vel_w_, vel_b_;   // momentum buffers

    // Device weight/bias pointers per layer (empty when CUDA disabled)
    std::vector<float*> d_w_;
    std::vector<float*> d_b_;

    // Host-side weights (CPU path, or kept only until upload succeeds)
    std::vector<std::vector<float>> h_w_;
    std::vector<std::vector<float>> h_b_;

    // Context used by the synchronous run_batch()
    mutable std::unique_ptr<GPUStreamContext> default_ctx_;
    mutable std::unique_ptr<std::mutex>       default_ctx_mutex_;

    void init_weights(unsigned seed);
    void upload_weights_to_device();
    void cpu_forward_batch(const float* inputs, size_t N, float* outputs,
                           const GPUStreamContext* ctx) const;
};

// -----------------------------------------------------------------------
// GPUVectorIndex
//   Brute-force cosine-similarity index that lives on the GPU.
//   * vectors are L2-normalised on insert and appended to a device matrix
//     [capacity x dim] (grows by doubling; a host mirror is kept so the
//     CPU fallback and rebuilds work without the device)
//   * search() uploads the query to the index's own stream, launches one
//     kernel that scores every stored vector, copies the scores back and
//     selects the top-k on the host (N*4 bytes - ~0.4 MB for 100k items)
//   * search_batch() scores B queries in one launch
//   Thread-safe: every call takes an internal mutex (one stream, one
//   scratch buffer).  This is what the memory expert and the reflection
//   stage use to search the vector memory.
// -----------------------------------------------------------------------
struct IndexHit {
    uint32_t index = 0;
    float    score = 0.f;   // cosine similarity
};

class GPUVectorIndex {
public:
    GPUVectorIndex(size_t dim, size_t initial_capacity = 8192, int priority = 60,
                   std::string owner = "vector-index");
    ~GPUVectorIndex();

    GPUVectorIndex(const GPUVectorIndex&)            = delete;
    GPUVectorIndex& operator=(const GPUVectorIndex&) = delete;

    bool   is_cuda()  const;
    size_t dim()      const;
    size_t size()     const;
    size_t capacity() const;

    // Append one vector (any length >= dim is accepted; extra values are
    // ignored, missing ones are zero).  Returns the item's index.
    uint32_t add(const float* vec, size_t len);
    // Replace the whole content (used after FIFO eviction / disk load)
    void rebuild(const std::vector<std::vector<float>>& vectors);
    void clear();

    std::vector<IndexHit> search(const float* query, size_t len, size_t k) const;
    // queries: B rows of `dim` floats
    std::vector<std::vector<IndexHit>> search_batch(const float* queries, size_t B, size_t k) const;

    double   last_search_us() const;   // wall time of the last search
    uint64_t searches()       const;
    uintptr_t native_handle() const;

    struct Impl;
    Impl* impl() const { return impl_.get(); }

private:
    std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------
// encode_input_v2
//   32-dimensional encoding.
//   Combines byte histogram + positional hash + length features.
// -----------------------------------------------------------------------
void encode_input_v2(const std::string& s,
                     float* out,          // caller provides float[32]
                     size_t dim = GPUNeuralNet::INPUT_DIM);

// Human readable backend name ("CUDA" / "CPU")
const char* gpu_backend_name();

// Free / total device memory in bytes of the selected CUDA device.
// Returns false (and zeros) when no CUDA device is available.
bool gpu_memory_info(size_t& free_bytes, size_t& total_bytes);

// -----------------------------------------------------------------------
// Concurrency self-test  (./openrai --gpu-selftest)
//   Proves that per-node streams really overlap on the device:
//     serial     : K forward passes on ONE stream, synchronised one by one
//     concurrent : K forward passes on K streams from one thread, then sync
//     threaded   : K threads, each driving its own stream (= expert nodes)
//   On a CUDA build "concurrent" must be clearly faster than "serial"
//   when the kernels are small enough to share the GPU.
// -----------------------------------------------------------------------
struct GPUConcurrencyReport {
    bool        cuda = false;
    int         device = -1;
    size_t      streams = 0;
    size_t      batch = 0;
    int         repeats = 0;
    NetArch     arch;
    double      serial_ms = 0.0;
    double      concurrent_ms = 0.0;
    double      threaded_ms = 0.0;
    double      speedup_concurrent = 0.0;   // serial / concurrent
    double      speedup_threaded = 0.0;     // serial / threaded
    double      max_result_diff = 0.0;      // async vs sync results
    std::vector<std::string> stream_info;   // one line per stream context
    // vector-memory index benchmark (GPU cosine search)
    bool        index_cuda = false;
    size_t      index_items = 0;
    double      index_search_us = 0.0;      // average per single query
    double      index_batch_us = 0.0;       // average per batch of 8 queries
    bool        index_results_match = false;
    std::string verdict;
    std::string describe() const;
};
GPUConcurrencyReport run_gpu_concurrency_selftest(const NetArch& arch,
                                                  size_t streams = 8,
                                                  size_t batch   = 8,
                                                  int    repeats = 20);

} // namespace openrai
