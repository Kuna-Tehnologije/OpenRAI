// -----------------------------------------------------------------------
// gpu_runtime.cu
//   CUDA implementation of GPUNeuralNet + GPUStreamContext.
//
//   * Weights: uploaded layer-by-layer (host copy released after each
//     upload so the 4B network needs ~2.4 GB of transient host memory
//     instead of 16 GB).
//   * Streams: one cudaStream_t per GPUStreamContext, created with
//     cudaStreamCreateWithPriority so a node's priority becomes a real
//     hardware scheduling priority.
//   * Contexts: all streams share the device's primary CUDA context
//     (kernels in different contexts cannot overlap; streams can).
//   * Zero allocation per tick: every context owns pinned host buffers
//     and device workspace sized for its max batch.
//
//   This file is compiled by nvcc when OPENRAI_CUDA_ENABLED is defined,
//   otherwise gpu_runtime_cpu.cpp provides the CPU-only implementation.
// -----------------------------------------------------------------------

#include <openrai/gpu_runtime.h>
#include <openrai/log.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>

// Kernel launchers (nn_kernels.cu)
extern "C" cudaError_t launch_layer_stream(
    const float* d_in, const float* d_W, const float* d_b, float* d_out,
    int batch, int in_dim, int out_dim, int use_sigmoid, cudaStream_t stream);
extern "C" cudaError_t launch_cosine_scores(
    const float* d_vecs, int N, int dim, const float* d_queries, int B,
    float* d_scores, cudaStream_t stream);

namespace openrai {

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t _e = (call);                                       \
        if (_e != cudaSuccess) {                                       \
            std::ostringstream _s;                                     \
            _s << "[CUDA] " << cudaGetErrorString(_e)                  \
               << " at " << __FILE__ << ":" << __LINE__;               \
            throw std::runtime_error(_s.str());                        \
        }                                                              \
    } while(0)

// -----------------------------------------------------------------------
// Device selection (OPENRAI_CUDA_DEVICE env var, default 0)
// -----------------------------------------------------------------------
static int select_device() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return -1;
    int dev = 0;
    if (const char* e = std::getenv("OPENRAI_CUDA_DEVICE")) {
        dev = std::atoi(e);
        if (dev < 0 || dev >= count) dev = 0;
    }
    return dev;
}

// -----------------------------------------------------------------------
// Xavier uniform init helper
// -----------------------------------------------------------------------
static void xavier_init(std::vector<float>& w,
                        size_t fan_in, size_t fan_out,
                        std::mt19937& rng)
{
    float lim = std::sqrt(6.f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> d(-lim, lim);
    for (auto& v : w) v = d(rng);
}

// -----------------------------------------------------------------------
// CPU layer (fallback when the upload failed)
// -----------------------------------------------------------------------
static void layer_cpu(const float* in,  size_t in_dim,
                      const float* W,   const float* b,
                      float*       out, size_t out_dim,
                      bool sigmoid_act)
{
    for (size_t o = 0; o < out_dim; ++o) {
        float acc = b[o];
        const float* w_row = W + o * in_dim;
        for (size_t i = 0; i < in_dim; ++i)
            acc += w_row[i] * in[i];
        if (sigmoid_act)
            out[o] = 1.f / (1.f + std::exp(-acc));
        else
            out[o] = acc >= 0.f ? acc : 0.01f * acc;  // LeakyReLU
    }
}

// -----------------------------------------------------------------------
// GPUStreamContext::Impl
// -----------------------------------------------------------------------
struct GPUStreamContext::Impl {
    std::string  owner;
    int          priority      = 50;
    int          cuda_priority = 0;
    int          device        = -1;
    size_t       max_batch     = 1;
    NetArch      arch;
    bool         cuda          = false;

    cudaStream_t stream = nullptr;
    cudaEvent_t  done   = nullptr;

    float* h_in  = nullptr;   // pinned  [max_batch x input_dim]
    float* h_out = nullptr;   // pinned  [max_batch x output_dim]
    float* d_in  = nullptr;   // device  [max_batch x input_dim]
    float* d_a   = nullptr;   // device  [max_batch x max_dim]  (ping)
    float* d_b   = nullptr;   // device  [max_batch x max_dim]  (pong)
    float* d_out = nullptr;   // device  [max_batch x output_dim]

    // CPU fallback workspace (used when cuda == false)
    std::vector<float> act_a, act_b, out;

    std::atomic<bool>     cancelled{false};
    std::atomic<bool>     busy{false};
    std::atomic<uint64_t> launches{0};

    void release_cuda() {
        if (stream) cudaStreamSynchronize(stream);
        if (h_in)  cudaFreeHost(h_in);
        if (h_out) cudaFreeHost(h_out);
        if (d_in)  cudaFree(d_in);
        if (d_a)   cudaFree(d_a);
        if (d_b)   cudaFree(d_b);
        if (d_out) cudaFree(d_out);
        if (done)   cudaEventDestroy(done);
        if (stream) cudaStreamDestroy(stream);
        h_in = h_out = d_in = d_a = d_b = d_out = nullptr;
        done = nullptr; stream = nullptr;
        cuda = false;
    }
};

GPUStreamContext::GPUStreamContext(std::string owner, int priority,
                                   size_t max_batch, const NetArch& arch)
    : impl_(std::make_unique<Impl>())
{
    auto& im     = *impl_;
    im.owner     = std::move(owner);
    im.priority  = std::clamp(priority, 0, 100);
    im.max_batch = std::max<size_t>(1, max_batch);
    im.arch      = arch;

    const size_t in_dim  = std::max<size_t>(1, arch.input_dim());
    const size_t out_dim = std::max<size_t>(1, arch.output_dim());
    const size_t max_dim = std::max<size_t>(1, arch.max_dim());

    im.device = select_device();
    if (im.device >= 0) {
        try {
            CUDA_CHECK(cudaSetDevice(im.device));

            // Map node priority (0..100, 100 = most important) onto the
            // device's stream priority range (numerically lower = higher).
            int least = 0, greatest = 0;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
            const double t = static_cast<double>(im.priority) / 100.0;
            im.cuda_priority = static_cast<int>(std::lround(least + (greatest - least) * t));

            CUDA_CHECK(cudaStreamCreateWithPriority(&im.stream, cudaStreamNonBlocking, im.cuda_priority));
            CUDA_CHECK(cudaEventCreateWithFlags(&im.done, cudaEventDisableTiming));

            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&im.h_in),
                                     im.max_batch * in_dim * sizeof(float), cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&im.h_out),
                                     im.max_batch * out_dim * sizeof(float), cudaHostAllocDefault));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_in),  im.max_batch * in_dim  * sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_a),   im.max_batch * max_dim * sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_b),   im.max_batch * max_dim * sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_out), im.max_batch * out_dim * sizeof(float)));

            // Record once so poll() is true before the first launch
            CUDA_CHECK(cudaEventRecord(im.done, im.stream));
            im.cuda = true;

            LOG("[CUDA] Stream context created"
                << " owner=" << im.owner
                << " device=" << im.device
                << " node_priority=" << im.priority
                << " stream_priority=" << im.cuda_priority
                << " (range " << least << ".." << greatest << ")"
                << " max_batch=" << im.max_batch
                << " workspace=" << ((2 * max_dim + in_dim + out_dim) * im.max_batch * sizeof(float) / 1024) << "KB");
        } catch (const std::exception& e) {
            LOG("[CUDA] Stream context for " << im.owner << " failed - CPU fallback: " << e.what());
            im.release_cuda();
        }
    }

    if (!im.cuda) {
        im.act_a.assign(im.max_batch * max_dim, 0.f);
        im.act_b.assign(im.max_batch * max_dim, 0.f);
    }
    im.out.assign(im.max_batch * out_dim, 0.f);
}

GPUStreamContext::~GPUStreamContext() {
    if (impl_) impl_->release_cuda();
}

bool GPUStreamContext::is_cuda() const { return impl_->cuda; }

bool GPUStreamContext::poll() const {
    if (!impl_->cuda) return !impl_->busy.load();
    return cudaEventQuery(impl_->done) == cudaSuccess;
}

void GPUStreamContext::synchronize() {
    if (impl_->cuda) cudaEventSynchronize(impl_->done);
}

void   GPUStreamContext::cancel()              { impl_->cancelled = true; }
bool   GPUStreamContext::cancelled()     const { return impl_->cancelled.load(); }
void   GPUStreamContext::reset_cancel()        { impl_->cancelled = false; }
int    GPUStreamContext::priority()      const { return impl_->priority; }
int    GPUStreamContext::cuda_priority() const { return impl_->cuda ? impl_->cuda_priority : 0; }
int    GPUStreamContext::device()        const { return impl_->cuda ? impl_->device : -1; }
size_t GPUStreamContext::max_batch()     const { return impl_->max_batch; }
const std::string& GPUStreamContext::owner() const { return impl_->owner; }
uint64_t GPUStreamContext::launches()    const { return impl_->launches.load(); }
uintptr_t GPUStreamContext::native_handle() const {
    return impl_->cuda ? reinterpret_cast<uintptr_t>(impl_->stream) : 0;
}

// -----------------------------------------------------------------------
// GPUNeuralNet
// -----------------------------------------------------------------------
GPUNeuralNet::GPUNeuralNet(NetArch arch, unsigned seed, bool trainable)
    : arch_(std::move(arch)),
      default_ctx_mutex_(std::make_unique<std::mutex>())
{
    if (!arch_.valid())
        throw std::invalid_argument("GPUNeuralNet: invalid NetArch");
    trainable_ = trainable;

    device_ = select_device();
    if (device_ < 0) {
        LOG("[CUDA] No CUDA device visible - CPU fallback for " << arch_.name);
        init_weights(seed);
        return;
    }

    // Fail early if the weights cannot fit
    size_t free_b = 0, total_b = 0;
    cudaSetDevice(device_);
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        const size_t need = arch_.weight_bytes() + (64u << 20);
        if (need > free_b) {
            LOG("[CUDA] " << arch_.name << " needs " << (need >> 20) << "MB but only "
                << (free_b >> 20) << "MB free on device " << device_ << " - CPU fallback");
            init_weights(seed);
            return;
        }
    }

    // Layer-by-layer: init on host -> upload -> release host copy
    // (trainable nets keep the host copy for train_batch()).
    std::mt19937 rng(seed);
    const size_t L = arch_.layer_count();
    d_w_.assign(L, nullptr);
    d_b_.assign(L, nullptr);
    if (trainable_) { h_w_.resize(L); h_b_.resize(L); }
    try {
        for (size_t l = 0; l < L; ++l) {
            const size_t in  = arch_.dims[l];
            const size_t out = arch_.dims[l + 1];
            std::vector<float> w(out * in), b(out, 0.f);
            xavier_init(w, in, out, rng);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_w_[l]), w.size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b_[l]), b.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_w_[l], w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b_[l], b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
            if (trainable_) { h_w_[l] = std::move(w); h_b_[l] = std::move(b); }
        }
        cuda_active_ = true;
        LOG("[CUDA] Weights uploaded  net=" << arch_.name
            << "  params=" << arch_.param_count() << " (~" << (arch_.param_count() / 1e9) << "B)"
            << "  arch=" << arch_.describe()
            << "  device=" << device_);
    } catch (const std::exception& e) {
        LOG("[CUDA] Upload failed for " << arch_.name << " - CPU fallback: " << e.what());
        for (auto p : d_w_) if (p) cudaFree(p);
        for (auto p : d_b_) if (p) cudaFree(p);
        d_w_.clear();
        d_b_.clear();
        cuda_active_ = false;
        init_weights(seed);
    }
}

void GPUNeuralNet::init_weights(unsigned seed) {
    std::mt19937 rng(seed);
    const size_t L = arch_.layer_count();
    h_w_.resize(L);
    h_b_.resize(L);
    for (size_t l = 0; l < L; ++l) {
        const size_t in  = arch_.dims[l];
        const size_t out = arch_.dims[l + 1];
        h_w_[l].resize(out * in);
        h_b_[l].assign(out, 0.f);
        xavier_init(h_w_[l], in, out, rng);
    }
}

void GPUNeuralNet::upload_weights_to_device() {
    // Upload of host-resident weights (only needed if init_weights() was
    // used first, e.g. after a failed constructor upload and a retry).
    if (cuda_active_ || device_ < 0 || h_w_.empty()) return;
    const size_t L = arch_.layer_count();
    d_w_.assign(L, nullptr);
    d_b_.assign(L, nullptr);
    try {
        cudaSetDevice(device_);
        for (size_t l = 0; l < L; ++l) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_w_[l]), h_w_[l].size() * sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b_[l]), h_b_[l].size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_w_[l], h_w_[l].data(), h_w_[l].size() * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_b_[l], h_b_[l].data(), h_b_[l].size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        cuda_active_ = true;
        if (!trainable_) { h_w_.clear(); h_b_.clear(); }
    } catch (const std::exception& e) {
        LOG("[CUDA] Upload failed - CPU fallback: " << e.what());
        for (auto p : d_w_) if (p) cudaFree(p);
        for (auto p : d_b_) if (p) cudaFree(p);
        d_w_.clear();
        d_b_.clear();
        cuda_active_ = false;
    }
}

void GPUNeuralNet::sync_weights_to_device() {
    if (!cuda_active_ || h_w_.size() != arch_.layer_count()) return;
    cudaSetDevice(device_);
    for (size_t l = 0; l < arch_.layer_count(); ++l) {
        cudaMemcpy(d_w_[l], h_w_[l].data(), h_w_[l].size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_[l], h_b_[l].data(), h_b_[l].size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
}

GPUNeuralNet::~GPUNeuralNet() {
    default_ctx_.reset();
    for (auto p : d_w_) if (p) cudaFree(p);
    for (auto p : d_b_) if (p) cudaFree(p);
}

// -----------------------------------------------------------------------
// CPU forward (fallback only)
// -----------------------------------------------------------------------
void GPUNeuralNet::cpu_forward_batch(const float* inputs, size_t N, float* outputs,
                                     const GPUStreamContext* ctx) const
{
    const size_t L       = arch_.layer_count();
    const size_t max_dim = arch_.max_dim();
    const size_t in_dim  = arch_.input_dim();
    const size_t out_dim = arch_.output_dim();

    std::vector<float> local_a, local_b;
    float* a = nullptr;
    float* b = nullptr;
    if (ctx && !ctx->impl()->cuda && ctx->impl()->act_a.size() >= N * max_dim) {
        a = ctx->impl()->act_a.data();
        b = ctx->impl()->act_b.data();
    } else {
        local_a.assign(N * max_dim, 0.f);
        local_b.assign(N * max_dim, 0.f);
        a = local_a.data();
        b = local_b.data();
    }

    for (size_t n = 0; n < N; ++n) {
        const float* src = inputs + n * in_dim;
        float* ping = a;
        float* pong = b;
        for (size_t l = 0; l < L; ++l) {
            if (ctx && ctx->cancelled()) {
                std::fill(outputs + n * out_dim, outputs + (n + 1) * out_dim, 0.f);
                return;
            }
            const bool last = (l + 1 == L);
            float* dst = last ? (outputs + n * out_dim) : ping;
            layer_cpu(src, arch_.dims[l], h_w_[l].data(), h_b_[l].data(), dst, arch_.dims[l + 1], last);
            src = dst;
            std::swap(ping, pong);
        }
    }
}

// -----------------------------------------------------------------------
// Asynchronous per-stream forward pass
// -----------------------------------------------------------------------
bool GPUNeuralNet::run_batch_async(const float* inputs, size_t N, GPUStreamContext& ctx) const {
    auto* im = ctx.impl();
    if (N == 0 || N > im->max_batch || im->cancelled.load()) return false;
    if (im->arch.max_dim() < arch_.max_dim() ||
        im->arch.output_dim() != arch_.output_dim() ||
        im->arch.input_dim()  != arch_.input_dim())
        return false;   // workspace sized for a different net

    const size_t in_dim  = arch_.input_dim();
    const size_t out_dim = arch_.output_dim();
    const size_t L       = arch_.layer_count();

    // --- CPU paths -------------------------------------------------------
    if (!cuda_active_) {
        im->busy = true;
        ++im->launches;
        cpu_forward_batch(inputs, N, im->out.data(), &ctx);
        im->busy = false;
        return true;
    }
    if (!im->cuda) {
        // Net on GPU but this context has no stream: run synchronously on
        // the shared default context and copy the result out.
        std::vector<NNResult> tmp;
        im->busy = true;
        ++im->launches;
        run_batch(inputs, N, tmp);
        for (size_t i = 0; i < N; ++i) {
            float* r = im->out.data() + i * out_dim;
            r[0] = tmp[i].relevance; r[1] = tmp[i].importance;
            r[2] = tmp[i].should_out ? 1.f : 0.f; r[3] = tmp[i].priority;
        }
        im->busy = false;
        return true;
    }

    // --- CUDA stream path ------------------------------------------------
    if (im->device != device_) {
        LOG("[CUDA] Context " << im->owner << " is on device " << im->device
            << " but weights live on device " << device_);
        return false;
    }

    cudaError_t e = cudaSetDevice(im->device);
    if (e != cudaSuccess) { LOG("[CUDA] cudaSetDevice: " << cudaGetErrorString(e)); return false; }

    // The previous launch must have finished before we overwrite h_in/d_in
    if (im->launches.load() > 0) cudaEventSynchronize(im->done);

    std::memcpy(im->h_in, inputs, N * in_dim * sizeof(float));
    e = cudaMemcpyAsync(im->d_in, im->h_in, N * in_dim * sizeof(float),
                        cudaMemcpyHostToDevice, im->stream);
    if (e != cudaSuccess) { LOG("[CUDA] H2D async: " << cudaGetErrorString(e)); return false; }

    const float* cur  = im->d_in;
    float*       ping = im->d_a;
    float*       pong = im->d_b;
    for (size_t l = 0; l < L; ++l) {
        const bool last = (l + 1 == L);
        float* dst = last ? im->d_out : ping;
        e = launch_layer_stream(cur, d_w_[l], d_b_[l], dst,
                                static_cast<int>(N),
                                static_cast<int>(arch_.dims[l]),
                                static_cast<int>(arch_.dims[l + 1]),
                                last ? 1 : 0, im->stream);
        if (e != cudaSuccess) {
            LOG("[CUDA] layer " << l << " launch failed on stream of " << im->owner
                << ": " << cudaGetErrorString(e));
            return false;
        }
        cur = dst;
        std::swap(ping, pong);
    }

    e = cudaMemcpyAsync(im->h_out, im->d_out, N * out_dim * sizeof(float),
                        cudaMemcpyDeviceToHost, im->stream);
    if (e != cudaSuccess) { LOG("[CUDA] D2H async: " << cudaGetErrorString(e)); return false; }

    e = cudaEventRecord(im->done, im->stream);
    if (e != cudaSuccess) { LOG("[CUDA] event record: " << cudaGetErrorString(e)); return false; }

    ++im->launches;
    return true;
}

void GPUNeuralNet::fetch_results(const GPUStreamContext& ctx, size_t N,
                                 std::vector<NNResult>& results) const
{
    auto* im = ctx.impl();
    const size_t out_dim = arch_.output_dim();
    N = std::min(N, im->max_batch);
    results.resize(N);

    const float* src = im->out.data();
    if (im->cuda && cuda_active_) {
        cudaEventSynchronize(im->done);   // no-op if already complete
        src = im->h_out;
    }
    for (size_t i = 0; i < N; ++i) {
        const float* r = src + i * out_dim;
        results[i] = { r[0], r[1], r[2] > 0.5f, r[3] };
    }
}

// -----------------------------------------------------------------------
// Synchronous API (shared default context, serialised by a mutex)
// -----------------------------------------------------------------------
void GPUNeuralNet::run_batch(const float*           inputs,
                              size_t                 N,
                              std::vector<NNResult>& results) const
{
    results.resize(N);
    if (N == 0) return;

    if (!cuda_active_) {
        const size_t out_dim = arch_.output_dim();
        std::vector<float> out(N * out_dim);
        cpu_forward_batch(inputs, N, out.data(), nullptr);
        for (size_t i = 0; i < N; ++i) {
            const float* r = out.data() + i * out_dim;
            results[i] = { r[0], r[1], r[2] > 0.5f, r[3] };
        }
        return;
    }

    std::lock_guard<std::mutex> lk(*default_ctx_mutex_);
    if (!default_ctx_ || default_ctx_->max_batch() < N) {
        default_ctx_ = std::make_unique<GPUStreamContext>(
            arch_.name + ":default", 50, std::max<size_t>(N, 32), arch_);
    }
    if (!run_batch_async(inputs, N, *default_ctx_)) {
        for (auto& r : results) r = NNResult{};
        return;
    }
    default_ctx_->synchronize();
    fetch_results(*default_ctx_, N, results);
}

NNResult GPUNeuralNet::run_single(const float* input) const {
    std::vector<NNResult> res;
    run_batch(input, 1, res);
    return res[0];
}

// -----------------------------------------------------------------------
// print_summary
// -----------------------------------------------------------------------
void GPUNeuralNet::print_summary() const {
    LOG("[GPUNeuralNet] " << (cuda_active_ ? "CUDA ACTIVE" : "CPU FALLBACK")
        << "  net=" << arch_.name
        << "  arch=" << arch_.describe()
        << "  params=" << arch_.param_count() << " (~" << (arch_.param_count() / 1e9) << "B)"
        << "  activations=LeakyReLU/Sigmoid"
        << "  device=" << device_);
}

// -----------------------------------------------------------------------
// GPUVectorIndex (CUDA)
// -----------------------------------------------------------------------
struct GPUVectorIndex::Impl {
    std::string        owner;
    size_t             dim = 32;
    size_t             n = 0;
    size_t             cap = 0;           // device capacity (vectors)
    std::vector<float> h_vecs;            // host mirror [n x dim], normalised
    bool               cuda = false;
    int                device = -1;
    int                cuda_priority = 0;
    cudaStream_t       stream = nullptr;
    float*             d_vecs = nullptr;  // [cap x dim]
    float*             d_q    = nullptr;  // [max_B x dim]
    float*             d_s    = nullptr;  // [max_B x cap]
    float*             h_q    = nullptr;  // pinned
    float*             h_s    = nullptr;  // pinned
    size_t             max_b = 16;
    mutable std::mutex mutex;
    mutable double     last_us = 0.0;
    mutable uint64_t   searches = 0;

    void release() {
        if (stream) cudaStreamSynchronize(stream);
        if (d_vecs) cudaFree(d_vecs);
        if (d_q)    cudaFree(d_q);
        if (d_s)    cudaFree(d_s);
        if (h_q)    cudaFreeHost(h_q);
        if (h_s)    cudaFreeHost(h_s);
        if (stream) cudaStreamDestroy(stream);
        d_vecs = d_q = d_s = h_q = h_s = nullptr;
        stream = nullptr;
        cuda = false;
    }

    // (re)allocate device storage for `new_cap` vectors and upload the mirror
    bool reserve_device(size_t new_cap) {
        if (!cuda) return false;
        if (new_cap <= cap && d_vecs) return true;
        float* nv = nullptr;
        float* ns = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&nv), new_cap * dim * sizeof(float)) != cudaSuccess) return false;
        if (cudaMalloc(reinterpret_cast<void**>(&ns), max_b * new_cap * sizeof(float)) != cudaSuccess) { cudaFree(nv); return false; }
        float* nh = nullptr;
        if (cudaHostAlloc(reinterpret_cast<void**>(&nh), max_b * new_cap * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
            cudaFree(nv); cudaFree(ns); return false;
        }
        if (n > 0) cudaMemcpy(nv, h_vecs.data(), n * dim * sizeof(float), cudaMemcpyHostToDevice);
        if (stream) cudaStreamSynchronize(stream);
        if (d_vecs) cudaFree(d_vecs);
        if (d_s)    cudaFree(d_s);
        if (h_s)    cudaFreeHost(h_s);
        d_vecs = nv; d_s = ns; h_s = nh; cap = new_cap;
        return true;
    }
};

static void normalise_into(const float* src, size_t len, float* dst, size_t dim) {
    double norm = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const float v = i < len ? src[i] : 0.f;
        dst[i] = v;
        norm += static_cast<double>(v) * v;
    }
    if (norm > 0.0) {
        const float inv = static_cast<float>(1.0 / std::sqrt(norm));
        for (size_t i = 0; i < dim; ++i) dst[i] *= inv;
    }
}

static void topk_from_scores(const float* scores, size_t n, size_t k, std::vector<IndexHit>& out) {
    out.clear();
    k = std::min(k, n);
    if (k == 0) return;
    std::vector<uint32_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = static_cast<uint32_t>(i);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [scores](uint32_t a, uint32_t b) { return scores[a] > scores[b]; });
    out.reserve(k);
    for (size_t i = 0; i < k; ++i) out.push_back(IndexHit{ idx[i], scores[idx[i]] });
}

GPUVectorIndex::GPUVectorIndex(size_t dim, size_t initial_capacity, int priority, std::string owner)
    : impl_(std::make_unique<Impl>())
{
    auto& im = *impl_;
    im.owner = std::move(owner);
    im.dim   = std::max<size_t>(1, dim);
    im.device = select_device();
    if (im.device >= 0) {
        try {
            CUDA_CHECK(cudaSetDevice(im.device));
            int least = 0, greatest = 0;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
            const double t = static_cast<double>(std::clamp(priority, 0, 100)) / 100.0;
            im.cuda_priority = static_cast<int>(std::lround(least + (greatest - least) * t));
            CUDA_CHECK(cudaStreamCreateWithPriority(&im.stream, cudaStreamNonBlocking, im.cuda_priority));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_q), im.max_b * im.dim * sizeof(float)));
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&im.h_q), im.max_b * im.dim * sizeof(float), cudaHostAllocDefault));
            im.cuda = true;
            if (!im.reserve_device(std::max<size_t>(256, initial_capacity)))
                throw std::runtime_error("cannot allocate index storage");
            LOG("[CUDA] Vector index created owner=" << im.owner << " dim=" << im.dim
                << " capacity=" << im.cap << " stream_priority=" << im.cuda_priority);
        } catch (const std::exception& e) {
            LOG("[CUDA] Vector index " << im.owner << " falls back to CPU: " << e.what());
            im.release();
        }
    }
    im.h_vecs.reserve(std::max<size_t>(256, initial_capacity) * im.dim);
}

GPUVectorIndex::~GPUVectorIndex() { if (impl_) impl_->release(); }

bool   GPUVectorIndex::is_cuda()  const { return impl_->cuda; }
size_t GPUVectorIndex::dim()      const { return impl_->dim; }
size_t GPUVectorIndex::size()     const { std::lock_guard<std::mutex> lk(impl_->mutex); return impl_->n; }
size_t GPUVectorIndex::capacity() const { std::lock_guard<std::mutex> lk(impl_->mutex); return impl_->cuda ? impl_->cap : impl_->h_vecs.capacity() / impl_->dim; }
double GPUVectorIndex::last_search_us() const { std::lock_guard<std::mutex> lk(impl_->mutex); return impl_->last_us; }
uint64_t GPUVectorIndex::searches() const { std::lock_guard<std::mutex> lk(impl_->mutex); return impl_->searches; }
uintptr_t GPUVectorIndex::native_handle() const { return impl_->cuda ? reinterpret_cast<uintptr_t>(impl_->stream) : 0; }

uint32_t GPUVectorIndex::add(const float* vec, size_t len) {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mutex);
    const size_t d = im.dim;
    im.h_vecs.resize((im.n + 1) * d);
    float* row = im.h_vecs.data() + im.n * d;
    normalise_into(vec, len, row, d);
    if (im.cuda) {
        if (im.n >= im.cap && !im.reserve_device(im.cap * 2)) {
            LOG("[CUDA] Vector index " << im.owner << " cannot grow - CPU fallback");
            im.release();
        }
    }
    if (im.cuda) {
        cudaSetDevice(im.device);
        cudaMemcpyAsync(im.d_vecs + im.n * d, row, d * sizeof(float), cudaMemcpyHostToDevice, im.stream);
    }
    return static_cast<uint32_t>(im.n++);
}

void GPUVectorIndex::rebuild(const std::vector<std::vector<float>>& vectors) {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mutex);
    const size_t d = im.dim;
    im.h_vecs.assign(vectors.size() * d, 0.f);
    for (size_t i = 0; i < vectors.size(); ++i)
        normalise_into(vectors[i].data(), vectors[i].size(), im.h_vecs.data() + i * d, d);
    im.n = vectors.size();
    if (im.cuda) {
        cudaSetDevice(im.device);
        if (im.n > im.cap && !im.reserve_device(std::max(im.n, im.cap * 2))) { im.release(); return; }
        if (im.n > 0)
            cudaMemcpyAsync(im.d_vecs, im.h_vecs.data(), im.n * d * sizeof(float), cudaMemcpyHostToDevice, im.stream);
        cudaStreamSynchronize(im.stream);   // h_vecs may be reused by the caller
    }
}

void GPUVectorIndex::clear() {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mutex);
    im.h_vecs.clear();
    im.n = 0;
}

std::vector<std::vector<IndexHit>> GPUVectorIndex::search_batch(const float* queries, size_t B, size_t k) const {
    auto& im = *impl_;
    std::lock_guard<std::mutex> lk(im.mutex);
    const auto t0 = std::chrono::steady_clock::now();
    const size_t d = im.dim, n = im.n;
    std::vector<std::vector<IndexHit>> out(B);
    if (B == 0 || n == 0) return out;

    bool done_on_gpu = false;
    if (im.cuda && B <= im.max_b) {
        cudaSetDevice(im.device);
        for (size_t b = 0; b < B; ++b) normalise_into(queries + b * d, d, im.h_q + b * d, d);
        cudaError_t e = cudaMemcpyAsync(im.d_q, im.h_q, B * d * sizeof(float), cudaMemcpyHostToDevice, im.stream);
        if (e == cudaSuccess)
            e = launch_cosine_scores(im.d_vecs, static_cast<int>(n), static_cast<int>(d),
                                     im.d_q, static_cast<int>(B), im.d_s, im.stream);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(im.h_s, im.d_s, B * n * sizeof(float), cudaMemcpyDeviceToHost, im.stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(im.stream);
        if (e == cudaSuccess) {
            for (size_t b = 0; b < B; ++b) topk_from_scores(im.h_s + b * n, n, k, out[b]);
            done_on_gpu = true;
        } else {
            LOG("[CUDA] Vector index search failed (" << cudaGetErrorString(e) << ") - CPU path");
        }
    }
    if (!done_on_gpu) {
        std::vector<float> q(d), scores(n);
        for (size_t b = 0; b < B; ++b) {
            normalise_into(queries + b * d, d, q.data(), d);
            for (size_t i = 0; i < n; ++i) {
                const float* v = im.h_vecs.data() + i * d;
                float acc = 0.f;
                for (size_t j = 0; j < d; ++j) acc += v[j] * q[j];
                scores[i] = acc;
            }
            topk_from_scores(scores.data(), n, k, out[b]);
        }
    }
    im.last_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    ++im.searches;
    return out;
}

std::vector<IndexHit> GPUVectorIndex::search(const float* query, size_t len, size_t k) const {
    std::vector<float> q(impl_->dim, 0.f);
    for (size_t i = 0; i < impl_->dim && i < len; ++i) q[i] = query[i];
    auto r = search_batch(q.data(), 1, k);
    return r.empty() ? std::vector<IndexHit>{} : std::move(r.front());
}

const char* gpu_backend_name() { return "CUDA"; }

bool gpu_memory_info(size_t& free_bytes, size_t& total_bytes) {
    free_bytes = total_bytes = 0;
    const int dev = select_device();
    if (dev < 0) return false;
    if (cudaSetDevice(dev) != cudaSuccess) return false;
    return cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess;
}

} // namespace openrai
