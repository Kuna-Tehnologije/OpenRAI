// -----------------------------------------------------------------------
// gpu_runtime_cpu.cpp
//   CPU-only implementation of GPUNeuralNet + GPUStreamContext.
//   Compiled when OPENRAI_CUDA_ENABLED is NOT defined.
//   The CUDA version (src/cuda/gpu_runtime.cu) provides the same interface.
//
//   "Streams" on the CPU: every OpenRAI node already runs on its own
//   thread, so run_batch_async() simply executes the forward pass inline
//   on the calling (node) thread.  Concurrency between nodes therefore
//   still exists; only the device-side overlap is missing.  Cancellation
//   is honoured between layers so kill() interrupts a running node.
// -----------------------------------------------------------------------

#ifndef OPENRAI_CUDA_ENABLED

#include <openrai/gpu_runtime.h>
#include <openrai/log.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>

namespace openrai {

// ---- Xavier init -------------------------------------------------------
static void xavier_init(std::vector<float>& w,
                        size_t fan_in, size_t fan_out,
                        std::mt19937& rng)
{
    float lim = std::sqrt(6.f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> d(-lim, lim);
    for (auto& v : w) v = d(rng);
}

// ---- Layer forward -----------------------------------------------------
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
// GPUStreamContext (CPU)
// -----------------------------------------------------------------------
struct GPUStreamContext::Impl {
    std::string        owner;
    int                priority  = 50;
    size_t             max_batch = 1;
    NetArch            arch;
    std::vector<float> act_a;       // ping
    std::vector<float> act_b;       // pong
    std::vector<float> out;         // [max_batch x output_dim]
    std::atomic<bool>  cancelled{false};
    std::atomic<bool>  busy{false};
    std::atomic<uint64_t> launches{0};
};

GPUStreamContext::GPUStreamContext(std::string owner, int priority,
                                   size_t max_batch, const NetArch& arch)
    : impl_(std::make_unique<Impl>())
{
    impl_->owner     = std::move(owner);
    impl_->priority  = std::clamp(priority, 0, 100);
    impl_->max_batch = std::max<size_t>(1, max_batch);
    impl_->arch      = arch;
    const size_t ws  = impl_->max_batch * std::max<size_t>(1, arch.max_dim());
    impl_->act_a.assign(ws, 0.f);
    impl_->act_b.assign(ws, 0.f);
    impl_->out.assign(impl_->max_batch * std::max<size_t>(1, arch.output_dim()), 0.f);
}

GPUStreamContext::~GPUStreamContext() = default;

bool   GPUStreamContext::is_cuda()       const { return false; }
bool   GPUStreamContext::poll()          const { return !impl_->busy.load(); }
void   GPUStreamContext::synchronize()         { /* inline execution - nothing pending */ }
void   GPUStreamContext::cancel()              { impl_->cancelled = true; }
bool   GPUStreamContext::cancelled()     const { return impl_->cancelled.load(); }
void   GPUStreamContext::reset_cancel()        { impl_->cancelled = false; }
int    GPUStreamContext::priority()      const { return impl_->priority; }
int    GPUStreamContext::cuda_priority() const { return 0; }
int    GPUStreamContext::device()        const { return -1; }
size_t GPUStreamContext::max_batch()     const { return impl_->max_batch; }
const std::string& GPUStreamContext::owner() const { return impl_->owner; }
uint64_t GPUStreamContext::launches()    const { return impl_->launches.load(); }
uintptr_t GPUStreamContext::native_handle() const { return 0; }

// -----------------------------------------------------------------------
// GPUNeuralNet  (CPU implementation)
// -----------------------------------------------------------------------
GPUNeuralNet::GPUNeuralNet(NetArch arch, unsigned seed, bool trainable)
    : arch_(std::move(arch)),
      default_ctx_mutex_(std::make_unique<std::mutex>())
{
    if (!arch_.valid())
        throw std::invalid_argument("GPUNeuralNet: invalid NetArch");
    trainable_ = trainable;
    init_weights(seed);
    cuda_active_ = false;
    LOG("[GPUNeuralNet] CPU fallback  arch=" << arch_.name << " " << arch_.describe()
        << "  params=" << arch_.param_count()
        << "  (install CUDA Toolkit to enable GPU streams)");
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

void GPUNeuralNet::upload_weights_to_device() {}   // no-op in CPU path
void GPUNeuralNet::sync_weights_to_device() {}     // host weights are the weights

GPUNeuralNet::~GPUNeuralNet() = default;

void GPUNeuralNet::cpu_forward_batch(const float* inputs, size_t N, float* outputs,
                                     const GPUStreamContext* ctx) const
{
    const size_t L       = arch_.layer_count();
    const size_t max_dim = arch_.max_dim();
    const size_t in_dim  = arch_.input_dim();
    const size_t out_dim = arch_.output_dim();

    // Use the context workspace if available (avoids allocation per tick)
    std::vector<float> local_a, local_b;
    float* a = nullptr;
    float* b = nullptr;
    if (ctx && ctx->impl()->act_a.size() >= N * max_dim) {
        a = ctx->impl()->act_a.data();
        b = ctx->impl()->act_b.data();
    } else {
        local_a.assign(N * max_dim, 0.f);
        local_b.assign(N * max_dim, 0.f);
        a = local_a.data();
        b = local_b.data();
    }

    for (size_t n = 0; n < N; ++n) {
        const float* x   = inputs + n * in_dim;
        float*       cur = a;
        float*       nxt = b;
        const float* src = x;
        for (size_t l = 0; l < L; ++l) {
            if (ctx && ctx->cancelled()) {
                std::fill(outputs + n * out_dim, outputs + (n + 1) * out_dim, 0.f);
                return;
            }
            const size_t li = arch_.dims[l];
            const size_t lo = arch_.dims[l + 1];
            const bool last = (l + 1 == L);
            float* dst = last ? (outputs + n * out_dim) : nxt;
            layer_cpu(src, li, h_w_[l].data(), h_b_[l].data(), dst, lo, last);
            src = dst;
            std::swap(cur, nxt);
        }
    }
}

void GPUNeuralNet::run_batch(const float*           inputs,
                              size_t                 N,
                              std::vector<NNResult>& results) const
{
    results.resize(N);
    if (N == 0) return;
    const size_t out_dim = arch_.output_dim();
    std::vector<float> out(N * out_dim);
    cpu_forward_batch(inputs, N, out.data(), nullptr);
    for (size_t i = 0; i < N; ++i) {
        const float* r = out.data() + i * out_dim;
        results[i] = { r[0], r[1], r[2] > 0.5f, r[3] };
    }
}

NNResult GPUNeuralNet::run_single(const float* input) const {
    std::vector<NNResult> res;
    run_batch(input, 1, res);
    return res[0];
}

bool GPUNeuralNet::run_batch_async(const float* inputs, size_t N, GPUStreamContext& ctx) const {
    auto* im = ctx.impl();
    if (N == 0 || N > im->max_batch || im->cancelled.load()) return false;
    if (im->arch.max_dim() < arch_.max_dim() || im->arch.output_dim() != arch_.output_dim())
        return false;   // context workspace was sized for a different net
    im->busy = true;
    ++im->launches;
    cpu_forward_batch(inputs, N, im->out.data(), &ctx);
    im->busy = false;
    return true;
}

void GPUNeuralNet::fetch_results(const GPUStreamContext& ctx, size_t N,
                                 std::vector<NNResult>& results) const
{
    const auto* im = ctx.impl();
    const size_t out_dim = arch_.output_dim();
    N = std::min(N, im->max_batch);
    results.resize(N);
    for (size_t i = 0; i < N; ++i) {
        const float* r = im->out.data() + i * out_dim;
        results[i] = { r[0], r[1], r[2] > 0.5f, r[3] };
    }
}

void GPUNeuralNet::print_summary() const {
    LOG("[GPUNeuralNet] CPU fallback"
        "  arch=" << arch_.name << " " << arch_.describe()
        << "  params=" << arch_.param_count()
        << " (~" << (static_cast<double>(arch_.param_count()) / 1e9) << "B)"
        << "  activations=LeakyReLU/Sigmoid");
}

// -----------------------------------------------------------------------
// GPUVectorIndex (CPU)
// -----------------------------------------------------------------------
struct GPUVectorIndex::Impl {
    std::string        owner;
    size_t             dim = 32;
    std::vector<float> vecs;        // [n x dim], normalised
    size_t             n = 0;
    mutable std::mutex mutex;
    mutable double     last_us = 0.0;
    mutable uint64_t   searches = 0;
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
    (void)priority;
    impl_->owner = std::move(owner);
    impl_->dim   = std::max<size_t>(1, dim);
    impl_->vecs.reserve(std::max<size_t>(1, initial_capacity) * impl_->dim);
}

GPUVectorIndex::~GPUVectorIndex() = default;

bool   GPUVectorIndex::is_cuda()  const { return false; }
size_t GPUVectorIndex::dim()      const { return impl_->dim; }
size_t GPUVectorIndex::size()     const { std::lock_guard<std::mutex> lk(impl_->mutex); return impl_->n; }
size_t GPUVectorIndex::capacity() const { std::lock_guard<std::mutex> lk(impl_->mutex); return impl_->vecs.capacity() / impl_->dim; }
double GPUVectorIndex::last_search_us() const { std::lock_guard<std::mutex> lk(impl_->mutex); return impl_->last_us; }
uint64_t GPUVectorIndex::searches() const { std::lock_guard<std::mutex> lk(impl_->mutex); return impl_->searches; }
uintptr_t GPUVectorIndex::native_handle() const { return 0; }

uint32_t GPUVectorIndex::add(const float* vec, size_t len) {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    const size_t d = impl_->dim;
    impl_->vecs.resize((impl_->n + 1) * d);
    normalise_into(vec, len, impl_->vecs.data() + impl_->n * d, d);
    return static_cast<uint32_t>(impl_->n++);
}

void GPUVectorIndex::rebuild(const std::vector<std::vector<float>>& vectors) {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    const size_t d = impl_->dim;
    impl_->vecs.assign(vectors.size() * d, 0.f);
    for (size_t i = 0; i < vectors.size(); ++i)
        normalise_into(vectors[i].data(), vectors[i].size(), impl_->vecs.data() + i * d, d);
    impl_->n = vectors.size();
}

void GPUVectorIndex::clear() {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    impl_->vecs.clear();
    impl_->n = 0;
}

std::vector<std::vector<IndexHit>> GPUVectorIndex::search_batch(const float* queries, size_t B, size_t k) const {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    const auto t0 = std::chrono::steady_clock::now();
    const size_t d = impl_->dim, n = impl_->n;
    std::vector<std::vector<IndexHit>> out(B);
    std::vector<float> q(d), scores(n);
    for (size_t b = 0; b < B; ++b) {
        normalise_into(queries + b * d, d, q.data(), d);
        for (size_t i = 0; i < n; ++i) {
            const float* v = impl_->vecs.data() + i * d;
            float acc = 0.f;
            for (size_t j = 0; j < d; ++j) acc += v[j] * q[j];
            scores[i] = acc;
        }
        topk_from_scores(scores.data(), n, k, out[b]);
    }
    impl_->last_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    ++impl_->searches;
    return out;
}

std::vector<IndexHit> GPUVectorIndex::search(const float* query, size_t len, size_t k) const {
    std::vector<float> q(impl_->dim, 0.f);
    for (size_t i = 0; i < impl_->dim && i < len; ++i) q[i] = query[i];
    auto r = search_batch(q.data(), 1, k);
    return r.empty() ? std::vector<IndexHit>{} : std::move(r.front());
}

const char* gpu_backend_name() { return "CPU"; }

bool gpu_memory_info(size_t& free_bytes, size_t& total_bytes) {
    free_bytes = total_bytes = 0;
    return false;
}

} // namespace openrai

#endif // !OPENRAI_CUDA_ENABLED
