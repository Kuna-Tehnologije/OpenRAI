// -----------------------------------------------------------------------
// gpu_runtime.cu
//   Implements GPUNeuralNet: weight init, upload, and batched inference.
//   Also provides the CPU fallback path used when CUDA is not available.
//
//   This file is compiled by nvcc when OPENRAI_CUDA_ENABLED is defined,
//   otherwise gpu_runtime_cpu.cpp provides the CPU-only implementation.
// -----------------------------------------------------------------------

#include <openrai/gpu_runtime.h>
#include <openrai/log.h>

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>

// Forward declaration of the C-interface kernel launcher
extern "C" void launch_layer(
    const float* d_in, const float* d_W, const float* d_b, float* d_out,
    int batch, int in_dim, int out_dim, int use_sigmoid);

namespace openrai {

// Layer dimensions
static constexpr int L0_IN  = 32,  L0_OUT = 256;
static constexpr int L1_IN  = 256, L1_OUT = 512;
static constexpr int L2_IN  = 512, L2_OUT = 256;
static constexpr int L3_IN  = 256, L3_OUT = 128;
static constexpr int L4_IN  = 128, L4_OUT = 4;

static constexpr int TOTAL_PARAMS =
    L0_IN*L0_OUT + L0_OUT +
    L1_IN*L1_OUT + L1_OUT +
    L2_IN*L2_OUT + L2_OUT +
    L3_IN*L3_OUT + L3_OUT +
    L4_IN*L4_OUT + L4_OUT;   // ~330k

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
// GPUNeuralNet constructor
// -----------------------------------------------------------------------
GPUNeuralNet::GPUNeuralNet() {
    init_weights(42);
    upload_weights_to_device();
}

void GPUNeuralNet::init_weights(unsigned seed) {
    std::mt19937 rng(seed);

    auto alloc_layer = [&](std::vector<float>& w, std::vector<float>& b,
                            size_t in, size_t out) {
        w.resize(out * in);
        b.resize(out, 0.f);
        xavier_init(w, in, out, rng);
    };

    alloc_layer(h_w0_, h_b0_, L0_IN,  L0_OUT);
    alloc_layer(h_w1_, h_b1_, L1_IN,  L1_OUT);
    alloc_layer(h_w2_, h_b2_, L2_IN,  L2_OUT);
    alloc_layer(h_w3_, h_b3_, L3_IN,  L3_OUT);
    alloc_layer(h_w4_, h_b4_, L4_IN,  L4_OUT);
}

// -----------------------------------------------------------------------
// Upload weights to device
// -----------------------------------------------------------------------
static float* upload(const std::vector<float>& h) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(),
                          h.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    return d;
}

void GPUNeuralNet::upload_weights_to_device() {
    try {
        d_w0_ = upload(h_w0_);  d_b0_ = upload(h_b0_);
        d_w1_ = upload(h_w1_);  d_b1_ = upload(h_b1_);
        d_w2_ = upload(h_w2_);  d_b2_ = upload(h_b2_);
        d_w3_ = upload(h_w3_);  d_b3_ = upload(h_b3_);
        d_w4_ = upload(h_w4_);  d_b4_ = upload(h_b4_);
        cuda_active_ = true;
        LOG("[CUDA] Weights uploaded  params=" << TOTAL_PARAMS
            << "  arch=32->256->512->256->128->4");
    } catch (const std::exception& e) {
        LOG("[CUDA] Upload failed — CPU fallback: " << e.what());
        cuda_active_ = false;
    }
}

// -----------------------------------------------------------------------
// Destructor — free device memory
// -----------------------------------------------------------------------
GPUNeuralNet::~GPUNeuralNet() {
    auto safe_free = [](void* p) { if (p) cudaFree(p); };
    safe_free(d_w0_); safe_free(d_b0_);
    safe_free(d_w1_); safe_free(d_b1_);
    safe_free(d_w2_); safe_free(d_b2_);
    safe_free(d_w3_); safe_free(d_b3_);
    safe_free(d_w4_); safe_free(d_b4_);
}

// -----------------------------------------------------------------------
// run_batch — CUDA path
// -----------------------------------------------------------------------
void GPUNeuralNet::run_batch(const float*           inputs,
                              size_t                 N,
                              std::vector<NNResult>& results) const
{
    results.resize(N);

    if (!cuda_active_) {
        // CPU fallback: process each sample sequentially
        std::vector<float> out(OUTPUT_DIM);
        for (size_t i = 0; i < N; ++i) {
            cpu_forward(inputs + i * INPUT_DIM, out.data());
            results[i] = { out[0], out[1], out[2] > 0.5f, out[3] };
        }
        return;
    }

    // Allocate device buffers for this batch
    float* d_in   = nullptr;
    float* d_a0   = nullptr;  // after layer 0 ? shape [N x 256]
    float* d_a1   = nullptr;  // after layer 1 ? shape [N x 512]
    float* d_a2   = nullptr;  // after layer 2 ? shape [N x 256]
    float* d_a3   = nullptr;  // after layer 3 ? shape [N x 128]
    float* d_out  = nullptr;  // after layer 4 ? shape [N x   4]

    size_t Ni = static_cast<int>(N);

    CUDA_CHECK(cudaMalloc(&d_in,  Ni * L0_IN  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a0,  Ni * L0_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a1,  Ni * L1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a2,  Ni * L2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a3,  Ni * L3_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, Ni * L4_OUT * sizeof(float)));

    // Upload batch inputs
    CUDA_CHECK(cudaMemcpy(d_in, inputs,
                          Ni * L0_IN * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Forward pass — layer by layer
    launch_layer(d_in,  (float*)d_w0_, d_b0_, d_a0, Ni, L0_IN,  L0_OUT, 0); // LeakyReLU
    launch_layer(d_a0,  (float*)d_w1_, d_b1_, d_a1, Ni, L1_IN,  L1_OUT, 0);
    launch_layer(d_a1,  (float*)d_w2_, d_b2_, d_a2, Ni, L2_IN,  L2_OUT, 0);
    launch_layer(d_a2,  (float*)d_w3_, d_b3_, d_a3, Ni, L3_IN,  L3_OUT, 0);
    launch_layer(d_a3,  (float*)d_w4_, d_b4_, d_out, Ni, L4_IN, L4_OUT, 1); // Sigmoid

    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    std::vector<float> h_out(Ni * L4_OUT);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          Ni * L4_OUT * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Unpack
    for (size_t i = 0; i < N; ++i) {
        const float* r = h_out.data() + i * L4_OUT;
        results[i] = { r[0], r[1], r[2] > 0.5f, r[3] };
    }

    // Free batch buffers
    cudaFree(d_in);
    cudaFree(d_a0); cudaFree(d_a1); cudaFree(d_a2); cudaFree(d_a3);
    cudaFree(d_out);
}

NNResult GPUNeuralNet::run_single(const float* input) const {
    std::vector<NNResult> res;
    run_batch(input, 1, res);
    return res[0];
}

// -----------------------------------------------------------------------
// CPU fallback forward pass
// -----------------------------------------------------------------------
static void layer_cpu(const float* in,  size_t in_dim,
                      const float* W,   const float* b,
                      float*       out, size_t out_dim,
                      bool sigmoid_act)
{
    for (size_t o = 0; o < out_dim; ++o) {
        float acc = b[o];
        for (size_t i = 0; i < in_dim; ++i)
            acc += W[o * in_dim + i] * in[i];
        if (sigmoid_act)
            out[o] = 1.f / (1.f + std::exp(-acc));
        else
            out[o] = acc >= 0.f ? acc : 0.01f * acc;  // LeakyReLU
    }
}

void GPUNeuralNet::cpu_forward(const float* input, float* output) const {
    std::vector<float> a0(L0_OUT), a1(L1_OUT), a2(L2_OUT), a3(L3_OUT);

    layer_cpu(input,    L0_IN,  h_w0_.data(), h_b0_.data(), a0.data(), L0_OUT, false);
    layer_cpu(a0.data(),L1_IN,  h_w1_.data(), h_b1_.data(), a1.data(), L1_OUT, false);
    layer_cpu(a1.data(),L2_IN,  h_w2_.data(), h_b2_.data(), a2.data(), L2_OUT, false);
    layer_cpu(a2.data(),L3_IN,  h_w3_.data(), h_b3_.data(), a3.data(), L3_OUT, false);
    layer_cpu(a3.data(),L4_IN,  h_w4_.data(), h_b4_.data(), output,    L4_OUT, true);
}

// -----------------------------------------------------------------------
// print_summary
// -----------------------------------------------------------------------
void GPUNeuralNet::print_summary() const {
    LOG("[GPUNeuralNet] " << (cuda_active_ ? "CUDA ACTIVE" : "CPU FALLBACK")
        << "  arch=32->256->512->256->128->4"
        << "  params=" << TOTAL_PARAMS
        << "  activations=LeakyReLU/Sigmoid");
}

// -----------------------------------------------------------------------
// encode_input_v2
//   32-dimensional encoding:
//   [0..7]   byte histogram (normalised, 8 buckets)
//   [8..23]  positional character hash (16 positions)
//   [24]     string length normalised
//   [25..31] bigram hash features (7 slots)
// -----------------------------------------------------------------------
void encode_input_v2(const std::string& s, float* out, size_t dim) {
    std::fill(out, out + dim, 0.f);
    if (s.empty()) return;

    const size_t n = s.size();

    // Byte histogram — 8 buckets
    for (size_t i = 0; i < n; ++i) {
        size_t bucket = static_cast<unsigned char>(s[i]) % 8;
        out[bucket] += 1.f;
    }

    // Positional hash — 16 slots
    for (size_t i = 0; i < std::min(n, size_t(16)); ++i) {
        out[8 + i] = static_cast<float>(static_cast<unsigned char>(s[i])) / 255.f;
    }

    // Length feature
    out[24] = std::min(1.f, static_cast<float>(n) / 128.f);

    // Bigram hash — 7 slots
    for (size_t i = 0; i + 1 < n && i < 14; i += 2) {
        size_t slot = 25 + (i / 2) % 7;
        out[slot] += static_cast<float>(
            (static_cast<unsigned char>(s[i]) * 31u +
             static_cast<unsigned char>(s[i+1])) & 0xFF) / 255.f;
    }

    // Normalise byte histogram
    float maxv = *std::max_element(out, out + 8);
    if (maxv > 0.f)
        for (size_t i = 0; i < 8; ++i)
            out[i] = (out[i] / maxv) * 2.f - 1.f;
}

} // namespace openrai
