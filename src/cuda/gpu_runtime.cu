// -----------------------------------------------------------------------
// gpu_runtime.cu
//   Implements GPUNeuralNet: weight init, upload, and batched inference.
//   Also provides the CPU fallback path used when CUDA is not available.
//
//   This file is compiled by nvcc when OPENRAI_CUDA_ENABLED is defined,
//   otherwise gpu_runtime_cpu.cpp provides the CPU-only implementation.
//
//   FP16 Support:
//     When use_fp16_ is enabled, weights are converted to half precision
//     and stored on device as __half. Inference still uses float32 for
//     activations to maintain numerical stability.
// -----------------------------------------------------------------------

#include <openrai/gpu_runtime.h>
#include <openrai/log.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// 12-layer architecture for ~4B parameters
static constexpr int L0_IN  = 32,    L0_OUT = 16384;
static constexpr int L1_IN  = 16384, L1_OUT = 24576;
static constexpr int L2_IN  = 24576, L2_OUT = 24576;
static constexpr int L3_IN  = 24576, L3_OUT = 24576;
static constexpr int L4_IN  = 24576, L4_OUT = 24576;
static constexpr int L5_IN  = 24576, L5_OUT = 16384;
static constexpr int L6_IN  = 16384, L6_OUT = 16384;
static constexpr int L7_IN  = 16384, L7_OUT = 8192;
static constexpr int L8_IN  = 8192,  L8_OUT = 4096;
static constexpr int L9_IN  = 4096,  L9_OUT = 2048;
static constexpr int L10_IN = 2048,  L10_OUT = 1024;
static constexpr int L11_IN = 1024,  L11_OUT = 4;

static constexpr size_t TOTAL_PARAMS =
    static_cast<size_t>(L0_IN)*L0_OUT + L0_OUT +
    static_cast<size_t>(L1_IN)*L1_OUT + L1_OUT +
    static_cast<size_t>(L2_IN)*L2_OUT + L2_OUT +
    static_cast<size_t>(L3_IN)*L3_OUT + L3_OUT +
    static_cast<size_t>(L4_IN)*L4_OUT + L4_OUT +
    static_cast<size_t>(L5_IN)*L5_OUT + L5_OUT +
    static_cast<size_t>(L6_IN)*L6_OUT + L6_OUT +
    static_cast<size_t>(L7_IN)*L7_OUT + L7_OUT +
    static_cast<size_t>(L8_IN)*L8_OUT + L8_OUT +
    static_cast<size_t>(L9_IN)*L9_OUT + L9_OUT +
    static_cast<size_t>(L10_IN)*L10_OUT + L10_OUT +
    static_cast<size_t>(L11_IN)*L11_OUT + L11_OUT;   // ~3.98B

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
// FP32 to FP16 conversion helpers
// -----------------------------------------------------------------------
static uint16_t float32_to_fp16(float f) {
    __half h = __float2half(f);
    return *reinterpret_cast<uint16_t*>(&h);
}

static void convert_vec_to_fp16(const std::vector<float>& src, std::vector<uint16_t>& dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = float32_to_fp16(src[i]);
    }
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
    alloc_layer(h_w5_, h_b5_, L5_IN,  L5_OUT);
    alloc_layer(h_w6_, h_b6_, L6_IN,  L6_OUT);
    alloc_layer(h_w7_, h_b7_, L7_IN,  L7_OUT);
    alloc_layer(h_w8_, h_b8_, L8_IN,  L8_OUT);
    alloc_layer(h_w9_, h_b9_, L9_IN,  L9_OUT);
    alloc_layer(h_w10_, h_b10_, L10_IN, L10_OUT);
    alloc_layer(h_w11_, h_b11_, L11_IN, L11_OUT);
}

// -----------------------------------------------------------------------
// convert_weights_to_fp16 - Convert all fp32 weights to fp16
// -----------------------------------------------------------------------
void GPUNeuralNet::convert_weights_to_fp16() {
    convert_vec_to_fp16(h_w0_, h_w0_fp16_);   convert_vec_to_fp16(h_b0_, h_b0_fp16_);
    convert_vec_to_fp16(h_w1_, h_w1_fp16_);   convert_vec_to_fp16(h_b1_, h_b1_fp16_);
    convert_vec_to_fp16(h_w2_, h_w2_fp16_);   convert_vec_to_fp16(h_b2_, h_b2_fp16_);
    convert_vec_to_fp16(h_w3_, h_w3_fp16_);   convert_vec_to_fp16(h_b3_, h_b3_fp16_);
    convert_vec_to_fp16(h_w4_, h_w4_fp16_);   convert_vec_to_fp16(h_b4_, h_b4_fp16_);
    convert_vec_to_fp16(h_w5_, h_w5_fp16_);   convert_vec_to_fp16(h_b5_, h_b5_fp16_);
    convert_vec_to_fp16(h_w6_, h_w6_fp16_);   convert_vec_to_fp16(h_b6_, h_b6_fp16_);
    convert_vec_to_fp16(h_w7_, h_w7_fp16_);   convert_vec_to_fp16(h_b7_, h_b7_fp16_);
    convert_vec_to_fp16(h_w8_, h_w8_fp16_);   convert_vec_to_fp16(h_b8_, h_b8_fp16_);
    convert_vec_to_fp16(h_w9_, h_w9_fp16_);   convert_vec_to_fp16(h_b9_, h_b9_fp16_);
    convert_vec_to_fp16(h_w10_, h_w10_fp16_); convert_vec_to_fp16(h_b10_, h_b10_fp16_);
    convert_vec_to_fp16(h_w11_, h_w11_fp16_); convert_vec_to_fp16(h_b11_, h_b11_fp16_);
}

// -----------------------------------------------------------------------
// set_device - Select GPU device
// -----------------------------------------------------------------------
void GPUNeuralNet::set_device(int gpu_id) {
    device_id_ = gpu_id;
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || gpu_id < 0 || gpu_id >= device_count) {
        LOG("[GPUNeuralNet] Invalid device ID " << gpu_id << ", using device 0");
        device_id_ = 0;
    }
    CUDA_CHECK(cudaSetDevice(device_id_));
    LOG("[GPUNeuralNet] Device set to GPU " << device_id_);
}

// -----------------------------------------------------------------------
// set_use_fp16 - Enable/disable FP16 mode
// -----------------------------------------------------------------------
void GPUNeuralNet::set_use_fp16(bool enable) {
    if (use_fp16_ == enable) return;

    use_fp16_ = enable;
    LOG("[GPUNeuralNet] FP16 mode " << (enable ? "enabled" : "disabled"));

    if (use_fp16_) {
        convert_weights_to_fp16();
    }

    // Re-upload weights if already on device
    if (cuda_active_) {
        free_device_weights();
        upload_weights_to_device();
    }
}

// -----------------------------------------------------------------------
// free_device_weights - Free all device weight memory
// -----------------------------------------------------------------------
void GPUNeuralNet::free_device_weights() {
    auto safe_free = [](void* p) { if (p) cudaFree(p); };
    safe_free(d_w0_); safe_free(d_b0_);
    safe_free(d_w1_); safe_free(d_b1_);
    safe_free(d_w2_); safe_free(d_b2_);
    safe_free(d_w3_); safe_free(d_b3_);
    safe_free(d_w4_); safe_free(d_b4_);
    safe_free(d_w5_); safe_free(d_b5_);
    safe_free(d_w6_); safe_free(d_b6_);
    safe_free(d_w7_); safe_free(d_b7_);
    safe_free(d_w8_); safe_free(d_b8_);
    safe_free(d_w9_); safe_free(d_b9_);
    safe_free(d_w10_); safe_free(d_b10_);
    safe_free(d_w11_); safe_free(d_b11_);
    d_w0_ = d_b0_ = nullptr;
    d_w1_ = d_b1_ = nullptr;
    d_w2_ = d_b2_ = nullptr;
    d_w3_ = d_b3_ = nullptr;
    d_w4_ = d_b4_ = nullptr;
    d_w5_ = d_b5_ = nullptr;
    d_w6_ = d_b6_ = nullptr;
    d_w7_ = d_b7_ = nullptr;
    d_w8_ = d_b8_ = nullptr;
    d_w9_ = d_b9_ = nullptr;
    d_w10_ = d_b10_ = nullptr;
    d_w11_ = d_b11_ = nullptr;
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

static void* upload_fp16(const std::vector<uint16_t>& h) {
    void* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d, h.data(),
                          h.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    return d;
}

void GPUNeuralNet::upload_weights_to_device() {
    try {
        CUDA_CHECK(cudaSetDevice(device_id_));

        if (use_fp16_) {
            // Upload fp16 weights
            if (h_w0_fp16_.empty()) {
                convert_weights_to_fp16();
            }
            d_w0_ = upload_fp16(h_w0_fp16_);  d_b0_ = upload(h_b0_);
            d_w1_ = upload_fp16(h_w1_fp16_);  d_b1_ = upload(h_b1_);
            d_w2_ = upload_fp16(h_w2_fp16_);  d_b2_ = upload(h_b2_);
            d_w3_ = upload_fp16(h_w3_fp16_);  d_b3_ = upload(h_b3_);
            d_w4_ = upload_fp16(h_w4_fp16_);  d_b4_ = upload(h_b4_);
            d_w5_ = upload_fp16(h_w5_fp16_);  d_b5_ = upload(h_b5_);
            d_w6_ = upload_fp16(h_w6_fp16_);  d_b6_ = upload(h_b6_);
            d_w7_ = upload_fp16(h_w7_fp16_);  d_b7_ = upload(h_b7_);
            d_w8_ = upload_fp16(h_w8_fp16_);  d_b8_ = upload(h_b8_);
            d_w9_ = upload_fp16(h_w9_fp16_);  d_b9_ = upload(h_b9_);
            d_w10_ = upload_fp16(h_w10_fp16_); d_b10_ = upload(h_b10_);
            d_w11_ = upload_fp16(h_w11_fp16_); d_b11_ = upload(h_b11_);
            cuda_active_ = true;
            LOG("[CUDA] FP16 weights uploaded  params=" << TOTAL_PARAMS << " (~" << (TOTAL_PARAMS/1e9) << "B)"
                << "  device=" << device_id_
                << "  arch=32->16384->24576->24576->24576->24576->16384->16384->8192->4096->2048->1024->4");
        } else {
            // Upload fp32 weights
            d_w0_ = upload(h_w0_);  d_b0_ = upload(h_b0_);
            d_w1_ = upload(h_w1_);  d_b1_ = upload(h_b1_);
            d_w2_ = upload(h_w2_);  d_b2_ = upload(h_b2_);
            d_w3_ = upload(h_w3_);  d_b3_ = upload(h_b3_);
            d_w4_ = upload(h_w4_);  d_b4_ = upload(h_b4_);
            d_w5_ = upload(h_w5_);  d_b5_ = upload(h_b5_);
            d_w6_ = upload(h_w6_);  d_b6_ = upload(h_b6_);
            d_w7_ = upload(h_w7_);  d_b7_ = upload(h_b7_);
            d_w8_ = upload(h_w8_);  d_b8_ = upload(h_b8_);
            d_w9_ = upload(h_w9_);  d_b9_ = upload(h_b9_);
            d_w10_ = upload(h_w10_); d_b10_ = upload(h_b10_);
            d_w11_ = upload(h_w11_); d_b11_ = upload(h_b11_);
            cuda_active_ = true;
            LOG("[CUDA] FP32 weights uploaded  params=" << TOTAL_PARAMS << " (~" << (TOTAL_PARAMS/1e9) << "B)"
                << "  device=" << device_id_
                << "  arch=32->16384->24576->24576->24576->24576->16384->16384->8192->4096->2048->1024->4");
        }
    } catch (const std::exception& e) {
        LOG("[CUDA] Upload failed — CPU fallback: " << e.what());
        cuda_active_ = false;
    }
}

// -----------------------------------------------------------------------
// Destructor — free device memory
// -----------------------------------------------------------------------
GPUNeuralNet::~GPUNeuralNet() {
    free_device_weights();
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

    // Allocate device buffers for this batch (12 layers)
    float* d_in   = nullptr;
    float* d_a0   = nullptr;  // after layer 0  [N x 16384]
    float* d_a1   = nullptr;  // after layer 1  [N x 24576]
    float* d_a2   = nullptr;  // after layer 2  [N x 24576]
    float* d_a3   = nullptr;  // after layer 3  [N x 24576]
    float* d_a4   = nullptr;  // after layer 4  [N x 24576]
    float* d_a5   = nullptr;  // after layer 5  [N x 16384]
    float* d_a6   = nullptr;  // after layer 6  [N x 16384]
    float* d_a7   = nullptr;  // after layer 7  [N x 8192]
    float* d_a8   = nullptr;  // after layer 8  [N x 4096]
    float* d_a9   = nullptr;  // after layer 9  [N x 2048]
    float* d_a10  = nullptr;  // after layer 10 [N x 1024]
    float* d_out  = nullptr;  // after layer 11 [N x 4]

    size_t Ni = static_cast<int>(N);

    CUDA_CHECK(cudaMalloc(&d_in,  Ni * L0_IN  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a0,  Ni * L0_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a1,  Ni * L1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a2,  Ni * L2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a3,  Ni * L3_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a4,  Ni * L4_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a5,  Ni * L5_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a6,  Ni * L6_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a7,  Ni * L7_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a8,  Ni * L8_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a9,  Ni * L9_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a10, Ni * L10_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, Ni * L11_OUT * sizeof(float)));

    // Upload batch inputs
    CUDA_CHECK(cudaMemcpy(d_in, inputs,
                          Ni * L0_IN * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Forward pass — layer by layer (12 layers)
    launch_layer(d_in,  (float*)d_w0_,  d_b0_,  d_a0,  Ni, L0_IN,  L0_OUT,  0); // LeakyReLU
    launch_layer(d_a0,  (float*)d_w1_,  d_b1_,  d_a1,  Ni, L1_IN,  L1_OUT,  0);
    launch_layer(d_a1,  (float*)d_w2_,  d_b2_,  d_a2,  Ni, L2_IN,  L2_OUT,  0);
    launch_layer(d_a2,  (float*)d_w3_,  d_b3_,  d_a3,  Ni, L3_IN,  L3_OUT,  0);
    launch_layer(d_a3,  (float*)d_w4_,  d_b4_,  d_a4,  Ni, L4_IN,  L4_OUT,  0);
    launch_layer(d_a4,  (float*)d_w5_,  d_b5_,  d_a5,  Ni, L5_IN,  L5_OUT,  0);
    launch_layer(d_a5,  (float*)d_w6_,  d_b6_,  d_a6,  Ni, L6_IN,  L6_OUT,  0);
    launch_layer(d_a6,  (float*)d_w7_,  d_b7_,  d_a7,  Ni, L7_IN,  L7_OUT,  0);
    launch_layer(d_a7,  (float*)d_w8_,  d_b8_,  d_a8,  Ni, L8_IN,  L8_OUT,  0);
    launch_layer(d_a8,  (float*)d_w9_,  d_b9_,  d_a9,  Ni, L9_IN,  L9_OUT,  0);
    launch_layer(d_a9,  (float*)d_w10_, d_b10_, d_a10, Ni, L10_IN, L10_OUT, 0);
    launch_layer(d_a10, (float*)d_w11_, d_b11_, d_out, Ni, L11_IN, L11_OUT, 1); // Sigmoid

    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    std::vector<float> h_out(Ni * L11_OUT);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          Ni * L11_OUT * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Unpack
    for (size_t i = 0; i < N; ++i) {
        const float* r = h_out.data() + i * L11_OUT;
        results[i] = { r[0], r[1], r[2] > 0.5f, r[3] };
    }

    // Free batch buffers
    cudaFree(d_in);
    cudaFree(d_a0); cudaFree(d_a1); cudaFree(d_a2); cudaFree(d_a3);
    cudaFree(d_a4); cudaFree(d_a5); cudaFree(d_a6); cudaFree(d_a7);
    cudaFree(d_a8); cudaFree(d_a9); cudaFree(d_a10);
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
    std::vector<float> a0(L0_OUT), a1(L1_OUT), a2(L2_OUT), a3(L3_OUT),
                       a4(L4_OUT), a5(L5_OUT), a6(L6_OUT), a7(L7_OUT),
                       a8(L8_OUT), a9(L9_OUT), a10(L10_OUT);

    layer_cpu(input,     L0_IN,  h_w0_.data(),  h_b0_.data(),  a0.data(),  L0_OUT,  false);
    layer_cpu(a0.data(), L1_IN,  h_w1_.data(),  h_b1_.data(),  a1.data(),  L1_OUT,  false);
    layer_cpu(a1.data(), L2_IN,  h_w2_.data(),  h_b2_.data(),  a2.data(),  L2_OUT,  false);
    layer_cpu(a2.data(), L3_IN,  h_w3_.data(),  h_b3_.data(),  a3.data(),  L3_OUT,  false);
    layer_cpu(a3.data(), L4_IN,  h_w4_.data(),  h_b4_.data(),  a4.data(),  L4_OUT,  false);
    layer_cpu(a4.data(), L5_IN,  h_w5_.data(),  h_b5_.data(),  a5.data(),  L5_OUT,  false);
    layer_cpu(a5.data(), L6_IN,  h_w6_.data(),  h_b6_.data(),  a6.data(),  L6_OUT,  false);
    layer_cpu(a6.data(), L7_IN,  h_w7_.data(),  h_b7_.data(),  a7.data(),  L7_OUT,  false);
    layer_cpu(a7.data(), L8_IN,  h_w8_.data(),  h_b8_.data(),  a8.data(),  L8_OUT,  false);
    layer_cpu(a8.data(), L9_IN,  h_w9_.data(),  h_b9_.data(),  a9.data(),  L9_OUT,  false);
    layer_cpu(a9.data(), L10_IN, h_w10_.data(), h_b10_.data(), a10.data(), L10_OUT, false);
    layer_cpu(a10.data(),L11_IN, h_w11_.data(), h_b11_.data(), output,     L11_OUT, true);
}

// -----------------------------------------------------------------------
// print_summary
// -----------------------------------------------------------------------
void GPUNeuralNet::print_summary() const {
    LOG("[GPUNeuralNet] " << (cuda_active_ ? "CUDA ACTIVE" : "CPU FALLBACK")
        << "  arch=32->16384->24576->24576->24576->24576->16384->16384->8192->4096->2048->1024->4"
        << "  params=" << TOTAL_PARAMS << " (~" << (TOTAL_PARAMS / 1e9) << "B)"
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
