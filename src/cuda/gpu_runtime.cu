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

// Global recursive mutex for CUDA operations across all GPUNeuralNet instances
// Recursive to allow cleanup code during exceptions without deadlock
static std::recursive_mutex global_cuda_mutex;

// 6-layer architecture for ~5M parameters
static constexpr int L0_IN  = 32,   L0_OUT = 512;
static constexpr int L1_IN  = 512,  L1_OUT = 1024;
static constexpr int L2_IN  = 1024, L2_OUT = 2048;
static constexpr int L3_IN  = 2048, L3_OUT = 1024;
static constexpr int L4_IN  = 1024, L4_OUT = 512;
static constexpr int L5_IN  = 512,  L5_OUT = 4;

static constexpr size_t TOTAL_PARAMS =
    static_cast<size_t>(L0_IN)*L0_OUT + L0_OUT +
    static_cast<size_t>(L1_IN)*L1_OUT + L1_OUT +
    static_cast<size_t>(L2_IN)*L2_OUT + L2_OUT +
    static_cast<size_t>(L3_IN)*L3_OUT + L3_OUT +
    static_cast<size_t>(L4_IN)*L4_OUT + L4_OUT +
    static_cast<size_t>(L5_IN)*L5_OUT + L5_OUT;   // ~5.26M

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
}

// -----------------------------------------------------------------------
// set_device - Select GPU device
// -----------------------------------------------------------------------
void GPUNeuralNet::set_device(int gpu_id) {
    std::lock_guard<std::recursive_mutex> lock(global_cuda_mutex);

    // Store old device for cleanup
    int old_device = device_id_;

    device_id_ = gpu_id;
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || gpu_id < 0 || gpu_id >= device_count) {
        LOG("[GPUNeuralNet] Invalid device ID " << gpu_id << ", using device 0");
        device_id_ = 0;
    }

    // If weights are already on a device, we need to move them
    if (cuda_active_ && old_device != device_id_) {
        // Free weights from old device
        CUDA_CHECK(cudaSetDevice(old_device));
        free_device_weights_internal();  // Internal call, mutex already held
        cuda_active_ = false;

        // Re-upload to new device
        CUDA_CHECK(cudaSetDevice(device_id_));
        upload_weights_to_device_internal();  // Internal call, mutex already held
    } else {
        CUDA_CHECK(cudaSetDevice(device_id_));
    }

    LOG("[GPUNeuralNet] Device set to GPU " << device_id_);
}

// -----------------------------------------------------------------------
// set_use_fp16 - Enable/disable FP16 mode
// -----------------------------------------------------------------------
void GPUNeuralNet::set_use_fp16(bool enable) {
    if (use_fp16_ == enable) return;

    std::lock_guard<std::recursive_mutex> lock(global_cuda_mutex);

    use_fp16_ = enable;
    LOG("[GPUNeuralNet] FP16 mode " << (enable ? "enabled" : "disabled"));

    if (use_fp16_) {
        convert_weights_to_fp16();
    }

    // Re-upload weights if already on device
    if (cuda_active_) {
        free_device_weights_internal();  // Internal call, mutex already held
        upload_weights_to_device_internal();  // Internal call, mutex already held
    }
}

// -----------------------------------------------------------------------
// free_device_weights_internal - Free all device weight memory (NO LOCK)
// -----------------------------------------------------------------------
void GPUNeuralNet::free_device_weights_internal() {
    // Make sure we're on the correct device before freeing
    if (cuda_active_) {
        cudaSetDevice(device_id_);
    }

    auto safe_free = [](void* p) { if (p) cudaFree(p); };
    safe_free(d_w0_); safe_free(d_b0_);
    safe_free(d_w1_); safe_free(d_b1_);
    safe_free(d_w2_); safe_free(d_b2_);
    safe_free(d_w3_); safe_free(d_b3_);
    safe_free(d_w4_); safe_free(d_b4_);
    safe_free(d_w5_); safe_free(d_b5_);
    d_w0_ = d_b0_ = nullptr;
    d_w1_ = d_b1_ = nullptr;
    d_w2_ = d_b2_ = nullptr;
    d_w3_ = d_b3_ = nullptr;
    d_w4_ = d_b4_ = nullptr;
    d_w5_ = d_b5_ = nullptr;
}

// -----------------------------------------------------------------------
// free_device_weights - Free all device weight memory (PUBLIC)
// -----------------------------------------------------------------------
void GPUNeuralNet::free_device_weights() {
    std::lock_guard<std::recursive_mutex> lock(global_cuda_mutex);
    free_device_weights_internal();
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

void GPUNeuralNet::upload_weights_to_device_internal() {
    LOG("[GPUNeuralNet] upload_weights_to_device_internal called - device_id=" << device_id_ << " use_fp16=" << use_fp16_);

    try {
        CUDA_CHECK(cudaSetDevice(device_id_));

        if (use_fp16_) {
            // Upload fp16 weights
            if (h_w0_fp16_.empty()) {
                LOG("[GPUNeuralNet] Converting weights to FP16...");
                convert_weights_to_fp16();
            }
            LOG("[GPUNeuralNet] Uploading FP16 weights...");
            d_w0_ = upload_fp16(h_w0_fp16_);  d_b0_ = upload(h_b0_);
            d_w1_ = upload_fp16(h_w1_fp16_);  d_b1_ = upload(h_b1_);
            d_w2_ = upload_fp16(h_w2_fp16_);  d_b2_ = upload(h_b2_);
            d_w3_ = upload_fp16(h_w3_fp16_);  d_b3_ = upload(h_b3_);
            d_w4_ = upload_fp16(h_w4_fp16_);  d_b4_ = upload(h_b4_);
            d_w5_ = upload_fp16(h_w5_fp16_);  d_b5_ = upload(h_b5_);
            cuda_active_ = true;
            LOG("[CUDA] FP16 weights uploaded  params=" << TOTAL_PARAMS << " (~" << (TOTAL_PARAMS/1e6) << "M)"
                << "  device=" << device_id_
                << "  arch=32->512->1024->2048->1024->512->4");
        } else {
            // Upload fp32 weights
            LOG("[GPUNeuralNet] Uploading FP32 weights...");
            d_w0_ = upload(h_w0_);  d_b0_ = upload(h_b0_);
            d_w1_ = upload(h_w1_);  d_b1_ = upload(h_b1_);
            d_w2_ = upload(h_w2_);  d_b2_ = upload(h_b2_);
            d_w3_ = upload(h_w3_);  d_b3_ = upload(h_b3_);
            d_w4_ = upload(h_w4_);  d_b4_ = upload(h_b4_);
            d_w5_ = upload(h_w5_);  d_b5_ = upload(h_b5_);
            cuda_active_ = true;
            LOG("[CUDA] FP32 weights uploaded  params=" << TOTAL_PARAMS << " (~" << (TOTAL_PARAMS/1e6) << "M)"
                << "  device=" << device_id_
                << "  arch=32->512->1024->2048->1024->512->4");
        }
    } catch (const std::exception& e) {
        LOG("[CUDA] Upload failed — CPU fallback: " << e.what());
        cuda_active_ = false;
    }
}

// -----------------------------------------------------------------------
// upload_weights_to_device - Public wrapper with lock
// -----------------------------------------------------------------------
void GPUNeuralNet::upload_weights_to_device() {
    std::lock_guard<std::recursive_mutex> lock(global_cuda_mutex);
    upload_weights_to_device_internal();
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
                                  std::lock_guard<std::recursive_mutex> lock(global_cuda_mutex);  // Global lock for all CUDA ops

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

    // Check if weights are uploaded
    if (!d_w0_ || !d_b0_ || !d_w1_ || !d_b1_ || !d_w2_ || !d_b2_ ||
        !d_w3_ || !d_b3_ || !d_w4_ || !d_b4_ || !d_w5_ || !d_b5_) {
        LOG("[GPUNeuralNet] ERROR: Device weights not uploaded! Using CPU fallback.");
        LOG("[GPUNeuralNet] d_w0_=" << d_w0_ << " d_w1_=" << d_w1_ << " d_w2_=" << d_w2_
            << " d_w3_=" << d_w3_ << " d_w4_=" << d_w4_ << " d_w5_=" << d_w5_);

        // Fallback to CPU
        std::vector<float> out(OUTPUT_DIM);
        for (size_t i = 0; i < N; ++i) {
            cpu_forward(inputs + i * INPUT_DIM, out.data());
            results[i] = { out[0], out[1], out[2] > 0.5f, out[3] };
        }
        return;
    }

    // Set the correct device before running inference
    CUDA_CHECK(cudaSetDevice(device_id_));

    // Allocate device buffers for this batch (6 layers)
    float* d_in   = nullptr;
    float* d_a0   = nullptr;  // after layer 0  [N x 512]
    float* d_a1   = nullptr;  // after layer 1  [N x 1024]
    float* d_a2   = nullptr;  // after layer 2  [N x 2048]
    float* d_a3   = nullptr;  // after layer 3  [N x 1024]
    float* d_a4   = nullptr;  // after layer 4  [N x 512]
    float* d_out  = nullptr;  // after layer 5  [N x 4]

    size_t Ni = static_cast<int>(N);

    CUDA_CHECK(cudaMalloc(&d_in,  Ni * L0_IN  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a0,  Ni * L0_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a1,  Ni * L1_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a2,  Ni * L2_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a3,  Ni * L3_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a4,  Ni * L4_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, Ni * L5_OUT * sizeof(float)));

    // Upload batch inputs
    CUDA_CHECK(cudaMemcpy(d_in, inputs,
                          Ni * L0_IN * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Forward pass — layer by layer (6 layers)
    launch_layer(d_in,  (float*)d_w0_,  d_b0_,  d_a0,  Ni, L0_IN,  L0_OUT,  0); // LeakyReLU
    launch_layer(d_a0,  (float*)d_w1_,  d_b1_,  d_a1,  Ni, L1_IN,  L1_OUT,  0);
    launch_layer(d_a1,  (float*)d_w2_,  d_b2_,  d_a2,  Ni, L2_IN,  L2_OUT,  0);
    launch_layer(d_a2,  (float*)d_w3_,  d_b3_,  d_a3,  Ni, L3_IN,  L3_OUT,  0);
    launch_layer(d_a3,  (float*)d_w4_,  d_b4_,  d_a4,  Ni, L4_IN,  L4_OUT,  0);
    launch_layer(d_a4,  (float*)d_w5_,  d_b5_,  d_out, Ni, L5_IN,  L5_OUT,  1); // Sigmoid

    CUDA_CHECK(cudaDeviceSynchronize());

    // Download results
    std::vector<float> h_out(Ni * L5_OUT);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          Ni * L5_OUT * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Unpack
    for (size_t i = 0; i < N; ++i) {
        const float* r = h_out.data() + i * L5_OUT;
        results[i] = { r[0], r[1], r[2] > 0.5f, r[3] };
    }

    // Free batch buffers
    cudaFree(d_in);
    cudaFree(d_a0); cudaFree(d_a1); cudaFree(d_a2); cudaFree(d_a3);
    cudaFree(d_a4);
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
    std::vector<float> a0(L0_OUT), a1(L1_OUT), a2(L2_OUT), a3(L3_OUT), a4(L4_OUT);

    layer_cpu(input,     L0_IN,  h_w0_.data(),  h_b0_.data(),  a0.data(),  L0_OUT,  false);
    layer_cpu(a0.data(), L1_IN,  h_w1_.data(),  h_b1_.data(),  a1.data(),  L1_OUT,  false);
    layer_cpu(a1.data(), L2_IN,  h_w2_.data(),  h_b2_.data(),  a2.data(),  L2_OUT,  false);
    layer_cpu(a2.data(), L3_IN,  h_w3_.data(),  h_b3_.data(),  a3.data(),  L3_OUT,  false);
    layer_cpu(a3.data(), L4_IN,  h_w4_.data(),  h_b4_.data(),  a4.data(),  L4_OUT,  false);
    layer_cpu(a4.data(), L5_IN,  h_w5_.data(),  h_b5_.data(),  output,     L5_OUT,  true);
}

// -----------------------------------------------------------------------
// print_summary
// -----------------------------------------------------------------------
void GPUNeuralNet::print_summary() const {
    LOG("[GPUNeuralNet] " << (cuda_active_ ? "CUDA ACTIVE" : "CPU FALLBACK")
        << "  arch=32->512->1024->2048->1024->512->4"
        << "  params=" << TOTAL_PARAMS << " (~" << (TOTAL_PARAMS / 1e6) << "M)"
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
