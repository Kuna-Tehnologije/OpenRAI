#pragma once

// -----------------------------------------------------------------------
// gpu_runtime.h
//   Public C++ interface to the CUDA neural-network backend.
//   Included from regular .cpp files — must stay free of CUDA types.
//
//   When OPENRAI_CUDA_ENABLED is NOT defined the implementation falls
//   back to the CPU NeuralNetwork so the build stays functional.
// -----------------------------------------------------------------------

#include <vector>
#include <string>
#include <cstddef>
#include <mutex>

namespace openrai {

// One inference result per node
struct NNResult {
    float relevance   = 0.f;   // output[0]
    float importance  = 0.f;   // output[1]
    bool  should_out  = false; // output[2] > 0.5
    float priority    = 0.f;   // output[3]
};

// -----------------------------------------------------------------------
// GPUNeuralNet
//   Manages a weight matrix on the device and runs batched inference.
//
//   Architecture (OPTIMIZED 5M PARAMS):
//     6 layers total
//     input  32
//     hidden 512    LeakyReLU
//     hidden 1024   LeakyReLU
//     hidden 2048   LeakyReLU
//     hidden 1024   LeakyReLU
//     hidden 512    LeakyReLU
//     output    4   Sigmoid
//
//   Total params: ~5.26M  (efficient mid-scale)
//
//   Batch inference:
//     run_batch(inputs, N) — runs N inputs concurrently on the GPU.
//     Each input is a float[INPUT_DIM] row in a flat array.
// -----------------------------------------------------------------------
class GPUNeuralNet {
public:
    static constexpr size_t INPUT_DIM  = 32;
    static constexpr size_t OUTPUT_DIM = 4;

    GPUNeuralNet();
    ~GPUNeuralNet();

    // Disable copy
    GPUNeuralNet(const GPUNeuralNet&)            = delete;
    GPUNeuralNet& operator=(const GPUNeuralNet&) = delete;

    // Run a batch of N inputs.
    // inputs  : flat float array, row-major, size = N * INPUT_DIM
    // results : output vector, size = N
    void run_batch(const float* inputs, size_t N,
                   std::vector<NNResult>& results) const;

    // Convenience: single input
    NNResult run_single(const float* input) const;

    bool is_cuda_active() const { return cuda_active_; }
    void print_summary()  const;

    void set_device(int gpu_id);
    void set_use_fp16(bool enable);
    bool is_fp16_enabled() const { return use_fp16_; }
    int get_device() const { return device_id_; }

private:
    bool  cuda_active_ = false;
    bool  use_fp16_ = false;
    int   device_id_ = 0;

    // ---- Device pointers (null when CUDA disabled) ----
    void* d_w0_ = nullptr;   float* d_b0_ = nullptr;   // 32->512
    void* d_w1_ = nullptr;   float* d_b1_ = nullptr;   // 512->1024
    void* d_w2_ = nullptr;   float* d_b2_ = nullptr;   // 1024->2048
    void* d_w3_ = nullptr;   float* d_b3_ = nullptr;   // 2048->1024
    void* d_w4_ = nullptr;   float* d_b4_ = nullptr;   // 1024->512
    void* d_w5_ = nullptr;   float* d_b5_ = nullptr;   // 512->4

    // ---- Host-side weight copies (CPU fallback) ----
    std::vector<float> h_w0_, h_b0_;
    std::vector<float> h_w1_, h_b1_;
    std::vector<float> h_w2_, h_b2_;
    std::vector<float> h_w3_, h_b3_;
    std::vector<float> h_w4_, h_b4_;
    std::vector<float> h_w5_, h_b5_;

    // ---- Host-side fp16 weight copies (for fp16 mode) ----
    std::vector<uint16_t> h_w0_fp16_, h_b0_fp16_;
    std::vector<uint16_t> h_w1_fp16_, h_b1_fp16_;
    std::vector<uint16_t> h_w2_fp16_, h_b2_fp16_;
    std::vector<uint16_t> h_w3_fp16_, h_b3_fp16_;
    std::vector<uint16_t> h_w4_fp16_, h_b4_fp16_;
    std::vector<uint16_t> h_w5_fp16_, h_b5_fp16_;

    void init_weights(unsigned seed);
    void upload_weights_to_device();
    void upload_weights_to_device_internal();  // Internal version without lock
    void cpu_forward(const float* input, float* output) const;
    void convert_weights_to_fp16();
    void free_device_weights();
    void free_device_weights_internal();  // Internal version without lock
};

// -----------------------------------------------------------------------
// encode_input_v2
//   32-dimensional encoding (up from 16 in Phase 1).
//   Combines byte histogram + positional hash + length features.
// -----------------------------------------------------------------------
void encode_input_v2(const std::string& s,
                     float* out,          // caller provides float[32]
                     size_t dim = GPUNeuralNet::INPUT_DIM);

} // namespace openrai
