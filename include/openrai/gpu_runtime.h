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
//   Architecture (deeper & wider than Phase-1 CPU net):
//     input  32
//     hidden 256  LeakyReLU
//     hidden 512  LeakyReLU
//     hidden 256  LeakyReLU
//     hidden 128  LeakyReLU
//     output   4  Sigmoid
//
//   Total params: ~330k  (vs. 1140 in Phase-1)
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

private:
    bool  cuda_active_ = false;

    // ---- Device pointers (null when CUDA disabled) ----
    void* d_w0_ = nullptr;   float* d_b0_ = nullptr;  // 32->256
    void* d_w1_ = nullptr;   float* d_b1_ = nullptr;  // 256->512
    void* d_w2_ = nullptr;   float* d_b2_ = nullptr;  // 512->256
    void* d_w3_ = nullptr;   float* d_b3_ = nullptr;  // 256->128
    void* d_w4_ = nullptr;   float* d_b4_ = nullptr;  // 128->4

    // ---- Host-side weight copies (CPU fallback) ----
    std::vector<float> h_w0_, h_b0_;
    std::vector<float> h_w1_, h_b1_;
    std::vector<float> h_w2_, h_b2_;
    std::vector<float> h_w3_, h_b3_;
    std::vector<float> h_w4_, h_b4_;

    void init_weights(unsigned seed);
    void upload_weights_to_device();
    void cpu_forward(const float* input, float* output) const;
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
