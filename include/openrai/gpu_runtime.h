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
//   Architecture (ULTRA-DEEP & WIDE - 3D AI Infrastructure):
//     12 layers total
//     input  32
//     hidden 16384  LeakyReLU
//     hidden 24576  LeakyReLU
//     hidden 24576  LeakyReLU
//     hidden 24576  LeakyReLU
//     hidden 24576  LeakyReLU
//     hidden 16384  LeakyReLU
//     hidden 16384  LeakyReLU
//     hidden 8192   LeakyReLU
//     hidden 4096   LeakyReLU
//     hidden 2048   LeakyReLU
//     hidden 1024   LeakyReLU
//     output    4   Sigmoid
//
//   Total params: ~3.98B  (4 billion scale)
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
    void* d_w0_ = nullptr;   float* d_b0_ = nullptr;   // 32->16384
    void* d_w1_ = nullptr;   float* d_b1_ = nullptr;   // 16384->24576
    void* d_w2_ = nullptr;   float* d_b2_ = nullptr;   // 24576->24576
    void* d_w3_ = nullptr;   float* d_b3_ = nullptr;   // 24576->24576
    void* d_w4_ = nullptr;   float* d_b4_ = nullptr;   // 24576->24576
    void* d_w5_ = nullptr;   float* d_b5_ = nullptr;   // 24576->16384
    void* d_w6_ = nullptr;   float* d_b6_ = nullptr;   // 16384->16384
    void* d_w7_ = nullptr;   float* d_b7_ = nullptr;   // 16384->8192
    void* d_w8_ = nullptr;   float* d_b8_ = nullptr;   // 8192->4096
    void* d_w9_ = nullptr;   float* d_b9_ = nullptr;   // 4096->2048
    void* d_w10_ = nullptr;  float* d_b10_ = nullptr;  // 2048->1024
    void* d_w11_ = nullptr;  float* d_b11_ = nullptr;  // 1024->4

    // ---- Host-side weight copies (CPU fallback) ----
    std::vector<float> h_w0_, h_b0_;
    std::vector<float> h_w1_, h_b1_;
    std::vector<float> h_w2_, h_b2_;
    std::vector<float> h_w3_, h_b3_;
    std::vector<float> h_w4_, h_b4_;
    std::vector<float> h_w5_, h_b5_;
    std::vector<float> h_w6_, h_b6_;
    std::vector<float> h_w7_, h_b7_;
    std::vector<float> h_w8_, h_b8_;
    std::vector<float> h_w9_, h_b9_;
    std::vector<float> h_w10_, h_b10_;
    std::vector<float> h_w11_, h_b11_;

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
