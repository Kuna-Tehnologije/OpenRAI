// -----------------------------------------------------------------------
// nn_kernels.cu
//   CUDA kernels for batched neural-network forward pass.
//
//   Architecture per sample (6 layers, ~5.26M params):
//     32 -> 512 (LeakyReLU)
//    512 -> 1024 (LeakyReLU)
//   1024 -> 2048 (LeakyReLU)
//   2048 -> 1024 (LeakyReLU)
//   1024 -> 512 (LeakyReLU)
//    512 -> 4 (Sigmoid)
//
//   Batch strategy:
//     1D grid launch: total threads = batch * out_dim
//     Each thread computes one output neuron for one sample.
//
//   Memory layout (row-major, all float32):
//     W[l] : [out_l x in_l]   device pointer
//     b[l] : [out_l]          device pointer
// -----------------------------------------------------------------------

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ---- Activation helpers -----------------------------------------------
__device__ __forceinline__ float leaky_relu(float x) {
    return x >= 0.f ? x : 0.01f * x;
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

// -----------------------------------------------------------------------
// generic_layer_kernel
//   1D grid launch where each thread handles one output neuron of one sample.
//
//   activations_in  : device ptr, [batch_size x in_dim]
//   W               : [out_dim x in_dim]
//   b               : [out_dim]
//   activations_out : [batch_size x out_dim]
//   in_dim, out_dim : layer dimensions
//   use_sigmoid     : 0 = LeakyReLU, 1 = Sigmoid
// -----------------------------------------------------------------------
__global__ void generic_layer_kernel(
        const float* __restrict__ activations_in,
        const float* __restrict__ W,
        const float* __restrict__ b,
        float*       __restrict__ activations_out,
        int batch_size,
        int in_dim, int out_dim,
        int use_sigmoid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_dim;

    if (idx >= total) return;

    int sample = idx / out_dim;
    int neuron = idx % out_dim;

    const float* x = activations_in + sample * in_dim;
    float*       y = activations_out + sample * out_dim;

    // dot product W[neuron] · x  +  b[neuron]
    const float* w_row = W + neuron * in_dim;
    float acc = b[neuron];

#pragma unroll 8
    for (int i = 0; i < in_dim; ++i)
        acc += w_row[i] * x[i];

    y[neuron] = use_sigmoid ? sigmoid(acc) : leaky_relu(acc);
}

// -----------------------------------------------------------------------
// Public C interface called from gpu_runtime.cpp
// -----------------------------------------------------------------------
extern "C" {

void launch_layer(
        const float* d_in,
        const float* d_W,
        const float* d_b,
        float*       d_out,
        int batch, int in_dim, int out_dim,
        int use_sigmoid)
{
    // 1D grid launch: total threads = batch * out_dim
    int total = batch * out_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    generic_layer_kernel<<<blocks, threads>>>(
        d_in, d_W, d_b, d_out,
        batch, in_dim, out_dim, use_sigmoid);
}

} // extern "C"
