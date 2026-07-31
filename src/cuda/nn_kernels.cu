// -----------------------------------------------------------------------
// nn_kernels.cu
//   CUDA kernels for batched neural-network forward pass.
//
//   Architecture per sample:
//     32 -> 256 (LeakyReLU)
//    256 -> 512 (LeakyReLU)
//    512 -> 256 (LeakyReLU)
//    256 -> 128 (LeakyReLU)
//    128 ->   4 (Sigmoid)
//
//   Batch strategy:
//     Each CUDA block handles one sample.
//     Threads within the block parallelise the neuron computations
//     of each layer using shared memory for activation vectors.
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
//   One block per sample, threads per output neuron.
//   Uses shared memory for the activation buffer.
//
//   grid  : (batch_size, 1, 1)
//   block : (MAX_NEURONS, 1, 1)   — launched with OUT_DIM active threads
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
        int in_dim, int out_dim,
        int use_sigmoid)
{
    const int sample = blockIdx.x;
    const int neuron = threadIdx.x;

    if (neuron >= out_dim) return;

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
    // One block per sample; threads = out_dim (padded to warp size)
    dim3 grid(batch);
    dim3 block(out_dim);   // max 1024; our largest layer is 512 — OK

    generic_layer_kernel<<<grid, block>>>(
        d_in, d_W, d_b, d_out,
        in_dim, out_dim, use_sigmoid);
}

} // extern "C"
