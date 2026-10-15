// -----------------------------------------------------------------------
// nn_kernels.cu
//   CUDA kernels for the batched fully-connected forward pass.
//
//   Generic layer:   out[sample][neuron] = act( W[neuron] . in[sample] + b[neuron] )
//   Activation:      LeakyReLU (hidden)  /  Sigmoid (last layer)
//
//   Launch strategy (warp-per-neuron):
//     grid  = ( ceil(out_dim / WARPS_PER_BLOCK), batch )
//     block = 256 threads = 8 warps
//     Every warp owns one output neuron of one sample; its 32 lanes walk
//     the weight row with stride 32 (fully coalesced 128-byte requests)
//     and the partial sums are reduced with __shfl_down_sync.
//
//   Why not one-thread-per-neuron with block = out_dim?
//     Layers here go up to 24576 neurons, far above the 1024 thread limit
//     of a block, so that configuration is an invalid launch.  The warp
//     kernel works for any out_dim and any in_dim.
//
//   Every launch takes an explicit cudaStream_t so each OpenRAI node can
//   drive the network from its own stream (see GPUStreamContext).
//
//   Memory layout (row-major, float32):
//     W : [out_dim x in_dim]
//     b : [out_dim]
//     in  : [batch x in_dim]
//     out : [batch x out_dim]
// -----------------------------------------------------------------------

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

namespace {

constexpr int kBlockThreads   = 256;
constexpr int kWarpsPerBlock  = kBlockThreads / 32;

__device__ __forceinline__ float leaky_relu(float x) {
    return x >= 0.f ? x : 0.01f * x;
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

__global__ void dense_layer_warp_kernel(
        const float* __restrict__ in,
        const float* __restrict__ W,
        const float* __restrict__ b,
        float*       __restrict__ out,
        int in_dim, int out_dim,
        int use_sigmoid)
{
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int neuron = blockIdx.x * kWarpsPerBlock + warp;
    const int sample = blockIdx.y;

    if (neuron >= out_dim) return;

    const float* x = in + static_cast<size_t>(sample) * in_dim;
    const float* w = W  + static_cast<size_t>(neuron) * in_dim;

    float acc = 0.f;

    // Vectorised path when the row is a multiple of 4 floats
    // (every layer width in OpenRAI is a multiple of 4; cudaMalloc
    //  guarantees 256-byte alignment of the buffers and row offsets
    //  are in_dim*4 bytes, so float4 loads are aligned).
    if ((in_dim & 3) == 0) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* w4 = reinterpret_cast<const float4*>(w);
        const int n4 = in_dim >> 2;
#pragma unroll 4
        for (int i = lane; i < n4; i += 32) {
            const float4 a = w4[i];
            const float4 v = x4[i];
            acc += a.x * v.x + a.y * v.y + a.z * v.z + a.w * v.w;
        }
    } else {
#pragma unroll 4
        for (int i = lane; i < in_dim; i += 32)
            acc += w[i] * x[i];
    }

    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        acc += b[neuron];
        out[static_cast<size_t>(sample) * out_dim + neuron] =
            use_sigmoid ? sigmoid(acc) : leaky_relu(acc);
    }
}

// -----------------------------------------------------------------------
// cosine_scores_kernel  (GPU vector memory search)
//   vecs    : [N x dim]  L2-normalised stored vectors
//   queries : [B x dim]  L2-normalised queries
//   scores  : [B x N]    dot products (= cosine similarity)
//   One thread per stored vector; B is small (1..16) so the query rows
//   are read through the L1 cache.
// -----------------------------------------------------------------------
__global__ void cosine_scores_kernel(
        const float* __restrict__ vecs,
        int N, int dim,
        const float* __restrict__ queries,
        int B,
        float* __restrict__ scores)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float* v = vecs + static_cast<size_t>(i) * dim;
    for (int b = 0; b < B; ++b) {
        const float* q = queries + static_cast<size_t>(b) * dim;
        float acc = 0.f;
#pragma unroll 8
        for (int j = 0; j < dim; ++j) acc += v[j] * q[j];
        scores[static_cast<size_t>(b) * N + i] = acc;
    }
}

} // namespace

// -----------------------------------------------------------------------
// Public C interface called from gpu_runtime.cu
// -----------------------------------------------------------------------
extern "C" {

cudaError_t launch_cosine_scores(
        const float* d_vecs, int N, int dim,
        const float* d_queries, int B,
        float* d_scores,
        cudaStream_t stream)
{
    if (N <= 0 || dim <= 0 || B <= 0) return cudaErrorInvalidValue;
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    cosine_scores_kernel<<<grid, block, 0, stream>>>(d_vecs, N, dim, d_queries, B, d_scores);
    return cudaGetLastError();
}

// Enqueue one dense layer on `stream`.  Returns the launch status.
cudaError_t launch_layer_stream(
        const float* d_in,
        const float* d_W,
        const float* d_b,
        float*       d_out,
        int batch, int in_dim, int out_dim,
        int use_sigmoid,
        cudaStream_t stream)
{
    if (batch <= 0 || in_dim <= 0 || out_dim <= 0) return cudaErrorInvalidValue;

    dim3 grid((out_dim + kWarpsPerBlock - 1) / kWarpsPerBlock, batch);
    dim3 block(kBlockThreads);

    dense_layer_warp_kernel<<<grid, block, 0, stream>>>(
        d_in, d_W, d_b, d_out, in_dim, out_dim, use_sigmoid);

    return cudaGetLastError();
}

// Legacy entry point (default stream) kept for compatibility.
void launch_layer(
        const float* d_in,
        const float* d_W,
        const float* d_b,
        float*       d_out,
        int batch, int in_dim, int out_dim,
        int use_sigmoid)
{
    (void)launch_layer_stream(d_in, d_W, d_b, d_out,
                              batch, in_dim, out_dim, use_sigmoid, 0);
}

} // extern "C"
