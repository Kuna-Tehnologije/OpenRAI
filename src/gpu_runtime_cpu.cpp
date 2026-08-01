// -----------------------------------------------------------------------
// gpu_runtime_cpu.cpp
//   CPU-only implementation of GPUNeuralNet.
//   Compiled when OPENRAI_CUDA_ENABLED is NOT defined.
//   The CUDA version (gpu_runtime.cu) provides the same interface.
// -----------------------------------------------------------------------

#ifndef OPENRAI_CUDA_ENABLED

#include <openrai/gpu_runtime.h>
#include <openrai/log.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

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
    static_cast<size_t>(L11_IN)*L11_OUT + L11_OUT;

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
static void layer_cpu(const float* in,  int in_dim,
                      const float* W,   const float* b,
                      float*       out, int out_dim,
                      bool sigmoid_act)
{
    for (int o = 0; o < out_dim; ++o) {
        float acc = b[o];
        for (int i = 0; i < in_dim; ++i)
            acc += W[o * in_dim + i] * in[i];
        if (sigmoid_act)
            out[o] = 1.f / (1.f + std::exp(-acc));
        else
            out[o] = acc >= 0.f ? acc : 0.01f * acc;  // LeakyReLU
    }
}

// -----------------------------------------------------------------------
// GPUNeuralNet  (CPU implementation)
// -----------------------------------------------------------------------
GPUNeuralNet::GPUNeuralNet() {
    init_weights(42);
    cuda_active_ = false;
    LOG("[GPUNeuralNet] CPU fallback  params=" << TOTAL_PARAMS
        << "  (install CUDA Toolkit to enable GPU)");
}

void GPUNeuralNet::init_weights(unsigned seed) {
    std::mt19937 rng(seed);

    auto alloc = [&](std::vector<float>& w, std::vector<float>& b,
                     size_t in, size_t out) {
        w.resize(out * in);
        b.resize(out, 0.f);
        xavier_init(w, in, out, rng);
    };

    alloc(h_w0_, h_b0_, L0_IN,  L0_OUT);
    alloc(h_w1_, h_b1_, L1_IN,  L1_OUT);
    alloc(h_w2_, h_b2_, L2_IN,  L2_OUT);
    alloc(h_w3_, h_b3_, L3_IN,  L3_OUT);
    alloc(h_w4_, h_b4_, L4_IN,  L4_OUT);
    alloc(h_w5_, h_b5_, L5_IN,  L5_OUT);
    alloc(h_w6_, h_b6_, L6_IN,  L6_OUT);
    alloc(h_w7_, h_b7_, L7_IN,  L7_OUT);
    alloc(h_w8_, h_b8_, L8_IN,  L8_OUT);
    alloc(h_w9_, h_b9_, L9_IN,  L9_OUT);
    alloc(h_w10_, h_b10_, L10_IN, L10_OUT);
    alloc(h_w11_, h_b11_, L11_IN, L11_OUT);
}

void GPUNeuralNet::upload_weights_to_device() {}   // no-op in CPU path

GPUNeuralNet::~GPUNeuralNet() {}

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

void GPUNeuralNet::run_batch(const float*           inputs,
                              size_t                 N,
                              std::vector<NNResult>& results) const
{
    results.resize(N);
    std::vector<float> out(L4_OUT);
    for (size_t i = 0; i < N; ++i) {
        cpu_forward(inputs + i * INPUT_DIM, out.data());
        results[i] = { out[0], out[1], out[2] > 0.5f, out[3] };
    }
}

NNResult GPUNeuralNet::run_single(const float* input) const {
    std::vector<NNResult> res;
    run_batch(input, 1, res);
    return res[0];
}

void GPUNeuralNet::print_summary() const {
    LOG("[GPUNeuralNet] CPU fallback"
        "  arch=32->16384->24576->24576->24576->24576->16384->16384->8192->4096->2048->1024->4"
        "  params=" << TOTAL_PARAMS << " (~" << (TOTAL_PARAMS / 1000000000.0) << "B)");
}

// -----------------------------------------------------------------------
// encode_input_v2
// -----------------------------------------------------------------------
void encode_input_v2(const std::string& s, float* out, size_t dim) {
    std::fill(out, out + dim, 0.f);
    if (s.empty()) return;
    const size_t n = s.size();

    for (size_t i = 0; i < n; ++i)
        out[static_cast<unsigned char>(s[i]) % 8] += 1.f;

    for (size_t i = 0; i < std::min(n, size_t(16)); ++i)
        out[8 + i] = static_cast<float>(static_cast<unsigned char>(s[i])) / 255.f;

    out[24] = std::min(1.f, static_cast<float>(n) / 128.f);

    for (size_t i = 0; i + 1 < n && i < 14; i += 2) {
        size_t slot = 25 + (i / 2) % 7;
        out[slot] += static_cast<float>(
            (static_cast<unsigned char>(s[i]) * 31u +
             static_cast<unsigned char>(s[i+1])) & 0xFF) / 255.f;
    }

    float maxv = *std::max_element(out, out + 8);
    if (maxv > 0.f)
        for (size_t i = 0; i < 8; ++i)
            out[i] = (out[i] / maxv) * 2.f - 1.f;
}

} // namespace openrai

#endif // !OPENRAI_CUDA_ENABLED
