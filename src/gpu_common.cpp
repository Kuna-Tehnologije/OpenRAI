// -----------------------------------------------------------------------
// gpu_common.cpp
//   Backend-independent parts of the GPU runtime:
//     * NetArch presets / parsing / parameter counting
//     * encode_input_v2  (32-dim text encoding)
//   Compiled for both the CUDA and the CPU build.
// -----------------------------------------------------------------------

#include <openrai/gpu_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <thread>

namespace openrai {

// -----------------------------------------------------------------------
// NetArch
// -----------------------------------------------------------------------
size_t NetArch::max_dim() const {
    size_t m = 0;
    for (size_t d : dims) m = std::max(m, d);
    return m;
}

size_t NetArch::param_count() const {
    size_t total = 0;
    for (size_t l = 0; l + 1 < dims.size(); ++l)
        total += dims[l] * dims[l + 1] + dims[l + 1];
    return total;
}

bool NetArch::valid() const {
    if (dims.size() < 2) return false;
    for (size_t d : dims) if (d == 0) return false;
    return true;
}

std::string NetArch::describe() const {
    std::ostringstream ss;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) ss << "->";
        ss << dims[i];
    }
    return ss.str();
}

NetArch NetArch::full_4b() {
    NetArch a;
    a.name = "full_4b";
    a.dims = { 32, 16384, 24576, 24576, 24576, 24576,
               16384, 16384, 8192, 4096, 2048, 1024, 4 };
    return a;
}

NetArch NetArch::expert_lite(size_t hidden) {
    if (hidden < 8) hidden = 8;
    NetArch a;
    a.name = "expert_lite";
    a.dims = { 32, hidden, hidden, hidden / 2, hidden / 4, 4 };
    for (auto& d : a.dims) if (d == 0) d = 4;
    return a;
}

NetArch NetArch::primary_lite() {
    NetArch a;
    a.name = "primary_lite";
    a.dims = { 32, 2048, 2048, 1024, 512, 256, 4 };
    return a;
}

NetArch NetArch::parse(const std::string& csv, const std::string& name) {
    NetArch a;
    a.name = name;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim
        tok.erase(tok.begin(), std::find_if(tok.begin(), tok.end(),
                  [](unsigned char c) { return c != ' ' && c != '\t'; }));
        tok.erase(std::find_if(tok.rbegin(), tok.rend(),
                  [](unsigned char c) { return c != ' ' && c != '\t'; }).base(), tok.end());
        if (tok.empty()) continue;
        try {
            long long v = std::stoll(tok);
            if (v > 0) a.dims.push_back(static_cast<size_t>(v));
        } catch (...) {
            // ignore junk tokens
        }
    }
    return a;
}

// -----------------------------------------------------------------------
// Training: CPU backprop on the host weights (both builds)
// -----------------------------------------------------------------------
float GPUNeuralNet::train_batch(const float* inputs, const float* targets, size_t N, float lr) {
    const size_t L = arch_.layer_count();
    if (!trainable_ || N == 0 || h_w_.size() != L) return -1.f;
    if (vel_w_.size() != L) {
        vel_w_.resize(L); vel_b_.resize(L);
        for (size_t l = 0; l < L; ++l) { vel_w_[l].assign(h_w_[l].size(), 0.f); vel_b_[l].assign(h_b_[l].size(), 0.f); }
    }
    const size_t in_dim = arch_.input_dim(), out_dim = arch_.output_dim();

    // gradient accumulators
    std::vector<std::vector<float>> gw(L), gb(L);
    for (size_t l = 0; l < L; ++l) { gw[l].assign(h_w_[l].size(), 0.f); gb[l].assign(h_b_[l].size(), 0.f); }

    // activations per layer (a[0] = input)
    std::vector<std::vector<float>> a(L + 1), delta(L + 1);
    for (size_t l = 0; l <= L; ++l) { a[l].resize(arch_.dims[l]); delta[l].resize(arch_.dims[l]); }

    double loss = 0.0;
    for (size_t n = 0; n < N; ++n) {
        std::copy(inputs + n * in_dim, inputs + (n + 1) * in_dim, a[0].begin());
        // forward
        for (size_t l = 0; l < L; ++l) {
            const size_t li = arch_.dims[l], lo = arch_.dims[l + 1];
            const bool last = (l + 1 == L);
            for (size_t o = 0; o < lo; ++o) {
                const float* w = h_w_[l].data() + o * li;
                float acc = h_b_[l][o];
                for (size_t i = 0; i < li; ++i) acc += w[i] * a[l][i];
                a[l + 1][o] = last ? 1.f / (1.f + std::exp(-acc)) : (acc >= 0.f ? acc : 0.01f * acc);
            }
        }
        // output delta: BCE with sigmoid -> (y - t)
        const float* t = targets + n * out_dim;
        for (size_t o = 0; o < out_dim; ++o) {
            const float y = std::clamp(a[L][o], 1e-6f, 1.f - 1e-6f);
            loss -= t[o] * std::log(y) + (1.f - t[o]) * std::log(1.f - y);
            delta[L][o] = a[L][o] - t[o];
        }
        // backward
        for (size_t l = L; l-- > 0;) {
            const size_t li = arch_.dims[l], lo = arch_.dims[l + 1];
            std::fill(delta[l].begin(), delta[l].end(), 0.f);
            for (size_t o = 0; o < lo; ++o) {
                const float d = delta[l + 1][o];
                if (d == 0.f) continue;
                gb[l][o] += d;
                float* g = gw[l].data() + o * li;
                const float* w = h_w_[l].data() + o * li;
                for (size_t i = 0; i < li; ++i) { g[i] += d * a[l][i]; delta[l][i] += d * w[i]; }
            }
            if (l > 0)
                for (size_t i = 0; i < li; ++i) delta[l][i] *= (a[l][i] >= 0.f ? 1.f : 0.01f);
        }
    }

    // SGD with momentum, gradient clipping per tensor
    const float scale = 1.f / static_cast<float>(N);
    const float mom = 0.9f;
    for (size_t l = 0; l < L; ++l) {
        double norm = 0.0;
        for (float g : gw[l]) norm += static_cast<double>(g) * g;
        const float clip = norm > 25.0 ? static_cast<float>(5.0 / std::sqrt(norm)) : 1.f;
        for (size_t i = 0; i < h_w_[l].size(); ++i) {
            vel_w_[l][i] = mom * vel_w_[l][i] - lr * gw[l][i] * scale * clip;
            h_w_[l][i] += vel_w_[l][i];
        }
        for (size_t i = 0; i < h_b_[l].size(); ++i) {
            vel_b_[l][i] = mom * vel_b_[l][i] - lr * gb[l][i] * scale;
            h_b_[l][i] += vel_b_[l][i];
        }
    }
    ++train_steps_;
    return static_cast<float>(loss / static_cast<double>(N * out_dim));
}

bool GPUNeuralNet::save_weights(const std::string& path) const {
    if (h_w_.size() != arch_.layer_count()) return false;
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write("ORAINET1", 8);
        const uint32_t L = static_cast<uint32_t>(arch_.dims.size());
        f.write(reinterpret_cast<const char*>(&L), 4);
        for (size_t d : arch_.dims) { const uint32_t v = static_cast<uint32_t>(d); f.write(reinterpret_cast<const char*>(&v), 4); }
        const uint32_t steps = static_cast<uint32_t>(train_steps_);
        f.write(reinterpret_cast<const char*>(&steps), 4);
        for (size_t l = 0; l < h_w_.size(); ++l) {
            f.write(reinterpret_cast<const char*>(h_w_[l].data()), static_cast<std::streamsize>(h_w_[l].size() * sizeof(float)));
            f.write(reinterpret_cast<const char*>(h_b_[l].data()), static_cast<std::streamsize>(h_b_[l].size() * sizeof(float)));
        }
        return static_cast<bool>(f);
    } catch (...) { return false; }
}

bool GPUNeuralNet::load_weights(const std::string& path) {
    if (!trainable_ || h_w_.size() != arch_.layer_count()) return false;
    try {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        char magic[8];
        f.read(magic, 8);
        if (std::string(magic, 8) != "ORAINET1") return false;
        uint32_t L = 0;
        f.read(reinterpret_cast<char*>(&L), 4);
        if (L != arch_.dims.size()) return false;
        for (size_t d : arch_.dims) { uint32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); if (v != d) return false; }
        uint32_t steps = 0;
        f.read(reinterpret_cast<char*>(&steps), 4);
        for (size_t l = 0; l < h_w_.size(); ++l) {
            f.read(reinterpret_cast<char*>(h_w_[l].data()), static_cast<std::streamsize>(h_w_[l].size() * sizeof(float)));
            f.read(reinterpret_cast<char*>(h_b_[l].data()), static_cast<std::streamsize>(h_b_[l].size() * sizeof(float)));
        }
        if (!f) return false;
        train_steps_ = steps;
        sync_weights_to_device();
        return true;
    } catch (...) { return false; }
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
    if (s.empty() || dim < 32) return;

    const size_t n = s.size();

    // Byte histogram - 8 buckets
    for (size_t i = 0; i < n; ++i) {
        size_t bucket = static_cast<unsigned char>(s[i]) % 8;
        out[bucket] += 1.f;
    }

    // Positional hash - 16 slots
    for (size_t i = 0; i < std::min(n, size_t(16)); ++i) {
        out[8 + i] = static_cast<float>(static_cast<unsigned char>(s[i])) / 255.f;
    }

    // Length feature
    out[24] = std::min(1.f, static_cast<float>(n) / 128.f);

    // Bigram hash - 7 slots
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

// -----------------------------------------------------------------------
// Concurrency self-test
// -----------------------------------------------------------------------
std::string GPUConcurrencyReport::describe() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2)
       << "GPU concurrency self-test\n"
       << "  backend     : " << (cuda ? "CUDA" : "CPU") << (cuda ? " device " + std::to_string(device) : "") << "\n"
       << "  network     : " << arch.name << " [" << arch.describe() << "] params=" << arch.param_count() << "\n"
       << "  streams     : " << streams << "  batch=" << batch << "  repeats=" << repeats << "\n";
    for (const auto& l : stream_info) ss << "    " << l << "\n";
    ss << "  serial      : " << serial_ms     << " ms   (" << streams * repeats << " forward passes on ONE stream)\n"
       << "  concurrent  : " << concurrent_ms << " ms   (same work on " << streams << " streams, one thread)   speedup x" << speedup_concurrent << "\n"
       << "  threaded    : " << threaded_ms   << " ms   (" << streams << " threads, each its own stream)       speedup x" << speedup_threaded << "\n"
       << "  max |async - sync| result difference: " << std::setprecision(6) << max_result_diff << "\n"
       << std::setprecision(1)
       << "  memory index: " << (index_cuda ? "GPU" : "CPU") << " cosine search over " << index_items << " vectors: "
       << index_search_us << " us/query, " << index_batch_us << " us per batch of 8"
       << (index_results_match ? " (matches CPU reference)" : " (MISMATCH vs CPU reference!)") << "\n"
       << "  verdict     : " << verdict << "\n";
    return ss.str();
}

GPUConcurrencyReport run_gpu_concurrency_selftest(const NetArch& arch_in,
                                                  size_t streams, size_t batch, int repeats)
{
    GPUConcurrencyReport r;
    r.arch    = arch_in.valid() ? arch_in : NetArch::expert_lite(1024);
    r.streams = std::max<size_t>(1, streams);
    r.batch   = std::max<size_t>(1, batch);
    r.repeats = std::max(1, repeats);

    GPUNeuralNet net(r.arch, 1234u);
    r.cuda = net.is_cuda_active();

    std::vector<std::unique_ptr<GPUStreamContext>> ctx;
    for (size_t k = 0; k < r.streams; ++k) {
        const int prio = static_cast<int>(100 - (k * 100) / std::max<size_t>(1, r.streams));
        ctx.push_back(std::make_unique<GPUStreamContext>("selftest:" + std::to_string(k), prio, r.batch, r.arch));
        std::ostringstream line;
        line << "stream " << k << ": handle=0x" << std::hex << ctx.back()->native_handle() << std::dec
             << " node_priority=" << ctx.back()->priority()
             << " cuda_priority=" << ctx.back()->cuda_priority()
             << " backend=" << (ctx.back()->is_cuda() ? "cuda" : "cpu");
        r.stream_info.push_back(line.str());
    }
    r.device = ctx.front()->device();

    const size_t in_dim = r.arch.input_dim();
    std::vector<float> inputs(r.batch * in_dim);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& v : inputs) v = d(rng);

    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // warm-up (module load, first-launch costs)
    for (auto& c : ctx) { net.run_batch_async(inputs.data(), r.batch, *c); c->synchronize(); }

    // serial: everything on stream 0, synchronised after every launch
    {
        auto t0 = clock::now();
        for (int rep = 0; rep < r.repeats; ++rep)
            for (size_t k = 0; k < r.streams; ++k) {
                net.run_batch_async(inputs.data(), r.batch, *ctx[0]);
                ctx[0]->synchronize();
            }
        r.serial_ms = ms(t0, clock::now());
    }

    // concurrent: K streams fed from one thread, sync at the end of each round
    {
        auto t0 = clock::now();
        for (int rep = 0; rep < r.repeats; ++rep) {
            for (size_t k = 0; k < r.streams; ++k) net.run_batch_async(inputs.data(), r.batch, *ctx[k]);
            for (size_t k = 0; k < r.streams; ++k) ctx[k]->synchronize();
        }
        r.concurrent_ms = ms(t0, clock::now());
    }

    // threaded: one thread per stream (exactly what expert nodes do)
    {
        auto t0 = clock::now();
        std::vector<std::thread> th;
        for (size_t k = 0; k < r.streams; ++k) {
            th.emplace_back([&, k] {
                for (int rep = 0; rep < r.repeats; ++rep) {
                    net.run_batch_async(inputs.data(), r.batch, *ctx[k]);
                    ctx[k]->synchronize();
                }
            });
        }
        for (auto& t : th) t.join();
        r.threaded_ms = ms(t0, clock::now());
    }

    r.speedup_concurrent = r.concurrent_ms > 0 ? r.serial_ms / r.concurrent_ms : 0.0;
    r.speedup_threaded   = r.threaded_ms   > 0 ? r.serial_ms / r.threaded_ms   : 0.0;

    // results must be identical regardless of the stream they ran on
    std::vector<NNResult> ref, got;
    net.run_batch(inputs.data(), r.batch, ref);
    for (auto& c : ctx) {
        net.fetch_results(*c, r.batch, got);
        for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
            r.max_result_diff = std::max(r.max_result_diff, static_cast<double>(std::fabs(ref[i].relevance  - got[i].relevance)));
            r.max_result_diff = std::max(r.max_result_diff, static_cast<double>(std::fabs(ref[i].importance - got[i].importance)));
            r.max_result_diff = std::max(r.max_result_diff, static_cast<double>(std::fabs(ref[i].priority   - got[i].priority)));
        }
    }

    // ---- vector memory index benchmark ----------------------------------------
    {
        const size_t dim = 32, items = 100000;
        GPUVectorIndex idx(dim, items, 60, "selftest-index");
        std::vector<std::vector<float>> vecs(items, std::vector<float>(dim));
        for (auto& vv : vecs) for (auto& x : vv) x = d(rng);
        idx.rebuild(vecs);
        r.index_cuda  = idx.is_cuda();
        r.index_items = idx.size();

        std::vector<float> q(8 * dim);
        for (auto& x : q) x = d(rng);
        // warm-up
        idx.search(q.data(), dim, 5);
        const int reps = 20;
        auto t0 = clock::now();
        for (int i = 0; i < reps; ++i) idx.search(q.data() + (i % 8) * dim, dim, 5);
        r.index_search_us = std::chrono::duration<double, std::micro>(clock::now() - t0).count() / reps;
        t0 = clock::now();
        for (int i = 0; i < reps; ++i) idx.search_batch(q.data(), 8, 5);
        r.index_batch_us = std::chrono::duration<double, std::micro>(clock::now() - t0).count() / reps;

        // CPU reference for the first query
        auto hits = idx.search(q.data(), dim, 1);
        double qn = 0; for (size_t j = 0; j < dim; ++j) qn += double(q[j]) * q[j];
        size_t best = 0; double best_s = -2;
        for (size_t i = 0; i < items; ++i) {
            double dot = 0, vn = 0;
            for (size_t j = 0; j < dim; ++j) { dot += double(vecs[i][j]) * q[j]; vn += double(vecs[i][j]) * vecs[i][j]; }
            const double sc = dot / (std::sqrt(vn) * std::sqrt(qn));
            if (sc > best_s) { best_s = sc; best = i; }
        }
        r.index_results_match = !hits.empty() && hits.front().index == best && std::fabs(hits.front().score - best_s) < 1e-3;
    }

    std::ostringstream v;
    if (r.cuda) {
        if (r.speedup_concurrent >= 1.25)
            v << "CUDA streams OVERLAP: " << r.streams << " streams finish x" << std::fixed << std::setprecision(2)
              << r.speedup_concurrent << " faster than one stream -> kernels of different nodes run concurrently";
        else
            v << "CUDA streams created, but little overlap measured (x" << std::fixed << std::setprecision(2)
              << r.speedup_concurrent << "). Either each kernel already saturates the GPU (use a smaller "
              << "OPENRAI_EXPERT_ARCH / batch) or the driver serialises (check MPS / exclusive mode)";
        if (r.max_result_diff > 1e-4) v << " | WARNING: results differ between streams (" << r.max_result_diff << ")";
    } else {
        v << "CPU build: no CUDA streams; " << r.streams << " node threads run in parallel (x"
          << std::fixed << std::setprecision(2) << r.speedup_threaded << " on this machine). "
          << "Build with the CUDA toolkit to get per-node streams.";
    }
    r.verdict = v.str();
    return r;
}

} // namespace openrai
