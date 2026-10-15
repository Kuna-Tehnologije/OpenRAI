#include <openrai/expert_model.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace openrai {

// =======================================================================
// TrainReport
// =======================================================================
std::string TrainReport::describe() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3)
       << "iterations=" << iterations << " batch=" << batch << " tokens=" << tokens
       << " loss " << first_loss << " -> " << last_loss << " (best " << best_loss << ")"
       << " in " << std::setprecision(2) << seconds << "s";
    return ss.str();
}

// =======================================================================
// tokenizer (generic)
// =======================================================================
std::vector<std::string> ExpertModel::words(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&] { if (!cur.empty()) { out.push_back(cur); cur.clear(); } };
    for (unsigned char c : text) {
        if (std::isalpha(c)) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (std::isdigit(c)) {
            // digits are single tokens so numbers are composable
            flush();
            out.push_back(std::string(1, static_cast<char>(c)));
        } else if (c == '+' || c == '-' || c == '*' || c == '/' || c == '=' || c == '%' ||
                   c == '^' || c == '(' || c == ')' || c == '?' || c == ':' || c == '.') {
            flush();
            out.push_back(std::string(1, static_cast<char>(c)));
        } else {
            flush();
        }
    }
    flush();
    return out;
}

std::vector<uint32_t> ExpertModel::encode(const std::string& text) const {
    std::vector<uint32_t> ids;
    for (const auto& w : words(text)) {
        auto it = word_to_id_.find(w);
        ids.push_back(it == word_to_id_.end() ? UNK : it->second);
    }
    return ids;
}

std::string ExpertModel::decode(const std::vector<uint32_t>& ids) const {
    std::string s;
    for (uint32_t id : ids) {
        if (id >= vocab_.size()) continue;
        if (!s.empty()) s += ' ';
        s += vocab_[id];
    }
    return s;
}

// =======================================================================
// construction / vocab / params
// =======================================================================
ExpertModel::ExpertModel(std::string name, Config cfg)
    : name_(std::move(name)), cfg_(cfg), rng_(cfg.seed)
{
    if (cfg_.vocab < 8) cfg_.vocab = 8;
    if (cfg_.context == 0) cfg_.context = 1;
    vocab_ = { "<bos>", "<unk>", "<eos>" };
    for (size_t i = 0; i < vocab_.size(); ++i) word_to_id_[vocab_[i]] = static_cast<uint32_t>(i);
    init_params();
}

void ExpertModel::build_vocab(const std::vector<std::string>& corpus) {
    std::unordered_map<std::string, size_t> freq;
    for (const auto& line : corpus)
        for (const auto& w : words(line)) ++freq[w];
    std::vector<std::pair<std::string, size_t>> sorted(freq.begin(), freq.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    vocab_ = { "<bos>", "<unk>", "<eos>" };
    for (const auto& p : sorted) {
        if (vocab_.size() >= cfg_.vocab) break;
        vocab_.push_back(p.first);
    }
    word_to_id_.clear();
    for (size_t i = 0; i < vocab_.size(); ++i) word_to_id_[vocab_[i]] = static_cast<uint32_t>(i);
    trained_ = false;
    total_iterations_ = 0;
    init_params();
}

void ExpertModel::init_params() {
    const size_t V = vocab_.size(), d = cfg_.dim, H = cfg_.hidden, C = cfg_.context * d;
    std::mt19937 rng(cfg_.seed);
    auto init = [&](std::vector<float>& w, size_t n, float scale) {
        std::normal_distribution<float> nd(0.f, scale);
        w.resize(n);
        for (auto& x : w) x = nd(rng);
    };
    init(emb_, V * d, 0.1f);
    init(w1_, H * C, std::sqrt(2.f / static_cast<float>(C)));
    b1_.assign(H, 0.f);
    init(w2_, V * H, std::sqrt(1.f / static_cast<float>(H)));
    b2_.assign(V, 0.f);
    auto zero = [](std::vector<float>& m, size_t n) { m.assign(n, 0.f); };
    zero(m_emb_, V * d); zero(v_emb_, V * d);
    zero(m_w1_, H * C);  zero(v_w1_, H * C);
    zero(m_b1_, H);      zero(v_b1_, H);
    zero(m_w2_, V * H);  zero(v_w2_, V * H);
    zero(m_b2_, V);      zero(v_b2_, V);
    adam_t_ = 0;
}

size_t ExpertModel::param_count() const {
    return emb_.size() + w1_.size() + b1_.size() + w2_.size() + b2_.size();
}

std::vector<uint32_t> ExpertModel::context_at(const std::vector<uint32_t>& seq, size_t pos) const {
    std::vector<uint32_t> ctx(cfg_.context, BOS);
    for (size_t k = 0; k < cfg_.context; ++k) {
        const long idx = static_cast<long>(pos) - static_cast<long>(cfg_.context) + static_cast<long>(k);
        if (idx >= 0) ctx[k] = seq[static_cast<size_t>(idx)];
    }
    return ctx;
}

// =======================================================================
// forward
// =======================================================================
void ExpertModel::forward(const uint32_t* ctx, std::vector<float>& x, std::vector<float>& h,
                          std::vector<float>& logp) const
{
    const size_t V = vocab_.size(), d = cfg_.dim, H = cfg_.hidden, C = cfg_.context * d;
    x.resize(C);
    for (size_t k = 0; k < cfg_.context; ++k)
        std::memcpy(x.data() + k * d, emb_.data() + static_cast<size_t>(ctx[k]) * d, d * sizeof(float));

    h.resize(H);
    for (size_t j = 0; j < H; ++j) {
        const float* w = w1_.data() + j * C;
        float acc = b1_[j];
        for (size_t i = 0; i < C; ++i) acc += w[i] * x[i];
        h[j] = acc > 0.f ? acc : 0.01f * acc;
    }

    logp.resize(V);
    float mx = -1e30f;
    for (size_t v = 0; v < V; ++v) {
        const float* w = w2_.data() + v * H;
        float acc = b2_[v];
        for (size_t j = 0; j < H; ++j) acc += w[j] * h[j];
        logp[v] = acc;
        mx = std::max(mx, acc);
    }
    double sum = 0.0;
    for (size_t v = 0; v < V; ++v) sum += std::exp(static_cast<double>(logp[v] - mx));
    const float lse = mx + static_cast<float>(std::log(sum));
    for (size_t v = 0; v < V; ++v) logp[v] -= lse;
}

// =======================================================================
// training (Adam)
// =======================================================================
TrainReport ExpertModel::train(const std::vector<std::string>& corpus, size_t iterations, size_t batch) {
    std::lock_guard<std::mutex> lk(mutex_);
    TrainReport rep;
    rep.iterations = iterations;
    rep.batch = std::max<size_t>(1, batch);
    const auto t0 = std::chrono::steady_clock::now();

    if (vocab_.size() <= 3) {
        // no build_vocab() call yet - do it here (unlocked helper would
        // re-init params, so inline the logic)
        std::unordered_map<std::string, size_t> freq;
        for (const auto& line : corpus) for (const auto& w : words(line)) ++freq[w];
        std::vector<std::pair<std::string, size_t>> sorted(freq.begin(), freq.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first; });
        vocab_ = { "<bos>", "<unk>", "<eos>" };
        for (const auto& p : sorted) { if (vocab_.size() >= cfg_.vocab) break; vocab_.push_back(p.first); }
        word_to_id_.clear();
        for (size_t i = 0; i < vocab_.size(); ++i) word_to_id_[vocab_[i]] = static_cast<uint32_t>(i);
        init_params();
    }

    // Every line is a sequence BOS-padded by the context window and closed
    // with EOS; contexts never cross line boundaries.
    std::vector<std::vector<uint32_t>> lines;
    for (const auto& line : corpus) {
        auto ids = encode(line);
        if (ids.empty()) continue;
        ids.push_back(EOS);
        lines.push_back(std::move(ids));
    }
    std::vector<std::pair<uint32_t, uint32_t>> pos;   // (line, index)
    for (uint32_t l = 0; l < lines.size(); ++l)
        for (uint32_t i = 0; i < lines[l].size(); ++i) pos.emplace_back(l, i);
    rep.tokens = pos.size();
    if (pos.empty() || iterations == 0) return rep;

    const size_t V = vocab_.size(), d = cfg_.dim, H = cfg_.hidden, C = cfg_.context * d;
    std::vector<float> x, h, logp, dh(H), dx(C);
    std::vector<float> g_emb(emb_.size()), g_w1(w1_.size()), g_b1(b1_.size()), g_w2(w2_.size()), g_b2(b2_.size());
    std::uniform_int_distribution<size_t> pick(0, pos.size() - 1);

    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    auto adam = [&](std::vector<float>& w, std::vector<float>& g, std::vector<float>& m, std::vector<float>& v, float scale) {
        const float lr_t = cfg_.lr * std::sqrt(1.f - std::pow(beta2, static_cast<float>(adam_t_))) /
                           (1.f - std::pow(beta1, static_cast<float>(adam_t_)));
        for (size_t i = 0; i < w.size(); ++i) {
            const float gi = g[i] * scale;
            if (gi == 0.f && m[i] == 0.f && v[i] == 0.f) continue;   // sparse rows untouched
            m[i] = beta1 * m[i] + (1.f - beta1) * gi;
            v[i] = beta2 * v[i] + (1.f - beta2) * gi * gi;
            w[i] -= lr_t * m[i] / (std::sqrt(v[i]) + eps);
        }
    };

    rep.losses.reserve(iterations);
    for (size_t it = 0; it < iterations; ++it) {
        std::fill(g_emb.begin(), g_emb.end(), 0.f);
        std::fill(g_w1.begin(), g_w1.end(), 0.f);
        std::fill(g_b1.begin(), g_b1.end(), 0.f);
        std::fill(g_w2.begin(), g_w2.end(), 0.f);
        std::fill(g_b2.begin(), g_b2.end(), 0.f);
        double loss = 0.0;

        for (size_t b = 0; b < rep.batch; ++b) {
            const auto [l, i] = pos[pick(rng_)];
            const auto& line = lines[l];
            auto ctx = context_at(line, i);
            const uint32_t target = line[i];

            forward(ctx.data(), x, h, logp);
            loss -= logp[target];

            // dlogits = softmax - onehot
            std::fill(dh.begin(), dh.end(), 0.f);
            for (size_t v = 0; v < V; ++v) {
                const float dl = std::exp(logp[v]) - (v == target ? 1.f : 0.f);
                if (dl == 0.f) continue;
                g_b2[v] += dl;
                float* gw = g_w2.data() + v * H;
                const float* w = w2_.data() + v * H;
                for (size_t j = 0; j < H; ++j) { gw[j] += dl * h[j]; dh[j] += dl * w[j]; }
            }
            // through LeakyReLU
            std::fill(dx.begin(), dx.end(), 0.f);
            for (size_t j = 0; j < H; ++j) {
                const float dpre = dh[j] * (h[j] > 0.f ? 1.f : 0.01f);
                if (dpre == 0.f) continue;
                g_b1[j] += dpre;
                float* gw = g_w1.data() + j * C;
                const float* w = w1_.data() + j * C;
                for (size_t c = 0; c < C; ++c) { gw[c] += dpre * x[c]; dx[c] += dpre * w[c]; }
            }
            for (size_t k = 0; k < cfg_.context; ++k) {
                float* ge = g_emb.data() + static_cast<size_t>(ctx[k]) * d;
                for (size_t e = 0; e < d; ++e) ge[e] += dx[k * d + e];
            }
        }

        ++adam_t_;
        const float scale = 1.f / static_cast<float>(rep.batch);
        adam(emb_, g_emb, m_emb_, v_emb_, scale);
        adam(w1_, g_w1, m_w1_, v_w1_, scale);
        adam(b1_, g_b1, m_b1_, v_b1_, scale);
        adam(w2_, g_w2, m_w2_, v_w2_, scale);
        adam(b2_, g_b2, m_b2_, v_b2_, scale);

        rep.losses.push_back(static_cast<float>(loss / rep.batch));
    }

    rep.first_loss = rep.losses.front();
    rep.best_loss  = *std::min_element(rep.losses.begin(), rep.losses.end());
    const size_t tail = std::min<size_t>(10, rep.losses.size());
    rep.last_loss = std::accumulate(rep.losses.end() - static_cast<std::ptrdiff_t>(tail), rep.losses.end(), 0.f) / tail;
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    trained_ = true;
    total_iterations_ += iterations;
    return rep;
}

// =======================================================================
// knowledge
// =======================================================================
ModelScore ExpertModel::score(const std::string& text) const {
    std::lock_guard<std::mutex> lk(mutex_);
    ModelScore s;
    auto ids = encode(text);
    if (ids.empty()) return s;
    ids.push_back(EOS);
    std::vector<float> x, h, logp;
    double total = 0.0;
    size_t known = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        auto ctx = context_at(ids, i);
        forward(ctx.data(), x, h, logp);
        total += logp[ids[i]];
        if (ids[i] != UNK) ++known;
    }
    s.tokens       = ids.size();
    s.mean_logprob = static_cast<float>(total / static_cast<double>(ids.size()));
    s.perplexity   = std::exp(-s.mean_logprob);
    s.coverage     = static_cast<float>(known) / static_cast<float>(ids.size());
    s.valid        = trained_;
    return s;
}

std::string ExpertModel::generate(const std::string& prompt, size_t max_tokens, float temperature) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto ids = encode(prompt);
    std::vector<uint32_t> out;
    std::vector<float> x, h, logp;
    std::mt19937 rng(static_cast<unsigned>(std::hash<std::string>{}(prompt)) ^ cfg_.seed);
    for (size_t n = 0; n < max_tokens; ++n) {
        auto ctx = context_at(ids, ids.size());
        forward(ctx.data(), x, h, logp);
        uint32_t next = UNK;
        if (temperature <= 0.f) {
            float best = -1e30f;
            for (size_t v = 0; v < logp.size(); ++v)
                if (v != UNK && v != BOS && logp[v] > best) { best = logp[v]; next = static_cast<uint32_t>(v); }
        } else {
            std::vector<double> p(logp.size());
            double sum = 0.0;
            for (size_t v = 0; v < logp.size(); ++v) {
                p[v] = (v == UNK || v == BOS) ? 0.0 : std::exp(static_cast<double>(logp[v]) / temperature);
                sum += p[v];
            }
            std::uniform_real_distribution<double> u(0.0, sum);
            double r = u(rng);
            next = EOS;
            for (size_t v = 0; v < p.size(); ++v) { r -= p[v]; if (r <= 0.0) { next = static_cast<uint32_t>(v); break; } }
        }
        if (next == EOS) break;
        ids.push_back(next);
        out.push_back(next);
    }
    return decode(out);
}

bool ExpertModel::trailing_integer(const std::string& text, long long& value) {
    auto w = words(text);
    size_t end = w.size();
    while (end > 0 && w[end - 1] == "<eos>") --end;
    size_t start = end;
    while (start > 0 && w[start - 1].size() == 1 && std::isdigit(static_cast<unsigned char>(w[start - 1][0]))) --start;
    if (start == end) return false;
    std::string digits;
    for (size_t i = start; i < end; ++i) digits += w[i];
    try { value = std::stoll(digits); } catch (...) { return false; }
    return true;
}

std::vector<std::pair<std::string, float>> ExpertModel::predict_next(const std::string& prompt, size_t k) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto ids = encode(prompt);
    std::vector<float> x, h, logp;
    auto ctx = context_at(ids, ids.size());
    forward(ctx.data(), x, h, logp);
    std::vector<uint32_t> idx(logp.size());
    std::iota(idx.begin(), idx.end(), 0u);
    k = std::min(k, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [&](uint32_t a, uint32_t b) { return logp[a] > logp[b]; });
    std::vector<std::pair<std::string, float>> out;
    for (size_t i = 0; i < k; ++i) out.emplace_back(vocab_[idx[i]], std::exp(logp[idx[i]]));
    return out;
}

// =======================================================================
// persistence  (binary: header, vocab, parameters)
// =======================================================================
bool ExpertModel::save(const std::string& path) const {
    std::lock_guard<std::mutex> lk(mutex_);
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        const char magic[8] = { 'O','R','A','I','M','D','L','1' };
        f.write(magic, 8);
        auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
        w32(static_cast<uint32_t>(vocab_.size()));
        w32(static_cast<uint32_t>(cfg_.dim));
        w32(static_cast<uint32_t>(cfg_.context));
        w32(static_cast<uint32_t>(cfg_.hidden));
        w32(static_cast<uint32_t>(total_iterations_));
        w32(trained_ ? 1u : 0u);
        for (const auto& w : vocab_) {
            w32(static_cast<uint32_t>(w.size()));
            f.write(w.data(), static_cast<std::streamsize>(w.size()));
        }
        auto wv = [&](const std::vector<float>& v) {
            w32(static_cast<uint32_t>(v.size()));
            f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
        };
        wv(emb_); wv(w1_); wv(b1_); wv(w2_); wv(b2_);
        return static_cast<bool>(f);
    } catch (...) {
        return false;
    }
}

bool ExpertModel::load(const std::string& path) {
    std::lock_guard<std::mutex> lk(mutex_);
    try {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        char magic[8];
        f.read(magic, 8);
        if (std::memcmp(magic, "ORAIMDL1", 8) != 0) return false;
        auto r32 = [&]() { uint32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; };
        const uint32_t V = r32(), d = r32(), C = r32(), H = r32(), iters = r32(), tr = r32();
        if (!f || V < 3 || d == 0 || C == 0 || H == 0) return false;
        cfg_.dim = d; cfg_.context = C; cfg_.hidden = H;
        if (cfg_.vocab < V) cfg_.vocab = V;
        vocab_.clear();
        vocab_.reserve(V);
        for (uint32_t i = 0; i < V; ++i) {
            const uint32_t n = r32();
            std::string w(n, ' ');
            f.read(w.data(), n);
            vocab_.push_back(std::move(w));
        }
        word_to_id_.clear();
        for (size_t i = 0; i < vocab_.size(); ++i) word_to_id_[vocab_[i]] = static_cast<uint32_t>(i);
        auto rv = [&](std::vector<float>& v, size_t expect) {
            const uint32_t n = r32();
            if (n != expect) return false;
            v.resize(n);
            f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
            return static_cast<bool>(f);
        };
        init_params();
        if (!rv(emb_, static_cast<size_t>(V) * d) || !rv(w1_, static_cast<size_t>(H) * C * d) ||
            !rv(b1_, H) || !rv(w2_, static_cast<size_t>(V) * H) || !rv(b2_, V))
            return false;
        total_iterations_ = iters;
        trained_ = tr != 0;
        return true;
    } catch (...) {
        return false;
    }
}

std::string ExpertModel::describe() const {
    std::ostringstream ss;
    ss << name_ << "{vocab=" << vocab_.size() << " dim=" << cfg_.dim << " ctx=" << cfg_.context
       << " hidden=" << cfg_.hidden << " params=" << param_count()
       << " trained=" << (trained_ ? "yes(" + std::to_string(total_iterations_) + " it)" : "no") << "}";
    return ss.str();
}

} // namespace openrai
