#pragma once

// -----------------------------------------------------------------------
// expert_model.h
//   ExpertModel - the knowledge of one expert node, as a trained model.
//
//   A small neural language model (Bengio-style neural n-gram):
//
//       tokens  t-3, t-2, t-1  --embed(d)-->  concat (3d)
//              --> hidden (LeakyReLU) --> softmax over the vocabulary
//
//   trained by cross-entropy on the expert's own corpus with Adam.
//   What it gives the node:
//     * score(text)     - how well the expert "understands" the text: mean
//                         log-likelihood per token under its model.  The
//                         primary compares the scores of all experts, so an
//                         expert is chosen because *its model* finds the
//                         input familiar - not because of a keyword list.
//     * generate(...)   - the expert's own continuation of a prompt in the
//                         vocabulary of its field (its "finding").
//     * train(...)      - learning from data; losses per iteration so the
//                         progress can be verified.
//     * save / load     - persistence in data/models/<domain>.model
//
//   The model tokenizer is generic (lower-cased words, digits as single
//   tokens, operator symbols as tokens) - it knows nothing about domains.
// -----------------------------------------------------------------------

#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace openrai {

struct ModelScore {
    float  mean_logprob = -1e9f;  // average log p(token | context)
    float  perplexity   = 0.f;
    float  coverage     = 0.f;    // fraction of tokens in the vocabulary
    size_t tokens       = 0;
    bool   valid        = false;
};

struct TrainReport {
    size_t iterations = 0;
    size_t batch = 0;
    size_t tokens = 0;             // training tokens in the corpus
    float  first_loss = 0.f;
    float  last_loss = 0.f;        // mean of the last 10 iterations
    float  best_loss = 0.f;
    double seconds = 0.0;
    std::vector<float> losses;     // one per iteration
    std::string describe() const;
};

struct ExpertModelConfig {
    size_t   vocab   = 2048;   // max vocabulary (incl. 3 specials)
    size_t   dim     = 32;     // embedding width
    size_t   context = 3;      // tokens of context
    size_t   hidden  = 128;
    float    lr      = 0.004f; // Adam learning rate
    unsigned seed    = 1;
};

class ExpertModel {
public:
    using Config = ExpertModelConfig;

    explicit ExpertModel(std::string name, Config cfg = Config());

    // ---- tokenizer (generic) ----
    static std::vector<std::string> words(const std::string& text);
    std::vector<uint32_t> encode(const std::string& text) const;   // unknown -> UNK
    std::string decode(const std::vector<uint32_t>& ids) const;

    // ---- data ----
    // Build the vocabulary from a corpus (most frequent words first).
    void build_vocab(const std::vector<std::string>& corpus);
    size_t vocab_size() const { return vocab_.size(); }
    size_t param_count() const;

    // ---- learning ----
    // `iterations` minibatch steps of `batch` random positions.
    TrainReport train(const std::vector<std::string>& corpus, size_t iterations, size_t batch = 64);
    bool trained() const { return trained_; }
    size_t train_iterations() const { return total_iterations_; }

    // ---- knowledge ----
    ModelScore  score(const std::string& text) const;
    std::string generate(const std::string& prompt, size_t max_tokens = 12, float temperature = 0.0f) const;
    // top-k next-token predictions after `prompt` (word, probability)
    std::vector<std::pair<std::string, float>> predict_next(const std::string& prompt, size_t k = 5) const;

    // Pull a trailing integer out of generated text ("equals 1 9" -> 19).
    // Returns false if the text does not end in a run of digit tokens.
    static bool trailing_integer(const std::string& text, long long& value);

    // ---- persistence ----
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    const std::string& name() const { return name_; }
    const Config& config() const { return cfg_; }
    std::string describe() const;

private:
    static constexpr uint32_t BOS = 0, UNK = 1, EOS = 2;

    std::string name_;
    Config cfg_;
    bool   trained_ = false;
    size_t total_iterations_ = 0;

    std::vector<std::string>                  vocab_;       // id -> word
    std::unordered_map<std::string, uint32_t> word_to_id_;

    // parameters (row-major)
    std::vector<float> emb_;   // [V x d]
    std::vector<float> w1_;    // [hidden x (context*d)]
    std::vector<float> b1_;    // [hidden]
    std::vector<float> w2_;    // [V x hidden]
    std::vector<float> b2_;    // [V]
    // Adam moments
    std::vector<float> m_emb_, v_emb_, m_w1_, v_w1_, m_b1_, v_b1_, m_w2_, v_w2_, m_b2_, v_b2_;
    size_t adam_t_ = 0;

    mutable std::mutex mutex_;
    mutable std::mt19937 rng_;

    void   init_params();
    // forward for one position; returns log-softmax into `logp` (size V)
    void   forward(const uint32_t* ctx, std::vector<float>& x, std::vector<float>& h, std::vector<float>& logp) const;
    std::vector<uint32_t> context_at(const std::vector<uint32_t>& seq, size_t pos) const;
};

} // namespace openrai
