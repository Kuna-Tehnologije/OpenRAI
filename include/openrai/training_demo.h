#pragma once

// -----------------------------------------------------------------------
// training_demo.h
//   Verifiable demonstration that expert knowledge is learned, not
//   hard-coded: N test nodes receive the *same* corpus (default ~10 000
//   words of one domain), each trains its own ExpertModel for a fixed
//   number of iterations on its own thread, and the report shows the loss
//   curves, what the models generate, how they score in-domain versus
//   foreign text, and the accuracy of the trained MLP scorer.
//
//   ./openrai --train-demo [--train-iters 200] [--train-nodes 3]
//                          [--train-corpus FILE] [--train-domain telemetry]
// -----------------------------------------------------------------------

#include <openrai/corpus.h>
#include <openrai/expert_model.h>

#include <string>
#include <vector>

namespace openrai {

struct TrainingDemoOptions {
    size_t      nodes       = 3;
    size_t      iterations  = 200;
    size_t      batch       = 64;
    size_t      words       = 10000;
    std::string domain      = "telemetry";     // synthetic corpus domain
    std::string corpus_file;                   // or: your own text file
    std::string foreign_domain = "math";       // contrast corpus
    bool        train_mlp   = true;
    size_t      mlp_hidden  = 256;
};

struct TrainingDemoNode {
    std::string  name;
    TrainReport  report;
    ModelScore   in_domain;        // mean over held-out in-domain lines
    ModelScore   foreign;          // mean over foreign-domain lines
    std::vector<std::string> samples;   // prompt -> generation
    float        mlp_accuracy = -1.f;   // domain classifier accuracy (-1 = not trained)
    float        mlp_first_loss = 0.f, mlp_last_loss = 0.f;
};

struct TrainingDemoReport {
    TrainingDemoOptions options;
    size_t corpus_lines = 0, corpus_words = 0, vocab = 0;
    std::vector<TrainingDemoNode> nodes;
    double seconds = 0.0;
    bool   passed = false;          // every node: loss fell >= 25 % and in-domain >> foreign
    std::string describe() const;
};

TrainingDemoReport run_training_demo(const TrainingDemoOptions& opt);

// -----------------------------------------------------------------------
// Large-scale confirmation run: a math model trained on up to ~500k mixed
// symbolic/worded arithmetic lines and a language model trained on up to
// ~500k words, both for (by default) 10000 iterations.  The report answers
// held-out questions using ONLY ExpertModel::generate() - no parser, no
// calculator - to prove the *model* learned arithmetic, not a tool.
// -----------------------------------------------------------------------
struct BigTrainOptions {
    size_t math_lines    = 500000;   // generated math corpus size (lines)
    size_t language_words = 500000;  // generated language corpus size (words)
    std::string language_domain = "telemetry";
    size_t iterations    = 10000;
    size_t batch         = 128;
    // math model: short vocabulary (digits/operators/a few words) but a
    // long context window so a full worded question fits before the answer
    size_t math_vocab = 64, math_dim = 16, math_hidden = 256, math_context = 14;
    // language model: bigger vocabulary, shorter context (plain n-gram LM)
    size_t lang_vocab = 2048, lang_dim = 32, lang_hidden = 128, lang_context = 4;
    size_t eval_questions = 300;     // held-out arithmetic questions to test
    unsigned seed = 2024;
    bool   verbose_progress = true;  // print a loss checkpoint every 10 %
};

struct BigTrainResult {
    TrainReport math_report, lang_report;
    size_t math_corpus_lines = 0, math_corpus_words = 0, math_vocab = 0;
    size_t lang_corpus_words = 0, lang_vocab = 0;
    // arithmetic-by-generation evaluation (model only, no parser)
    size_t eval_total = 0, eval_correct = 0;
    std::vector<std::string> eval_samples;     // "question -> generated (expected) [ok/FAIL]"
    std::vector<std::string> lang_samples;     // language model generations
    double seconds = 0.0;
    bool   passed = false;
    std::string describe() const;
};

BigTrainResult run_big_training_demo(const BigTrainOptions& opt);

} // namespace openrai
