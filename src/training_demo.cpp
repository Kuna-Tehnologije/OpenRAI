#include <openrai/training_demo.h>
#include <openrai/gpu_runtime.h>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>

namespace openrai {

std::string TrainingDemoReport::describe() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3)
       << "Expert knowledge training demo\n"
       << "  data        : " << corpus_lines << " lines / " << corpus_words << " words"
       << (options.corpus_file.empty() ? " (synthetic '" + options.domain + "' corpus)" : " (" + options.corpus_file + ")")
       << ", vocabulary " << vocab << "\n"
       << "  nodes       : " << nodes.size() << " test experts, identical data, " << options.iterations
       << " iterations x batch " << options.batch << " each, one thread per node\n"
       << "  contrast    : '" << options.foreign_domain << "' text the experts never saw\n";
    for (const auto& n : nodes) {
        ss << "  -- " << n.name << " --\n"
           << "     loss " << n.report.first_loss << " -> " << n.report.last_loss
           << " (best " << n.report.best_loss << ", " << std::setprecision(2) << n.report.seconds << " s)\n     curve:";
        for (size_t i = 0; i < n.report.losses.size(); i += std::max<size_t>(1, n.report.losses.size() / 8))
            ss << " " << std::setprecision(2) << n.report.losses[i];
        ss << "\n" << std::setprecision(2)
           << "     in-domain  logp " << n.in_domain.mean_logprob << "  ppl " << n.in_domain.perplexity << "  coverage " << n.in_domain.coverage << "\n"
           << "     foreign    logp " << n.foreign.mean_logprob   << "  ppl " << n.foreign.perplexity   << "  coverage " << n.foreign.coverage << "\n";
        for (const auto& s : n.samples) ss << "     generate: " << s << "\n";
        if (n.mlp_accuracy >= 0.f)
            ss << "     MLP scorer: loss " << n.mlp_first_loss << " -> " << n.mlp_last_loss
               << ", domain accuracy " << std::setprecision(1) << (100.f * n.mlp_accuracy) << " %\n";
    }
    ss << std::setprecision(2) << "  total       : " << seconds << " s\n"
       << "  verdict     : " << (passed ? "PASS - every node learned the data (loss down >= 25 %, in-domain text >= 2 nats more likely than foreign text)"
                                        : "FAIL - see the numbers above") << "\n";
    return ss.str();
}

TrainingDemoReport run_training_demo(const TrainingDemoOptions& opt) {
    TrainingDemoReport rep;
    rep.options = opt;
    const auto t0 = std::chrono::steady_clock::now();

    std::vector<CorpusLine> corpus = opt.corpus_file.empty()
        ? generate_domain_corpus(opt.domain, opt.words, 777)
        : load_corpus_file(opt.corpus_file);
    if (corpus.empty()) corpus = generate_domain_corpus(opt.domain, opt.words, 777);
    std::vector<CorpusLine> foreign = generate_domain_corpus(opt.foreign_domain, std::max<size_t>(1000, opt.words / 5), 778);

    // hold out 10 % of the data for scoring
    std::mt19937 rng(4242);
    std::shuffle(corpus.begin(), corpus.end(), rng);
    const size_t held = std::max<size_t>(1, corpus.size() / 10);
    std::vector<CorpusLine> test(corpus.begin(), corpus.begin() + static_cast<std::ptrdiff_t>(held));
    std::vector<CorpusLine> train(corpus.begin() + static_cast<std::ptrdiff_t>(held), corpus.end());
    const auto train_texts = corpus_texts(train);
    rep.corpus_lines = corpus.size();
    rep.corpus_words = corpus_word_count(corpus);

    rep.nodes.resize(std::max<size_t>(1, opt.nodes));
    std::vector<std::thread> workers;
    for (size_t k = 0; k < rep.nodes.size(); ++k) {
        workers.emplace_back([&, k] {
            TrainingDemoNode& n = rep.nodes[k];
            n.name = "T" + std::to_string(k) + ":test-expert";
            ExpertModel::Config mc;
            mc.seed = 100 + static_cast<unsigned>(k);      // different init, same data
            ExpertModel model(n.name, mc);
            model.build_vocab(train_texts);
            n.report = model.train(train_texts, opt.iterations, opt.batch);

            auto mean_score = [&](const std::vector<CorpusLine>& lines) {
                ModelScore m; double lp = 0, cov = 0; size_t cnt = 0;
                for (size_t i = 0; i < lines.size() && i < 64; ++i) {
                    auto s = model.score(lines[i].text);
                    if (!s.valid) continue;
                    lp += s.mean_logprob; cov += s.coverage; ++cnt;
                }
                if (cnt) { m.mean_logprob = static_cast<float>(lp / cnt); m.coverage = static_cast<float>(cov / cnt);
                           m.perplexity = std::exp(-m.mean_logprob); m.valid = true; }
                return m;
            };
            n.in_domain = mean_score(test);
            n.foreign   = mean_score(foreign);

            for (size_t i = 0; i < 3 && i < test.size(); ++i) {
                auto w = ExpertModel::words(test[i].text);
                std::string prompt;
                for (size_t j = 0; j < w.size() && j < 3; ++j) prompt += (j ? " " : "") + w[j];
                n.samples.push_back("\"" + prompt + "\" -> \"" + model.generate(prompt, 10) + "\"");
            }

            if (opt.train_mlp) {
                GPUNeuralNet net(NetArch::expert_lite(opt.mlp_hidden), 50 + static_cast<unsigned>(k), true);
                std::mt19937 r2(9 + static_cast<unsigned>(k));
                const size_t B = 64;
                std::vector<float> in(B * 32), tg(B * 4);
                for (size_t it = 0; it < opt.iterations; ++it) {
                    for (size_t b = 0; b < B; ++b) {
                        const bool mine = (b % 2 == 0);
                        const auto& src = mine ? train : foreign;
                        const auto& cl = src[r2() % src.size()];
                        encode_input_v2(cl.text, in.data() + b * 32, 32);
                        float* t = tg.data() + b * 4;
                        t[0] = mine ? 1.f : 0.f; t[1] = cl.importance; t[2] = t[0] * (cl.importance >= 0.5f ? 1.f : 0.f); t[3] = cl.importance;
                    }
                    const float loss = net.train_batch(in.data(), tg.data(), B, 0.02f);
                    if (it == 0) n.mlp_first_loss = loss;
                    n.mlp_last_loss = loss;
                }
                net.sync_weights_to_device();
                size_t correct = 0, total = 0;
                for (size_t i = 0; i < 200; ++i) {
                    const bool mine = (i % 2 == 0);
                    const auto& src = mine ? test : foreign;
                    const auto& cl = src[i % src.size()];
                    float x[32];
                    encode_input_v2(cl.text, x, 32);
                    if ((net.run_single(x).relevance > 0.5f) == mine) ++correct;
                    ++total;
                }
                n.mlp_accuracy = static_cast<float>(correct) / static_cast<float>(total);
            }
        });
    }
    for (auto& w : workers) w.join();

    // vocabulary size (from one throwaway model)
    { ExpertModel v("vocab"); v.build_vocab(train_texts); rep.vocab = v.vocab_size(); }

    rep.passed = true;
    for (const auto& n : rep.nodes) {
        const bool loss_ok = n.report.first_loss > 0.f && n.report.last_loss <= 0.75f * n.report.first_loss;
        const bool sep_ok  = n.in_domain.valid && n.foreign.valid && (n.in_domain.mean_logprob - n.foreign.mean_logprob) >= 2.f;
        const bool mlp_ok  = !opt.train_mlp || n.mlp_accuracy >= 0.85f;
        if (!(loss_ok && sep_ok && mlp_ok)) rep.passed = false;
    }
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return rep;
}

// =======================================================================
// Large-scale confirmation run
// =======================================================================
std::string BigTrainResult::describe() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3)
       << "Large-scale expert knowledge confirmation\n"
       << "=== MATH MODEL (no parser, no calculator - generation only) ===\n"
       << "  corpus   : " << math_corpus_lines << " lines / " << math_corpus_words << " words, vocabulary " << math_vocab << "\n"
       << "  training : " << math_report.describe() << "\n  curve:";
    for (size_t i = 0; i < math_report.losses.size(); i += std::max<size_t>(1, math_report.losses.size() / 10))
        ss << " " << std::setprecision(2) << math_report.losses[i];
    ss << "\n  held-out arithmetic questions answered by ExpertModel::generate() alone:\n";
    for (const auto& s : eval_samples) ss << "    " << s << "\n";
    ss << std::setprecision(1) << "  accuracy : " << eval_correct << "/" << eval_total
       << " = " << (eval_total ? 100.0 * eval_correct / eval_total : 0.0) << " %\n"
       << "\n=== LANGUAGE MODEL ===\n"
       << "  corpus   : " << lang_corpus_words << " words (domain), vocabulary " << lang_vocab << "\n"
       << std::setprecision(3) << "  training : " << lang_report.describe() << "\n  curve:";
    for (size_t i = 0; i < lang_report.losses.size(); i += std::max<size_t>(1, lang_report.losses.size() / 10))
        ss << " " << std::setprecision(2) << lang_report.losses[i];
    ss << "\n";
    for (const auto& s : lang_samples) ss << "  " << s << "\n";
    ss << std::setprecision(1) << "\n  total time: " << std::setprecision(2) << seconds << " s\n"
       << "  verdict   : " << (passed ? "PASS - the math model answers held-out word+number questions correctly "
                                        "by generation alone, and the language model's loss fell sharply."
                                      : "FAIL - see the numbers above") << "\n";
    return ss.str();
}

BigTrainResult run_big_training_demo(const BigTrainOptions& opt) {
    BigTrainResult res;
    const auto t0 = std::chrono::steady_clock::now();

    // ---- 1. math model: mixed symbolic + worded arithmetic, no parser ----
    auto math_corpus = generate_domain_corpus_lines("math", opt.math_lines, opt.seed);
    res.math_corpus_lines = math_corpus.size();
    res.math_corpus_words = corpus_word_count(math_corpus);
    auto math_texts = corpus_texts(math_corpus);

    ExpertModel::Config mc;
    mc.vocab = opt.math_vocab; mc.dim = opt.math_dim; mc.hidden = opt.math_hidden; mc.context = opt.math_context;
    mc.seed = opt.seed;
    ExpertModel math_model("math-big", mc);
    math_model.build_vocab(math_texts);
    res.math_vocab = math_model.vocab_size();

    if (opt.verbose_progress) {
        // train in checkpoints so progress can be printed without changing
        // the learning trajectory (same total iterations/batches)
        const size_t steps = 10;
        for (size_t k = 0; k < steps; ++k) {
            const size_t n = opt.iterations / steps + (k == steps - 1 ? opt.iterations % steps : 0);
            auto part = math_model.train(math_texts, n, opt.batch);
            res.math_report.iterations += part.iterations;
            res.math_report.tokens = part.tokens;
            res.math_report.losses.insert(res.math_report.losses.end(), part.losses.begin(), part.losses.end());
            res.math_report.seconds += part.seconds;
            std::cout << "  [math]  iter " << res.math_report.losses.size() << "/" << opt.iterations
                      << "  loss " << part.last_loss << "  (" << std::fixed << std::setprecision(1) << res.math_report.seconds << "s)" << std::endl;
        }
        res.math_report.first_loss = res.math_report.losses.front();
        res.math_report.best_loss = *std::min_element(res.math_report.losses.begin(), res.math_report.losses.end());
        const size_t tail = std::min<size_t>(10, res.math_report.losses.size());
        res.math_report.last_loss = std::accumulate(res.math_report.losses.end() - static_cast<std::ptrdiff_t>(tail),
                                                     res.math_report.losses.end(), 0.f) / tail;
        res.math_report.batch = opt.batch;
    } else {
        res.math_report = math_model.train(math_texts, opt.iterations, opt.batch);
    }

    // Evaluate: held-out questions, ExpertModel::generate() only.
    auto questions = generate_math_questions(opt.eval_questions, opt.seed + 1);
    res.eval_total = questions.size();
    for (size_t i = 0; i < questions.size(); ++i) {
        const auto& q = questions[i];
        const std::string generated = math_model.generate(q.prompt, 6, 0.f);
        long long got = 0;
        const bool parsed = ExpertModel::trailing_integer(generated, got);
        const bool ok = parsed && got == q.expected;
        if (ok) ++res.eval_correct;
        if (i < 20) {
            std::ostringstream s;
            s << "\"" << q.prompt << "\" -> model says \"" << generated << "\" (expected " << q.expected << ") "
              << (ok ? "[ok]" : "[FAIL]");
            res.eval_samples.push_back(s.str());
        }
    }

    // ---- 2. language model: one domain at ~500k words ----
    auto lang_corpus = generate_domain_corpus(opt.language_domain, opt.language_words, opt.seed + 2);
    res.lang_corpus_words = corpus_word_count(lang_corpus);
    auto lang_texts = corpus_texts(lang_corpus);

    ExpertModel::Config lc;
    lc.vocab = opt.lang_vocab; lc.dim = opt.lang_dim; lc.hidden = opt.lang_hidden; lc.context = opt.lang_context;
    lc.seed = opt.seed + 3;
    ExpertModel lang_model(opt.language_domain + "-big", lc);
    lang_model.build_vocab(lang_texts);
    res.lang_vocab = lang_model.vocab_size();

    if (opt.verbose_progress) {
        const size_t steps = 10;
        for (size_t k = 0; k < steps; ++k) {
            const size_t n = opt.iterations / steps + (k == steps - 1 ? opt.iterations % steps : 0);
            auto part = lang_model.train(lang_texts, n, opt.batch);
            res.lang_report.iterations += part.iterations;
            res.lang_report.tokens = part.tokens;
            res.lang_report.losses.insert(res.lang_report.losses.end(), part.losses.begin(), part.losses.end());
            res.lang_report.seconds += part.seconds;
            std::cout << "  [lang]  iter " << res.lang_report.losses.size() << "/" << opt.iterations
                      << "  loss " << part.last_loss << "  (" << std::fixed << std::setprecision(1) << res.lang_report.seconds << "s)" << std::endl;
        }
        res.lang_report.first_loss = res.lang_report.losses.front();
        res.lang_report.best_loss = *std::min_element(res.lang_report.losses.begin(), res.lang_report.losses.end());
        const size_t tail = std::min<size_t>(10, res.lang_report.losses.size());
        res.lang_report.last_loss = std::accumulate(res.lang_report.losses.end() - static_cast<std::ptrdiff_t>(tail),
                                                     res.lang_report.losses.end(), 0.f) / tail;
        res.lang_report.batch = opt.batch;
    } else {
        res.lang_report = lang_model.train(lang_texts, opt.iterations, opt.batch);
    }

    for (size_t i = 0; i < 5 && i < lang_corpus.size(); ++i) {
        auto w = ExpertModel::words(lang_corpus[i].text);
        std::string prompt;
        for (size_t j = 0; j < w.size() && j < 3; ++j) prompt += (j ? " " : "") + w[j];
        res.lang_samples.push_back("generate: \"" + prompt + "\" -> \"" + lang_model.generate(prompt, 12) + "\"");
    }

    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double acc = res.eval_total ? static_cast<double>(res.eval_correct) / res.eval_total : 0.0;
    res.passed = acc >= 0.6 && res.math_report.last_loss < 0.5f * res.math_report.first_loss &&
                res.lang_report.last_loss < 0.5f * res.lang_report.first_loss;
    return res;
}

} // namespace openrai
