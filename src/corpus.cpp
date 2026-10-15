#include <openrai/corpus.h>
#include <openrai/expert_model.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>

namespace openrai {

namespace {

struct Gen {
    std::mt19937 rng;
    explicit Gen(unsigned seed) : rng(seed) {}
    int    i(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }
    double f(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(rng); }
    template <typename T> const T& pick(const std::vector<T>& v) { return v[static_cast<size_t>(i(0, static_cast<int>(v.size()) - 1))]; }
    std::string node() { return "node " + std::to_string(i(0, 31)); }
    std::string cluster() { return std::string("cluster ") + static_cast<char>('A' + i(0, 5)); }
    std::string pct() { return std::to_string(i(40, 99)) + " percent"; }
    std::string ms() { return std::to_string(i(3, 900)) + " ms"; }
    std::string num(int lo, int hi) { return std::to_string(i(lo, hi)); }
    std::string fmt(double v) { std::ostringstream ss; ss.precision(2); ss << std::fixed << v; return ss.str(); }
};

using Template = std::function<CorpusLine(Gen&)>;

std::vector<Template> templates_for(const std::string& domain) {
    std::vector<Template> t;
    auto L = [](std::string s, float imp) { return CorpusLine{ std::move(s), imp }; };

    if (domain == "stability") {
        t = {
            [&](Gen& g) { return L("error: service crashed on " + g.node() + " restarting worker " + g.num(1, 16), 0.9f); },
            [&](Gen& g) { return L("stability check passed for " + g.cluster() + " all workers healthy", 0.2f); },
            [&](Gen& g) { return L("failover completed " + g.node() + " recovered after " + g.ms(), 0.7f); },
            [&](Gen& g) { return L("watchdog: " + g.node() + " unresponsive for " + g.num(2, 60) + " seconds", 0.8f); },
            [&](Gen& g) { return L("overload protection engaged on " + g.cluster() + " shedding " + g.pct() + " of load", 0.8f); },
            [&](Gen& g) { return L("restart scheduled for " + g.node() + " to recover from memory leak", 0.6f); },
            [&](Gen& g) { return L("safe mode active: " + g.cluster() + " running with reduced capacity", 0.7f); },
            [&](Gen& g) { return L("uptime " + g.num(1, 400) + " days on " + g.node() + " no failures recorded", 0.1f); },
            [&](Gen& g) { return L("critical: disk failure on " + g.node() + " failing over to replica", 0.95f); },
            [&](Gen& g) { return L("recovery plan: isolate " + g.node() + " then rebalance " + g.cluster(), 0.6f); },
        };
    } else if (domain == "anomaly") {
        t = {
            [&](Gen& g) { return L("anomaly score " + g.fmt(g.f(0.3, 0.99)) + " above threshold on " + g.node(), 0.9f); },
            [&](Gen& g) { return L("spike detected in request rate on " + g.node() + " " + g.num(2, 20) + "x the baseline", 0.8f); },
            [&](Gen& g) { return L("outlier: latency sample " + g.ms() + " deviates " + g.num(3, 9) + " sigma from normal", 0.8f); },
            [&](Gen& g) { return L("trigger: unusual login pattern detected from " + g.cluster(), 0.7f); },
            [&](Gen& g) { return L("anomaly detector retrained baseline window " + g.num(100, 9000) + " samples", 0.3f); },
            [&](Gen& g) { return L("alert threshold raised to " + g.fmt(g.f(0.5, 0.95)) + " for " + g.cluster() + " to reduce false positives", 0.4f); },
            [&](Gen& g) { return L("no anomalies in the last " + g.num(1, 24) + " hours on " + g.node(), 0.1f); },
            [&](Gen& g) { return L("drift detected: feature distribution shifted on " + g.node(), 0.6f); },
            [&](Gen& g) { return L("burst of " + g.num(50, 5000) + " errors within " + g.num(1, 10) + " seconds flagged as anomaly", 0.9f); },
        };
    } else if (domain == "memory") {
        t = {
            [&](Gen& g) { return L("recall: similar context found in memory segment " + g.num(1, 512) + " similarity " + g.fmt(g.f(0.5, 0.99)), 0.5f); },
            [&](Gen& g) { return L("remember that " + g.node() + " was restarted " + g.num(1, 9) + " times last week", 0.5f); },
            [&](Gen& g) { return L("retrieve history of " + g.cluster() + " from the vector store", 0.4f); },
            [&](Gen& g) { return L("stored " + g.num(10, 900) + " new embeddings in long term memory", 0.2f); },
            [&](Gen& g) { return L("what did we learn earlier about " + g.node() + " and its latency", 0.5f); },
            [&](Gen& g) { return L("memory consolidation: merged " + g.num(2, 40) + " duplicate contexts", 0.3f); },
            [&](Gen& g) { return L("earlier answer recalled with confidence " + g.fmt(g.f(0.4, 0.99)), 0.5f); },
            [&](Gen& g) { return L("context window holds the last " + g.num(4, 64) + " exchanges about " + g.cluster(), 0.3f); },
            [&](Gen& g) { return L("forget stale entries older than " + g.num(1, 90) + " days", 0.3f); },
        };
    } else if (domain == "throughput") {
        t = {
            [&](Gen& g) { return L("latency on the gpu queue exceeded " + g.ms() + " batch size " + g.num(8, 2048), 0.7f); },
            [&](Gen& g) { return L("throughput " + g.num(100, 90000) + " requests per second on " + g.cluster(), 0.3f); },
            [&](Gen& g) { return L("stream " + g.num(1, 32) + " connected on port " + g.num(9000, 9999), 0.2f); },
            [&](Gen& g) { return L("queue depth " + g.num(0, 5000) + " workers " + g.num(1, 64) + " utilization " + g.pct(), 0.4f); },
            [&](Gen& g) { return L("batch inference took " + g.ms() + " for " + g.num(8, 512) + " samples", 0.3f); },
            [&](Gen& g) { return L("bandwidth saturated on " + g.node() + " dropping to " + g.pct() + " efficiency", 0.7f); },
            [&](Gen& g) { return L("thread pool resized to " + g.num(2, 64) + " threads to keep latency under " + g.ms(), 0.4f); },
            [&](Gen& g) { return L("pipeline stage " + g.num(1, 9) + " is the bottleneck at " + g.ms() + " per item", 0.6f); },
        };
    } else if (domain == "planning") {
        t = {
            [&](Gen& g) { return L("goal: prioritize stability over throughput for " + g.cluster(), 0.6f); },
            [&](Gen& g) { return L("plan step " + g.num(1, 9) + ": increase thread priority on " + g.node(), 0.5f); },
            [&](Gen& g) { return L("schedule maintenance for " + g.cluster() + " in " + g.num(1, 72) + " hours", 0.4f); },
            [&](Gen& g) { return L("command: decrease batch size to " + g.num(8, 256) + " and observe", 0.5f); },
            [&](Gen& g) { return L("strategy: rebalance load from " + g.node() + " to " + g.cluster(), 0.5f); },
            [&](Gen& g) { (void)g; return L("how to reduce latency: first profile then cache then scale", 0.4f); },
            [&](Gen& g) { return L("objective for this window: keep errors below " + g.pct() + " of requests", 0.5f); },
            [&](Gen& g) { return L("next action: verify the fix on " + g.node() + " before rollout", 0.5f); },
            [&](Gen& g) { (void)g; return L("priority order: safety then correctness then speed", 0.4f); },
        };
    } else if (domain == "reflection") {
        t = {
            [&](Gen& g) { return L("reflect: compare current output with earlier memory segment " + g.num(1, 256), 0.4f); },
            [&](Gen& g) { return L("conclusion supported by " + g.num(1, 9) + " pieces of evidence confidence " + g.fmt(g.f(0.3, 0.99)), 0.5f); },
            [&](Gen& g) { return L("counterpoint: the evidence from " + g.node() + " is weak", 0.5f); },
            [&](Gen& g) { return L("reasoning chain has " + g.num(2, 8) + " steps and no contradiction", 0.4f); },
            [&](Gen& g) { (void)g; return L("judge the conclusion: should it become the final output", 0.5f); },
            [&](Gen& g) { return L("self check: the last " + g.num(2, 30) + " outputs were consistent", 0.3f); },
            [&](Gen& g) { (void)g; return L("the previous answer was uncertain so gather more evidence", 0.5f); },
            [&](Gen& g) { return L("verify assumption that " + g.cluster() + " is healthy before concluding", 0.5f); },
        };
    } else if (domain == "telemetry") {
        t = {
            [&](Gen& g) { return L("sensor: temperature spike detected at " + g.node() + " value " + g.num(60, 95) + "c", 0.8f); },
            [&](Gen& g) { return L("probe: cpu usage " + g.pct() + " on " + g.node(), 0.4f); },
            [&](Gen& g) { return L("measurement: fan speed " + g.num(800, 4000) + " rpm on " + g.node(), 0.2f); },
            [&](Gen& g) { return L("alert: memory usage exceeded " + g.pct() + " on " + g.node(), 0.8f); },
            [&](Gen& g) { return L("telemetry sample " + g.num(1, 99999) + " voltage " + g.fmt(g.f(11.4, 12.6)) + " volts", 0.2f); },
            [&](Gen& g) { return L("gpu temperature " + g.num(40, 90) + "c power draw " + g.num(50, 350) + " watts", 0.4f); },
            [&](Gen& g) { return L("disk usage " + g.pct() + " inode usage " + g.pct() + " on " + g.node(), 0.4f); },
            [&](Gen& g) { return L("network probe: packet loss " + g.fmt(g.f(0.0, 5.0)) + " percent on " + g.cluster(), 0.5f); },
        };
    } else if (domain == "language") {
        t = {
            [&](Gen& g) { return L("user: summarize current system state for window " + g.num(0, 127), 0.5f); },
            [&](Gen& g) { return L("query: explain why " + g.node() + " was restarted", 0.5f); },
            [&](Gen& g) { return L("please describe the plan for " + g.cluster() + " in simple words", 0.4f); },
            [&](Gen& g) { return L("feed: new labelled sample batch received size " + g.num(64, 4096), 0.3f); },
            [&](Gen& g) { return L("user asks what happened on " + g.node() + " yesterday", 0.5f); },
            [&](Gen& g) { return L("write a short report about the last " + g.num(1, 24) + " hours", 0.4f); },
            [&](Gen& g) { return L("tell me how many requests failed on " + g.cluster(), 0.5f); },
            [&](Gen& g) { (void)g; return L("chat: hello can you help me understand the alerts", 0.4f); },
        };
    } else if (domain == "math") {
        // Mixed symbolic + worded phrasing, small operand ranges so a
        // short-context neural model can actually learn the mapping
        // (addition/subtraction up to 2 digits, small multiplication,
        // exact division) instead of just memorising a parser shortcut.
        // Small, finite operand ranges on purpose: a short-context neural
        // model has to learn a (still sizeable) lookup-like mapping rather
        // than a general algorithm, so keeping the space small enough to
        // be covered many times over by the corpus is what makes this
        // learnable at all without a parser.
        t = {
            [&](Gen& g) { int a = g.i(0, 49), b = g.i(0, 49); return L(std::to_string(a) + " + " + std::to_string(b) + " = " + std::to_string(a + b), 0.5f); },
            [&](Gen& g) { int a = g.i(0, 49), b = g.i(0, 49); return L("what is " + std::to_string(a) + " + " + std::to_string(b) + " = " + std::to_string(a + b), 0.5f); },
            [&](Gen& g) { int a = g.i(0, 49), b = g.i(0, 49); return L("what is " + std::to_string(a) + " plus " + std::to_string(b) + " equals " + std::to_string(a + b), 0.6f); },
            [&](Gen& g) { int a = g.i(0, 49), b = g.i(0, 49); return L(std::to_string(a) + " plus " + std::to_string(b) + " is " + std::to_string(a + b), 0.5f); },
            [&](Gen& g) { int a = g.i(0, 49), b = g.i(0, 49); return L("sum of " + std::to_string(a) + " and " + std::to_string(b) + " is " + std::to_string(a + b), 0.5f); },
            [&](Gen& g) { int a = g.i(0, 49), b = g.i(0, 49); return L("calculate " + std::to_string(a) + " plus " + std::to_string(b) + " the answer is " + std::to_string(a + b), 0.6f); },
            [&](Gen& g) { int a = g.i(0, 49), b = g.i(0, a); return L(std::to_string(a) + " - " + std::to_string(b) + " = " + std::to_string(a - b), 0.5f); },
            [&](Gen& g) { int a = g.i(0, 49), b = g.i(0, a); return L("what is " + std::to_string(a) + " minus " + std::to_string(b) + " equals " + std::to_string(a - b), 0.6f); },
            [&](Gen& g) { int a = g.i(0, 49), b = g.i(0, a); return L(std::to_string(a) + " minus " + std::to_string(b) + " is " + std::to_string(a - b), 0.5f); },
            [&](Gen& g) { int a = g.i(0, 12), b = g.i(0, 12); return L(std::to_string(a) + " * " + std::to_string(b) + " = " + std::to_string(a * b), 0.5f); },
            [&](Gen& g) { int a = g.i(0, 12), b = g.i(0, 12); return L("what is " + std::to_string(a) + " times " + std::to_string(b) + " equals " + std::to_string(a * b), 0.6f); },
            [&](Gen& g) { int a = g.i(0, 12), b = g.i(0, 12); return L(std::to_string(a) + " times " + std::to_string(b) + " is " + std::to_string(a * b), 0.5f); },
            [&](Gen& g) { int b = g.i(1, 12), c = g.i(0, 12); return L(std::to_string(b * c) + " / " + std::to_string(b) + " = " + std::to_string(c), 0.5f); },
            [&](Gen& g) { int b = g.i(1, 12), c = g.i(0, 12); return L("what is " + std::to_string(b * c) + " divided by " + std::to_string(b) + " equals " + std::to_string(c), 0.6f); },
        };
    } else {
        t = {
            [&](Gen& g) { return L("system state for " + g.node() + " is nominal", 0.2f); },
            [&](Gen& g) { return L("new data stream connected on " + g.cluster(), 0.3f); },
            [&](Gen& g) { return L("input received for window " + g.num(0, 999), 0.2f); },
            [&](Gen& g) { return L("current status of " + g.node() + " updated", 0.2f); },
        };
    }
    return t;
}

} // namespace

std::vector<CorpusLine> generate_domain_corpus(const std::string& domain, size_t target_words, unsigned seed) {
    Gen g(seed);
    auto templates = templates_for(domain);
    std::vector<CorpusLine> out;
    size_t words = 0;
    while (words < target_words) {
        CorpusLine line = g.pick(templates)(g);
        words += ExpertModel::words(line.text).size();
        out.push_back(std::move(line));
    }
    return out;
}

std::vector<CorpusLine> generate_domain_corpus_lines(const std::string& domain, size_t n_lines, unsigned seed) {
    Gen g(seed);
    auto templates = templates_for(domain);
    std::vector<CorpusLine> out;
    out.reserve(n_lines);
    for (size_t i = 0; i < n_lines; ++i) out.push_back(g.pick(templates)(g));
    return out;
}

std::vector<MathQuestion> generate_math_questions(size_t n, unsigned seed) {
    Gen g(seed);
    std::vector<MathQuestion> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        MathQuestion q;
        switch (g.i(0, 3)) {
        case 0: { int a = g.i(0, 49), b = g.i(0, 49);
                   q.prompt = "what is " + std::to_string(a) + " plus " + std::to_string(b) + " equals";
                   q.expected = a + b; break; }
        case 1: { int a = g.i(0, 49), b = g.i(0, a);
                   q.prompt = "what is " + std::to_string(a) + " minus " + std::to_string(b) + " equals";
                   q.expected = a - b; break; }
        case 2: { int a = g.i(0, 12), b = g.i(0, 12);
                   q.prompt = "what is " + std::to_string(a) + " times " + std::to_string(b) + " equals";
                   q.expected = static_cast<long long>(a) * b; break; }
        default: { int b = g.i(1, 12), c = g.i(0, 12);
                   q.prompt = "what is " + std::to_string(b * c) + " divided by " + std::to_string(b) + " equals";
                   q.expected = c; break; }
        }
        out.push_back(std::move(q));
    }
    return out;
}

std::vector<CorpusLine> load_corpus_file(const std::string& path) {
    std::vector<CorpusLine> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        CorpusLine cl;
        const auto tab = line.find('\t');
        if (tab != std::string::npos) {
            try { cl.importance = std::stof(line.substr(0, tab)); cl.text = line.substr(tab + 1); }
            catch (...) { cl.text = line; }
        } else {
            cl.text = line;
        }
        if (!cl.text.empty()) out.push_back(std::move(cl));
    }
    return out;
}

bool save_corpus_file(const std::string& path, const std::vector<CorpusLine>& lines) {
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream f(path, std::ios::trunc);
        if (!f) return false;
        for (const auto& l : lines) f << l.importance << '\t' << l.text << '\n';
        return static_cast<bool>(f);
    } catch (...) {
        return false;
    }
}

std::vector<CorpusLine> load_or_generate_corpus(const std::string& dir, const std::string& domain,
                                                size_t target_words, unsigned seed) {
    const std::string path = dir + "/" + domain + ".txt";
    auto lines = load_corpus_file(path);
    if (!lines.empty()) return lines;
    lines = generate_domain_corpus(domain, target_words, seed);
    save_corpus_file(path, lines);
    return lines;
}

size_t corpus_word_count(const std::vector<CorpusLine>& lines) {
    size_t n = 0;
    for (const auto& l : lines) n += ExpertModel::words(l.text).size();
    return n;
}

std::vector<std::string> corpus_texts(const std::vector<CorpusLine>& lines) {
    std::vector<std::string> out;
    out.reserve(lines.size());
    for (const auto& l : lines) out.push_back(l.text);
    return out;
}

} // namespace openrai
