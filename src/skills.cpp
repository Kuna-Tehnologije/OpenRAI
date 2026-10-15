#include <openrai/skills.h>
#include <openrai/semantic.h>
#include <openrai/experience_store.h>
#include <openrai/lm_adapter.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace openrai {

// =======================================================================
// Tokens
// =======================================================================
namespace {

enum class Tok { Number, Op, LParen, RParen, Percent, Acc, Sqrt, Word, End };

struct Token {
    Tok         type = Tok::End;
    double      value = 0.0;
    char        op = 0;          // + - * / ^
    std::string text;
};

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// Replace a whole word / phrase only (bounded by non-alphanumerics)
void replace_phrase(std::string& s, const std::string& phrase, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(phrase, pos)) != std::string::npos) {
        const bool left_ok  = pos == 0 || !std::isalnum(static_cast<unsigned char>(s[pos - 1]));
        const size_t end    = pos + phrase.size();
        const bool right_ok = end >= s.size() || !std::isalnum(static_cast<unsigned char>(s[end]));
        if (left_ok && right_ok) {
            s.replace(pos, phrase.size(), to);
            pos += to.size();
        } else {
            pos += phrase.size();
        }
    }
}

// Natural language -> operator symbols.  Longest phrases first.
std::string normalise(std::string s) {
    s = to_lower(std::move(s));
    replace_all(s, "\n", " ");
    replace_all(s, "\t", " ");
    replace_all(s, "×", "*");
    replace_all(s, "÷", "/");
    replace_all(s, "−", "-");

    static const std::vector<std::pair<const char*, const char*>> phrases = {
        { "multiplied by",       " * " },
        { "multiply it by",      " * " },
        { "multiply that by",    " * " },
        { "multiply by",         " * " },
        { "multiplied with",     " * " },
        { "times",               " * " },
        { "divided by",          " / " },
        { "divide it by",        " / " },
        { "divide that by",      " / " },
        { "divide by",           " / " },
        { "to the power of",     " ^ " },
        { "raised to the power of", " ^ " },
        { "raised to",           " ^ " },
        { "power of",            " ^ " },
        { "square root of",      " sqrt " },
        { "sqrt of",             " sqrt " },
        { "root of",             " sqrt " },
        { "squared",             " ^ 2 " },
        { "cubed",               " ^ 3 " },
        { "half of",             " 0.5 * " },
        { "a third of",          " (1/3) * " },
        { "a quarter of",        " 0.25 * " },
        { "double",              " 2 * " },
        { "triple",              " 3 * " },
        { "per cent",            " % " },
        { "percent",             " % " },
        { "plus",                " + " },
        { "added to",            " + " },
        { "add",                 " + " },
        { "increased by",        " + " },
        { "minus",               " - " },
        { "subtract",            " - " },
        { "subtracted by",       " - " },
        { "take away",           " - " },
        { "decreased by",        " - " },
        { "reduced by",          " - " },
        { "less",                " - " },
        { "the sum",             " # " },
        { "the result",          " # " },
        { "the total",           " # " },
        { "the answer",          " # " },
        { "everything",          " # " },
        { "sum",                 " # " },
        { "result",              " # " },
        { "total",               " # " },
        { "it",                  " # " },
        { "that",                " # " },
        { "this",                " # " },
        // fillers
        { "what is",             " " },
        { "what's",              " " },
        { "whats",               " " },
        { "how much is",         " " },
        { "how many is",         " " },
        { "calculate",           " " },
        { "compute",             " " },
        { "evaluate",            " " },
        { "solve",               " " },
        { "please",              " " },
        { "tell me",             " " },
        { "give me",             " " },
        { "equals",              " " },
        { "equal to",            " " },
        { "equal",               " " },
        { "is",                  " " },
        { "the",                 " " },
        { "a",                   " " },
        { "an",                  " " },
        { "by",                  " " },
        { "with",                " " },
        { "all",                 " " },
        { "to",                  " " },
    };
    for (const auto& p : phrases) replace_phrase(s, p.first, p.second);
    replace_all(s, "=", " ");
    replace_all(s, "?", " ");
    return s;
}

bool is_value_end(const std::vector<Token>& t) {
    if (t.empty()) return false;
    const Tok k = t.back().type;
    return k == Tok::Number || k == Tok::RParen || k == Tok::Percent || k == Tok::Acc;
}

std::vector<Token> tokenize(const std::string& clause, std::vector<std::string>& ignored) {
    std::vector<Token> out;
    size_t i = 0;
    const size_t n = clause.size();
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(clause[i]);
        if (std::isspace(c)) { ++i; continue; }

        // number: 1,234.5  or .5
        if (std::isdigit(c) || (c == '.' && i + 1 < n && std::isdigit(static_cast<unsigned char>(clause[i + 1])))) {
            std::string num;
            while (i < n) {
                const unsigned char d = static_cast<unsigned char>(clause[i]);
                if (std::isdigit(d) || d == '.') { num.push_back(static_cast<char>(d)); ++i; }
                else if (d == ',' && i + 1 < n && std::isdigit(static_cast<unsigned char>(clause[i + 1])) &&
                         !num.empty()) { ++i; }   // thousands separator
                else break;
            }
            Token t; t.type = Tok::Number; t.text = num;
            try { t.value = std::stod(num); } catch (...) { t.value = 0.0; }
            out.push_back(t);
            continue;
        }

        if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
            Token t; t.type = Tok::Op; t.op = static_cast<char>(c); t.text = std::string(1, static_cast<char>(c));
            out.push_back(t); ++i; continue;
        }
        if (c == 'x' && is_value_end(out) && i + 1 < n && !std::isalpha(static_cast<unsigned char>(clause[i + 1]))) {
            Token t; t.type = Tok::Op; t.op = '*'; t.text = "*";
            out.push_back(t); ++i; continue;
        }
        if (c == '(' || c == '[') { Token t; t.type = Tok::LParen; t.text = "("; out.push_back(t); ++i; continue; }
        if (c == ')' || c == ']') { Token t; t.type = Tok::RParen; t.text = ")"; out.push_back(t); ++i; continue; }
        if (c == '%') { Token t; t.type = Tok::Percent; t.text = "%"; out.push_back(t); ++i; continue; }
        if (c == '#') { Token t; t.type = Tok::Acc; t.text = "#"; out.push_back(t); ++i; continue; }

        if (std::isalpha(c)) {
            std::string w;
            while (i < n && (std::isalpha(static_cast<unsigned char>(clause[i])) || clause[i] == '\'')) w.push_back(clause[i++]);
            if (w == "sqrt") { Token t; t.type = Tok::Sqrt; t.text = "sqrt"; out.push_back(t); continue; }
            if (w == "of") {
                // "20% of 50", "3/4 of 100"  -> multiplication
                if (is_value_end(out)) { Token t; t.type = Tok::Op; t.op = '*'; t.text = "*"; out.push_back(t); }
                continue;
            }
            if (w == "pi")  { Token t; t.type = Tok::Number; t.value = 3.14159265358979323846; t.text = "pi"; out.push_back(t); continue; }
            if (w == "e")   { Token t; t.type = Tok::Number; t.value = 2.71828182845904523536; t.text = "e";  out.push_back(t); continue; }
            ignored.push_back(w);
            continue;
        }
        ++i;   // any other symbol
    }
    return out;
}

std::string tokens_to_string(const std::vector<Token>& t) {
    std::string s;
    for (const auto& k : t) {
        if (!s.empty() && k.type != Tok::Percent) s += ' ';
        s += k.text;
    }
    return s;
}

// =======================================================================
// Recursive-descent evaluator
// =======================================================================
struct Value {
    double v = 0.0;
    bool   percent_literal = false;   // "10%" written directly (for relative +/-)
};

class Parser {
public:
    Parser(const std::vector<Token>& toks, double acc, std::vector<std::string>& trace)
        : t_(toks), acc_(acc), trace_(trace) {}

    double parse_all() {
        Value v = expr();
        if (peek().type != Tok::End) throw std::runtime_error("unexpected '" + peek().text + "'");
        return v.v;
    }

private:
    const std::vector<Token>& t_;
    size_t pos_ = 0;
    double acc_;
    std::vector<std::string>& trace_;

    const Token& peek() const {
        static const Token end_tok{};
        return pos_ < t_.size() ? t_[pos_] : end_tok;
    }
    Token next() { return pos_ < t_.size() ? t_[pos_++] : Token{}; }
    bool accept_op(char op) {
        if (peek().type == Tok::Op && peek().op == op) { ++pos_; return true; }
        return false;
    }

    Value expr() {
        Value left = term();
        while (peek().type == Tok::Op && (peek().op == '+' || peek().op == '-')) {
            const char op = next().op;
            Value right = term();
            if (right.percent_literal) {
                // "100 - 10%" -> 100 * (1 - 0.10)  (relative change)
                const double factor = op == '+' ? 1.0 + right.v : 1.0 - right.v;
                trace_.push_back(fmt(left.v) + " " + op + " " + fmt(right.v * 100.0) + "% (relative) = " + fmt(left.v * factor));
                left.v *= factor;
            } else {
                const double r = op == '+' ? left.v + right.v : left.v - right.v;
                trace_.push_back(fmt(left.v) + " " + op + " " + fmt(right.v) + " = " + fmt(r));
                left.v = r;
            }
            left.percent_literal = false;
        }
        return left;
    }

    Value term() {
        Value left = unary();
        while (peek().type == Tok::Op && (peek().op == '*' || peek().op == '/')) {
            const char op = next().op;
            Value right = unary();
            if (op == '/' && right.v == 0.0) throw std::runtime_error("division by zero");
            const double r = op == '*' ? left.v * right.v : left.v / right.v;
            trace_.push_back(fmt(left.v) + " " + op + " " + fmt(right.v) + " = " + fmt(r));
            left.v = r;
            left.percent_literal = false;
        }
        return left;
    }

    Value unary() {
        if (accept_op('-')) { Value v = unary(); v.v = -v.v; v.percent_literal = false; return v; }
        if (accept_op('+')) { return unary(); }
        return power();
    }

    Value power() {
        Value base = postfix();
        if (accept_op('^')) {
            Value ex = unary();
            const double r = std::pow(base.v, ex.v);
            trace_.push_back(fmt(base.v) + " ^ " + fmt(ex.v) + " = " + fmt(r));
            base.v = r;
            base.percent_literal = false;
        }
        return base;
    }

    Value postfix() {
        Value v = primary();
        bool was_number = (pos_ > 0 && t_[pos_ - 1].type == Tok::Number);
        while (peek().type == Tok::Percent) {
            ++pos_;
            trace_.push_back(fmt(v.v) + "% = " + fmt(v.v / 100.0));
            v.v /= 100.0;
            v.percent_literal = was_number;
            was_number = false;
        }
        return v;
    }

    Value primary() {
        const Token tk = next();
        switch (tk.type) {
        case Tok::Number: return Value{ tk.value, false };
        case Tok::Acc:    return Value{ acc_, false };
        case Tok::LParen: {
            Value v = expr();
            if (next().type != Tok::RParen) throw std::runtime_error("missing ')'");
            v.percent_literal = false;
            return v;
        }
        case Tok::Sqrt: {
            Value v = unary();
            if (v.v < 0.0) throw std::runtime_error("square root of a negative number");
            const double r = std::sqrt(v.v);
            trace_.push_back("sqrt(" + fmt(v.v) + ") = " + fmt(r));
            return Value{ r, false };
        }
        case Tok::End:    throw std::runtime_error("expression ended too early");
        default:          throw std::runtime_error("unexpected '" + tk.text + "'");
        }
    }

public:
    static std::string fmt(double v) {
        if (std::isnan(v)) return "nan";
        if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
        if (std::fabs(v - std::round(v)) < 1e-9 && std::fabs(v) < 1e15) {
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(0) << std::round(v);
            return ss.str();
        }
        std::ostringstream ss;
        ss << std::setprecision(10) << v;
        return ss.str();
    }
};

// split on clause separators
std::vector<std::string> split_clauses(const std::string& s) {
    std::vector<std::string> clauses;
    std::string cur = s;
    // unify separators
    static const char* seps[] = { " and then ", " and ", " then ", " or ", " also ", " after that ", " afterwards " };
    for (const char* sep : seps) replace_all(cur, sep, " ; ");
    // a comma is a clause separator unless it sits between digits (1,000)
    for (size_t i = 0; i < cur.size(); ++i) {
        if (cur[i] != ',') continue;
        const bool between_digits = i > 0 && i + 1 < cur.size() &&
            std::isdigit(static_cast<unsigned char>(cur[i - 1])) &&
            std::isdigit(static_cast<unsigned char>(cur[i + 1]));
        if (!between_digits) cur[i] = ';';
    }
    std::stringstream ss(cur);
    std::string part;
    while (std::getline(ss, part, ';')) {
        // trim
        const auto b = part.find_first_not_of(" \t");
        const auto e = part.find_last_not_of(" \t");
        if (b == std::string::npos) continue;
        clauses.push_back(part.substr(b, e - b + 1));
    }
    return clauses;
}

bool has_type(const std::vector<Token>& t, Tok k) {
    for (const auto& x : t) if (x.type == k) return true;
    return false;
}

size_t count_numbers(const std::vector<Token>& t) {
    size_t n = 0;
    for (const auto& x : t) if (x.type == Tok::Number) ++n;
    return n;
}

} // namespace

// =======================================================================
// Public API
// =======================================================================
double evaluate_arithmetic(const std::string& expression, double acc) {
    std::vector<std::string> ignored, trace;
    auto toks = tokenize(normalise(expression), ignored);
    Parser p(toks, acc, trace);
    return p.parse_all();
}

SkillResult solve_math_text(const std::string& text) {
    SkillResult r;
    const std::string norm = normalise(text);
    const auto clauses = split_clauses(norm);
    const auto raw_clauses = split_clauses(to_lower(text));

    r.steps.push_back("input: \"" + text + "\"");

    double acc = 0.0;
    bool have_acc = false;
    int solved = 0, failed = 0;
    size_t total_numbers = 0, total_ops = 0;
    std::string last_error;
    std::vector<std::string> all_ignored;

    for (size_t ci = 0; ci < clauses.size(); ++ci) {
        std::vector<std::string> ignored;
        std::vector<Token> toks = tokenize(clauses[ci], ignored);
        all_ignored.insert(all_ignored.end(), ignored.begin(), ignored.end());
        total_numbers += count_numbers(toks);
        for (const auto& t : toks)
            if (t.type == Tok::Op || t.type == Tok::Percent || t.type == Tok::Sqrt) ++total_ops;

        if (toks.empty() || (!has_type(toks, Tok::Number) && !has_type(toks, Tok::Acc))) {
            r.steps.push_back("clause " + std::to_string(ci + 1) + ": \"" +
                              (ci < raw_clauses.size() ? raw_clauses[ci] : clauses[ci]) + "\" -> no arithmetic, skipped");
            continue;
        }

        // A follow-up clause that starts with an operator ("add 3/4",
        // "divided by 2") or mentions the running result ("sum ...")
        // operates on the accumulator.
        const bool starts_with_op = toks.front().type == Tok::Op;
        const bool uses_acc       = has_type(toks, Tok::Acc);
        if (have_acc && starts_with_op && !uses_acc) {
            Token a; a.type = Tok::Acc; a.text = "#";
            toks.insert(toks.begin(), a);
        }
        if (!have_acc && uses_acc) {
            // "sum divided by 2" as first clause -> nothing to refer to; use 0
            r.steps.push_back("note: '#' (the result) referenced before anything was computed, using 0");
        }

        std::vector<std::string> trace;
        try {
            Parser p(toks, acc, trace);
            const double v = p.parse_all();
            std::ostringstream step;
            step << "clause " << (ci + 1) << ": \""
                 << (ci < raw_clauses.size() ? raw_clauses[ci] : clauses[ci]) << "\""
                 << " -> " << tokens_to_string(toks);
            if (uses_acc || (have_acc && starts_with_op)) step << "   (# = " << Parser::fmt(acc) << ")";
            step << " = " << Parser::fmt(v);
            r.steps.push_back(step.str());
            for (const auto& t : trace) r.steps.push_back("    " + t);
            acc = v;
            have_acc = true;
            ++solved;
        } catch (const std::exception& e) {
            r.steps.push_back("clause " + std::to_string(ci + 1) + ": \"" + clauses[ci] + "\" -> cannot evaluate (" + e.what() + ")");
            last_error = e.what();
            ++failed;
        }
    }

    // A lone number without any operator is not a calculation
    // ("temperature on node 3") - leave it to the other experts.
    const bool is_arithmetic = total_ops > 0 || total_numbers >= 2;

    if (!have_acc) {
        const bool math_error = last_error.rfind("division", 0) == 0 || last_error.rfind("square root", 0) == 0;
        if (total_ops > 0 && failed > 0 && math_error) {
            // e.g. "5 / 0": recognised, but has no value
            r.handled    = true;
            r.answer     = "undefined (" + last_error + ")";
            r.confidence = 0.6f;
            r.steps.push_back("answer: " + r.answer);
        } else {
            r.handled = false;
            r.confidence = 0.f;
            r.thinking = "no arithmetic found";
            return r;
        }
    } else if (!is_arithmetic) {
        r.handled = false;
        r.confidence = 0.f;
        r.thinking = "a number, but nothing to compute";
        return r;
    } else {
        r.handled = true;
        r.answer  = Parser::fmt(acc);
        r.confidence = failed == 0 ? 0.95f : std::max(0.4f, 0.95f - 0.2f * static_cast<float>(failed));
    }
    if (!all_ignored.empty()) {
        std::string ig;
        for (size_t i = 0; i < all_ignored.size() && i < 8; ++i) { if (i) ig += ", "; ig += all_ignored[i]; }
        r.steps.push_back("ignored words: " + ig);
        r.confidence -= std::min(0.25f, 0.03f * static_cast<float>(all_ignored.size()));
    }
    if (have_acc) r.steps.push_back("answer: " + r.answer);

    std::ostringstream th;
    for (size_t i = 0; i < r.steps.size(); ++i) {
        if (i) th << '\n';
        th << r.steps[i];
    }
    r.thinking = th.str();
    return r;
}

Skill make_math_skill() {
    Skill s;
    s.name = "math";
    s.run = [](const SkillContext& c) { return solve_math_text(c.text); };
    return s;
}

// =======================================================================
// Memory skill
// =======================================================================
Skill make_memory_skill(VectorMemory* memory, ExperienceStore* store,
                        const Tokenizer* tokenizer, const EmbeddingLayer* embedder,
                        MemorySkillOptions opt)
{
    Skill s;
    s.name = "memory";
    s.run = [memory, store, tokenizer, embedder, opt](const SkillContext& c) {
        SkillResult r;
        if (!memory || !tokenizer || !embedder) { r.thinking = "memory skill not wired"; return r; }
        const auto q = embedder->embed(tokenizer->tokenize(c.text));

        // 1. learned experiences (question -> answer) first
        auto exp_hits = memory->search_tagged(q, "experience:", opt.top_k);
        // 2. general memory for context
        auto hits = memory->search(q, opt.top_k);

        std::ostringstream th;
        th << "memory search: " << memory->size() << " items, "
           << (memory->gpu_index_active() ? "GPU index" : "CPU index")
           << ", " << std::fixed << std::setprecision(0) << memory->last_search_us() << " us";
        r.steps.push_back(th.str());

        for (const auto& h : hits) {
            if (h.similarity < 0.15f) continue;
            std::ostringstream line;
            line << std::fixed << std::setprecision(2) << "[" << h.similarity << "] " << h.text;
            r.context.push_back(line.str());
        }
        if (r.context.empty()) r.steps.push_back("no related memories");
        else {
            r.steps.push_back("closest memories:");
            for (const auto& l : r.context) r.steps.push_back("  " + l);
        }

        // Recall a learned answer?
        for (const auto& h : exp_hits) {
            if (h.similarity < opt.recall_threshold) break;
            int64_t id = 0;
            try { id = std::stoll(h.tag.substr(std::string("experience:").size())); } catch (...) { id = 0; }
            Experience e;
            if (!id || !store || !store->get(id, e)) continue;
            if (e.reward < 0.f) {
                r.steps.push_back("similar earlier question (" + Parser::fmt(h.similarity) + ") but its answer was marked bad - not reusing");
                continue;
            }
            r.handled       = true;
            r.answer        = e.answer;
            r.experience_id = e.id;
            r.recalled_immediate = e.immediate.empty() ? e.answer : e.immediate;
            r.confidence    = std::clamp(h.similarity * (0.85f + 0.05f * std::min(3.f, e.reward)), 0.f, 0.99f);
            std::ostringstream step;
            step << std::fixed << std::setprecision(2)
                 << "RECALL experience #" << e.id << " (similarity " << h.similarity
                 << ", asked " << e.uses << "x, reward " << e.reward << ", from " << e.expert << "): \""
                 << e.question << "\" -> \"" << e.answer << "\"";
            r.steps.push_back(step.str());
            store->touch(e.id);
            break;
        }

        std::ostringstream all;
        for (size_t i = 0; i < r.steps.size(); ++i) { if (i) all << '\n'; all << r.steps[i]; }
        r.thinking = all.str();
        return r;
    };
    return s;
}

// =======================================================================
// Language skill (slow)
// =======================================================================
Skill make_language_skill(LMAdapter* lm) {
    Skill s;
    s.name = "language";
    s.slow = true;
    s.run = [lm](const SkillContext& c) {
        SkillResult r;
        if (!lm || !lm->is_ready()) { r.thinking = "language model not ready"; return r; }
        const std::string prompt = LMAdapter::build_prompt(c.text, c.findings, c.memory, c.goals,
                                                           lm->config().max_prompt_chars);
        LMResponse resp = lm->generate(prompt);
        std::ostringstream th;
        th << "LM " << lm->model_info() << " | prompt " << prompt.size() << " chars | "
           << std::fixed << std::setprecision(0) << resp.latency_ms << " ms";
        r.steps.push_back(th.str());
        if (!c.findings.empty()) {
            r.steps.push_back("findings given to the LM:");
            for (const auto& f : c.findings) r.steps.push_back("  - " + f);
        }
        if (!c.memory.empty()) {
            r.steps.push_back("memory given to the LM:");
            for (const auto& m : c.memory) r.steps.push_back("  - " + m);
        }
        if (resp.ok) {
            r.handled    = true;
            r.answer     = resp.text;
            r.confidence = resp.confidence;
        } else {
            r.handled = false;
            r.steps.push_back("LM error: " + resp.error);
        }
        std::ostringstream all;
        for (size_t i = 0; i < r.steps.size(); ++i) { if (i) all << '\n'; all << r.steps[i]; }
        r.thinking = all.str();
        return r;
    };
    return s;
}

} // namespace openrai
