#include <openrai/lm_adapter.h>
#include <openrai/log.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#  define OPENRAI_POPEN  _popen
#  define OPENRAI_PCLOSE _pclose
#else
#  define OPENRAI_POPEN  popen
#  define OPENRAI_PCLOSE pclose
#endif

namespace openrai {

// =======================================================================
// helpers
// =======================================================================
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else { out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
}

bool json_extract_string(const std::string& json, const std::string& key, std::string& out) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = json.find(needle, pos)) != std::string::npos) {
        size_t p = pos + needle.size();
        while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == '\r')) ++p;
        if (p >= json.size() || json[p] != ':') { pos = p; continue; }
        ++p;
        while (p < json.size() && (json[p] == ' ' || json[p] == '\t' || json[p] == '\n' || json[p] == '\r')) ++p;
        if (p >= json.size() || json[p] != '"') { pos = p; continue; }
        ++p;
        out.clear();
        while (p < json.size()) {
            const char c = json[p];
            if (c == '"') return true;
            if (c == '\\' && p + 1 < json.size()) {
                const char e = json[p + 1];
                switch (e) {
                    case 'n': out += '\n'; p += 2; continue;
                    case 'r': out += '\r'; p += 2; continue;
                    case 't': out += '\t'; p += 2; continue;
                    case 'b': out += '\b'; p += 2; continue;
                    case 'f': out += '\f'; p += 2; continue;
                    case '"': out += '"';  p += 2; continue;
                    case '\\': out += '\\'; p += 2; continue;
                    case '/': out += '/';  p += 2; continue;
                    case 'u': {
                        if (p + 5 < json.size()) {
                            unsigned cp = static_cast<unsigned>(std::strtoul(json.substr(p + 2, 4).c_str(), nullptr, 16));
                            p += 6;
                            // surrogate pair
                            if (cp >= 0xD800 && cp <= 0xDBFF && p + 5 < json.size() && json[p] == '\\' && json[p + 1] == 'u') {
                                unsigned lo = static_cast<unsigned>(std::strtoul(json.substr(p + 2, 4).c_str(), nullptr, 16));
                                if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); p += 6; }
                            }
                            append_utf8(out, cp);
                            continue;
                        }
                        p += 2; continue;
                    }
                    default: out += e; p += 2; continue;
                }
            }
            out += c;
            ++p;
        }
        return false;   // unterminated
    }
    return false;
}

int run_command_capture(const std::string& cmd, std::string& output) {
    output.clear();
    FILE* pipe = OPENRAI_POPEN(cmd.c_str(), "r");
    if (!pipe) return -1;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) output.append(buf, n);
    const int rc = OPENRAI_PCLOSE(pipe);
#if defined(_WIN32)
    return rc;
#else
    if (rc == -1) return -1;
    return (rc >> 8) & 0xFF;   // WEXITSTATUS without <sys/wait.h>
#endif
}

static std::string trim(std::string s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// =======================================================================
// Mock backend
// =======================================================================
class MockBackend : public ILanguageModel {
public:
    bool load(const LMConfig&) override { loaded_ = true; return true; }
    LMResponse generate(const std::string&, const std::string& user) override {
        LMResponse r;
        r.ok = true;
        r.confidence = 0.5f;
        // Echo the most useful line of the prompt so tests can verify the
        // context made it through.
        std::string finding;
        std::istringstream in(user);
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("- ", 0) == 0) { finding = line.substr(2); break; }
        }
        r.text = "[mock-lm] " + (finding.empty() ? std::string("no expert findings") : finding);
        r.tokens_used = static_cast<int>(r.text.size() / 4);
        return r;
    }
    bool is_loaded() const override { return loaded_; }
    std::string model_info() const override { return "mock"; }
private:
    bool loaded_ = false;
};

// =======================================================================
// HTTP backend (system curl)
// =======================================================================
class HttpCurlBackend : public ILanguageModel {
public:
    bool load(const LMConfig& cfg) override {
        cfg_ = cfg;
        while (!cfg_.url.empty() && cfg_.url.back() == '/') cfg_.url.pop_back();
        if (cfg_.api == "ollama")      endpoint_ = "/api/generate";
        else if (cfg_.api == "openai") endpoint_ = "/v1/completions";
        else { cfg_.api = "openai-chat"; endpoint_ = "/v1/chat/completions"; }

        // Probe: GET /v1/models (or /api/tags for ollama) with a short timeout
        const std::string probe = cfg_.api == "ollama" ? "/api/tags" : "/v1/models";
        std::string out;
        std::ostringstream cmd;
        cmd << curl_bin() << " -s -S --max-time 2 \"" << cfg_.url << probe << "\"";
        const int rc = run_command_capture(cmd.str(), out);
        loaded_ = (rc == 0 && !out.empty());
        if (loaded_) {
            std::string id;
            if (cfg_.model.empty() && json_extract_string(out, "id", id)) {
                cfg_.model = id;    // first model the server lists
                LOG("[LM] server default model: " << cfg_.model);
            }
            if (cfg_.model.empty() && json_extract_string(out, "name", id)) cfg_.model = id;
            LOG("[LM] http backend ready url=" << cfg_.url << " api=" << cfg_.api
                << " model=" << (cfg_.model.empty() ? "(server default)" : cfg_.model));
        } else {
            LOG("[LM] http backend NOT reachable at " << cfg_.url << probe
                << " (rc=" << rc << ") - language expert disabled. Start your local LLM server or set OPENRAI_LLM=off");
        }
        return loaded_;
    }

    LMResponse generate(const std::string& system, const std::string& user) override {
        LMResponse r;
        if (!loaded_) { r.error = "backend not loaded"; return r; }
        const auto t0 = std::chrono::steady_clock::now();

        std::ostringstream body;
        if (cfg_.api == "openai-chat") {
            body << "{";
            if (!cfg_.model.empty()) body << "\"model\":\"" << json_escape(cfg_.model) << "\",";
            body << "\"messages\":[{\"role\":\"system\",\"content\":\"" << json_escape(system) << "\"},"
                 << "{\"role\":\"user\",\"content\":\"" << json_escape(user) << "\"}],"
                 << "\"max_tokens\":" << cfg_.max_tokens << ","
                 << "\"temperature\":" << cfg_.temperature << ","
                 << "\"stream\":false}";
        } else if (cfg_.api == "openai") {
            body << "{";
            if (!cfg_.model.empty()) body << "\"model\":\"" << json_escape(cfg_.model) << "\",";
            body << "\"prompt\":\"" << json_escape(system + "\n\n" + user + "\n\nAnswer:") << "\","
                 << "\"max_tokens\":" << cfg_.max_tokens << ","
                 << "\"temperature\":" << cfg_.temperature << ","
                 << "\"stream\":false}";
        } else {  // ollama
            body << "{";
            if (!cfg_.model.empty()) body << "\"model\":\"" << json_escape(cfg_.model) << "\",";
            body << "\"system\":\"" << json_escape(system) << "\","
                 << "\"prompt\":\"" << json_escape(user) << "\","
                 << "\"options\":{\"num_predict\":" << cfg_.max_tokens << ",\"temperature\":" << cfg_.temperature << "},"
                 << "\"stream\":false}";
        }

        // Request body through a temp file: robust against shell quoting on
        // every platform.
        const std::string tmp = temp_path();
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) { r.error = "cannot write " + tmp; return r; }
            f << body.str();
        }

        std::ostringstream cmd;
        cmd << curl_bin() << " -s -S --max-time " << cfg_.timeout_s
            << " -X POST \"" << cfg_.url << endpoint_ << "\""
            << " -H \"Content-Type: application/json\""
            << " --data-binary \"@" << tmp << "\" 2>&1";

        std::string out;
        const int rc = run_command_capture(cmd.str(), out);
        std::remove(tmp.c_str());
        r.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        if (rc != 0) {
            r.error = "curl exit " + std::to_string(rc) + ": " + trim(out).substr(0, 200);
            return r;
        }

        std::string text;
        bool got = false;
        if (cfg_.api == "openai-chat") got = json_extract_string(out, "content", text);
        else if (cfg_.api == "openai") got = json_extract_string(out, "text", text);
        else got = json_extract_string(out, "response", text);
        if (!got) {
            std::string err;
            if (json_extract_string(out, "message", err) || json_extract_string(out, "error", err))
                r.error = "server: " + err.substr(0, 200);
            else
                r.error = "unexpected response: " + trim(out).substr(0, 200);
            return r;
        }
        r.text = trim(text);
        r.ok = !r.text.empty();
        if (!r.ok) r.error = "empty completion";
        r.confidence = r.ok ? 0.75f : 0.f;
        r.tokens_used = static_cast<int>(r.text.size() / 4);
        return r;
    }

    bool is_loaded() const override { return loaded_; }
    std::string model_info() const override {
        return "http " + cfg_.api + " @ " + cfg_.url + (cfg_.model.empty() ? "" : " model=" + cfg_.model);
    }

private:
    LMConfig    cfg_;
    std::string endpoint_;
    bool        loaded_ = false;

    static const char* curl_bin() {
#if defined(_WIN32)
        return "curl.exe";
#else
        return "curl";
#endif
    }
    static std::string temp_path() {
        static std::atomic<unsigned> counter{0};
        std::ostringstream ss;
#if defined(_WIN32)
        const char* t = std::getenv("TEMP");
        ss << (t ? t : ".") << "\\";
#else
        ss << "/tmp/";
#endif
        ss << "openrai_lm_" << std::hash<std::thread::id>{}(std::this_thread::get_id()) % 100000
           << "_" << counter++ << ".json";
        return ss.str();
    }
};

// =======================================================================
// LMAdapter
// =======================================================================
LMAdapter::LMAdapter(LMConfig cfg) : cfg_(std::move(cfg)) {}
LMAdapter::~LMAdapter() = default;

bool LMAdapter::initialize() {
    ready_ = false;
    if (cfg_.backend == "off" || cfg_.backend.empty()) {
        LOG("[LM] language model disabled (OPENRAI_LLM=off)");
        return false;
    }
    if (cfg_.backend == "mock") backend_ = std::make_unique<MockBackend>();
    else                        backend_ = std::make_unique<HttpCurlBackend>();
    ready_ = backend_->load(cfg_);
    return ready_;
}

std::string LMAdapter::model_info() const {
    return backend_ ? backend_->model_info() : "none";
}

LMResponse LMAdapter::generate(const std::string& user_prompt) {
    if (!ready_ || !backend_) { LMResponse r; r.error = "language model not ready"; return r; }
    return backend_->generate(cfg_.system_prompt, user_prompt);
}

std::string LMAdapter::build_prompt(const std::string& question,
                                    const std::vector<std::string>& expert_findings,
                                    const std::vector<std::string>& memory_context,
                                    const std::vector<std::string>& goals,
                                    size_t max_chars)
{
    std::ostringstream p;
    if (!expert_findings.empty()) {
        p << "Expert node findings:\n";
        for (const auto& f : expert_findings) p << "- " << f << "\n";
        p << "\n";
    }
    if (!memory_context.empty()) {
        p << "Relevant memory:\n";
        for (const auto& m : memory_context) p << "- " << m << "\n";
        p << "\n";
    }
    if (!goals.empty()) {
        p << "Current goals: ";
        for (size_t i = 0; i < goals.size(); ++i) { if (i) p << ", "; p << goals[i]; }
        p << "\n\n";
    }
    p << "User: " << question << "\n";
    std::string s = p.str();
    if (s.size() > max_chars) s = s.substr(s.size() - max_chars);   // keep the tail (question)
    return s;
}

LMResponse LMAdapter::generate_with_context(const std::string& question,
                                            const std::vector<std::string>& expert_findings,
                                            const std::vector<std::string>& memory_context,
                                            const std::vector<std::string>& goals)
{
    return generate(build_prompt(question, expert_findings, memory_context, goals, cfg_.max_prompt_chars));
}

} // namespace openrai
