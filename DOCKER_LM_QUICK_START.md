# Quick Docker LM Connection for OpenRAI

## Your Current Setup

You're pulling Qwen2.5-7B-Instruct via Docker. Let's connect OpenRAI directly to it!

## Step 1: Check what's running

```powershell
docker ps
docker logs <container_id>
```

Look for the API endpoint (usually port 8000, 8080, or 5000).

## Step 2: Simple HTTP Backend (no dependencies)

We'll use Windows built-in `Invoke-WebRequest` - no libcurl needed!

Update `src/lm_adapter.cpp` with this simple HTTP backend:

```cpp
// Simple HTTP Backend using system curl/PowerShell
class SimpleHTTPBackend : public ILanguageModel {
public:
    bool load(const LMConfig& config) override {
        api_url_ = config.model_path; // e.g. "http://localhost:8000"
        std::cout << "[SimpleHTTPBackend] API URL: " << api_url_ << "\n";
        loaded_ = true;
        return true;
    }

    LMResponse generate(const std::string& prompt) override {
        LMResponse response;

        // Escape quotes in prompt
        std::string escaped_prompt = prompt;
        size_t pos = 0;
        while ((pos = escaped_prompt.find("\"", pos)) != std::string::npos) {
            escaped_prompt.replace(pos, 1, "\\\"");
            pos += 2;
        }

        // Build JSON request
        std::string json = "{\"prompt\":\"" + escaped_prompt + 
                          "\",\"max_tokens\":512,\"temperature\":0.7}";

        // Use system curl (available on Windows 10+)
        std::string cmd = "curl -s -X POST " + api_url_ + "/v1/completions "
                         + "-H \"Content-Type: application/json\" "
                         + "-d \"" + json + "\"";

        // Execute and capture output
        FILE* pipe = _popen(cmd.c_str(), "r");
        if (!pipe) {
            response.text = "[Error] Failed to execute curl";
            return response;
        }

        char buffer[128];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
            result += buffer;
        }
        _pclose(pipe);

        // Simple JSON parsing (extract "text" field)
        size_t text_start = result.find("\"text\":\"");
        if (text_start != std::string::npos) {
            text_start += 8; // Skip "text":"
            size_t text_end = result.find("\"", text_start);
            if (text_end != std::string::npos) {
                response.text = result.substr(text_start, text_end - text_start);
                response.confidence = 0.9f;
            }
        } else {
            response.text = result; // Return full response if parsing fails
        }

        return response;
    }

    std::vector<float> embed(const std::string& text) override {
        return std::vector<float>(768, 0.5f);
    }

    bool is_loaded() const override { return loaded_; }

    std::string model_info() const override {
        return "Simple HTTP Backend @ " + api_url_;
    }

private:
    std::string api_url_;
    bool loaded_ = false;
};
```

## Step 3: Update LMFactory

```cpp
std::unique_ptr<ILanguageModel> LMFactory::create(const std::string& model_type) {
    if (model_type == "http" || model_type == "docker") {
        return std::make_unique<SimpleHTTPBackend>();
    }
    // ... rest
}
```

## Step 4: Update src/core.cpp config

```cpp
LMConfig lm_config;
lm_config.model_type = "docker";
lm_config.model_path = "http://localhost:8000";  // Check your Docker port!
```

## Step 5: Find your Docker API endpoint

Most common setups:

### vLLM (most likely)
```bash
docker run -d --gpus all -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct

# API: http://localhost:8000/v1/completions
```

### text-generation-webui
```bash
docker run -d -p 5000:5000 \
    atinoda/text-generation-webui:latest

# API: http://localhost:5000/api/v1/generate
```

### Ollama
```bash
docker run -d -p 11434:11434 ollama/ollama
ollama pull qwen2.5:7b

# API: http://localhost:11434/api/generate
```

## Step 6: Test the connection

```powershell
# Test if API is responding
curl http://localhost:8000/v1/models

# Or
Invoke-WebRequest -Uri "http://localhost:8000/v1/models" | Select-Object Content
```

## Step 7: Rebuild and run

```powershell
cd C:\dev\openRAI
cmake --build build --config Debug
.\build\Debug\openrai.exe
```

---

## Why this approach?

? **No external dependencies** (uses system curl)
? **Works with any Docker LM** (vLLM, Ollama, text-gen-webui)
? **Simple to debug** (just test with `curl` first)
? **Fast integration** (5 minutes, not hours)

---

## What's your Docker running?

Check with:
```powershell
docker ps --format "table {{.Image}}\t{{.Ports}}\t{{.Status}}"
```

Tell me the output and I'll give you the exact API URL to use!
