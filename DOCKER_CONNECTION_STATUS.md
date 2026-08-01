# OpenRAI + Docker Model Runner Connection

## ? CURRENT STATUS

OpenRAI is now configured to connect to Docker Model Runner:
- **Port**: `12434` (Docker Desktop AI settings)
- **Model**: `docker.io/ai/smollm2:latest`
- **API**: Ollama-compatible `/api/generate`
- **Endpoint**: `http://localhost:12434`

### Build Status
```
? openrai_chat_gui.exe compiled successfully
? SimpleHTTPBackend configured for Docker Model Runner
? LMConfig set to "docker" type with correct endpoint
```

## ?? DOCKER MODEL RUNNER TESTING

### 1. Verify Model is Available
```bash
curl.exe -s http://localhost:12434/api/tags
```

**Expected Output**:
```json
{
  "models": [{
    "name": "docker.io/ai/smollm2:latest",
    "size": 0,
    "digest": "sha256:354bf30d0aa...",
    "details": {
      "format": "gguf",
      "family": "llama",
      "parameter_size": "361.82 M",
      "quantization_level": "IQ2_XXS/Q4_K_M"
    }
  }]
}
```

### 2. Test Generation (if model is responsive)
```bash
curl.exe -s -X POST http://localhost:12434/api/generate ^
  -H "Content-Type: application/json" ^
  --data-binary "@test_request.json"
```

Where `test_request.json` contains:
```json
{"model":"docker.io/ai/smollm2:latest","prompt":"Hello, what is AI?","stream":false}
```

## ?? KNOWN ISSUE

Docker Model Runner encountered an internal error during testing:
```
unable to load runner: llama.cpp failed: couldn't bind HTTP server socket
hostname: C:/Users/NevenJoviæ/AppData/Local/Docker/run/inference-0.sock
```

This appears to be a **Unicode path issue** with the Docker socket binding on your system (non-ASCII character "æ" in username).

## ??? TROUBLESHOOTING

### Issue: Docker Model Runner Not Responding

**Symptoms**:
- `/api/tags` works (shows model list)
- `/api/generate` fails or times out
- Error: "llama.cpp terminated unexpectedly"

**Possible Causes**:
1. **Unicode Path Issue**: Windows username contains non-ASCII characters
2. **GPU Memory**: Insufficient VRAM for model loading
3. **Model State**: Model not fully initialized

**Solutions**:

#### Option 1: Restart Docker Model Runner
1. Open Docker Desktop
2. Go to **Settings** ? **AI** ? **Docker Model Runner**
3. **Disable** Docker Model Runner
4. Wait 10 seconds
5. **Enable** Docker Model Runner again
6. Test with `/api/tags` and `/api/generate`

#### Option 2: Use Alternative Local Model Server
If Docker Model Runner continues to have issues, consider:

**A) Ollama (Recommended)**
```bash
# Install Ollama from https://ollama.ai
ollama pull smollm2
ollama serve
```
Then change OpenRAI config to:
```cpp
lm_config.model_path = "http://localhost:11434";
```

**B) llama.cpp Server**
```bash
# Download llama.cpp and start server
llama-server.exe -m smollm2.gguf --host localhost --port 8080
```
Then change OpenRAI config to:
```cpp
lm_config.model_path = "http://localhost:8080";
```

#### Option 3: Check Docker Desktop Logs
1. Docker Desktop ? **Troubleshoot** ? **Get support**
2. Check logs for llama.cpp errors
3. Look for GPU/CUDA initialization failures

## ?? RUNNING OPENRAI WITH DOCKER MODEL RUNNER

### Start the Chat GUI
```cmd
cd C:\dev\openRAI\build\Release
openrai_chat_gui.exe
```

### Expected Behavior
1. Window opens with OpenRAI logo and chat interface
2. Console shows: `[SimpleHTTPBackend] Connecting to Docker LM at http://localhost:12434`
3. You can type messages and send them
4. OpenRAI's cognition layer processes context
5. Docker Model Runner generates natural language responses
6. Responses appear in the chat history

### What's Happening Under the Hood
```
User Input
    ?
OpenRAI Cognition Layer
    ??? Vector Memory (semantic search)
    ??? Knowledge Graph (entity/relation extraction)
    ??? Goal System (intent alignment)
    ??? Context Assembly
        ?
SimpleHTTPBackend (HTTP POST to Docker Model Runner)
    ?
Docker Model Runner (smollm2 inference on GPU)
    ?
Response parsed and integrated back into OpenRAI
    ?
Chat Display + Memory Storage
```

## ?? CODE CHANGES SUMMARY

### `src/core.cpp` - Line 172-179
```cpp
LMConfig lm_config;
lm_config.model_type = "docker";  // Docker Model Runner
lm_config.model_path = "http://localhost:12434";  // Docker Model Runner port
lm_config.context_length = 4096;
lm_config.max_tokens = 512;
lm_config.temperature = 0.7f;
lm_config.use_gpu = true;
```

### `src/lm_adapter.cpp` - Line 157-163
```cpp
std::ostringstream cmd;
cmd << "curl.exe -s -X POST " << api_url_ << "/api/generate "
    << "-H \"Content-Type: application/json\" "
    << "-d \"{\\\"model\\\":\\\"docker.io/ai/smollm2:latest\\\","
    << "\\\"prompt\\\":\\\"" << escaped_prompt << "\\\","
    << "\\\"stream\\\":false}\" 2>&1";
```

### `src/lm_adapter.cpp` - Line 190-210
```cpp
// Simple JSON parsing - extract "response" field (Docker Model Runner / Ollama format)
size_t text_start = result.find("\"response\":\"");
if (text_start != std::string::npos) {
    text_start += 12; // Skip "response":"
    size_t text_end = result.find("\"", text_start);
    if (text_end != std::string::npos) {
        response.text = result.substr(text_start, text_end - text_start);
        // ...
    }
}
```

## ?? NEXT STEPS

1. **Restart Docker Model Runner** in Docker Desktop settings
2. **Test endpoint**: `curl.exe -s http://localhost:12434/api/tags`
3. **Run GUI**: `build\Release\openrai_chat_gui.exe`
4. **Send test message**: Type "Hello" and click Send
5. **Check console output** for connection status and API responses

If Docker Model Runner continues to have issues due to the Unicode path problem, **Ollama** is the recommended alternative (same API format, more stable on Windows).

## ?? ARCHITECTURE NOTES

OpenRAI is a **hybrid cognition + language system**:
- **Cognition Layer**: OpenRAI (memory, reasoning, goals, reflection)
- **Language Layer**: External model (Docker Model Runner, Ollama, llama.cpp)

This separation allows OpenRAI to focus on **thinking** while delegating **natural language generation** to specialized models. The `LMAdapter` provides a clean abstraction for switching between different language model backends.
