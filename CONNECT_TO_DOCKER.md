# Connect OpenRAI to Your Docker Qwen Model - 3 Steps!

## Step 1: Find Docker API Port

```powershell
# Check what's running
docker ps

# Look for port mapping like: 0.0.0.0:8000->8000/tcp
```

Most common ports:
- **8000** - vLLM (most likely for Qwen)
- **11434** - Ollama
- **5000** - text-generation-webui

## Step 2: Test API

```powershell
# Test if API responds
curl http://localhost:8000/v1/models

# Or test generation
curl -X POST http://localhost:8000/v1/completions `
  -H "Content-Type: application/json" `
  -d '{\"prompt\":\"Hello!\",\"max_tokens\":50}'
```

If you get JSON response = API is working! ?

## Step 3: Update OpenRAI Config

Open `src/core.cpp` and change:

```cpp
// Find this section (around line 165):
LMConfig lm_config;
lm_config.model_type = "mock";  // <-- CHANGE THIS LINE

// Change to:
lm_config.model_type = "docker";  // or "http"
lm_config.model_path = "http://localhost:8000";  // <-- YOUR PORT HERE
```

## Step 4: Rebuild and Run

```powershell
cd C:\dev\openRAI
cmake --build build --config Debug
.\build\Debug\openrai.exe
```

You should see:
```
[SimpleHTTPBackend] Connecting to Docker LM at http://localhost:8000
[SimpleHTTPBackend] Using system curl (Windows 10+ built-in)
[SimpleHTTPBackend] Ready to connect!
[LMAdapter] Initialized: Simple HTTP Backend (curl) @ http://localhost:8000
[OpenRAI] Engine initialized ... language=docker
```

## Test It!

The main loop will automatically call the LM when generating outputs.

To manually test, you can inject a query:

```cpp
// In your code or modify main.cpp:
rai.inject_input("query: summarize system health");
```

---

## Troubleshooting

### "Failed to execute curl"
- curl should be built into Windows 10+
- Check: `curl --version`
- If missing, install: `winget install cURL.cURL`

### "Connection refused"
- Docker container not running: `docker ps`
- Wrong port: Check `docker ps` output
- API not ready: `docker logs <container_id>`

### "Unexpected response format"
- Different API format (not OpenAI-compatible)
- Check logs - OpenRAI will print full response
- May need to adjust JSON parsing in `SimpleHTTPBackend::generate()`

---

## What's Happening?

1. OpenRAI gathers cognitive context (memory + graph)
2. Builds enhanced prompt with context
3. Calls your Docker LM via HTTP POST
4. LM generates natural language response
5. OpenRAI stores response in memory + graph
6. Result: **Cognition + Language hybrid**!

---

**Ready?** Your Qwen model download should be done soon. Then just follow these 3 steps!
