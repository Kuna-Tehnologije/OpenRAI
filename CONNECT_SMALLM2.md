# Quick Connect: OpenRAI ? Docker Smallm2

## 1. Provjeri Docker API

```powershell
# Check if container is running
docker ps | Select-String "Smallm2"

# Get container ID
$containerId = docker ps --filter "ancestor=354bf30d0aa" --format "{{.ID}}"

# Check logs to find API port
docker logs $containerId | Select-String -Pattern "port|listening|http"

# Most common ports for LM containers:
# - 8000 (vLLM / FastAPI)
# - 11434 (Ollama)
# - 5000 (text-generation-webui)
# - 8080 (generic HTTP)
```

## 2. Test API

```powershell
# Test if API responds (try common ports)
curl http://localhost:8000/health
curl http://localhost:11434/api/tags
curl http://localhost:5000/api/v1/model

# Or use Invoke-WebRequest
Invoke-WebRequest -Uri "http://localhost:8000/v1/models" | Select-Object Content
```

## 3. Update OpenRAI Config

Open `src/core.cpp` and find this section (around line 165):

```cpp
// Find this:
LMConfig lm_config;
lm_config.model_type = "mock";  // <-- CHANGE THIS

// Change to:
lm_config.model_type = "docker";  // or "http"
lm_config.model_path = "http://localhost:8000";  // <-- YOUR PORT HERE
```

## 4. Rebuild OpenRAI

```powershell
cd C:\dev\openRAI
cmake --build build --config Debug

# Or just rebuild chat executable
cmake --build build --config Debug --target openrai_chat
cmake --build build --config Debug --target openrai_chat_gui
```

## 5. Run and Test!

### Console version:
```powershell
.\build\Debug\openrai_chat.exe
> Hello!
```

### GUI version:
```powershell
.\build\Debug\openrai_chat_gui.exe
# Windows GUI opens with chat interface!
```

---

## Expected Output

When connected successfully:

```
[OpenRAI] Engine initialized ... language=docker
[SimpleHTTPBackend] Connecting to Docker LM at http://localhost:8000
[SimpleHTTPBackend] Using system curl (Windows 10+ built-in)
[SimpleHTTPBackend] Ready to connect!
[LMAdapter] Initialized: Simple HTTP Backend (curl) @ http://localhost:8000

> Hello!
[YOU]: Hello!

[OpenRAI]: Thinking...
[SimpleHTTPBackend] Calling API...
[SimpleHTTPBackend] ? Generated 145 chars

[OpenRAI]: Hello! I'm OpenRAI, enhanced with Smallm2. How can I help you today?

(Response time: 0.287s)
```

---

## Troubleshooting

### "Connection refused"
```powershell
# Check if Docker container is actually running
docker ps

# Check container logs
docker logs $containerId

# Try different port
curl http://localhost:8000
curl http://localhost:11434
curl http://localhost:5000
```

### "curl failed"
```powershell
# Check curl is available
curl --version

# If missing, install:
winget install cURL.cURL
```

### "Unknown API format"
Different containers use different API formats. Check logs in OpenRAI:
- It will print the full API response
- Adjust `SimpleHTTPBackend` if needed

---

## What Port Does Smallm2 Use?

Smallm2 is likely using one of these setups:

### Option A: Ollama (most likely for GGUF)
```
Port: 11434
Endpoint: http://localhost:11434/api/generate
```

### Option B: vLLM
```
Port: 8000
Endpoint: http://localhost:8000/v1/completions
```

### Option C: llama.cpp server
```
Port: 8080
Endpoint: http://localhost:8080/completion
```

Run `docker logs` to confirm!

---

## Quick Test Script

Save as `test_docker_lm.ps1`:

```powershell
$ports = @(8000, 11434, 5000, 8080)

foreach ($port in $ports) {
    Write-Host "Testing port $port..." -ForegroundColor Yellow
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$port/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "? Found API on port $port" -ForegroundColor Green
            Write-Host "Response: $($response.Content)" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "? Port $port not responding" -ForegroundColor DarkGray
    }
}
```

Run: `.\test_docker_lm.ps1`

---

## Next: Connect!

Once you find the port:

1. Update `src/core.cpp`:
   ```cpp
   lm_config.model_type = "docker";
   lm_config.model_path = "http://localhost:YOUR_PORT";
   ```

2. Rebuild:
   ```powershell
   cmake --build build --config Debug --target openrai_chat_gui
   ```

3. Run GUI:
   ```powershell
   .\build\Debug\openrai_chat_gui.exe
   ```

4. **Chat with real AI!** ??

---

**Status:** 
- ? Docker model ready (Smallm2, 361M params)
- ? Need to find API port
- ? Update OpenRAI config
- ? Rebuild and test

**Let me know the port and I'll give you exact config!**
