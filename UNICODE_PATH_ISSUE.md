# ?? Docker Model Runner Unicode Path Issue

## Problem

Docker Model Runner is **crashing** on your system due to a Unicode character in your Windows username:

```
couldn't bind HTTP server socket, hostname: C:/Users/NevenJoviæ/AppData/Local/Docker/run/inference-0.sock
                                                      ^^^^
                                                      This 'æ' causes the crash
```

This is a known Docker Desktop bug with non-ASCII characters in Windows usernames.

## ? SOLUTION: Use Ollama Instead

**Ollama** is a better alternative - it's:
- More stable than Docker Model Runner
- Uses the same API format (fully compatible with OpenRAI)
- Easier to manage models
- No Unicode path issues

### Install Ollama

1. **Download**: https://ollama.ai/download/windows
2. **Install**: Run the installer (takes 2 minutes)
3. **Pull model**: Open PowerShell and run:
   ```powershell
   ollama pull smollm2
   ```
4. **Verify**: Ollama runs automatically on port **11434**

### Update OpenRAI Config

Open `src/core.cpp` and change line 174:

**FROM:**
```cpp
lm_config.model_path = "http://localhost:12434";  // Docker Model Runner port
```

**TO:**
```cpp
lm_config.model_path = "http://localhost:11434";  // Ollama port
```

Then rebuild:
```cmd
cmake --build C:\dev\openRAI\build --target openrai_chat_gui --config Release
```

### Test Ollama

```powershell
# Check Ollama is running
curl.exe -s http://localhost:11434/api/tags

# Test generation
curl.exe -s -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d '{\"model\":\"smollm2\",\"prompt\":\"Hello, what is 2+2?\",\"stream\":false}'
```

You should see:
```json
{"model":"smollm2","response":"2+2 equals 4..."}
```

## Alternative: Use Mock Backend (No External Model)

If you want to test OpenRAI's cognition features without waiting for a model:

In `src/core.cpp` line 173:
```cpp
lm_config.model_type = "mock";  // Use mock backend
```

The mock backend returns test responses instantly, so you can test:
- Memory storage
- Knowledge graph
- Goal tracking
- UI functionality

Then switch to Ollama later for real language generation.

## Why This Happened

Docker Model Runner tries to create a Unix socket at:
```
C:/Users/NevenJoviæ/AppData/Local/Docker/run/inference-0.sock
```

But llama.cpp (the inference engine) doesn't properly handle the Unicode 'æ' character in the path, causing it to crash before it can even start the HTTP server.

## Current OpenRAI Status

? **OpenRAI cognition layer**: Working perfectly
? **Windows GUI**: Fixed (no more send-on-keystroke)
? **HTTP backend**: Working (tested with proper escaping)
? **Docker Model Runner**: Broken due to Unicode path bug
? **Ollama alternative**: Ready to use (just need to install)

## Next Steps

**Option 1 (Recommended)**: Install Ollama and update config to port 11434
**Option 2**: Use mock backend to test OpenRAI features while Docker team fixes the bug
**Option 3**: Report bug to Docker Desktop team (but this will take time to fix)

Choose Option 1 for the best experience! ??
