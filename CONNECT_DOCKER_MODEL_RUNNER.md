# BRZO SPAJANJE: OpenRAI ? Docker Model Runner (Smallm2)

## ?? Tvoj Setup

? Docker Model Runner aktivan  
? Port: **12434**  
? Model: Smallm2 (361M params)  
? GPU inference: Enabled  

---

## Korak 1: Test API-ja

```powershell
# Provjeri da li model runner radi
curl http://localhost:12434/api/tags

# Trebao bi dobiti JSON s listom modela, ukljuèujuæi Smallm2
```

---

## Korak 2: Update OpenRAI Config

Otvori `C:\dev\openRAI\src\core.cpp` u editoru i naði ovaj dio (oko linije 165):

```cpp
// NAÐI OVO:
LMConfig lm_config;
lm_config.model_type = "mock";  // <-- PROMIJENI
lm_config.model_path = "";

// PROMIJENI U:
LMConfig lm_config;
lm_config.model_type = "docker";  // ili "ollama"
lm_config.model_path = "http://localhost:12434";  // TVOJ PORT!
```

Toèna lokacija u fajlu:

```cpp
// Around line 160-170 in src/core.cpp
LOG("[InputPool] CSV=" << STEP_INPUT_POOL_PATH << " count=" << STEP_INPUT_POOL.size());

// Initialize LM Adapter (Cognition + Language Layer)
LMConfig lm_config;
lm_config.model_type = "docker";  // <-- CHANGE THIS LINE
lm_config.model_path = "http://localhost:12434";  // <-- AND THIS ONE
lm_config.context_length = 4096;
lm_config.max_tokens = 512;
lm_config.temperature = 0.7f;
lm_config.use_gpu = true;
```

---

## Korak 3: Rebuild

```powershell
cd C:\dev\openRAI

# Rebuild all chat executables
cmake --build build --config Debug --target openrai_chat
cmake --build build --config Debug --target openrai_chat_gui
```

---

## Korak 4: Pokreni GUI! ??

```powershell
.\build\Debug\openrai_chat_gui.exe
```

Trebao bi se otvoriti **Windows GUI** s chat interfaceom!

---

## Oèekivani Output

Kada se spoji uspješno:

```
[OpenRAI] Engine initialized nodes=20 threads=20 gpu_slots=128 semantic=on language=docker
[SimpleHTTPBackend] Connecting to Docker LM at http://localhost:12434
[SimpleHTTPBackend] Using system curl (Windows 10+ built-in)
[SimpleHTTPBackend] Ready to connect!
[LMAdapter] Initialized: Simple HTTP Backend (curl) @ http://localhost:12434
```

Zatim u GUI prozoru:

```
????????????????????????????????????????????????????
? OpenRAI Chat - Hybrid Cognition + Language AI   ?
????????????????????????????????????????????????????
? [OpenRAI]: Hello! I'm OpenRAI with Smallm2.     ?
?            How can I help you today?             ?
?                                                  ?
? [YOU]: What can you do?                          ?
?                                                  ?
? [OpenRAI]: I'm a hybrid AI that combines         ?
?            cognitive processing with natural     ?
?            language. I can remember our          ?
?            conversations, build knowledge        ?
?            graphs, and provide thoughtful        ?
?            responses using the Smallm2 model.    ?
?                                                  ?
????????????????????????????????????????????????????
? Type here...                          [Send]    ?
????????????????????????????????????????????????????
? Messages: 4 | Last response: 0.345s             ?
????????????????????????????????????????????????????
```

---

## Docker Model Runner API Format

Docker Model Runner koristi **Ollama-kompatibilni API**:

### Endpoint za generation:
```
POST http://localhost:12434/api/generate
Content-Type: application/json

{
  "model": "smallm2",
  "prompt": "Hello!",
  "stream": false
}
```

### Response format:
```json
{
  "model": "smallm2",
  "created_at": "2024-01-08T12:34:56Z",
  "response": "Hello! How can I help you today?",
  "done": true
}
```

---

## Ako SimpleHTTPBackend ne radi odmah

Možda trebaš prilagoditi API format. Otvori `src/lm_adapter.cpp` i naði `SimpleHTTPBackend::generate()` oko linije 180:

```cpp
// Current code (vLLM/OpenAI format):
cmd << "curl -s -X POST " << api_url_ << "/v1/completions "
    << "-H \"Content-Type: application/json\" "
    << "-d \"{\\\"prompt\\\":\\\"" << escaped_prompt << "\\\","
    << "\\\"max_tokens\\\":512,"
    << "\\\"temperature\\\":0.7}\" 2>&1";

// Za Ollama/Docker Model Runner format, promijeni u:
cmd << "curl -s -X POST " << api_url_ << "/api/generate "
    << "-H \"Content-Type: application/json\" "
    << "-d \"{\\\"model\\\":\\\"smallm2\\\","
    << "\\\"prompt\\\":\\\"" << escaped_prompt << "\\\","
    << "\\\"stream\\\":false}\" 2>&1";
```

I promijeni parsing:

```cpp
// Current: looks for "text" field
size_t text_start = result.find("\"text\":\"");

// Za Ollama: promijeni u "response" field
size_t text_start = result.find("\"response\":\"");
```

Ali prvo probaj bez promjena - možda veæ radi!

---

## Troubleshooting

### 1. "Connection refused"
```powershell
# Provjeri da li je Docker Model Runner ukljuèen
# Docker Desktop ? Settings ? AI ? Enable Docker Model Runner ?
```

### 2. "Model not found"
```powershell
# Provjeri dostupne modele
curl http://localhost:12434/api/tags
```

### 3. "Slow responses"
- Normalno za 361M model na CPU: 1-3s
- S GPU: 0.3-0.8s
- Provjeraš: "Enable GPU-backed inference" je ?

---

## Quick Test

Probaj direktno iz terminala:

```powershell
curl -X POST http://localhost:12434/api/generate `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"smallm2\",\"prompt\":\"Hello!\",\"stream\":false}'
```

Ako dobiješ JSON response ? **API radi!** ?

---

## Sada uradi ovo:

1. **Update `src/core.cpp`**:
   ```cpp
   lm_config.model_type = "docker";
   lm_config.model_path = "http://localhost:12434";
   ```

2. **Rebuild**:
   ```powershell
   cmake --build build --config Debug --target openrai_chat_gui
   ```

3. **Run**:
   ```powershell
   .\build\Debug\openrai_chat_gui.exe
   ```

4. **Chat with Smallm2!** ??

---

**Status**: READY TO CONNECT! ??  
Port: 12434  
Model: Smallm2 (361M)  
GPU: Enabled  

**Next**: Edit `src/core.cpp`, rebuild, run GUI!
