# OpenRAI Chat Interface

## Quick Start

### Build Chat Interface

```powershell
cd C:\dev\openRAI
cmake --build build --config Debug --target openrai_chat
```

### Run Chat

```powershell
.\build\Debug\openrai_chat.exe
```

---

## Features

### 1. Interactive Chat
- Type messages and get AI responses
- Real-time response generation
- Conversation history

### 2. Cognitive Context
- AI uses memory from previous messages
- Knowledge graph tracks entities
- Goal-aligned responses

### 3. Commands

| Command | Description |
|---------|-------------|
| `/quit` or `/exit` | Close chat |
| `/clear` | Clear chat history |
| `/stats` | Show statistics |
| `/memory` | Show memory status |

---

## Screenshot (Console Mode)

```
??????????????????????????????????????????????????????????????????
?           OpenRAI Chat Interface (Console Mode)                ?
?                                                                ?
?  Hybrid Cognition + Language AI                                ?
?  Type your message and press Enter                             ?
?  Commands: /quit, /clear, /stats, /memory                      ?
??????????????????????????????????????????????????????????????????

[OpenRAI]: Hello! I'm OpenRAI, a hybrid AI with memory, knowledge 
graph, and language capabilities. How can I help you today?

> What can you do?

[YOU]: What can you do?

[OpenRAI]: Thinking...
[SimpleHTTPBackend] Calling API...
[SimpleHTTPBackend] ? Generated 245 chars

[OpenRAI]: I'm a hybrid AI system that combines cognitive 
processing with natural language. I can:

1. Remember our conversation (persistent memory)
2. Build knowledge graphs from our discussions
3. Align responses with goals
4. Reflect on my reasoning
5. Process information in parallel

What would you like to explore?

(Response time: 0.523s)

> /stats

??????????????????? STATISTICS ???????????????????
? Total messages:     4
? User messages:      2
? AI responses:       2
? Last response time: 0.523s
? Chat history size:  4 messages
??????????????????????????????????????????????????

> /quit

Goodbye!
```

---

## Architecture

```
User types message
    ?
ChatGUI::send_message()
    ?
OpenRAI::inject_input() ? Stores in current memory
    ?
OpenRAI::generate_response()
    ?? Gathers memory context
    ?? Queries knowledge graph
    ?? Builds enhanced prompt
    ?? Calls LM (Docker/HTTP)
    ?
LM generates response
    ?
OpenRAI post-processing
    ?? Stores in vector memory
    ?? Updates knowledge graph
    ?? Triggers reflection
    ?
ChatGUI displays response
    ?
User sees AI response + stats
```

---

## Configuration

Edit `src/core.cpp` to change LM backend:

```cpp
// Use Docker LM (Qwen, Llama, etc.)
LMConfig lm_config;
lm_config.model_type = "docker";
lm_config.model_path = "http://localhost:8000";

// Or use mock for testing
lm_config.model_type = "mock";
```

---

## Advanced: ImGui Integration (Future)

The current version uses console I/O. To add a proper GUI:

### Option 1: Dear ImGui (Recommended)

```bash
# Add ImGui
vcpkg install imgui[core,win32-binding,dx11-binding]
```

Update `CMakeLists.txt`:
```cmake
find_package(imgui CONFIG REQUIRED)
target_link_libraries(openrai_chat PRIVATE imgui::imgui)
```

Update `src/chat_gui.cpp`:
```cpp
// Use ImGui rendering instead of console I/O
ImGui::Begin("OpenRAI Chat");
ImGui::Text("Conversation:");
ImGui::BeginChild("ChatHistory");
for (auto& msg : chat_history_) {
    ImGui::TextColored(
        msg.is_user ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : ImVec4(0.8f, 0.8f, 0.2f, 1.0f),
        "%s: %s",
        msg.is_user ? "YOU" : "OpenRAI",
        msg.text.c_str()
    );
}
ImGui::EndChild();
ImGui::InputText("Message", input_buffer_, sizeof(input_buffer_));
ImGui::SameLine();
if (ImGui::Button("Send")) {
    send_message();
}
ImGui::End();
```

### Option 2: Native Windows UI

```cpp
// Create Win32 window with text boxes
HWND hChatBox = CreateWindowEx(...);
HWND hInputBox = CreateWindowEx(...);
// Handle WM_COMMAND for send button
```

---

## Multi-Turn Conversation Example

```
> Tell me about memory systems

[OpenRAI]: Memory systems in AI typically have three levels:
1. Short-term (working memory)
2. Long-term (persistent storage)
3. Episodic (experience-based)

I use all three with vector embeddings for semantic search.

> How does vector search work?

[OpenRAI]: [Recalls previous message about memory from context]

Vector search uses embeddings - numerical representations of text.
Similar concepts have similar vectors (measured by cosine similarity).

When you asked about memory earlier, I stored that as a vector.
Now when you ask about "vector search", I retrieve related memories.

> /memory

??????????????????? MEMORY STATUS ???????????????????
? Vector memory: 8 entries                          ?
? Knowledge graph: 12 entities, 15 relations        ?
? Short-term memory: 4 messages                     ?
? Topics: memory, vector search, embeddings, AI     ?
?????????????????????????????????????????????????????
```

---

## Troubleshooting

### "Failed to initialize chat interface"
- Check that OpenRAI core is properly built
- Verify LM backend is configured (mock or docker)

### No AI responses
- Check LM backend status
- For Docker: Verify container is running (`docker ps`)
- For Mock: Should work immediately

### Slow responses
- Docker LM: Normal (100-500ms for Qwen 7B)
- Mock: Should be instant (<1ms)
- Check Docker container logs if very slow

---

## Keyboard Shortcuts (Console)

- **Enter** - Send message
- **Ctrl+C** - Force quit
- **Up/Down** - (Future) Navigate history

---

## Next Steps

1. ? Test console chat with mock backend
2. ? Connect to Docker LM for real responses
3. ?? (Optional) Integrate ImGui for visual interface
4. ?? (Optional) Add conversation export/import
5. ?? (Optional) Add voice input/output

---

**Current Status**: Console chat interface ready! ??

Try it:
```powershell
.\build\Debug\openrai_chat.exe
```
