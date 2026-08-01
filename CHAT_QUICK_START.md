# ?? OpenRAI Chat Interface - Ready to Use!

## Quick Start (3 Commands!)

```powershell
# 1. Build chat interface
cd C:\dev\openRAI
cmake --build build --config Debug --target openrai_chat

# 2. Run it!
.\build\Debug\openrai_chat.exe

# 3. Start chatting!
> Hello!
```

---

## What You Get

### Interactive Chat
- Type messages, get AI responses
- Conversation history
- Response timing stats

### Cognitive AI
- **Memory**: Remembers previous messages
- **Knowledge Graph**: Tracks entities and relations
- **Goal Alignment**: Value-driven responses
- **Self-Reflection**: Explains reasoning

### Commands
- `/quit` - Exit chat
- `/clear` - Clear history
- `/stats` - Show statistics
- `/memory` - Memory status

---

## Example Session

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

> What's the difference between you and ChatGPT?

[YOU]: What's the difference between you and ChatGPT?

[OpenRAI]: Thinking...
[SimpleHTTPBackend] Calling API...
[SimpleHTTPBackend] ? Generated 312 chars

[OpenRAI]: Great question! The key differences are:

1. **Persistent Memory**: I remember across sessions (vector DB)
2. **Knowledge Graph**: I build explicit entity/relation maps
3. **Goal System**: I align responses with defined goals
4. **Self-Reflection**: I analyze my own reasoning
5. **Hybrid Architecture**: I combine cognition (OpenRAI) with 
   language (Qwen/Llama)

ChatGPT excels at language, but I add structured memory and reasoning!

(Response time: 0.425s)

> Tell me more about your memory

[YOU]: Tell me more about your memory

[OpenRAI]: Thinking...

[OpenRAI]: [Retrieves context from previous message about memory]

I use three memory layers:

1. **Current Memory**: Active conversation (last 10 messages)
2. **Short-term Memory**: Important recent items (top 4)
3. **Vector Memory**: Persistent semantic storage (5000 embeddings)

When you asked "what's the difference" earlier, I stored that as 
a vector. Now when you ask about memory, I retrieve related context.

This is similar to human episodic memory!

(Response time: 0.387s)

> /stats

??????????????????? STATISTICS ???????????????????
? Total messages:     6
? User messages:      3
? AI responses:       3
? Last response time: 0.387s
? Chat history size:  6 messages
??????????????????????????????????????????????????

> /quit

Goodbye!

Thank you for using OpenRAI!
```

---

## Configuration

### Using Mock Backend (Default)

Works immediately, no setup:

```cpp
// src/core.cpp (default)
LMConfig lm_config;
lm_config.model_type = "mock";
```

### Using Docker LM (Real AI!)

When your Qwen model download finishes:

```cpp
// src/core.cpp
LMConfig lm_config;
lm_config.model_type = "docker";
lm_config.model_path = "http://localhost:8000";
```

See `CONNECT_TO_DOCKER.md` for details.

---

## Architecture Flow

```
[User types in console]
        ?
[ChatGUI captures input]
        ?
[OpenRAI::inject_input()] ? Stores in memory
        ?
[OpenRAI::generate_response()]
  ?? Gathers memory context (last 5 messages)
  ?? Queries knowledge graph (related entities)
  ?? Builds enhanced prompt with context
  ?? Calls LM backend (Mock or Docker)
        ?
[LM generates response]
        ?
[OpenRAI post-processing]
  ?? Stores response in vector memory
  ?? Updates knowledge graph
  ?? Triggers self-reflection
  ?? Returns final text
        ?
[ChatGUI displays response]
        ?
[User sees: AI message + timing + stats]
```

---

## Two Executables

### 1. `openrai.exe` (Original)
- Automated loop
- Batch processing
- No user interaction
- For testing/benchmarking

### 2. `openrai_chat.exe` (New!)
- Interactive chat
- User input
- Conversation interface
- For real usage

---

## Future: Visual GUI (ImGui)

Current version uses console. To add visual interface:

1. **Install ImGui**:
   ```bash
   vcpkg install imgui[core,win32-binding,dx11-binding]
   ```

2. **Update CMakeLists.txt**:
   ```cmake
   find_package(imgui CONFIG REQUIRED)
   target_link_libraries(openrai_chat PRIVATE imgui::imgui)
   add_compile_definitions(OPENRAI_USE_IMGUI)
   ```

3. **Use ImGui implementation**:
   See `src/chat_gui_imgui.cpp` (already prepared!)

Result: Modern Windows app with text boxes, buttons, styling!

---

## Keyboard Shortcuts

- **Enter** - Send message
- **Ctrl+C** - Force quit
- **/command** - Special commands

---

## Tips

### Multi-turn Conversations
The AI remembers context:
```
> What is AI?
> Can you explain that in simpler terms?  ? Refers to previous answer
> Give me an example  ? Continues the thread
```

### Using Commands
```
/stats   ? See message count, timing
/memory  ? Check memory status
/clear   ? Start fresh conversation
/quit    ? Exit
```

### Response Timing
- Mock: <1ms (instant)
- Docker Qwen 7B: 100-500ms (depends on load)
- Shows after each response: `(Response time: 0.425s)`

---

## Troubleshooting

### Chat not responding
- Check LM backend in `src/core.cpp`
- For Docker: `docker ps` to verify container
- For Mock: Should always work

### No color in output
- Console mode doesn't have colors yet
- Wait for ImGui version for styled output

### Slow responses
- Normal for Docker LM (100-500ms)
- Mock should be instant
- Check Docker logs if very slow

---

## Next Steps

1. ? **Try it now** - Works with mock backend
2. ? **Connect to Docker** - When Qwen download finishes
3. ?? **Add ImGui** - For visual interface (optional)
4. ?? **Export conversations** - Save chat history (future)
5. ?? **Voice I/O** - Speak to AI (future)

---

## Files

- `include/openrai/chat_gui.h` - Interface
- `src/chat_gui.cpp` - Console implementation
- `src/chat_gui_imgui.cpp` - ImGui template (future)
- `src/main_chat.cpp` - Chat executable entry point
- `CHAT_INTERFACE.md` - Full documentation

---

**Status**: ? Console chat ready to use!

```powershell
.\build\Debug\openrai_chat.exe
```

Enjoy chatting with OpenRAI! ??
