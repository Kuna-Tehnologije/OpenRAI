# OpenRAI Chat Interface - Opcije

## ?? Tri naèina korištenja

### 1. Console Mode (Trenutno - Radi odmah!) ?

```powershell
.\build\Debug\openrai_chat.exe
```

**Prednosti:**
- ? Nema dependencies
- ? Brz setup (veæ build-ano)
- ? Radi na svim terminalima
- ? Lagan (~2 MB exe)

**Izgled:**
```
> Hello!
[OpenRAI]: Hello! How can I help you today?

> What's your memory like?
[OpenRAI]: I use three memory layers...
```

---

### 2. ImGui Visual UI (15 min setup) ??

```powershell
# Run automatic setup
.\scripts\setup_imgui.ps1

# Then run
.\build\Debug\openrai_chat.exe
```

**Prednosti:**
- ? Profesionalni Windows UI
- ? Scrollable chat history
- ? Click-to-send button
- ? Color-coded messages
- ? Resizable window
- ? Modern dark theme

**Izgled:**
```
?????????????????????????????????????????
? OpenRAI Chat                     ?  ?
?????????????????????????????????????????
? ?? Hello! How can I help?            ?
? ?? What's special about you?         ?
? ?? I combine cognition with...       ?
?                                  ?   ?
?                                  ?   ?
?????????????????????????????????????????
? [Type here...         ]    [Send]   ?
?????????????????????????????????????????
? 6 messages | 0.42s | Memory: 8      ?
?????????????????????????????????????????
```

**Setup guide:** `IMGUI_ACTIVATION_GUIDE.md`

---

### 3. Programski (API) ??

```cpp
#include <openrai/core.h>

int main() {
    openrai::OpenRAI rai;
    rai.start();

    // Inject input
    rai.inject_input("What is AI?");

    // Generate response
    std::string response = rai.generate_response("Explain AI");

    std::cout << response << "\n";

    rai.stop();
    return 0;
}
```

**Prednosti:**
- ? Potpuna kontrola
- ? Integracija u druge aplikacije
- ? Batch processing
- ? Automation

**Example:** `examples/lm_example.cpp`

---

## ?? Usporedba

| Feature | Console | ImGui | API |
|---------|---------|-------|-----|
| **Setup time** | 0 min | 15 min | Custom |
| **Dependencies** | None | vcpkg+ImGui | Custom |
| **UI** | Text | Visual | None |
| **Ease of use** | Medium | Easy | Expert |
| **File size** | ~2 MB | ~5 MB | Variable |
| **Platform** | Any terminal | Windows | Any |
| **Best for** | Quick test | Daily use | Integration |

---

## ?? Koji odabrati?

### Ako želiš brzo testirati OpenRAI:
? **Console mode** (radi odmah)

### Ako želiš koristiti svaki dan:
? **ImGui UI** (15 min setup, profesionalni izgled)

### Ako integraš u postojeæi projekt:
? **API mode** (potpuna kontrola)

---

## ?? Quick Start Guide

### Console (0 min)
```powershell
.\build\Debug\openrai_chat.exe
```

### ImGui (15 min)
```powershell
.\scripts\setup_imgui.ps1
.\build\Debug\openrai_chat.exe
```

### API (custom)
```cpp
// See examples/lm_example.cpp
```

---

## ?? Dokumentacija

| Dokument | Svrha |
|----------|-------|
| `CHAT_QUICK_START.md` | Console mode guide |
| `IMGUI_ACTIVATION_GUIDE.md` | ImGui setup |
| `CHAT_INTERFACE.md` | Detailed docs |
| `examples/lm_example.cpp` | API examples |

---

## ?? ImGui Features (kada aktiviran)

- **Chat history panel**: Scrollable, color-coded
- **Input box**: With Enter-to-send
- **Send button**: Click to send message
- **Stats bar**: Messages, timing, memory
- **Dark theme**: Professional look
- **Resizable**: Window adapts to screen

---

## ?? Switching Modes

### Console ? ImGui
```powershell
# Run setup script
.\scripts\setup_imgui.ps1

# Rebuild
cmake --build build --config Debug --target openrai_chat

# Run - now with GUI!
.\build\Debug\openrai_chat.exe
```

### ImGui ? Console
Remove `-DOPENRAI_USE_IMGUI` from CMakeLists.txt and rebuild.

### Build Both
```cmake
# In CMakeLists.txt
add_executable(openrai_chat_console src/main_chat.cpp src/chat_gui.cpp ...)
add_executable(openrai_chat_gui src/main_chat.cpp src/chat_gui_imgui.cpp ...)
target_compile_definitions(openrai_chat_gui PRIVATE OPENRAI_USE_IMGUI)
```

---

## ?? Preporuka

1. **Testiraj s Console** (trenutno radi)
2. **Spoji na Docker Qwen** (kada download završi)
3. **Aktiviraj ImGui** (za profesionalni izgled)

---

**Status:**
- ? Console: Ready
- ?? ImGui: Template ready, needs vcpkg
- ? API: Working
- ? Docker: Waiting for Qwen download

Odaberi naèin koji ti odgovara! ??
