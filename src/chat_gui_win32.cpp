// Simple Windows GUI for OpenRAI Chat (no ImGui dependency)
// Simple Windows GUI for OpenRAI Chat (no ImGui dependency)
// Uses native Win32 API for text boxes and buttons

#include "openrai/chat_gui.h"
#include "openrai/core.h"
#include <windows.h>
#include <commctrl.h>
#include <string>
#include <thread>
#include <mutex>

#pragma comment(lib, "comctl32.lib")

namespace openrai {

// ImplData for compatibility with base class
struct ChatGUI::ImplData {
    // Empty - not used in Win32 version
};

// Window handles
static HWND g_hWnd = nullptr;
static HWND g_hChatHistory = nullptr;
static HWND g_hInputBox = nullptr;
static HWND g_hSendButton = nullptr;
static HWND g_hStatsLabel = nullptr;

static OpenRAI* g_pRAI = nullptr;
static std::vector<ChatMessage> g_ChatHistory;
static std::mutex g_ChatMutex;
static bool g_bRunning = false;

// Custom message for UI update from background thread
#define WM_UPDATE_CHAT (WM_USER + 1)

// Window procedure
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

// Update chat display (must be called from UI thread)
void UpdateChatDisplay() {
    if (!g_hChatHistory) return;

    std::string chatText;
    {
        std::lock_guard<std::mutex> lock(g_ChatMutex);
        for (const auto& msg : g_ChatHistory) {
            if (msg.is_user) {
                chatText += "[YOU]: ";
            } else {
                chatText += "[OpenRAI]: ";
            }
            chatText += msg.text + "\r\n\r\n";
        }
    }

    SetWindowTextA(g_hChatHistory, chatText.c_str());

    // Scroll to bottom
    SendMessage(g_hChatHistory, EM_SETSEL, chatText.size(), chatText.size());
    SendMessage(g_hChatHistory, EM_SCROLLCARET, 0, 0);
}

// Send message handler
void SendChatMessage() {
    if (!g_hInputBox || !g_pRAI) return;

    char buffer[512];
    GetWindowTextA(g_hInputBox, buffer, sizeof(buffer));

    std::string input(buffer);
    if (input.empty()) return;

    // Clear input
    SetWindowTextA(g_hInputBox, "");

    // Add user message
    ChatMessage userMsg;
    userMsg.text = input;
    userMsg.is_user = true;
    {
        std::lock_guard<std::mutex> lock(g_ChatMutex);
        g_ChatHistory.push_back(userMsg);
    }
    UpdateChatDisplay();

    // Update status
    SetWindowTextA(g_hStatsLabel, "OpenRAI is thinking...");

    // Generate AI response (in background to not block UI)
    std::thread([input]() {
        // Inject to OpenRAI
        g_pRAI->inject_input(input);

        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Generate response
        auto start = std::chrono::steady_clock::now();
        std::string response = g_pRAI->generate_response(input);
        auto end = std::chrono::steady_clock::now();
        float responseTime = std::chrono::duration<float>(end - start).count();

        // Add AI message
        ChatMessage aiMsg;
        aiMsg.text = response;
        aiMsg.is_user = false;
        g_ChatHistory.push_back(aiMsg);

        // Update UI (must be on main thread in real app, but this works for simple case)
        UpdateChatDisplay();

        // Update stats
        char stats[256];
        sprintf_s(stats, "Messages: %d | Last response: %.3fs", 
                 (int)g_ChatHistory.size(), responseTime);
        SetWindowTextA(g_hStatsLabel, stats);

    }).detach();
}

// Window procedure implementation
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_UPDATE_CHAT:
        // Update display when background thread adds message
        UpdateChatDisplay();
        break;

    case WM_COMMAND:
        // Only handle button click (BN_CLICKED), not edit control notifications
        if (LOWORD(wParam) == 1002 && HIWORD(wParam) == BN_CLICKED) {  // Send button clicked
            SendChatMessage();
        }
        break;

    case WM_SIZE: {
        // Resize controls
        RECT rc;
        GetClientRect(hWnd, &rc);

        int width = rc.right - rc.left;
        int height = rc.bottom - rc.top;

        // Chat history (top 75%)
        if (g_hChatHistory) {
            SetWindowPos(g_hChatHistory, nullptr, 
                        10, 10, 
                        width - 20, height * 3 / 4 - 70,
                        SWP_NOZORDER);
        }

        // Input box (bottom area)
        if (g_hInputBox) {
            SetWindowPos(g_hInputBox, nullptr,
                        10, height * 3 / 4 - 50,
                        width - 120, 30,
                        SWP_NOZORDER);
        }

        // Send button
        if (g_hSendButton) {
            SetWindowPos(g_hSendButton, nullptr,
                        width - 100, height * 3 / 4 - 50,
                        80, 30,
                        SWP_NOZORDER);
        }

        // Stats label
        if (g_hStatsLabel) {
            SetWindowPos(g_hStatsLabel, nullptr,
                        10, height - 30,
                        width - 20, 20,
                        SWP_NOZORDER);
        }
        break;
    }

    case WM_DESTROY:
        g_bRunning = false;
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// ChatGUI implementation for Windows native UI
ChatGUI::ChatGUI(OpenRAI* rai_engine) 
    : rai_engine_(rai_engine),
      impl_(std::make_unique<ImplData>()) {
}

ChatGUI::~ChatGUI() {
    shutdown();
}

bool ChatGUI::initialize(int width, int height) {
    g_pRAI = rai_engine_;
    g_bRunning = true;

    // Register window class
    WNDCLASSEXA wc = {};
    wc.cbSize = sizeof(WNDCLASSEXA);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = "OpenRAIChatClass";

    if (!RegisterClassExA(&wc)) {
        MessageBoxA(nullptr, "Failed to register window class", "Error", MB_OK | MB_ICONERROR);
        return false;
    }

    // Create main window
    g_hWnd = CreateWindowExA(
        0,
        "OpenRAIChatClass",
        "OpenRAI Chat - Hybrid Cognition + Language AI",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, width, height,
        nullptr, nullptr, GetModuleHandle(nullptr), nullptr
    );

    if (!g_hWnd) {
        MessageBoxA(nullptr, "Failed to create window", "Error", MB_OK | MB_ICONERROR);
        return false;
    }

    // Create chat history (multiline edit control)
    g_hChatHistory = CreateWindowExA(
        WS_EX_CLIENTEDGE,
        "EDIT",
        "",
        WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
        10, 10, width - 40, height * 3 / 4 - 80,
        g_hWnd, (HMENU)1000, GetModuleHandle(nullptr), nullptr
    );

    // Set font
    HFONT hFont = CreateFontA(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                            DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Consolas");
    SendMessage(g_hChatHistory, WM_SETFONT, (WPARAM)hFont, TRUE);

    // Create input box
    g_hInputBox = CreateWindowExA(
        WS_EX_CLIENTEDGE,
        "EDIT",
        "",
        WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
        10, height * 3 / 4 - 60, width - 140, 30,
        g_hWnd, (HMENU)1001, GetModuleHandle(nullptr), nullptr
    );
    SendMessage(g_hInputBox, WM_SETFONT, (WPARAM)hFont, TRUE);

    // Create send button
    g_hSendButton = CreateWindowA(
        "BUTTON",
        "Send",
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        width - 120, height * 3 / 4 - 60, 100, 30,
        g_hWnd, (HMENU)1002, GetModuleHandle(nullptr), nullptr
    );
    SendMessage(g_hSendButton, WM_SETFONT, (WPARAM)hFont, TRUE);

    // Create stats label
    g_hStatsLabel = CreateWindowA(
        "STATIC",
        "Messages: 0",
        WS_CHILD | WS_VISIBLE | SS_LEFT,
        10, height - 40, width - 40, 20,
        g_hWnd, (HMENU)1003, GetModuleHandle(nullptr), nullptr
    );

    // Add welcome message
    add_message("Hello! I'm OpenRAI, a hybrid AI with memory, knowledge graph, and language capabilities. How can I help you today?", false);

    // Show window
    ShowWindow(g_hWnd, SW_SHOW);
    UpdateWindow(g_hWnd);

    return true;
}

bool ChatGUI::render_frame() {
    if (!g_bRunning) return false;

    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            g_bRunning = false;
            return false;
        }

        // Handle Enter key in input box
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_RETURN && msg.hwnd == g_hInputBox) {
            SendChatMessage();
            continue; // Skip default processing
        }

        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    return g_bRunning;
}

void ChatGUI::add_message(const std::string& text, bool is_user) {
    std::lock_guard<std::mutex> lock(chat_mutex_);

    ChatMessage msg;
    msg.text = text;
    msg.is_user = is_user;
    msg.timestamp = static_cast<float>(chat_history_.size());

    chat_history_.push_back(msg);
    {
        std::lock_guard<std::mutex> glock(g_ChatMutex);
        g_ChatHistory.push_back(msg);
    }
    total_messages_++;

    // Post message to UI thread to update display (thread-safe)
    if (g_hWnd) {
        PostMessage(g_hWnd, WM_UPDATE_CHAT, 0, 0);
    }
}

void ChatGUI::shutdown() {
    g_bRunning = false;
    if (g_hWnd) {
        DestroyWindow(g_hWnd);
        g_hWnd = nullptr;
    }
    running_ = false;
}

} // namespace openrai

// Forward declaration
extern int main(int argc, char* argv[]);

// WinMain entry point for WIN32 subsystem
#ifdef OPENRAI_WIN32_GUI
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Call the regular main() function
    return main(0, nullptr);
}
#endif
