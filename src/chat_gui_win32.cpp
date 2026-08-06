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
#include <fstream>

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
#define WM_UPDATE_STATS (WM_USER + 2)

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

// Update stats label with OpenRAI statistics
void UpdateStatsLabel() {
    if (!g_hStatsLabel || !g_pRAI) return;

    int iteration = g_pRAI->get_iteration_count();
    std::string latest_output = g_pRAI->get_latest_output();
    float confidence = g_pRAI->get_latest_confidence();

    // Skrati output ako je predugaèak
    if (latest_output.length() > 60) {
        latest_output = latest_output.substr(0, 57) + "...";
    }

    char stats[512];
    if (latest_output.empty()) {
        sprintf_s(stats, "Iter: %d | Messages: %d | Latest: (none)", 
                 iteration, (int)g_ChatHistory.size());
    } else {
        sprintf_s(stats, "Iter: %d | Messages: %d | Latest: %s (conf: %.2f)", 
                 iteration, (int)g_ChatHistory.size(), latest_output.c_str(), confidence);
    }

    SetWindowTextA(g_hStatsLabel, stats);
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

    // Inject input to OpenRAI (ne generišemo response ovde, OpenRAI æe automatski poslati kroz callback)
    std::thread([input]() {
        // Inject to OpenRAI
        g_pRAI->inject_input(input);

        // OpenRAI æe automatski generisati i poslati response kroz callback kada je confidence >= 0.9
        // Nema potrebe za dodatnim generate_response() pozivom ovde

    }).detach();
}

// Window procedure implementation
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_UPDATE_CHAT:
        // Update display when background thread adds message
        UpdateChatDisplay();
        break;

    case WM_UPDATE_STATS:
        // Update stats label
        UpdateStatsLabel();
        break;

    case WM_TIMER:
        // Periodic stats update (every 500ms)
        if (wParam == 1) {
            UpdateStatsLabel();
        }
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

    // Unregister class if it exists (from previous run)
    UnregisterClassA("OpenRAIChatClass", GetModuleHandle(nullptr));

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
        DWORD error = GetLastError();
        char msg[256];
        sprintf_s(msg, "Failed to register window class. Error code: %lu", error);
        MessageBoxA(nullptr, msg, "Error", MB_OK | MB_ICONERROR);

        std::ofstream log("chat_gui_debug.log", std::ios::app);
        log << "ERROR: Failed to register window class, error=" << error << "\n";
        log.flush();
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
        DWORD error = GetLastError();
        char msg[256];
        sprintf_s(msg, "Failed to create window. Error code: %lu", error);
        MessageBoxA(nullptr, msg, "Error", MB_OK | MB_ICONERROR);

        std::ofstream log("chat_gui_debug.log", std::ios::app);
        log << "ERROR: Failed to create window, error=" << error << "\n";
        log.flush();
        return false;
    }

    // Log window created
    {
        std::ofstream log("chat_gui_debug.log", std::ios::app);
        log << "Window created successfully: hwnd=" << g_hWnd << "\n";
        log.flush();
    }

    // Create chat history (multiline edit control)
    {
        std::ofstream log("chat_gui_debug.log", std::ios::app);
        log << "Creating chat history control...\n";
        log.flush();
    }

    g_hChatHistory = CreateWindowExA(
        WS_EX_CLIENTEDGE,
        "EDIT",
        "",
        WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY,
        10, 10, width - 40, height * 3 / 4 - 80,
        g_hWnd, (HMENU)1000, GetModuleHandle(nullptr), nullptr
    );

    if (!g_hChatHistory) {
        std::ofstream log("chat_gui_debug.log", std::ios::app);
        log << "ERROR: Failed to create chat history control\n";
        log.flush();
        MessageBoxA(nullptr, "Failed to create chat history control", "Error", MB_OK);
        return false;
    }

    // Set font
    {
        std::ofstream log("chat_gui_debug.log", std::ios::app);
        log << "Creating font...\n";
        log.flush();
    }

    HFONT hFont = CreateFontA(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                            DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, "Consolas");
    SendMessage(g_hChatHistory, WM_SETFONT, (WPARAM)hFont, TRUE);

    // Create input box
    {
        std::ofstream log("chat_gui_debug.log", std::ios::app);
        log << "Creating input box...\n";
        log.flush();
    }

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
    {
        std::ofstream log("chat_gui_debug.log", std::ios::app);
        log << "Creating send button...\n";
        log.flush();
    }

    g_hSendButton = CreateWindowA(
        "BUTTON",
        "Send",
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        width - 120, height * 3 / 4 - 60, 100, 30,
        g_hWnd, (HMENU)1002, GetModuleHandle(nullptr), nullptr
    );
    SendMessage(g_hSendButton, WM_SETFONT, (WPARAM)hFont, TRUE);

    // Create stats label
    {
        std::ofstream log("chat_gui_debug.log", std::ios::app);
        log << "Creating stats label...\n";
        log.flush();
    }

    g_hStatsLabel = CreateWindowA(
        "STATIC",
        "Messages: 0",
        WS_CHILD | WS_VISIBLE | SS_LEFT,
        10, height - 40, width - 40, 20,
        g_hWnd, (HMENU)1003, GetModuleHandle(nullptr), nullptr
    );

    // Add welcome message
    add_message("Hello! I'm OpenRAI, a hybrid AI with memory, knowledge graph, and language capabilities. How can I help you today?", false);

    // Set timer for periodic stats update (500ms)
    SetTimer(g_hWnd, 1, 500, nullptr);

    // Show window
    ShowWindow(g_hWnd, SW_SHOW);
    UpdateWindow(g_hWnd);

    // Log success
    std::ofstream log("chat_gui_debug.log", std::ios::app);
    log << "Win32 GUI initialized successfully, window shown\n";
    log << "g_bRunning = " << g_bRunning << "\n";
    log.flush();

    running_ = true;
    return true;
}

bool ChatGUI::render_frame() {
    if (!g_bRunning) {
        // Write to debug log
        std::ofstream log("chat_gui_debug.log", std::ios::app);
        log << "render_frame: g_bRunning = false, exiting\n";
        log.flush();
        return false;
    }

    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            g_bRunning = false;
            std::ofstream log("chat_gui_debug.log", std::ios::app);
            log << "render_frame: WM_QUIT received, exiting\n";
            log.flush();
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
        KillTimer(g_hWnd, 1);
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
