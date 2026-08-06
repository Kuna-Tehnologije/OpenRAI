// Future: ImGui-based Windows GUI for OpenRAI
// This is a template for when ImGui is integrated
// Current version uses console I/O (see chat_gui.cpp)

#ifdef OPENRAI_USE_IMGUI

#include "openrai/chat_gui.h"
#include "openrai/core.h"

// ImGui headers
#include <imgui.h>
#include <imgui_impl_win32.h>
#include <imgui_impl_dx11.h>

#include <d3d11.h>
#include <tchar.h>

namespace openrai {

// DirectX 11 state
static ID3D11Device*            g_pd3dDevice = nullptr;
static ID3D11DeviceContext*     g_pd3dDeviceContext = nullptr;
static IDXGISwapChain*          g_pSwapChain = nullptr;
static ID3D11RenderTargetView*  g_mainRenderTargetView = nullptr;

// Forward declare window procedure
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

bool ChatGUI::initialize(int width, int height) {
    // Create application window
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, 
                      GetModuleHandle(NULL), NULL, NULL, NULL, NULL, 
                      _T("OpenRAI"), NULL };
    ::RegisterClassEx(&wc);

    HWND hwnd = ::CreateWindow(wc.lpszClassName, _T("OpenRAI Chat - Hybrid Cognition + Language AI"), 
                              WS_OVERLAPPEDWINDOW, 100, 100, width, height, 
                              NULL, NULL, wc.hInstance, NULL);

    // Initialize Direct3D
    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
        return false;
    }

    // Show the window
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    running_ = true;

    // Add welcome message
    add_message("Hello! I'm OpenRAI, a hybrid AI with memory, knowledge graph, and language capabilities.", false);

    return true;
}

bool ChatGUI::render_frame() {
    if (!running_) return false;

    // Poll and handle messages
    MSG msg;
    while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
        ::TranslateMessage(&msg);
        ::DispatchMessage(&msg);
        if (msg.message == WM_QUIT)
            running_ = false;
    }

    if (!running_) return false;

    // Start the Dear ImGui frame
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    // Main window
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("OpenRAI Chat", nullptr, 
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    // Chat history panel (75% of window)
    ImGui::BeginChild("ChatHistory", ImVec2(0, -80), true);

    std::lock_guard<std::mutex> lock(chat_mutex_);
    for (auto& msg : chat_history_) {
        ImVec4 color = msg.is_user ? ImVec4(0.2f, 0.8f, 0.3f, 1.0f) : ImVec4(1.0f, 0.8f, 0.2f, 1.0f);
        const char* label = msg.is_user ? "[YOU]" : "[OpenRAI]";

        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::TextWrapped("%s: %s", label, msg.text.c_str());
        ImGui::PopStyleColor();

        ImGui::Spacing();
    }

    // Auto-scroll to bottom
    if (scroll_to_bottom_) {
        ImGui::SetScrollHereY(1.0f);
        scroll_to_bottom_ = false;
    }

    ImGui::EndChild();

    // Input panel (bottom 25%)
    ImGui::Separator();
    ImGui::Text("Message:");
    ImGui::SameLine();

    ImGui::PushItemWidth(-100);
    if (ImGui::InputText("##input", input_buffer_, sizeof(input_buffer_), 
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
        send_message();
    }
    ImGui::PopItemWidth();

    ImGui::SameLine();
    if (ImGui::Button("Send", ImVec2(80, 0))) {
        send_message();
    }

    // Stats bar
    ImGui::Separator();
    ImGui::Text("Messages: %d | Last response: %.3fs | Memory: [Stats here]", 
                total_messages_, last_response_time_);

    ImGui::End();

    // Rendering
    ImGui::Render();
    g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);

    const float clear_color[4] = {0.1f, 0.1f, 0.1f, 1.0f};
    g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color);

    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    g_pSwapChain->Present(1, 0);

    return running_;
}

void ChatGUI::shutdown() {
    // Cleanup
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();

    running_ = false;
}

void ChatGUI::add_message(const std::string& text, bool is_user) {
    std::lock_guard<std::mutex> lock(chat_mutex_);

    ChatMessage msg;
    msg.text = text;
    msg.is_user = is_user;
    msg.timestamp = static_cast<float>(chat_history_.size());

    chat_history_.push_back(msg);
    total_messages_++;
    scroll_to_bottom_ = true;
}

void ChatGUI::send_message() {
    std::string input(input_buffer_);
    if (input.empty()) return;

    // Clear input buffer
    memset(input_buffer_, 0, sizeof(input_buffer_));

    // Add user message
    add_message(input, true);

    // Inject to OpenRAI engine
    if (rai_engine_) {
        rai_engine_->inject_input(input);
    }
}

void ChatGUI::render_chat_window() {
    // Implemented inline in render_frame()
}

void ChatGUI::render_input_area() {
    // Implemented inline in render_frame()
}

void ChatGUI::render_stats_panel() {
    // Implemented inline in render_frame()
}

void ChatGUI::render_memory_panel() {
    // Future implementation
}

// Helper functions for D3D11
bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };

    if (D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags, 
                                      featureLevelArray, 2, D3D11_SDK_VERSION, &sd, 
                                      &g_pSwapChain, &g_pd3dDevice, &featureLevel, 
                                      &g_pd3dDeviceContext) != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = NULL; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = NULL; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
}

void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget() {
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = NULL; }
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
    case WM_SIZE:
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU)
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}

} // namespace openrai

#endif // OPENRAI_USE_IMGUI
