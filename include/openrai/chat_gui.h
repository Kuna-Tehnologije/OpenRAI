#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace openrai {

// Forward declaration
class OpenRAI;

// Chat message structure
struct ChatMessage {
    std::string text;
    bool is_user;  // true = user, false = AI
    float timestamp;
    float relevance = 0.0f;
    float importance = 0.0f;
};

// GUI state
class ChatGUI {
public:
    explicit ChatGUI(OpenRAI* rai_engine);
    ~ChatGUI();

    // Initialize GUI window
    bool initialize(int width = 1280, int height = 720);

    // Main GUI loop (returns false when window should close)
    bool render_frame();

    // Cleanup
    void shutdown();

    // Add message to chat history
    void add_message(const std::string& text, bool is_user);

    // Check if GUI is running
    bool is_running() const { return running_; }

private:
    OpenRAI* rai_engine_;
    bool running_ = false;

    // Chat state
    std::vector<ChatMessage> chat_history_;
    char input_buffer_[512] = {0};
    bool scroll_to_bottom_ = false;
    std::mutex chat_mutex_;

    // Stats
    int total_messages_ = 0;
    float last_response_time_ = 0.0f;

    // Internal rendering
    void render_chat_window();
    void render_input_area();
    void render_stats_panel();
    void render_memory_panel();

    void send_message();

    // ImGui/Win32 state
    struct ImplData;
    std::unique_ptr<ImplData> impl_;
};

} // namespace openrai
