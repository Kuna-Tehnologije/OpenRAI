#include "openrai/chat_gui.h"
#include "openrai/core.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>

// Include ImGui - we'll use Dear ImGui for cross-platform GUI
// Note: This requires imgui to be added to the project
// For now, this is a simple Windows console-based chat interface

namespace openrai {

struct ChatGUI::ImplData {
    // Placeholder for ImGui state when integrated
    bool use_imgui = false;
};

ChatGUI::ChatGUI(OpenRAI* rai_engine)
    : rai_engine_(rai_engine),
      impl_(std::make_unique<ImplData>()) {
}

ChatGUI::~ChatGUI() {
    shutdown();
}

bool ChatGUI::initialize(int width, int height) {
    std::cout << "\n";
    std::cout << "??????????????????????????????????????????????????????????????????\n";
    std::cout << "?           OpenRAI Chat Interface (Console Mode)                ?\n";
    std::cout << "?                                                                ?\n";
    std::cout << "?  Hybrid Cognition + Language AI                                ?\n";
    std::cout << "?  Type your message and press Enter                             ?\n";
    std::cout << "?  Commands: /quit, /clear, /stats, /memory                      ?\n";
    std::cout << "??????????????????????????????????????????????????????????????????\n";
    std::cout << "\n";

    // Add welcome message
    add_message("Hello! I'm OpenRAI, a hybrid AI with memory, knowledge graph, and language capabilities. How can I help you today?", false);

    running_ = true;
    return true;
}

void ChatGUI::add_message(const std::string& text, bool is_user) {
    std::lock_guard<std::mutex> lock(chat_mutex_);

    ChatMessage msg;
    msg.text = text;
    msg.is_user = is_user;
    msg.timestamp = static_cast<float>(chat_history_.size());

    chat_history_.push_back(msg);
    total_messages_++;

    // Print to console
    if (is_user) {
        std::cout << "\n[YOU]: " << text << "\n";
    } else {
        std::cout << "[OpenRAI]: " << text << "\n";
    }
}

void ChatGUI::send_message() {
    std::string input(input_buffer_);
    if (input.empty()) return;

    // Clear input buffer
    memset(input_buffer_, 0, sizeof(input_buffer_));

    // Handle commands
    if (input[0] == '/') {
        if (input == "/quit" || input == "/exit") {
            running_ = false;
            std::cout << "\nGoodbye!\n";
            return;
        } else if (input == "/clear") {
            chat_history_.clear();
            total_messages_ = 0;
            std::cout << "\n[Chat cleared]\n";
            return;
        } else if (input == "/stats") {
            render_stats_panel();
            return;
        } else if (input == "/memory") {
            render_memory_panel();
            return;
        } else {
            std::cout << "\nUnknown command. Available: /quit, /clear, /stats, /memory\n";
            return;
        }
    }

    // Add user message
    add_message(input, true);

    // Generate AI response: one ask() through the 7.82 Hz loop (primary ->
    // experts -> arbiter -> optional language expert).  generate_response()
    // blocks until the final answer is available.
    auto start = std::chrono::steady_clock::now();

    std::cout << "[OpenRAI]: thinking..." << std::endl;
    std::string response = rai_engine_->generate_response(input);

    auto end = std::chrono::steady_clock::now();
    last_response_time_ = std::chrono::duration<float>(end - start).count();

    // Add AI response
    add_message(response, false);

    std::cout << "\n(Response time: " << last_response_time_ << "s)\n";
}

void ChatGUI::render_stats_panel() {
    std::cout << "\n??????????????????? STATISTICS ???????????????????\n";
    std::cout << "? Total messages:     " << total_messages_ << "\n";
    std::cout << "? User messages:      " << (total_messages_ / 2) << "\n";
    std::cout << "? AI responses:       " << (total_messages_ / 2) << "\n";
    std::cout << "? Last response time: " << last_response_time_ << "s\n";
    std::cout << "? Chat history size:  " << chat_history_.size() << " messages\n";
    std::cout << "??????????????????????????????????????????????????\n\n";
}

void ChatGUI::render_memory_panel() {
    const auto st = rai_engine_->stats();
    std::cout << "\n=== MEMORY STATUS ===\n"
              << "  vector memory items : " << st.vector_memory << " (" << (st.gpu_memory_index ? "GPU" : "CPU") << " index)\n"
              << "  learned experiences : " << st.experiences << "  recalls=" << st.recalls << "\n"
              << "  language model      : " << st.llm << "  answers=" << st.language_answers << "\n";
    for (const auto& l : rai_engine_->recent_experiences(5)) std::cout << "  " << l << "\n";
    std::cout << "\n";
}

bool ChatGUI::render_frame() {
    if (!running_) return false;

    // Console input mode
    std::cout << "\n> ";
    std::cout.flush();

    // Read line
    if (!std::cin.getline(input_buffer_, sizeof(input_buffer_))) {
        running_ = false;
        return false;
    }

    send_message();

    return running_;
}

void ChatGUI::shutdown() {
    running_ = false;
    std::cout << "\nChat interface closed.\n";
}

} // namespace openrai
