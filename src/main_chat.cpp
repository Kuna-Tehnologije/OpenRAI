// OpenRAI Chat Interface
// Interactive chat with hybrid Cognition + Language AI

#include <openrai/core.h>
#include <openrai/chat_gui.h>
#include <iostream>
#include <thread>
#include <chrono>

int main(int argc, char* argv[]) {
    std::cout << "=== OpenRAI Chat Application ===\n\n";
    std::cout << "Initializing OpenRAI engine...\n";

    // Initialize OpenRAI
    openrai::OpenRAI rai;
    rai.start();

    // Give engine time to initialize
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "OpenRAI engine ready!\n";

    // Initialize Chat GUI
    openrai::ChatGUI chat_gui(&rai);

    if (!chat_gui.initialize(1280, 720)) {
        std::cerr << "Failed to initialize chat interface\n";
        return 1;
    }

    // Main chat loop
    while (chat_gui.render_frame()) {
        // GUI rendering happens in render_frame()
    }

    // Cleanup
    chat_gui.shutdown();
    rai.stop();

    std::cout << "\nThank you for using OpenRAI!\n";

    return 0;
}
