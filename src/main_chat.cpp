// OpenRAI Chat Interface
// Interactive chat with hybrid Cognition + Language AI

#include <openrai/core.h>
#include <openrai/chat_gui.h>
#include <openrai/log.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <memory>
#include <atomic>

int main(int argc, char* argv[]) {
    // Create log file for debugging GUI issues
    std::ofstream debugLog("chat_gui_debug.log", std::ios::out | std::ios::trunc);
    debugLog << "=== OpenRAI Chat Application Debug Log ===\n";
    debugLog.flush();

    // Engine logs go to openrai.log only; the chat shows answers.
    openrai::Logger::get().set_console(false);

    std::cout << "=== OpenRAI Chat Application ===\n\n";
    std::cout << "Initializing OpenRAI engine...\n";
    debugLog << "Starting initialization...\n";
    debugLog.flush();

    try {
        // Initialize OpenRAI in background thread to avoid blocking GUI
        debugLog << "Creating OpenRAI instance in background...\n";
        debugLog.flush();

        std::unique_ptr<openrai::OpenRAI> rai;
        std::atomic<bool> rai_ready{false};
        std::atomic<bool> rai_failed{false};
        std::string error_message;

        // Start OpenRAI initialization in background thread
        std::thread init_thread([&]() {
            try {
                debugLog << "[BG] Constructing OpenRAI...\n";
                debugLog.flush();

                rai = std::make_unique<openrai::OpenRAI>();

                debugLog << "[BG] Starting OpenRAI engine...\n";
                debugLog.flush();

                rai->start();

                debugLog << "[BG] OpenRAI ready!\n";
                debugLog.flush();

                rai_ready = true;
            } catch (const std::exception& e) {
                debugLog << "[BG] EXCEPTION during init: " << e.what() << "\n";
                debugLog.flush();
                error_message = e.what();
                rai_failed = true;
            } catch (...) {
                debugLog << "[BG] UNKNOWN EXCEPTION during init\n";
                debugLog.flush();
                error_message = "Unknown error";
                rai_failed = true;
            }
        });

        // Wait for initialization with timeout (30 seconds)
        debugLog << "Waiting for OpenRAI initialization (max 30 seconds)...\n";
        debugLog.flush();

        int wait_count = 0;
        while (!rai_ready && !rai_failed && wait_count < 300) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            wait_count++;
            if (wait_count % 10 == 0) {
                std::cout << ".";
                std::cout.flush();
                debugLog << ".";
                debugLog.flush();
            }
        }
        std::cout << "\n";
        debugLog << "\n";
        debugLog << "Wait loop finished: ready=" << rai_ready << " failed=" << rai_failed << " count=" << wait_count << "\n";
        debugLog.flush();

        if (rai_failed) {
            std::cerr << "Failed to initialize OpenRAI: " << error_message << "\n";
            debugLog << "ERROR: Failed to initialize OpenRAI: " << error_message << "\n";
            debugLog.flush();
            init_thread.join();
            return 1;
        }

        if (!rai_ready) {
            std::cerr << "OpenRAI initialization timeout\n";
            debugLog << "ERROR: OpenRAI initialization timeout\n";
            debugLog.flush();
            init_thread.detach(); // Let it finish in background
            return 1;
        }

        debugLog << "Joining init thread...\n";
        debugLog.flush();

        init_thread.join();

        std::cout << "OpenRAI engine ready!\n";
        debugLog << "OpenRAI engine ready!\n";
        debugLog.flush();

        // Initialize Chat GUI
        debugLog << "Creating ChatGUI instance...\n";
        debugLog.flush();

        openrai::ChatGUI chat_gui(rai.get());

        debugLog << "Initializing ChatGUI window...\n";
        debugLog.flush();

        if (!chat_gui.initialize(1280, 720)) {
            std::cerr << "Failed to initialize chat interface\n";
            debugLog << "ERROR: Failed to initialize chat interface\n";
            debugLog.flush();
            return 1;
        }

        debugLog << "ChatGUI initialized successfully\n";
        debugLog.flush();

        // Postavi callback za slanje outputa iz core-a u GUI
        debugLog << "Setting up output callback...\n";
        debugLog.flush();

        rai->set_output_callback([&chat_gui](const std::string& output) {
            chat_gui.add_message(output, false);
        });

        debugLog << "Entering main GUI loop...\n";
        debugLog.flush();

        // Main chat loop
        int frameCount = 0;
        while (chat_gui.render_frame()) {
            // GUI rendering happens in render_frame()
            frameCount++;
            if (frameCount % 60 == 0) {
                debugLog << "Frame: " << frameCount << "\n";
                debugLog.flush();
            }
        }

        debugLog << "Exiting main loop...\n";
        debugLog.flush();

        // Cleanup
        chat_gui.shutdown();
        rai->stop();

        std::cout << "\nThank you for using OpenRAI!\n";
        debugLog << "Shutdown complete\n";
        debugLog.flush();

    } catch (const std::exception& e) {
        debugLog << "EXCEPTION: " << e.what() << "\n";
        debugLog.flush();
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        debugLog << "UNKNOWN EXCEPTION\n";
        debugLog.flush();
        std::cerr << "Unknown exception\n";
        return 1;
    }

    return 0;
}
