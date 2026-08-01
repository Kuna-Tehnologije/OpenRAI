// Example: Using OpenRAI with Language Model Adapter
// Demonstrates hybrid Cognition + Language architecture

#include <openrai/core.h>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "=== OpenRAI + Language Model Example ===\n\n";

    // Initialize OpenRAI (includes LM adapter with mock backend)
    openrai::OpenRAI rai;
    rai.start();

    // Give the system time to initialize
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Example 1: Generate response with cognitive context
    std::cout << "Example 1: Simple generation\n";
    std::cout << "Prompt: \"Summarize current system state\"\n\n";

    std::string response1 = rai.generate_response("Summarize current system state");
    std::cout << "Response: " << response1 << "\n\n";

    // Inject some system events to build memory context
    std::cout << "Example 2: Building cognitive context\n";
    rai.inject_input("alert: memory usage exceeded 75 percent on node 3");
    rai.inject_input("sensor: temperature spike detected at node 7 value 82C");
    rai.inject_input("event: new data stream connected on port 9042");

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Generate response with rich memory context
    std::cout << "Prompt: \"What are the critical issues?\"\n\n";
    std::string response2 = rai.generate_response("What are the critical issues?");
    std::cout << "Response: " << response2 << "\n\n";

    // Example 3: Chain of reasoning
    std::cout << "Example 3: Multi-turn conversation\n";

    std::cout << "Turn 1: \"Explain the memory alert\"\n";
    std::string response3 = rai.generate_response("Explain the memory alert");
    std::cout << "Response: " << response3 << "\n\n";

    std::cout << "Turn 2: \"What should we do about it?\"\n";
    std::string response4 = rai.generate_response("What should we do about it?");
    std::cout << "Response: " << response4 << "\n\n";

    // Example 4: Goal-driven response
    rai.inject_input("goal: prioritize stability over throughput");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cout << "Example 4: With goal alignment\n";
    std::cout << "Prompt: \"Recommend system optimization strategy\"\n\n";
    std::string response5 = rai.generate_response("Recommend system optimization strategy");
    std::cout << "Response: " << response5 << "\n\n";

    std::cout << "=== Note ===\n";
    std::cout << "Currently using MockBackend for testing.\n";
    std::cout << "To use real language models:\n";
    std::cout << "1. Integrate llama.cpp: Change model_type to \"llama\"\n";
    std::cout << "2. Integrate ONNX Runtime: Change model_type to \"qwen\"\n";
    std::cout << "3. Provide model path in LMConfig\n";
    std::cout << "\nSee HYBRID_ARCHITECTURE.md for integration guide.\n";

    rai.stop();
    return 0;
}
