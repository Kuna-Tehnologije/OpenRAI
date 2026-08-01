#include <openrai/core.h>
#include <openrai/log.h>

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <string>

int main() {
    std::cout << "=========================================\n"
              << "  OpenRAI  |  raw / real / reimagined AI\n"
              << "  nodes=300  threads=300  gpu_slots=128\n"
              << "  one injected input per iteration\n"
              << "  log -> openrai.log\n"
              << "=========================================\n\n";

    auto engine = std::make_unique<openrai::OpenRAI>();
    engine->start();

    LOG("[Main] Engine running for 120 seconds; each iteration processes one injected input and one learning cycle...");
    std::this_thread::sleep_for(std::chrono::seconds(1200));

    engine->stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(600));

    std::cout << "\n=========================================\n"
              << "  OpenRAI Shutdown Complete\n"
              << "  Full log saved to: openrai.log\n"
              << "=========================================\n";

    return 0;
}
