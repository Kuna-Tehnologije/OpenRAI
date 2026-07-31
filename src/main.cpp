#include <openrai/core.h>
#include <openrai/log.h>

#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <atomic>

static const std::vector<std::string> INPUT_POOL = {
    "user: what is the current system state",
    "sensor: temperature spike detected at node 7",
    "alert: memory usage exceeded 80 percent",
    "query: summarize last 10 reasoning outputs",
    "event: new data stream connected on port 9900",
    "command: increase thread priority on node 3",
    "signal: external pattern match on cluster B",
    "request: run deep analysis on short-term memory",
    "trigger: anomaly score above threshold 0.9",
    "note: iteration convergence detected in long-term",
    "input: re-evaluate node 5 with updated weights",
    "probe: latency measurement on gpu queue exceeded 50ms",
    "flag: importance drift detected across iterations",
    "hint: node 9 outputs consistently high relevance",
    "push: external knowledge injection reinforcement signal",
    "sys: watchdog heartbeat check passed",
    "net: packet loss detected on uplink channel",
    "db: vector index compaction triggered",
    "sched: task graph rebalance requested",
    "model: weight gradient vanishing at layer 2",
    "user: override reasoning threshold to 0.4",
    "env: GPU temperature 87C fan speed 90 percent",
    "audit: 500 long-term entries written this session",
    "ctrl: emergency stop requested by operator",
    "feed: new labelled sample batch received size 256",
};

int main() {
    std::cout << "=========================================\n"
              << "  OpenRAI  |  raw / real / reimagined AI\n"
              << "  nodes=100  threads=100  gpu_slots=128\n"
              << "  log -> openrai.log\n"
              << "=========================================\n\n";

    auto engine = std::make_unique<openrai::OpenRAI>();
    engine->start();

    // ------------------------------------------------------------------
    // Random input injector — fires every 5-12 seconds
    // ------------------------------------------------------------------
    std::atomic<bool> injector_running{true};

    std::thread injector([&]() {
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int>    delay_dist(5, 12);
        std::uniform_int_distribution<size_t> pick_dist(0, INPUT_POOL.size() - 1);

        while (injector_running) {
            int wait_sec = delay_dist(rng);
            for (int t = 0; t < wait_sec * 2 && injector_running; ++t)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

            if (!injector_running) break;
            engine->inject_input(INPUT_POOL[pick_dist(rng)]);
        }
    });

    // ------------------------------------------------------------------
    // Run for 2 minutes (120 seconds)
    // ------------------------------------------------------------------
    LOG("[Main] Engine running for 120 seconds (~60 iterations)...");
    std::this_thread::sleep_for(std::chrono::seconds(120));

    injector_running = false;
    injector.join();

    engine->stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(600));

    std::cout << "\n=========================================\n"
              << "  OpenRAI Shutdown Complete\n"
              << "  Full log saved to: openrai.log\n"
              << "=========================================\n";

    return 0;
}
