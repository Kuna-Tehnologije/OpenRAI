#pragma once

#include <openrai/core.h>
#include <openrai/thread_pool.h>`n#include <openrai/memory.h>`n#include <openrai/gpu_queue.h>`n#include <openrai/reasoning.h>`n#include <thread>`n#include <chrono>`n#include <cassert>
#include <iostream>

namespace openrai::tests {

// Test: ThreadPool executes tasks
void test_thread_pool() {
    std::cout << "\n[TEST] ThreadPool..." << std::endl;

    auto pool = std::make_unique<ThreadPool>(10);
    int counter = 0;

    for (int i = 0; i < 100; ++i) {
        pool->enqueue([&counter]() { counter++; });
    }

    pool->wait_all();
    assert(counter == 100);
    std::cout << "? ThreadPool executed all 100 tasks" << std::endl;
}

// Test: MemorySystem layering
void test_memory_layers() {
    std::cout << "\n[TEST] Memory Layers..." << std::endl;

    auto memory = std::make_unique<MemorySystem>();

    // Add 10 outputs to current memory
    for (int i = 0; i < 10; ++i) {
        Output out;
        out.data = "output_" + std::to_string(i);
        out.relevance_score = 0.1f * (i % 5);  // Varying scores
        out.thread_id = i;
        memory->add_current_output(out);
    }

    assert(memory->get_current_outputs().size() == 10);
    std::cout << "? Current memory holds 10 outputs" << std::endl;

    // Filter to short-term
    memory->filter_to_short_term();
    auto st = memory->get_short_term_memory();
    assert(st.size() == 4);
    std::cout << "? Short-term memory filtered to 4 top entries" << std::endl;

    // Add to long-term
    MemoryEntry entry;
    entry.content = "test thought";
    entry.importance = 0.8f;
    entry.created_at = std::chrono::system_clock::now();
    memory->add_to_long_term(entry);

    auto lt = memory->retrieve_from_long_term("thought", 10);
    assert(lt.size() == 1);
    std::cout << "? Long-term memory stores and retrieves" << std::endl;
}

// Test: ReasoningModule evaluation
void test_reasoning() {
    std::cout << "\n[TEST] Reasoning Module..." << std::endl;

    auto memory = std::make_unique<MemorySystem>();
    auto reasoning = std::make_unique<ReasoningModule>(memory.get());

    auto result = reasoning->evaluate_thought(
        "important: system processed 1000 requests successfully"
    );

    assert(result.importance_score > 0.5f);
    assert(result.should_output == true);
    std::cout << "? Reasoning correctly evaluates important thoughts" << std::endl;
}

// Test: GPUJobQueue
void test_gpu_queue() {
    std::cout << "\n[TEST] GPU Job Queue..." << std::endl;

    auto queue = std::make_unique<GPUJobQueue>(5);

    // Enqueue 5 jobs
    for (int i = 0; i < 5; ++i) {
        GPUJob job;
        job.type = "inference";
        job.data = "input_" + std::to_string(i);
        queue->enqueue_job(job);
    }

    assert(queue->queue_size() <= 5);
    std::cout << "? GPU queue enqueued jobs (size <= 5)" << std::endl;

    // Dequeue
    GPUJob out;
    assert(queue->try_dequeue_job(out));
    assert(out.data == "input_0");
    std::cout << "? GPU queue dequeues in FIFO order" << std::endl;
}

// Test: Full loop
void test_full_loop() {
    std::cout << "\n[TEST] Full OpenRAI Loop (3 seconds)..." << std::endl;

    auto engine = std::make_unique<OpenRAI>();
    engine->start();

    std::this_thread::sleep_for(std::chrono::seconds(3));

    engine->stop();
    std::cout << "? OpenRAI loop completed without crash" << std::endl;
}

// Run all tests
inline void run_all_tests() {
    std::cout << "\n??????????????????????????????????????????" << std::endl;
    std::cout << "?        OpenRAI Unit Tests (PoC)        ?" << std::endl;
    std::cout << "??????????????????????????????????????????" << std::endl;

    try {
        test_thread_pool();
        test_memory_layers();
        test_reasoning();
        test_gpu_queue();
        test_full_loop();

        std::cout << "\n??????????????????????????????????????????" << std::endl;
        std::cout << "?      ? All Tests Passed!              ?" << std::endl;
        std::cout << "??????????????????????????????????????????" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n? Test failed: " << e.what() << std::endl;
        throw;
    }
}

} // namespace openrai::tests
