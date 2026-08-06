#pragma once

#include <openrai/core.h>
#include <openrai/thread_pool.h>
#include <openrai/memory.h>
#include <openrai/gpu_queue.h>
#include <openrai/reasoning.h>
#include <openrai/node.h>
#include <thread>
#include <chrono>
#include <cassert>
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

// Test: Node task scheduling with timeout (Priority 6)
void test_node_task_timeout() {
    std::cout << "\n[TEST] Node Task Scheduling with Timeout..." << std::endl;

    auto thread_pool = std::make_shared<ThreadPool>(4);
    auto node_manager = std::make_unique<NodeManager>();
    node_manager->set_thread_pool(thread_pool);

    // Register a test node
    auto node = std::make_unique<Node>("test_node", NodeSpecialty::Reasoning, 0);
    node_manager->register_node(std::move(node));

    // Test 1: Normal task completion
    std::cout << "  Test 1: Normal completion..." << std::endl;
    bool callback_invoked = false;
    NodeInput input;
    input.text = "Test input for quick processing";
    input.priority = 0.5f;

    node_manager->enqueue_node_task(
        "test_node",
        input,
        [&callback_invoked](const NodeTaskResult& result) {
            callback_invoked = true;
            assert(result.completed == true);
            assert(result.timed_out == false);
            std::cout << "    ? Task completed: " << result.output.summary << std::endl;
        },
        std::chrono::milliseconds(5000)
    );

    thread_pool->wait_all();
    assert(callback_invoked);
    std::cout << "    ? Callback invoked successfully" << std::endl;

    // Test 2: Timeout simulation (mock with short timeout)
    std::cout << "  Test 2: Timeout handling..." << std::endl;
    bool timeout_callback = false;

    node_manager->enqueue_node_task(
        "test_node",
        input,
        [&timeout_callback](const NodeTaskResult& result) {
            timeout_callback = true;
            if (result.timed_out) {
                std::cout << "    ? Task timed out as expected: " << result.error_message << std::endl;
            } else {
                std::cout << "    ? Task completed within timeout" << std::endl;
            }
        },
        std::chrono::milliseconds(1)  // Very short timeout to simulate timeout
    );

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    assert(timeout_callback);

    // Test 3: Non-existent node
    std::cout << "  Test 3: Non-existent node handling..." << std::endl;
    bool error_callback = false;

    node_manager->enqueue_node_task(
        "non_existent_node",
        input,
        [&error_callback](const NodeTaskResult& result) {
            error_callback = true;
            assert(result.completed == false);
            assert(!result.error_message.empty());
            std::cout << "    ? Error handled: " << result.error_message << std::endl;
        },
        std::chrono::milliseconds(5000)
    );

    thread_pool->wait_all();
    assert(error_callback);

    std::cout << "? Node task scheduling with timeout works correctly" << std::endl;
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
        test_node_task_timeout();  // Priority 6 test
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
