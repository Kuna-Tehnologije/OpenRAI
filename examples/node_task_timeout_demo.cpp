// -----------------------------------------------------------------------
// node_task_timeout_demo.cpp
//   Demonstration of Priority 6: Task scheduling with timeout handling
//
//   Features demonstrated:
//   - Async node task scheduling using ThreadPool
//   - Timeout handling for slow nodes
//   - Callback-based result notification
// -----------------------------------------------------------------------

#include <openrai/node.h>
#include <openrai/thread_pool.h>
#include <openrai/log.h>

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>

using namespace openrai;

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Priority 6: Node Task Timeout Demo" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create ThreadPool with 4 workers
    auto thread_pool = std::make_shared<ThreadPool>(4);
    std::cout << "[Demo] Created ThreadPool with 4 workers\n" << std::endl;

    // Create NodeManager and assign ThreadPool
    auto node_manager = std::make_unique<NodeManager>();
    node_manager->set_thread_pool(thread_pool);
    std::cout << "[Demo] NodeManager initialized with ThreadPool\n" << std::endl;

    // Register multiple nodes with different specialties
    node_manager->register_node(std::make_unique<Node>("reasoning_node", NodeSpecialty::Reasoning, 0));
    node_manager->register_node(std::make_unique<Node>("planning_node", NodeSpecialty::Planning, 0));
    node_manager->register_node(std::make_unique<Node>("memory_node", NodeSpecialty::Memory, 0));
    std::cout << "[Demo] Registered 3 specialized nodes\n" << std::endl;

    // -----------------------------------------------------------------------
    // Test 1: Normal task completion
    // -----------------------------------------------------------------------
    std::cout << "\n--- Test 1: Normal Task Completion ---" << std::endl;

    NodeInput input1;
    input1.text = "Analyze the logical implications of quantum computing on cryptography";
    input1.priority = 0.8f;

    int completed_count = 0;

    node_manager->enqueue_node_task(
        "reasoning_node",
        input1,
        [&completed_count](const NodeTaskResult& result) {
            if (result.completed) {
                std::cout << "\n[Callback] Task completed successfully!" << std::endl;
                std::cout << "  Summary: " << result.output.summary << std::endl;
                std::cout << "  Confidence: " << result.output.confidence << std::endl;
                completed_count++;
            } else {
                std::cout << "\n[Callback] Task failed: " << result.error_message << std::endl;
            }
        },
        std::chrono::milliseconds(5000)
    );

    // -----------------------------------------------------------------------
    // Test 2: Multiple parallel tasks
    // -----------------------------------------------------------------------
    std::cout << "\n--- Test 2: Parallel Task Execution ---" << std::endl;

    NodeInput input2;
    input2.text = "Create a step-by-step plan for implementing a neural network";
    input2.priority = 0.7f;

    NodeInput input3;
    input3.text = "Store this important fact: OpenRAI uses distributed cognitive architecture";
    input3.priority = 0.6f;

    node_manager->enqueue_node_task(
        "planning_node",
        input2,
        [&completed_count](const NodeTaskResult& result) {
            if (result.completed) {
                std::cout << "\n[Callback] Planning task completed!" << std::endl;
                std::cout << "  Summary: " << result.output.summary << std::endl;
                completed_count++;
            }
        },
        std::chrono::milliseconds(5000)
    );

    node_manager->enqueue_node_task(
        "memory_node",
        input3,
        [&completed_count](const NodeTaskResult& result) {
            if (result.completed) {
                std::cout << "\n[Callback] Memory task completed!" << std::endl;
                std::cout << "  Summary: " << result.output.summary << std::endl;
                completed_count++;
            }
        },
        std::chrono::milliseconds(5000)
    );

    // -----------------------------------------------------------------------
    // Test 3: Timeout simulation
    // -----------------------------------------------------------------------
    std::cout << "\n--- Test 3: Timeout Handling (1ms timeout) ---" << std::endl;

    node_manager->enqueue_node_task(
        "reasoning_node",
        input1,
        [](const NodeTaskResult& result) {
            if (result.timed_out) {
                std::cout << "\n[Callback] Task timed out as expected!" << std::endl;
                std::cout << "  Error: " << result.error_message << std::endl;
            } else if (result.completed) {
                std::cout << "\n[Callback] Task completed before timeout" << std::endl;
            }
        },
        std::chrono::milliseconds(1)  // Very short timeout
    );

    // -----------------------------------------------------------------------
    // Test 4: Non-existent node error handling
    // -----------------------------------------------------------------------
    std::cout << "\n--- Test 4: Error Handling (Non-existent Node) ---" << std::endl;

    node_manager->enqueue_node_task(
        "non_existent_node",
        input1,
        [](const NodeTaskResult& result) {
            if (!result.completed && !result.timed_out) {
                std::cout << "\n[Callback] Error handled correctly!" << std::endl;
                std::cout << "  Error: " << result.error_message << std::endl;
            }
        },
        std::chrono::milliseconds(5000)
    );

    // Wait for all tasks to complete
    std::cout << "\n[Demo] Waiting for all tasks to complete..." << std::endl;
    thread_pool->wait_all();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Demo Complete!" << std::endl;
    std::cout << "  Successfully completed: " << completed_count << "/3 normal tasks" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
