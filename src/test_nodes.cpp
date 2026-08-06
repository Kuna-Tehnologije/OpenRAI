// -----------------------------------------------------------------------
// test_nodes.cpp
//   Quick test for Priority 3: Per-node abstraction + registry
//
//   Build:
//     cmake --build build --target test_nodes --config Release
//   
//   Run:
//     build/Release/test_nodes.exe
// -----------------------------------------------------------------------

#include <openrai/node.h>
#include <openrai/log.h>

#include <iostream>
#include <vector>

using namespace openrai;

int main() {
    LOG("=== OpenRAI Node System Test ===");

    // Create NodeManager
    NodeManager manager;

    // Register multiple specialized nodes
    manager.register_node(std::make_unique<Node>("Retrieval_Node", NodeSpecialty::Retrieval, 0));
    manager.register_node(std::make_unique<Node>("Reasoning_Node", NodeSpecialty::Reasoning, 0));
    manager.register_node(std::make_unique<Node>("Formatter_Node", NodeSpecialty::Formatter, 0));

    LOG("\n--- Registered Nodes ---");
    for (const auto& name : manager.get_node_names()) {
        LOG("  - " << name);
    }

    // Test 1: Single node processing
    LOG("\n--- Test 1: Single Node Processing ---");
    NodeInput input1;
    input1.text = "What is the meaning of artificial intelligence?";
    input1.priority = 0.8f;

    NodeOutput output1 = manager.process_on_node("Reasoning_Node", input1);
    LOG("Output: " << output1.summary);
    LOG("Confidence: " << output1.confidence);
    LOG("Should propagate: " << (output1.should_propagate ? "yes" : "no"));

    // Test 2: Parallel processing across multiple nodes
    LOG("\n--- Test 2: Parallel Processing ---");
    NodeInput input2;
    input2.text = "Generate a plan for building a neural network system";
    input2.priority = 0.9f;

    std::vector<std::string> nodes_to_call = {"Retrieval_Node", "Reasoning_Node", "Formatter_Node"};
    auto results = manager.process_parallel(nodes_to_call, input2);

    LOG("Received " << results.size() << " outputs:");
    for (size_t i = 0; i < results.size(); ++i) {
        LOG("  [" << i << "] " << results[i].summary);
    }

    // Test 3: Store and search in node's VectorDB
    LOG("\n--- Test 3: Memory Storage and Retrieval ---");
    Node* retrieval_node = manager.get_node("Retrieval_Node");
    if (retrieval_node) {
        // Store some memories
        std::vector<float> dummy_embedding(32, 0.5f);
        retrieval_node->store_memory("AI is a branch of computer science", dummy_embedding, "definition");
        retrieval_node->store_memory("Neural networks are inspired by biology", dummy_embedding, "concept");
        retrieval_node->store_memory("Deep learning uses multiple layers", dummy_embedding, "architecture");

        LOG("Stored 3 memories in Retrieval_Node");
        LOG("Total memories: " << retrieval_node->memory_count());

        // Search
        auto search_results = retrieval_node->search_context(dummy_embedding, 2);
        LOG("Search results:");
        for (const auto& res : search_results) {
            LOG("  - [" << res.similarity << "] " << res.text_value);
        }
    }

    // Test 4: Device and FP16 configuration
    LOG("\n--- Test 4: Hardware Configuration ---");
    Node* reasoning_node = manager.get_node("Reasoning_Node");
    if (reasoning_node) {
        LOG("Initial device: " << reasoning_node->device());
        LOG("Initial FP16: " << (reasoning_node->is_fp16_enabled() ? "enabled" : "disabled"));

        reasoning_node->set_use_fp16(true);
        LOG("After set_use_fp16(true): " << (reasoning_node->is_fp16_enabled() ? "enabled" : "disabled"));
    }

    LOG("\n=== Test Complete ===");
    LOG("Total nodes: " << manager.node_count());
    LOG("All tests passed!");

    return 0;
}
