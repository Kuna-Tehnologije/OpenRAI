#pragma once

#include <vector>
#include <string>
#include <random>
#include <cstdint>

namespace openrai {

enum class Activation { ReLU, Sigmoid, Linear };

// -----------------------------------------------------------------------
// NeuralNetwork
//   Pure C++ feedforward network.
//   No external dependencies - runs on every GPUJob as the inference engine.
//
//   Output layout (4 neurons):
//     [0] relevance_score   (Sigmoid 0..1)
//     [1] importance_score  (Sigmoid 0..1)
//     [2] should_output     (Sigmoid 0..1, threshold 0.5)
//     [3] priority          (Sigmoid 0..1)
// -----------------------------------------------------------------------
class NeuralNetwork {
public:
    struct LayerConfig {
        size_t   size;
        Activation activation;
    };

    // Constructs the network and initialises weights with Xavier init.
    NeuralNetwork(size_t input_size,
                  const std::vector<LayerConfig>& layers,
                  uint32_t seed = 42);

    // Forward pass.  Returns output vector of size output_size().
    std::vector<float> forward(const std::vector<float>& input) const;

    size_t input_size()  const { return input_size_; }
    size_t output_size() const;
    size_t num_layers()  const { return layers_.size(); }

    void print_summary() const;

private:
    struct Layer {
        std::vector<std::vector<float>> weights;  // [out][in]
        std::vector<float>              bias;
        Activation                      activation;
    };

    size_t              input_size_;
    std::vector<Layer>  layers_;

    static float              activate    (float x, Activation act);
    static std::vector<float> activate_vec(const std::vector<float>& z, Activation act);
};

// -----------------------------------------------------------------------
// encode_input
//   Converts arbitrary string data into a normalised float vector.
//   Used to produce input tensors from node output text.
// -----------------------------------------------------------------------
std::vector<float> encode_input(const std::string& data, size_t dim = 16);

} // namespace openrai
