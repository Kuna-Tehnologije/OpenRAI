#include <openrai/neural_net.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace openrai {

// -----------------------------------------------------------------------
// NeuralNetwork
// -----------------------------------------------------------------------

NeuralNetwork::NeuralNetwork(size_t input_size,
                             const std::vector<LayerConfig>& layer_configs,
                             uint32_t seed)
    : input_size_(input_size)
{
    if (layer_configs.empty())
        throw std::invalid_argument("[NN] Must have at least one layer.");

    std::mt19937 rng(seed);

    size_t prev = input_size;
    for (const auto& cfg : layer_configs) {
        Layer layer;
        layer.activation = cfg.activation;

        float limit = std::sqrt(6.0f / static_cast<float>(prev + cfg.size));
        std::uniform_real_distribution<float> dist(-limit, limit);

        layer.weights.resize(cfg.size, std::vector<float>(prev));
        layer.bias.resize(cfg.size, 0.0f);

        for (auto& row : layer.weights)
            for (auto& w : row)
                w = dist(rng);

        layers_.push_back(std::move(layer));
        prev = cfg.size;
    }
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) const {
    if (input.size() != input_size_)
        throw std::invalid_argument("[NN] Input size mismatch.");

    std::vector<float> activations = input;

    for (const auto& layer : layers_) {
        std::vector<float> z(layer.bias);

        for (size_t o = 0; o < layer.weights.size(); ++o)
            for (size_t i = 0; i < activations.size(); ++i)
                z[o] += layer.weights[o][i] * activations[i];

        activations = activate_vec(z, layer.activation);
    }

    return activations;
}

size_t NeuralNetwork::output_size() const {
    return layers_.empty() ? 0 : layers_.back().weights.size();
}

void NeuralNetwork::print_summary() const {
    std::cout << "[NN] Architecture  input=" << input_size_;
    size_t prev = input_size_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        size_t out = layers_[i].weights.size();
        std::cout << " -> " << out;
        prev = out;
    }
    std::cout << "  total_params=";
    size_t params = 0;
    size_t p = input_size_;
    for (const auto& l : layers_) {
        size_t o = l.weights.size();
        params += o * p + o;
        p = o;
    }
    std::cout << params << std::endl;
}

float NeuralNetwork::activate(float x, Activation act) {
    switch (act) {
        case Activation::ReLU:    return x > 0.0f ? x : 0.0f;
        case Activation::Sigmoid: return 1.0f / (1.0f + std::exp(-x));
        case Activation::Linear:  return x;
    }
    return x;
}

std::vector<float> NeuralNetwork::activate_vec(const std::vector<float>& z,
                                               Activation act) {
    std::vector<float> out(z.size());
    std::transform(z.begin(), z.end(), out.begin(),
                   [act](float v) { return activate(v, act); });
    return out;
}

std::vector<float> encode_input(const std::string& data, size_t dim) {
    std::vector<float> vec(dim, 0.0f);
    if (data.empty()) return vec;

    for (size_t i = 0; i < data.size(); ++i) {
        size_t bucket = i % dim;
        vec[bucket] += static_cast<float>(static_cast<unsigned char>(data[i]));
    }

    float maxv = *std::max_element(vec.begin(), vec.end());
    if (maxv > 0.0f)
        for (auto& v : vec)
            v = (v / maxv) * 2.0f - 1.0f;

    return vec;
}

} // namespace openrai
