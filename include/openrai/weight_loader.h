#pragma once

#include <string>
#include <vector>
#include <memory>

namespace openrai {

// Layer metadata from manifest
struct LayerManifest {
    std::string name;
    int in_dim;
    int out_dim;
    std::string dtype;  // "fp32", "fp16", "int4", "int8"
    std::string file_path;
};

// Model manifest
struct ModelManifest {
    std::string model_name;
    std::string dtype;
    std::vector<LayerManifest> layers;
};

// Weight buffer (host-side)
struct WeightBuffer {
    std::string layer_name;
    void* data;          // Host pointer to weights
    size_t size_bytes;   // Total size in bytes
    std::string dtype;   // Data type
    int in_dim;
    int out_dim;

    WeightBuffer() : data(nullptr), size_bytes(0), in_dim(0), out_dim(0) {}
    ~WeightBuffer();

    // Create from float32 array
    static std::shared_ptr<WeightBuffer> create_from_float32(
        const std::string& name,
        const float* data,
        int in_dim,
        int out_dim);

    // Create from fp16 array
    static std::shared_ptr<WeightBuffer> create_from_fp16(
        const std::string& name,
        const uint16_t* data,
        int in_dim,
        int out_dim);

    // Helper: convert float32 to fp16 if needed
    static std::vector<uint16_t> float32_to_fp16(const float* data, size_t count);
};

// Weight Loader
class WeightLoader {
public:
    WeightLoader();
    ~WeightLoader();

    // Load weights from manifest
    // Returns vector of weight buffers (one per layer)
    std::vector<std::shared_ptr<WeightBuffer>> 
    load_weights_from_manifest(const std::string& manifest_path, 
                                bool prefer_fp16 = false);

    // Save weights to manifest + binary files
    bool save_weights_to_manifest(
        const std::string& manifest_path,
        const std::string& model_name,
        const std::vector<std::shared_ptr<WeightBuffer>>& weights);

    // Parse manifest JSON
    static ModelManifest parse_manifest(const std::string& manifest_path);

    // Load single layer binary
    static std::shared_ptr<WeightBuffer> 
    load_layer_binary(const LayerManifest& layer, bool convert_to_fp16 = false);

    // Save single layer binary
    static bool save_layer_binary(
        const std::string& file_path,
        const WeightBuffer& buffer);

private:
    std::string base_path_;  // Base directory for relative paths
};

} // namespace openrai
