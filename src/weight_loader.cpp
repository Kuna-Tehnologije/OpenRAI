#include "openrai/weight_loader.h"
#include "openrai/log.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstring>
#include <cstdint>

// Simple JSON parser (minimal for manifest)
#include <algorithm>

namespace openrai {

WeightBuffer::~WeightBuffer() {
    if (data) {
        free(data);
        data = nullptr;
    }
}

std::vector<uint16_t> WeightBuffer::float32_to_fp16(const float* data, size_t count) {
    std::vector<uint16_t> fp16(count);

    // Simple float32 ? fp16 conversion
    for (size_t i = 0; i < count; ++i) {
        float val = data[i];
        uint32_t f32;
        std::memcpy(&f32, &val, sizeof(float));

        // Extract sign, exponent, mantissa
        uint32_t sign = (f32 >> 31) & 0x1;
        int32_t exp = ((f32 >> 23) & 0xFF) - 127;
        uint32_t mant = f32 & 0x7FFFFF;

        uint16_t fp16_val;

        // Handle special cases
        if (exp > 15) {
            // Overflow ? infinity
            fp16_val = (sign << 15) | 0x7C00;
        } else if (exp < -14) {
            // Underflow ? zero
            fp16_val = (sign << 15);
        } else {
            // Normal case
            uint16_t fp16_exp = (exp + 15) & 0x1F;
            uint16_t fp16_mant = (mant >> 13) & 0x3FF;
            fp16_val = (sign << 15) | (fp16_exp << 10) | fp16_mant;
        }

        fp16[i] = fp16_val;
    }

    return fp16;
}

std::shared_ptr<WeightBuffer> WeightBuffer::create_from_float32(
    const std::string& name,
    const float* data,
    int in_dim,
    int out_dim) {

    auto buffer = std::make_shared<WeightBuffer>();
    buffer->layer_name = name;
    buffer->in_dim = in_dim;
    buffer->out_dim = out_dim;
    buffer->dtype = "fp32";

    size_t count = static_cast<size_t>(in_dim) * out_dim;
    buffer->size_bytes = count * sizeof(float);
    buffer->data = malloc(buffer->size_bytes);

    std::memcpy(buffer->data, data, buffer->size_bytes);

    return buffer;
}

std::shared_ptr<WeightBuffer> WeightBuffer::create_from_fp16(
    const std::string& name,
    const uint16_t* data,
    int in_dim,
    int out_dim) {

    auto buffer = std::make_shared<WeightBuffer>();
    buffer->layer_name = name;
    buffer->in_dim = in_dim;
    buffer->out_dim = out_dim;
    buffer->dtype = "fp16";

    size_t count = static_cast<size_t>(in_dim) * out_dim;
    buffer->size_bytes = count * sizeof(uint16_t);
    buffer->data = malloc(buffer->size_bytes);

    std::memcpy(buffer->data, data, buffer->size_bytes);

    return buffer;
}

WeightLoader::WeightLoader() {}

WeightLoader::~WeightLoader() {}

// Minimal JSON parser for manifest
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\"");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r\"");
    return str.substr(first, last - first + 1);
}

static std::string extract_value(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";

    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";

    size_t start = json.find_first_not_of(" \t\n\r:", pos + 1);
    if (start == std::string::npos) return "";

    char ch = json[start];
    if (ch == '"') {
        // String value
        size_t end = json.find('"', start + 1);
        return json.substr(start + 1, end - start - 1);
    } else if (ch == '[' || ch == '{') {
        // Object/array - skip for now
        return "";
    } else {
        // Number
        size_t end = json.find_first_of(",}\n", start);
        return trim(json.substr(start, end - start));
    }
}

ModelManifest WeightLoader::parse_manifest(const std::string& manifest_path) {
    ModelManifest manifest;

    std::ifstream file(manifest_path);
    if (!file.is_open()) {
        LOG("[WeightLoader] Failed to open manifest: " << manifest_path);
        return manifest;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    // Extract model_name and dtype
    manifest.model_name = extract_value(json, "model_name");
    manifest.dtype = extract_value(json, "dtype");

    // Extract layers array
    size_t layers_start = json.find("\"layers\"");
    if (layers_start == std::string::npos) {
        LOG("[WeightLoader] No layers found in manifest");
        return manifest;
    }

    size_t array_start = json.find('[', layers_start);
    size_t array_end = json.find(']', array_start);

    std::string layers_json = json.substr(array_start + 1, array_end - array_start - 1);

    // Parse each layer object
    size_t pos = 0;
    while (true) {
        size_t obj_start = layers_json.find('{', pos);
        if (obj_start == std::string::npos) break;

        size_t obj_end = layers_json.find('}', obj_start);
        std::string obj = layers_json.substr(obj_start, obj_end - obj_start + 1);

        LayerManifest layer;
        layer.name = extract_value(obj, "name");
        layer.in_dim = std::stoi(extract_value(obj, "in"));
        layer.out_dim = std::stoi(extract_value(obj, "out"));
        layer.dtype = extract_value(obj, "dtype");
        layer.file_path = extract_value(obj, "path");

        manifest.layers.push_back(layer);

        pos = obj_end + 1;
    }

    LOG("[WeightLoader] Parsed manifest: " << manifest.model_name 
        << " dtype=" << manifest.dtype 
        << " layers=" << manifest.layers.size());

    return manifest;
}

std::shared_ptr<WeightBuffer> WeightLoader::load_layer_binary(
    const LayerManifest& layer, bool convert_to_fp16) {

    auto buffer = std::make_shared<WeightBuffer>();
    buffer->layer_name = layer.name;
    buffer->in_dim = layer.in_dim;
    buffer->out_dim = layer.out_dim;

    // Open binary file
    std::ifstream file(layer.file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG("[WeightLoader] Failed to open layer file: " << layer.file_path);
        return nullptr;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read into temporary buffer
    std::vector<char> temp(file_size);
    file.read(temp.data(), file_size);
    file.close();

    // Determine if conversion is needed
    bool is_fp32 = (layer.dtype == "fp32");
    bool need_conversion = is_fp32 && convert_to_fp16;

    if (need_conversion) {
        // Convert float32 ? fp16
        size_t float_count = file_size / sizeof(float);
        auto fp16_data = WeightBuffer::float32_to_fp16(
            reinterpret_cast<const float*>(temp.data()), float_count);

        buffer->size_bytes = fp16_data.size() * sizeof(uint16_t);
        buffer->data = malloc(buffer->size_bytes);
        std::memcpy(buffer->data, fp16_data.data(), buffer->size_bytes);
        buffer->dtype = "fp16";

        LOG("[WeightLoader] Loaded & converted " << layer.name 
            << " fp32?fp16 size=" << buffer->size_bytes);
    } else {
        // Direct copy
        buffer->size_bytes = file_size;
        buffer->data = malloc(buffer->size_bytes);
        std::memcpy(buffer->data, temp.data(), file_size);
        buffer->dtype = layer.dtype;

        LOG("[WeightLoader] Loaded " << layer.name 
            << " dtype=" << layer.dtype 
            << " size=" << buffer->size_bytes);
    }

    return buffer;
}

std::vector<std::shared_ptr<WeightBuffer>> 
WeightLoader::load_weights_from_manifest(const std::string& manifest_path, bool prefer_fp16) {

    std::vector<std::shared_ptr<WeightBuffer>> buffers;

    // Parse manifest
    auto manifest = parse_manifest(manifest_path);
    if (manifest.layers.empty()) {
        LOG("[WeightLoader] No layers in manifest");
        return buffers;
    }

    // Store base path for relative paths
    std::filesystem::path manifest_file(manifest_path);
    base_path_ = manifest_file.parent_path().string();

    // Load each layer
    for (const auto& layer : manifest.layers) {
        // Resolve relative path
        std::filesystem::path layer_path(layer.file_path);
        if (layer_path.is_relative()) {
            layer_path = std::filesystem::path(base_path_) / layer_path;
        }

        LayerManifest resolved_layer = layer;
        resolved_layer.file_path = layer_path.string();

        auto buffer = load_layer_binary(resolved_layer, prefer_fp16);
        if (buffer) {
            buffers.push_back(buffer);
        }
    }

    LOG("[WeightLoader] Loaded " << buffers.size() << "/" << manifest.layers.size() << " layers");

    return buffers;
}

bool WeightLoader::save_layer_binary(const std::string& file_path, const WeightBuffer& buffer) {
    if (!buffer.data || buffer.size_bytes == 0) {
        LOG("[WeightLoader] Cannot save empty buffer: " << buffer.layer_name);
        return false;
    }

    // Create directory if needed
    std::filesystem::path path(file_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    // Write binary file
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG("[WeightLoader] Failed to open file for writing: " << file_path);
        return false;
    }

    file.write(reinterpret_cast<const char*>(buffer.data), buffer.size_bytes);
    file.close();

    LOG("[WeightLoader] Saved " << buffer.layer_name 
        << " to " << file_path 
        << " size=" << buffer.size_bytes 
        << " dtype=" << buffer.dtype);

    return true;
}

bool WeightLoader::save_weights_to_manifest(
    const std::string& manifest_path,
    const std::string& model_name,
    const std::vector<std::shared_ptr<WeightBuffer>>& weights) {

    if (weights.empty()) {
        LOG("[WeightLoader] No weights to save");
        return false;
    }

    // Determine base path for weights directory
    std::filesystem::path manifest_file(manifest_path);
    std::string weights_dir = manifest_file.parent_path().string() + "/weights";

    // Create weights directory
    std::filesystem::create_directories(weights_dir);

    // Build manifest JSON manually (simple approach)
    std::ostringstream json;
    json << "{\n";
    json << "  \"model_name\": \"" << model_name << "\",\n";

    // Determine common dtype
    std::string common_dtype = weights[0]->dtype;
    json << "  \"dtype\": \"" << common_dtype << "\",\n";

    json << "  \"layers\": [\n";

    // Save each layer
    for (size_t i = 0; i < weights.size(); ++i) {
        const auto& buffer = weights[i];

        // Generate filename
        std::string filename = buffer->layer_name + ".bin";
        std::string file_path = weights_dir + "/" + filename;

        // Save binary
        if (!save_layer_binary(file_path, *buffer)) {
            LOG("[WeightLoader] Failed to save layer: " << buffer->layer_name);
            continue;
        }

        // Add to manifest
        json << "    {\"name\":\"" << buffer->layer_name << "\",";
        json << "\"in\":" << buffer->in_dim << ",";
        json << "\"out\":" << buffer->out_dim << ",";
        json << "\"dtype\":\"" << buffer->dtype << "\",";
        json << "\"path\":\"weights/" << filename << "\"}";

        if (i < weights.size() - 1) {
            json << ",";
        }
        json << "\n";
    }

    json << "  ]\n";
    json << "}\n";

    // Write manifest JSON
    std::ofstream manifest_file_stream(manifest_path);
    if (!manifest_file_stream.is_open()) {
        LOG("[WeightLoader] Failed to open manifest for writing: " << manifest_path);
        return false;
    }

    manifest_file_stream << json.str();
    manifest_file_stream.close();

    LOG("[WeightLoader] Saved manifest: " << manifest_path 
        << " with " << weights.size() << " layers");

    return true;
}

} // namespace openrai
