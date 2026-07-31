#pragma once

#include <queue>
#include <mutex>
#include <semaphore>
#include <vector>
#include <string>
#include <functional>

namespace openrai {

// -----------------------------------------------------------------------
// GPUJob
//   type       : "inference" | "embedding" | "matrix_op"
//   input_data : raw string payload from node output
//   input_tensor: encoded float vector (filled before enqueue)
//   output_tensor: NN forward-pass result (filled after execution)
//     [0] relevance_score  (0..1)
//     [1] importance_score (0..1)
//     [2] should_output    (0..1, threshold 0.5)
//     [3] priority         (0..1)
// -----------------------------------------------------------------------
struct GPUJob {
    std::string        type;          // job type
    std::string        input_data;    // raw payload
    int                priority = 0;

    std::vector<float> input_tensor;  // encoded input (16 floats)
    std::vector<float> output_tensor; // NN inference result (4 floats)

    // Convenience accessors (valid after execution)
    float relevance()       const { return output_tensor.size() > 0 ? output_tensor[0] : 0.f; }
    float importance()      const { return output_tensor.size() > 1 ? output_tensor[1] : 0.f; }
    bool  should_output()   const { return output_tensor.size() > 2 && output_tensor[2] > 0.5f; }
    float job_priority()    const { return output_tensor.size() > 3 ? output_tensor[3] : 0.f; }
};

// -----------------------------------------------------------------------
// GPUJobQueue
//   Thread-safe FIFO queue with semaphore-bounded concurrency.
//   Execution is handled by a dedicated worker thread that owns the NN.
// -----------------------------------------------------------------------
class GPUJobQueue {
public:
    explicit GPUJobQueue(size_t max_concurrent_jobs = 10);

    void enqueue_job(const GPUJob& job);
    bool try_dequeue_job(GPUJob& job);
    size_t queue_size() const;

private:
    std::queue<GPUJob>       queue_;
    mutable std::mutex       mutex_;
    std::counting_semaphore<128> available_slots_;
    size_t                   max_concurrent_;
};

} // namespace openrai
