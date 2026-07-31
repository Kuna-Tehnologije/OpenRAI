#include <openrai/gpu_queue.h>
#include <openrai/log.h>

namespace openrai {

GPUJobQueue::GPUJobQueue(size_t max_concurrent_jobs)
    : available_slots_(static_cast<std::ptrdiff_t>(max_concurrent_jobs)),
      max_concurrent_(max_concurrent_jobs) {
    LOG("[GPUQueue] Initialized  slots=" << max_concurrent_jobs);
}

void GPUJobQueue::enqueue_job(const GPUJob& job) {
    available_slots_.acquire();
    { std::lock_guard<std::mutex> lock(mutex_); queue_.push(job); }
    LOG("[GPUQueue] Enqueued  type=" << job.type << "  node=" << job.input_data);
}

bool GPUJobQueue::try_dequeue_job(GPUJob& job) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) return false;
    job = queue_.front();
    queue_.pop();
    available_slots_.release();
    return true;
}

size_t GPUJobQueue::queue_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

} // namespace openrai
