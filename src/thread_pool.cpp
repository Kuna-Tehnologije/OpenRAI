#include <openrai/thread_pool.h>
#include <openrai/log.h>

namespace openrai {

ThreadPool::ThreadPool(size_t num_threads) {
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
        workers_.emplace_back([this] { worker(); });
    LOG("[ThreadPool] Started with " << num_threads << " worker threads");
}

ThreadPool::~ThreadPool() {
    shutdown_ = true;
    condition_.notify_all();
    for (auto& w : workers_)
        if (w.joinable()) w.join();
    LOG("[ThreadPool] Shutdown complete");
}

void ThreadPool::enqueue(Task task) {
    { std::lock_guard<std::mutex> lock(queue_mutex_); tasks_.push(std::move(task)); }
    condition_.notify_one();
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    condition_.wait(lock, [this] { return tasks_.empty() && active_tasks_ == 0; });
}

size_t ThreadPool::active_tasks() const { return active_tasks_.load(); }

void ThreadPool::worker() {
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait(lock, [this] { return !tasks_.empty() || shutdown_; });
        if (shutdown_ && tasks_.empty()) break;
        if (!tasks_.empty()) {
            Task task = std::move(tasks_.front());
            tasks_.pop();
            ++active_tasks_;
            lock.unlock();
            try { task(); } catch (...) {}
            --active_tasks_;
            condition_.notify_all();
        }
    }
}

} // namespace openrai
