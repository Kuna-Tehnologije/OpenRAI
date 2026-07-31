#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

namespace openrai {

// Task type
using Task = std::function<void()>;

// Thread Pool za paralelnu obradu (10 radnih niti)
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = 10);
    ~ThreadPool();

    void enqueue(Task task);
    void wait_all();
    size_t active_tasks() const;

private:
    std::vector<std::thread> workers_;
    std::queue<Task> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> shutdown_{false};
    std::atomic<size_t> active_tasks_{0};

    void worker();
};

} // namespace openrai
