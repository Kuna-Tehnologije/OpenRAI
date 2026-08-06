#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <future>
#include <chrono>

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

    template<typename F, typename... Args>
    auto enqueue_with_timeout(std::chrono::milliseconds timeout, F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>;

private:
    std::vector<std::thread> workers_;
    std::queue<Task> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> shutdown_{false};
    std::atomic<size_t> active_tasks_{0};

    void worker();
};

template<typename F, typename... Args>
auto ThreadPool::enqueue_with_timeout(std::chrono::milliseconds timeout, F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type>
{
    using return_type = typename std::invoke_result<F, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> result = task->get_future();

    enqueue([task]() { (*task)(); });

    return result;
}

} // namespace openrai
