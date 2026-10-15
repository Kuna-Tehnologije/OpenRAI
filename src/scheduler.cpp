#include <openrai/scheduler.h>

#include <algorithm>
#include <thread>

namespace openrai {

TickScheduler::TickScheduler(duration period, duration compute_budget)
    : period_(period), budget_(compute_budget)
{
    if (period_.count() <= 0) period_ = std::chrono::nanoseconds(127877238);   // 7.82 Hz
    if (budget_.count() <= 0 || budget_ > period_) budget_ = period_ * 8 / 10;
    start();
}

void TickScheduler::start() {
    anchor_ = clock::now();
    next_   = anchor_;
    index_  = 0;
    overruns_ = resyncs_ = 0;
    last_busy_ = worst_busy_ = total_busy_ = duration{0};
}

TickScheduler::Tick TickScheduler::begin_tick() {
    Tick t;
    t.index            = ++index_;
    tick_start_        = clock::now();
    t.start            = tick_start_;
    t.compute_deadline = tick_start_ + budget_;
    t.end              = next_ + period_;
    return t;
}

TickScheduler::duration TickScheduler::end_tick() {
    const time_point now = clock::now();
    last_busy_  = std::chrono::duration_cast<duration>(now - tick_start_);
    worst_busy_ = std::max(worst_busy_, last_busy_);
    total_busy_ += last_busy_;

    next_ += period_;
    duration overrun{0};
    if (now > next_) {
        ++overruns_;
        overrun = std::chrono::duration_cast<duration>(now - next_);
        if (overrun > period_) {
            // Hopelessly behind: re-anchor instead of bursting.
            ++resyncs_;
            next_ = now;
        }
        return overrun;
    }

    std::this_thread::sleep_until(next_);
    return overrun;
}

double TickScheduler::measured_hz() const {
    // (ticks - 1) boundaries between the first and the latest tick start
    if (index_ < 2) return 0.0;
    const auto elapsed = std::chrono::duration_cast<duration>(tick_start_ - anchor_).count();
    if (elapsed <= 0) return 0.0;
    return static_cast<double>(index_ - 1) * 1e9 / static_cast<double>(elapsed);
}

double TickScheduler::avg_busy_ms() const {
    if (index_ == 0) return 0.0;
    return static_cast<double>(total_busy_.count()) / 1e6 / static_cast<double>(index_);
}

} // namespace openrai
