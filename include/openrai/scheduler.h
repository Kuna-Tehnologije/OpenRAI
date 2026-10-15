#pragma once

// -----------------------------------------------------------------------
// scheduler.h
//   TickScheduler - fixed-rate heartbeat of the engine (default 7.82 Hz,
//   i.e. one tick every 127.877 ms).
//
//   * drift-free: the next boundary is computed from the previous boundary,
//     not from "now", so jitter does not accumulate
//   * every tick exposes a compute deadline (start + compute_budget) that
//     the primary hands to the experts, and a hard end (start + period)
//   * overruns are counted; if the loop falls more than one full period
//     behind, the schedule is re-synchronised instead of trying to catch
//     up with a burst of back-to-back ticks
// -----------------------------------------------------------------------

#include <chrono>
#include <cstdint>

namespace openrai {

class TickScheduler {
public:
    using clock      = std::chrono::steady_clock;
    using time_point = clock::time_point;
    using duration   = std::chrono::nanoseconds;

    struct Tick {
        uint64_t   index = 0;
        time_point start;
        time_point compute_deadline;   // experts must answer by this point
        time_point end;                // next tick begins here
    };

    TickScheduler(duration period, duration compute_budget);

    void   start();                        // (re)anchor the schedule at now
    Tick   begin_tick();                   // call at the top of the loop
    // Call at the bottom of the loop.  Sleeps until the tick boundary.
    // Returns the overrun (zero when the tick finished in time).
    duration end_tick();

    duration period()         const { return period_; }
    duration compute_budget() const { return budget_; }
    double   target_hz()      const { return 1e9 / static_cast<double>(period_.count()); }
    double   measured_hz()    const;       // over the whole run
    uint64_t ticks()          const { return index_; }
    uint64_t overruns()       const { return overruns_; }
    uint64_t resyncs()        const { return resyncs_; }
    duration last_busy()      const { return last_busy_; }
    duration worst_busy()     const { return worst_busy_; }
    double   avg_busy_ms()    const;

private:
    duration   period_;
    duration   budget_;
    time_point anchor_;        // first tick start
    time_point next_;          // next boundary
    time_point tick_start_;
    uint64_t   index_     = 0;
    uint64_t   overruns_  = 0;
    uint64_t   resyncs_   = 0;
    duration   last_busy_{0};
    duration   worst_busy_{0};
    duration   total_busy_{0};
};

} // namespace openrai
