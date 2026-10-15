#pragma once

#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <ctime>

namespace openrai {

// -----------------------------------------------------------------------
// Logger
//   Thread-safe. Every entry goes to:
//     1. std::cout       (console, real-time)
//     2. openrai.log     (file, appended; each run gets a timestamp header)
// -----------------------------------------------------------------------
class Logger {
public:
    static Logger& get() {
        static Logger instance;
        return instance;
    }

    void write(const std::string& msg) {
        // Build timestamp  HH:MM:SS.mmm
        auto now  = std::chrono::system_clock::now();
        auto tt   = std::chrono::system_clock::to_time_t(now);
        auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) % 1000;
        std::tm tm_buf{};
#if defined(_WIN32)
        localtime_s(&tm_buf, &tt);
#else
        localtime_r(&tt, &tm_buf);
#endif
        std::ostringstream ts;
        ts << std::put_time(&tm_buf, "%H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << ms.count();

        const std::string line = "[" + ts.str() + "] " + msg;

        std::lock_guard<std::mutex> lk(mutex_);
        if (console_) std::cout << line << "\n";
        if (file_.is_open())
            file_ << line << "\n" << std::flush;
    }

    // Console echo on/off (the file log is always written).  Used by the
    // interactive console / GUI so answers are not buried in engine logs.
    void set_console(bool on) { std::lock_guard<std::mutex> lk(mutex_); console_ = on; }
    bool console() const { return console_; }

private:
    Logger() {
        file_.open("openrai.log", std::ios::app);
        if (file_.is_open()) {
            auto now  = std::chrono::system_clock::now();
            auto tt   = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf{};
#if defined(_WIN32)
            localtime_s(&tm_buf, &tt);
#else
            localtime_r(&tt, &tm_buf);
#endif
            file_ << "\n====== RUN START "
                  << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
                  << " ======\n" << std::flush;
        }
    }

    ~Logger() {
        if (file_.is_open()) {
            file_ << "====== RUN END ======\n\n";
        }
    }

    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;

    std::mutex    mutex_;
    std::ofstream file_;
    bool          console_ = true;
};

} // namespace openrai

// -----------------------------------------------------------------------
// LOG macro — assembles the full message, then calls Logger::write once
// -----------------------------------------------------------------------
#define LOG(msg) \
    do { \
        std::ostringstream _oss; \
        _oss << msg; \
        openrai::Logger::get().write(_oss.str()); \
    } while(0)
