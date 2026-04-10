#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }

    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

#endif // TIMER_H
