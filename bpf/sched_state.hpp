#pragma once
#include "sched_event.h"
#include <cstdint>
#include <cstddef>
#include <map>
#include <optional>
#include <string>

namespace reqscope::sched {
struct State {
    uint64_t since = 0, previous = 0;
    std::string state;
    unsigned waker = 0;
};
struct Interval { State state; uint64_t end = 0; };
enum class Result { applied, ignored, out_of_order, thread_limit, invalid_kind };
struct Transition {
    Result result = Result::applied;
    std::optional<Interval> closed;
    uint64_t previous_since = 0;
};

// Userspace state only. A wakeup_new begins a new TID lifetime. In particular,
// never close an earlier lifetime's last state using the new task's timestamp.
class StateMachine {
    std::map<unsigned, State> states_;
    size_t limit_;
public:
    explicit StateMachine(size_t limit = 8192) : limit_(limit) {}
    void clear() { states_.clear(); }
    Transition observe(const reqscope_sched_event& event) {
        if (event.kind < REQSCOPE_SWITCH_OUT || event.kind > REQSCOPE_WAKEUP_NEW)
            return {Result::invalid_kind, {}, 0};
        auto it = states_.find(event.tid);
        if (it != states_.end() && event.timestamp < it->second.since)
            return {Result::out_of_order, {}, it->second.since};
        if (event.kind == REQSCOPE_WAKEUP_NEW && it != states_.end()) {
            states_.erase(it);
            it = states_.end();
        }
        if (it == states_.end() && states_.size() >= limit_)
            return {Result::thread_limit, {}, 0};
        if (event.kind == REQSCOPE_WAKEUP && it != states_.end() && it->second.state == "on_cpu")
            return {Result::ignored, {}, 0};
        Transition result;
        if (it != states_.end()) result.closed = Interval{it->second, event.timestamp};
        states_[event.tid] = {event.timestamp, event.prev_state,
            event.kind == REQSCOPE_SWITCH_IN ? "on_cpu" :
            event.kind == REQSCOPE_WAKEUP || event.kind == REQSCOPE_WAKEUP_NEW || event.prev_state == 0
                ? "runnable" : "sleeping", event.waker_tid};
        return result;
    }
};
}
