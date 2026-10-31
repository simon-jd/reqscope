#include "bpf/options.hpp"
#include "bpf/sched_state.hpp"
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, message);
}
reqscope_sched_event event(unsigned kind, uint64_t time, unsigned tid = 7, uint64_t previous = 0) {
    return {time, previous, 1, tid, 9, kind};
}
}
int main() {
    using namespace reqscope::sched;
    try {
        StateMachine machine;
        check(!machine.observe(event(REQSCOPE_SWITCH_IN, 10)).closed, "first event cannot close an interval");
        auto sleeping = machine.observe(event(REQSCOPE_SWITCH_OUT, 20, 7, 1));
        check(sleeping.closed && sleeping.closed->state.since == 10 && sleeping.closed->end == 20 &&
            sleeping.closed->state.state == "on_cpu", "running interval");
        auto new_task = machine.observe(event(REQSCOPE_WAKEUP_NEW, 100));
        check(!new_task.closed, "TID reuse must not close the old task's sleeping interval");
        auto new_cpu = machine.observe(event(REQSCOPE_SWITCH_IN, 110));
        check(new_cpu.closed && new_cpu.closed->state.since == 100 && new_cpu.closed->end == 110 &&
            new_cpu.closed->state.state == "runnable", "new lifetime starts at wakeup_new");
        check(machine.observe(event(REQSCOPE_WAKEUP, 115)).result == Result::ignored,
            "ordinary wakeup cannot split an on-cpu interval");
        check(machine.observe(event(REQSCOPE_WAKEUP_NEW, 100)).result == Result::out_of_order,
            "late wakeup_new cannot erase a newer observed state");
        auto out_of_order = machine.observe(event(REQSCOPE_SWITCH_OUT, 105));
        check(out_of_order.result == Result::out_of_order && out_of_order.previous_since == 110,
            "out-of-order event must not replace the latest state");
        auto retained = machine.observe(event(REQSCOPE_SWITCH_OUT, 120));
        check(retained.closed && retained.closed->state.since == 110, "out-of-order state preserved");
        machine.clear();
        check(!machine.observe(event(REQSCOPE_SWITCH_IN, 200)).closed,
            "loss reset must not connect intervals across missing events");
        StateMachine limited(1);
        limited.observe(event(REQSCOPE_SWITCH_IN, 1));
        check(limited.observe(event(REQSCOPE_SWITCH_IN, 2, 8)).result == Result::thread_limit, "bounded state map");
        check(limited.observe(event(REQSCOPE_WAKEUP_NEW, 3)).result == Result::applied,
            "existing TID replacement does not need another state slot");
        check(limited.observe(event(999, 4)).result == Result::invalid_kind, "unknown kind rejected");

        for (const auto* invalid : {"", "-1", "+1", " 1", "1 ", "1tail", "18446744073709551616"})
            rejects([&] { unsigned_option(invalid, UINT64_MAX, "test"); }, "strict unsigned parsing");
        rejects([] { unsigned_option("4294967297", UINT32_MAX, "test"); }, "unsigned narrowing overflow");
        check(unsigned_option("18446744073709551615", UINT64_MAX, "test") == UINT64_MAX,
            "uint64 maximum parses exactly");
        check(deadline_after_seconds(123, 0) == UINT64_MAX, "zero duration is unlimited");
        check(deadline_after_seconds(123, 1) == 1000000123ULL, "duration units");
        check(deadline_after_seconds(UINT64_MAX - 1000000000ULL, 1) == UINT64_MAX,
            "maximum representable deadline");
        rejects([] { deadline_after_seconds(UINT64_MAX - 999999999ULL, 1); }, "deadline addition overflow");
        rejects([] { deadline_after_seconds(0, UINT64_MAX / 1000000000ULL + 1); }, "duration multiplication overflow");
        std::cout << "scheduler state and numeric boundary regressions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
