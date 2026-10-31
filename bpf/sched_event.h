#pragma once
/* Shared wire layout; independent of libc and generated kernel headers. */
enum reqscope_sched_kind {
    REQSCOPE_SWITCH_OUT = 1, REQSCOPE_SWITCH_IN = 2, REQSCOPE_WAKEUP = 3,
    REQSCOPE_WAKEUP_NEW = 4
};
struct reqscope_sched_event {
    unsigned long long timestamp;
    unsigned long long prev_state;
    unsigned int tgid;
    unsigned int tid;
    unsigned int waker_tid;
    unsigned int kind;
};
