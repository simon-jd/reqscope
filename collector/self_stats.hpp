#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace reqscope::collector {
uint64_t monotonic_ns() noexcept;

// Where the collector's own host time goes. Each site counts its calls, the
// wall time spent in them and the longest call, and how many calls ran on an
// engine thread (one that has pushed a step id, i.e. launches the steps).
// Relaxed atomics only: safe and bounded inside CUPTI callbacks. The writer
// emits the totals as cumulative CollectorStats records.
enum class Site : unsigned {
    buffer_request,     // CUPTI asks for an activity buffer
    buffer_complete,    // CUPTI returns a filled activity buffer
    resource_callback,  // CUPTI graph resource callback
    emit_json,          // a plugin record through the C ABI
    push_step,          // CUPTI external correlation push
    pop_step,           // CUPTI external correlation pop
    process_buffer,     // writer: one CUPTI buffer into records, write_record included
    write_record,       // writer: frame one record into the segment buffer
    rotate_segment,     // writer: close a segment and enforce retention
    probe_tick,         // clock probe: one bracketed copy
    switch_granularity, // sampled mode: kernel tracing on or off at a step push
    sampled_flush,      // sampled mode: the writer's periodic forced CUPTI flush
    count };
inline constexpr const char* site_names[] = {"buffer_request", "buffer_complete", "resource_callback",
    "emit_json", "push_step", "pop_step", "process_buffer", "write_record", "rotate_segment", "probe_tick", "switch_granularity", "sampled_flush"};
static_assert(std::size(site_names) == static_cast<size_t>(Site::count));

struct SiteStats {
    std::atomic<uint64_t> calls{0}, ns{0}, max_ns{0}, engine_calls{0};
};
struct SelfStats {
    std::array<SiteStats, static_cast<size_t>(Site::count)> sites;
    std::atomic<uint64_t> timestamp_callbacks{0}, cupti_records{0}, cupti_bytes{0};
    std::atomic<uint64_t> probe_cpu_ns{0};
};
SelfStats& self_stats() noexcept;
// True on a thread that has pushed a step id.
bool on_engine_thread() noexcept;
void mark_engine_thread() noexcept;

inline void record_site(Site site, uint64_t ns) noexcept {
    auto& s = self_stats().sites[static_cast<size_t>(site)];
    s.calls.fetch_add(1, std::memory_order_relaxed);
    s.ns.fetch_add(ns, std::memory_order_relaxed);
    if (on_engine_thread()) s.engine_calls.fetch_add(1, std::memory_order_relaxed);
    uint64_t seen = s.max_ns.load(std::memory_order_relaxed);
    while (ns > seen && !s.max_ns.compare_exchange_weak(seen, ns, std::memory_order_relaxed)) {}
}
class Timed {
public:
    explicit Timed(Site site) noexcept : site_(site), begin_(monotonic_ns()) {}
    ~Timed() { record_site(site_, monotonic_ns() - begin_); }
    Timed(const Timed&) = delete;
    Timed& operator=(const Timed&) = delete;
private:
    Site site_;
    uint64_t begin_;
};
}
