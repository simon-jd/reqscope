#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace reqscope::collector {

// Which engine steps sampled mode traces kernel by kernel; the others get
// light mode's records without graph envelopes. Two rules, either or both:
// - periodic: one step in every `every` new steps;
// - triggered: when the interval from one step's start to the next reaches
//   `trigger_percent` of the median of recent such intervals, the next
//   `window` steps. Only intervals of a step that, like the step before it,
//   had no prefill count: under async scheduling a step's interval also
//   holds the wait for the previous step's output, so a prefill makes the
//   next interval long too.
// Kernel tracing stays on for one more step after each run of traced steps
// (the guard step, `kernels` without a reason): when CUPTI's kernel tracing
// is off at a graph's next launch, the records of its last traced launch get
// the end times of later launches of the same graph (docs/design.md,
// "Sampled tracing"). A guard step's own kernel records are not valid.
// on_push runs on the engine thread at every step push; a step pushes twice
// (execute_model, sample_tokens) with the same id and keeps its decision.
// on_step runs on the writer thread for each Step record (which arrives
// after the step ends) and only raises a request the engine thread picks up
// at its next new step. No locks, no allocation.
struct SamplerConfig {
    uint64_t every = 0;            // 0: no periodic steps
    uint64_t trigger_percent = 0;  // 0: no triggered windows
    uint64_t window = 32;          // steps traced after a trigger
};

class TraceSampler {
public:
    enum class Reason { none, periodic, triggered };
    // kernels: kernel tracing on for this step (a traced step or a guard step).
    struct Decision { Reason reason = Reason::none; bool new_step = false; bool kernels = false; };
    static constexpr size_t kHistory = 64;     // decode-only intervals in the median
    static constexpr size_t kMinHistory = 16;  // before any trigger

    explicit TraceSampler(SamplerConfig config = {}) : config_(config) {}
    // Before the first on_push or on_step.
    void configure(SamplerConfig config) { config_ = config; }
    const SamplerConfig& config() const { return config_; }

    // Engine thread.
    Decision on_push(uint64_t step_id) {
        if (started_ && step_id == last_step_) return {last_reason_, false, last_kernels_};
        started_ = true;
        last_step_ = step_id;
        const uint64_t requests = requests_.load(std::memory_order_acquire);
        if (requests != seen_requests_ && config_.window) {
            seen_requests_ = requests;
            window_end_ = step_id + config_.window;  // exclusive
            window_open_ = true;
        }
        Reason reason = Reason::none;
        if (window_open_ && step_id < window_end_) reason = Reason::triggered;
        else {
            window_open_ = false;
            if (config_.every && pushes_ % config_.every == 0) reason = Reason::periodic;
        }
        ++pushes_;
        const bool kernels = reason != Reason::none || last_reason_ != Reason::none;
        if (kernels && reason == Reason::none) guards_.fetch_add(1, std::memory_order_relaxed);
        last_reason_ = reason;
        last_kernels_ = kernels;
        if (reason != Reason::none) {
            last_traced_push_.store(pushes_, std::memory_order_relaxed);
            traced_.fetch_add(1, std::memory_order_relaxed);
        }
        pushed_.store(pushes_, std::memory_order_relaxed);
        return {reason, true, kernels};
    }
    // Any thread: at least `steps` new steps were pushed since the last
    // traced one (or none was ever traced), so its GPU work has finished
    // when at most steps - 1 steps run ahead of the GPU.
    bool settled(uint64_t steps) const {
        const uint64_t traced = last_traced_push_.load(std::memory_order_relaxed);
        return !traced || pushed_.load(std::memory_order_relaxed) >= traced + steps;
    }

    // Writer thread: a Step record's id, start time and whether any member
    // was in prefill. Returns true when it raised a trigger.
    bool on_step(uint64_t step_id, uint64_t t_begin, bool prefill) {
        bool raised = false;
        const bool consecutive = have_previous_ && step_id == previous_id_ + 1;
        if (consecutive && previous_decode_run_ >= 2 && t_begin > previous_begin_) {
            const uint64_t interval = t_begin - previous_begin_;
            if (config_.trigger_percent && count_ >= kMinHistory) {
                const uint64_t median = median_interval();
                if (median && interval * 100 >= median * config_.trigger_percent) {
                    requests_.fetch_add(1, std::memory_order_release);
                    ++triggers_;
                    raised = true;
                }
            }
            history_[next_] = interval;
            next_ = (next_ + 1) % kHistory;
            count_ = std::min(count_ + 1, kHistory);
        }
        // Consecutive steps without prefill, ending with this one.
        previous_decode_run_ = prefill ? 0 : consecutive ? previous_decode_run_ + 1 : 1;
        have_previous_ = true;
        previous_id_ = step_id;
        previous_begin_ = t_begin;
        return raised;
    }
    uint64_t triggers() const { return triggers_; }
    // Any thread: steps decided so far, how many of them were traced, and
    // how many were guard steps.
    uint64_t steps() const { return pushed_.load(std::memory_order_relaxed); }
    uint64_t traced() const { return traced_.load(std::memory_order_relaxed); }
    uint64_t guards() const { return guards_.load(std::memory_order_relaxed); }
    uint64_t median_interval() const {
        std::array<uint64_t, kHistory> copy{};
        std::copy(history_.begin(), history_.begin() + count_, copy.begin());
        auto middle = copy.begin() + count_ / 2;
        std::nth_element(copy.begin(), middle, copy.begin() + count_);
        return count_ ? *middle : 0;
    }

private:
    SamplerConfig config_;
    std::atomic<uint64_t> requests_{0}, pushed_{0}, last_traced_push_{0}, traced_{0}, guards_{0};
    // Engine thread only.
    bool started_ = false, window_open_ = false, last_kernels_ = false;
    uint64_t last_step_ = 0, pushes_ = 0, seen_requests_ = 0, window_end_ = 0;
    Reason last_reason_ = Reason::none;
    // Writer thread only.
    bool have_previous_ = false;
    uint64_t previous_id_ = 0, previous_begin_ = 0, previous_decode_run_ = 0, triggers_ = 0;
    std::array<uint64_t, kHistory> history_{};
    size_t next_ = 0, count_ = 0;
};

inline const char* reason_name(TraceSampler::Reason reason) {
    return reason == TraceSampler::Reason::periodic ? "periodic"
        : reason == TraceSampler::Reason::triggered ? "triggered" : "none";
}

}  // namespace reqscope::collector
