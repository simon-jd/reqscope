// Which steps sampled mode traces: TraceSampler alone, no CUDA.
#include "collector/trace_sampler.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using reqscope::collector::SamplerConfig;
using reqscope::collector::TraceSampler;
using Reason = TraceSampler::Reason;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Pushes a step twice (execute_model and sample_tokens), as the plugin does.
Reason step(TraceSampler& sampler, uint64_t id) {
    const auto first = sampler.on_push(id);
    const auto second = sampler.on_push(id);
    check(first.new_step && !second.new_step, "second push of a step is not a new step");
    check(first.reason == second.reason, "a step keeps its decision across its two pushes");
    return first.reason;
}

void periodic() {
    TraceSampler sampler(SamplerConfig{4, 0, 32});
    int traced = 0;
    for (uint64_t id = 100; id < 200; ++id) {
        const auto reason = step(sampler, id);
        check(reason == ((id - 100) % 4 == 0 ? Reason::periodic : Reason::none), "every fourth step");
        traced += reason != Reason::none;
    }
    check(traced == 25, "25 of 100 steps traced");
    check(sampler.steps() == 100 && sampler.traced() == 25, "counts of steps and traced steps");
}

// The writer's forced flush waits until three steps were pushed after the
// last traced one.
void settled_after_traced_steps() {
    TraceSampler sampler(SamplerConfig{4, 0, 32});
    check(sampler.settled(3), "nothing traced yet");
    step(sampler, 1);               // traced (first push)
    check(!sampler.settled(3), "a traced step was just pushed");
    step(sampler, 2);
    step(sampler, 3);
    check(!sampler.settled(3), "two steps after it");
    step(sampler, 4);
    check(sampler.settled(3), "three steps after it");
    step(sampler, 5);               // traced again
    check(!sampler.settled(3), "the next traced step");
}

void nothing_without_rules() {
    TraceSampler sampler(SamplerConfig{0, 0, 32});
    for (uint64_t id = 1; id < 50; ++id) check(step(sampler, id) == Reason::none, "no rule, no trace");
}

// Steps of 5 ms; one 20 ms interval triggers a window of 8 steps that starts
// at the next new step the engine pushes.
void triggered() {
    TraceSampler sampler(SamplerConfig{0, 200, 8});
    uint64_t t = 1'000'000'000, id = 1;
    for (; id <= 40; ++id, t += 5'000'000) {
        check(step(sampler, id) == Reason::none, "no trigger in steady steps");
        check(!sampler.on_step(id, t, false), "steady intervals stay below the trigger");
    }
    // Step 41 starts 20 ms after step 40: step 40 was slow.
    check(step(sampler, id) == Reason::none, "step 41 was decided before its record arrived");
    t += 15'000'000;
    check(sampler.on_step(id, t, false), "a 4x interval triggers");
    ++id; t += 5'000'000;
    for (int k = 0; k < 8; ++k, ++id, t += 5'000'000) {
        check(step(sampler, id) == Reason::triggered, "the next 8 steps are traced");
        sampler.on_step(id, t, false);
    }
    check(step(sampler, id) == Reason::none, "the window ends after 8 steps");
    check(sampler.triggers() == 1, "one trigger counted");
}

// A slow step with prefill is interference: membership explains it, so it
// does not trigger tracing, and neither does the step after it, whose
// interval holds the wait for the prefill's output. Nor do intervals across
// a missing step id.
void prefill_and_gaps_do_not_trigger() {
    TraceSampler sampler(SamplerConfig{0, 200, 8});
    uint64_t t = 1'000'000'000, id = 1;
    for (; id <= 40; ++id, t += 5'000'000) sampler.on_step(id, t, false);
    sampler.on_step(id, t, true);   // step 41 has prefill and takes 30 ms
    ++id; t += 30'000'000;
    check(!sampler.on_step(id, t, false), "a slow prefill step does not trigger");
    ++id; t += 30'000'000;          // step 42 waited for step 41's output
    check(!sampler.on_step(id, t, false), "nor the step after a prefill");
    id += 2; t += 40'000'000;       // step 44 missing: no interval
    check(!sampler.on_step(id, t, false), "no interval across a missing step");
    ++id; t += 40'000'000;          // one step after the gap: not yet two decode steps
    check(!sampler.on_step(id, t, false), "a run restarts after a missing step");
    ++id; t += 40'000'000;
    check(sampler.on_step(id, t, false), "two decode steps in a row again: slow interval triggers");
    check(sampler.triggers() == 1, "one trigger counted");
}

void no_trigger_before_history() {
    TraceSampler sampler(SamplerConfig{0, 200, 8});
    uint64_t t = 0;
    for (uint64_t id = 1; id <= TraceSampler::kMinHistory; ++id, t += 5'000'000) sampler.on_step(id, t, false);
    t += 50'000'000;
    check(!sampler.on_step(TraceSampler::kMinHistory + 1, t, false), "too few intervals for a median");
}

// A trigger inside a window extends it from the step where it is seen.
void window_extends() {
    TraceSampler sampler(SamplerConfig{0, 200, 4});
    uint64_t t = 0, id = 1;
    for (; id <= 40; ++id, t += 5'000'000) sampler.on_step(id, t, false);
    t += 20'000'000;
    sampler.on_step(id, t, false);
    check(step(sampler, 50) == Reason::triggered, "window opens at the next push");
    check(step(sampler, 51) == Reason::triggered, "window continues");
    ++id; t += 20'000'000;
    sampler.on_step(id, t, false);
    for (uint64_t s = 52; s < 56; ++s) check(step(sampler, s) == Reason::triggered, "extended window");
    check(step(sampler, 56) == Reason::none, "extended window ends");
}

void both_rules() {
    TraceSampler sampler(SamplerConfig{10, 200, 3});
    uint64_t t = 0, id = 1;
    for (; id <= 40; ++id, t += 5'000'000) sampler.on_step(id, t, false);
    check(step(sampler, 1) == Reason::periodic, "first step periodic");
    check(step(sampler, 2) == Reason::none, "second not");
    t += 20'000'000;
    sampler.on_step(id, t, false);
    check(step(sampler, 3) == Reason::triggered, "trigger wins");
    check(step(sampler, 4) == Reason::triggered, "window");
    check(step(sampler, 5) == Reason::triggered, "window");
    check(step(sampler, 6) == Reason::none, "after window");
    for (uint64_t s = 7; s <= 10; ++s) check(step(sampler, s) == Reason::none, "until the tenth push");
    check(step(sampler, 11) == Reason::periodic, "eleventh push is periodic");
}
}  // namespace

int main() {
    try {
        periodic();
        settled_after_traced_steps();
        nothing_without_rules();
        triggered();
        prefill_and_gaps_do_not_trigger();
        no_trigger_before_history();
        window_extends();
        both_rules();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "trace sampler tests passed\n";
    return 0;
}
