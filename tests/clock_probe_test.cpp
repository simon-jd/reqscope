// Drives ClockProbeCore with a fake CUDA driver: no GPU needed.
#include "collector/clock_probe.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using reqscope::collector::ClockProbeCore;
using reqscope::collector::ProbeDriver;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Fake driver state. A call named in fail_call returns error 700 once it is
// reached for the fail_on-th time (1-based); every other call succeeds.
struct Fake {
    bool context_active = true;
    std::string fail_call;
    int fail_on = 1;
    int seen = 0;
    int streams_created = 0, streams_destroyed = 0, host_allocs = 0, host_frees = 0;
    int device_allocs = 0, device_frees = 0, retains = 0, releases = 0, copies = 0;
} fake;

int fail(const char* call) {
    if (fake.fail_call != call) return 0;
    return ++fake.seen >= fake.fail_on ? 700 : 0;
}
int device_get(int* device, int) { *device = 0; return 0; }
int primary_state(int, unsigned* flags, int* active) { *flags = 0; *active = fake.context_active; return 0; }
int primary_retain(void** context, int) {
    if (int error = fail("retain")) return error;
    ++fake.retains; *context = reinterpret_cast<void*>(0x10); return 0;
}
int primary_release(int) { ++fake.releases; return 0; }
int set_current(void*) { return 0; }
int stream_create(void** stream, unsigned) {
    if (int error = fail("stream_create")) return error;
    ++fake.streams_created; *stream = reinterpret_cast<void*>(0x20); return 0;
}
int stream_destroy(void*) { ++fake.streams_destroyed; return 0; }
int host_alloc(void** host, size_t) {
    if (int error = fail("host_alloc")) return error;
    ++fake.host_allocs; *host = reinterpret_cast<void*>(0x30); return 0;
}
int host_free(void*) { ++fake.host_frees; return 0; }
int device_alloc(unsigned long long* target, size_t) {
    if (int error = fail("device_alloc")) return error;
    ++fake.device_allocs; *target = 0x40; return 0;
}
int device_free(unsigned long long) { ++fake.device_frees; return 0; }
int copy_async(unsigned long long target, const void* host, size_t, void* stream) {
    check(target && host && stream, "a copy was issued without a stream and both buffers");
    if (int error = fail("copy")) return error;
    ++fake.copies; return 0;
}
int synchronize(void*) { return fail("synchronize"); }
int exchange_capture_mode(int*) { return 0; }
int context_id(void*, uint32_t* id) { *id = 1; return 0; }
int stream_id(void*, void*, uint32_t* id) { *id = 7; return 0; }

uint64_t clock_value = 1000;
uint64_t now() { return clock_value += 10; }

ProbeDriver driver() {
    ProbeDriver d;
    d.device_get = device_get; d.primary_state = primary_state; d.primary_retain = primary_retain;
    d.primary_release = primary_release; d.set_current = set_current; d.stream_create = stream_create;
    d.stream_destroy = stream_destroy; d.host_alloc = host_alloc; d.host_free = host_free;
    d.device_alloc = device_alloc; d.device_free = device_free; d.copy_async = copy_async;
    d.synchronize = synchronize; d.exchange_capture_mode = exchange_capture_mode;
    d.context_id = context_id; d.stream_id = stream_id;
    return d;
}

struct Run {
    std::vector<std::string> records;
    ClockProbeCore core{driver(), [this](const std::string& r) { records.push_back(r); }, now};
    int count(const std::string& type) const {
        int n = 0;
        for (const auto& r : records) n += r.find("\"type\":\"" + type + "\"") != std::string::npos;
        return n;
    }
    bool has(const std::string& text) const {
        for (const auto& r : records) if (r.find(text) != std::string::npos) return true;
        return false;
    }
};

void balanced(const std::string& what) {
    check(fake.streams_created == fake.streams_destroyed, what + ": stream leaked");
    check(fake.host_allocs == fake.host_frees, what + ": host buffer leaked");
    check(fake.device_allocs == fake.device_frees, what + ": device buffer leaked");
    check(fake.retains == fake.releases, what + ": context reference leaked");
}

void probes_while_the_driver_works() {
    fake = Fake{};
    fake.context_active = false;
    Run run;
    for (int i = 0; i < 3; ++i) check(run.core.tick(), "waiting for a context is not a failure");
    check(run.records.empty(), "no records while the context does not exist");
    fake.context_active = true;
    for (int i = 0; i < 3; ++i) check(run.core.tick(), "a working probe keeps running");
    check(run.count("ClockProbe") == 3 && run.count("Diagnostic") == 0, "three probes, no diagnostic");
    check(run.has("\"ctx\":1,\"stream\":7"), "probe records carry CUPTI's context and stream ids");
    run.core.release();
    balanced("normal run");
}

void failed_allocation_never_makes_the_stream_ready() {
    for (const char* call : {"host_alloc", "device_alloc"}) {
        fake = Fake{};
        fake.fail_call = call;
        Run run;
        check(!run.core.tick(), std::string(call) + " failure must stop the probe");
        for (int i = 0; i < 5; ++i) check(!run.core.tick(), "a stopped probe stays stopped");
        check(fake.copies == 0 && run.count("ClockProbe") == 0,
              std::string("no copy may run after ") + call + " failed, although the stream exists");
        check(run.count("Diagnostic") == 1 && run.has("\"stopped\":true") && run.has("\"error\":700"),
              "exactly one stopping Diagnostic with the driver error");
        balanced(call);
    }
}

void a_probe_that_stops_mid_run_says_so() {
    for (const char* call : {"copy", "synchronize"}) {
        fake = Fake{};
        fake.fail_call = call;
        fake.fail_on = 3;
        Run run;
        check(run.core.tick() && run.core.tick(), "the first two probes work");
        check(!run.core.tick(), std::string(call) + " failure must stop the probe");
        check(!run.core.tick(), "a stopped probe stays stopped");
        check(run.count("ClockProbe") == 2, "only the two completed probes are recorded");
        check(run.count("Diagnostic") == 1 && run.has("\"stopped\":true") && run.has("\"probes\":2"),
              "the stopping Diagnostic says how many probes completed");
        balanced(call);
    }
}

void a_long_wait_is_reported_once_without_stopping() {
    fake = Fake{};
    fake.context_active = false;
    Run run;
    for (unsigned i = 0; i < ClockProbeCore::report_wait_after + 5; ++i) check(run.core.tick(), "still waiting");
    check(run.count("Diagnostic") == 1 && run.has("\"stopped\":false"), "one non-stopping wait Diagnostic");
    fake.context_active = true;
    check(run.core.tick() && run.count("ClockProbe") == 1, "the probe starts once the context appears");
}

void missing_driver_stops_with_a_diagnostic() {
    fake = Fake{};
    Run run;
    run.core.unavailable();
    check(run.core.stopped() && !run.core.tick(), "no driver means no probe");
    check(run.count("Diagnostic") == 1 && run.has("\"stopped\":true"), "and a stopping Diagnostic");
}
}  // namespace

int main() {
    try {
        probes_while_the_driver_works();
        failed_allocation_never_makes_the_stream_ready();
        a_probe_that_stops_mid_run_says_so();
        a_long_wait_is_reported_once_without_stopping();
        missing_driver_stops_with_a_diagnostic();
    } catch (const std::exception& error) {
        std::cerr << "clock probe test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "clock probe tests passed\n";
    return 0;
}
