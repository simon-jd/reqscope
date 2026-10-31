#pragma once
// The clock probe's per-interval logic, separate from its thread and from the
// real CUDA driver so it can be driven with a fake driver in tests.
//
// Every interval the probe brackets a 4-byte host-to-device copy on its own
// stream between two host clock reads and emits a ClockProbe record. It waits
// for the application's primary context and never creates one. Any failure
// after that context exists stops the probe for good: it releases what it
// acquired and emits a Diagnostic with "stopped": true, because a probe that
// stops silently would leave the clock bound looking measured when it is not.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>

namespace reqscope::collector {

// The CUDA driver and CUPTI entry points the probe uses. Each returns 0 on
// success (CUDA_SUCCESS / CUPTI_SUCCESS).
struct ProbeDriver {
    int (*device_get)(int*, int) = nullptr;
    int (*primary_state)(int, unsigned*, int*) = nullptr;
    int (*primary_retain)(void**, int) = nullptr;
    int (*primary_release)(int) = nullptr;
    int (*set_current)(void*) = nullptr;
    int (*stream_create)(void**, unsigned) = nullptr;
    int (*stream_destroy)(void*) = nullptr;
    int (*host_alloc)(void**, size_t) = nullptr;
    int (*host_free)(void*) = nullptr;
    int (*device_alloc)(unsigned long long*, size_t) = nullptr;
    int (*device_free)(unsigned long long) = nullptr;
    int (*copy_async)(unsigned long long, const void*, size_t, void*) = nullptr;
    int (*synchronize)(void*) = nullptr;
    int (*exchange_capture_mode)(int*) = nullptr;
    int (*context_id)(void* context, uint32_t* id) = nullptr;
    int (*stream_id)(void* context, void* stream, uint32_t* id) = nullptr;
};

class ClockProbeCore {
public:
    using Emit = std::function<void(const std::string&)>;
    using Clock = uint64_t (*)();
    static constexpr unsigned report_wait_after = 20;

    ClockProbeCore(const ProbeDriver& driver, Emit emit, Clock now)
        : d(driver), emit(std::move(emit)), now(now) {}
    ~ClockProbeCore() { release(); }
    ClockProbeCore(const ClockProbeCore&) = delete;
    ClockProbeCore& operator=(const ClockProbeCore&) = delete;

    // One interval. Returns false once the probe has stopped.
    bool tick() {
        if (state == State::stopped) return false;
        if (state == State::waiting) {
            int device = 0, is_active = 0;
            unsigned flags = 0;
            if (d.device_get(&device, 0) || d.primary_state(device, &flags, &is_active) || !is_active) {
                // Before the application creates its context, waiting is normal.
                if (++waited == report_wait_after)
                    diagnose("no active primary context after 20 intervals; clock bound unknown until it appears",
                             false, 0);
                return true;
            }
            if (!prepare(device)) return false;
            state = State::running;
        }
        const uint64_t t0 = now();
        const int copied = d.copy_async(target, host, 4, stream);
        const int synced = copied ? 0 : d.synchronize(stream);
        const uint64_t t1 = now();
        if (copied) return stop("cuMemcpyHtoDAsync", copied);
        if (synced) return stop("cuStreamSynchronize", synced);
        char record[256];
        std::snprintf(record, sizeof record,
                      "{\"type\":\"ClockProbe\",\"probe\":%llu,\"t0\":%llu,\"t1\":%llu,\"ctx\":%u,\"stream\":%u}",
                      static_cast<unsigned long long>(++probes), static_cast<unsigned long long>(t0),
                      static_cast<unsigned long long>(t1), context_id, stream_id);
        emit(record);
        return true;
    }

    // Report that the driver entry points could not be found. The probe
    // stops; nothing was acquired.
    void unavailable() {
        if (state == State::stopped) return;
        state = State::stopped;
        diagnose("CUDA driver entry points not found; clock bound unknown", true, 0);
    }

    // Synchronize and free the stream, buffers and context reference. Safe
    // to call more than once.
    void release() {
        if (stream) {
            if (context) d.set_current(context);
            d.synchronize(stream);
            d.stream_destroy(stream);
            stream = nullptr;
        }
        if (target) { d.device_free(target); target = 0; }
        if (host) { d.host_free(host); host = nullptr; }
        if (retained) { d.primary_release(retained - 1); retained = 0; }
        context = nullptr;
    }

    bool stopped() const { return state == State::stopped; }
    uint64_t probe_count() const { return probes; }

private:
    enum class State { waiting, running, stopped };

    bool prepare(int device) {
        if (int error = d.primary_retain(&context, device)) { context = nullptr; return stop("cuDevicePrimaryCtxRetain", error); }
        retained = device + 1;
        if (int error = d.set_current(context)) return stop("cuCtxSetCurrent", error);
        // vLLM captures CUDA graphs in global capture mode, where an
        // allocation or synchronization on any other thread invalidates the
        // capture (cudaErrorStreamCaptureInvalidated at engine startup). In
        // relaxed mode this thread's calls are allowed and leave captures
        // alone; the probe never touches a capturing stream.
        int relaxed = 2;  // CU_STREAM_CAPTURE_MODE_RELAXED
        if (int error = d.exchange_capture_mode(&relaxed)) return stop("cuThreadExchangeStreamCaptureMode", error);
        if (int error = d.stream_create(&stream, 1 /* CU_STREAM_NON_BLOCKING */)) {
            stream = nullptr;
            return stop("cuStreamCreate", error);
        }
        if (int error = d.host_alloc(&host, 64)) { host = nullptr; return stop("cuMemAllocHost", error); }
        if (int error = d.device_alloc(&target, 64)) { target = 0; return stop("cuMemAlloc", error); }
        if (int error = d.context_id(context, &context_id)) return stop("cuptiGetContextId", error);
        if (int error = d.stream_id(context, stream, &stream_id)) return stop("cuptiGetStreamIdEx", error);
        return true;
    }

    bool stop(const char* call, int error) {
        state = State::stopped;
        release();
        diagnose((std::string(call) + " failed; probe stopped; clock bound unknown").c_str(), true, error);
        return false;
    }

    void diagnose(const char* reason, bool is_stopped, int error) {
        char record[384];
        std::snprintf(record, sizeof record,
                      "{\"type\":\"Diagnostic\",\"source\":\"clock_probe\",\"reason\":\"%s\",\"stopped\":%s,"
                      "\"error\":%d,\"probes\":%llu}",
                      reason, is_stopped ? "true" : "false", error, static_cast<unsigned long long>(probes));
        emit(record);
    }

    ProbeDriver d;
    Emit emit;
    Clock now;
    State state = State::waiting;
    unsigned waited = 0;
    uint64_t probes = 0;
    void* context = nullptr;
    void* stream = nullptr;
    void* host = nullptr;
    unsigned long long target = 0;
    int retained = 0;
    uint32_t context_id = 0, stream_id = 0;
};

}  // namespace reqscope::collector
