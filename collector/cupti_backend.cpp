#include "collector/backend.hpp"
#include "collector/clock_probe.hpp"
#include "collector/record_text.hpp"
#include "collector/self_stats.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <pthread.h>
#include <time.h>
#include <filesystem>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

#ifdef REQSCOPE_HAS_CUPTI
#include <cupti.h>
#include <cupti_activity.h>
#include <cupti_driver_cbid.h>
#include <cupti_runtime_cbid.h>
#endif

extern "C" int reqscope_emit_json(const char* record);

namespace reqscope::collector {
namespace {
std::atomic<bool> active{false};
bool graph_mode = false;
thread_local unsigned correlation_depth = 0;
// Sampled mode: light mode's records without graph envelopes, with kernel
// tracing switched on for the steps the sampler picks (docs/design.md,
// "Sampled tracing"). Only the engine thread switches.
bool sampled_mode = false;
TraceSampler sampler;
#ifdef REQSCOPE_HAS_CUPTI
constexpr size_t buffer_size = 256 * 1024;
size_t buffer_count = 0;
struct alignas(16) Buffer {
    std::atomic<bool> used{false};
    alignas(16) std::array<uint8_t, buffer_size> data{};
};
// Allocate and initialize a bounded pool before registering any callbacks.
// The pointer and count remain immutable while callbacks can run.
// Intentionally process-lifetime, like the collector singleton. CUDA libraries
// can run late teardown callbacks after our atexit drain; there is no verified
// callback-unregister/quiescence contract that would make delete[] safe.
Buffer* buffers = nullptr;
bool kernels_on = false;    // sampled mode: kernel tracing is on now
CUpti_SubscriberHandle subscriber{};
bool subscribed = false;
std::vector<CUpti_ActivityKind> enabled;
struct EnabledApi { CUpti_CallbackDomain domain; CUpti_CallbackId cbid; };
std::vector<EnabledApi> enabled_apis;
CUptiResult enable_api(EnabledApi api, uint8_t enable) {
    return api.domain == CUPTI_CB_DOMAIN_DRIVER_API
        ? cuptiActivityEnableDriverApi(api.cbid, enable)
        : cuptiActivityEnableRuntimeApi(api.cbid, enable);
}
void disable_api_subset() noexcept {
    for (const auto api : enabled_apis)
        if (enable_api(api, 0) != CUPTI_SUCCESS) note_loss(LossReason::cupti_parse_error);
    enabled_apis.clear();
}
bool light_api_name(std::string_view name) {
    // Only workload submission, data movement and explicit synchronization.
    // Prefixes deliberately include ABI version/PTSZ/PTDS variants in the
    // supported 130202 API catalog. Query, allocation, getter, and graph
    // construction APIs are excluded; this is a native CUPTI filter, not a
    // writer-side discard. Enumerated names/CBIDs are persisted in the header.
    constexpr std::array prefixes = {
        "cuLaunch", "cudaLaunch", "cuGraphLaunch", "cudaGraphLaunch",
        "cuGraphUpload", "cudaGraphUpload", "cuMemcpy", "cudaMemcpy",
        "cuMemset", "cudaMemset", "cuCtxSynchronize", "cudaDeviceSynchronize",
        "cudaThreadSynchronize", "cuStreamSynchronize", "cudaStreamSynchronize",
        "cuEventSynchronize", "cudaEventSynchronize", "cuEventRecord", "cudaEventRecord",
        "cuStreamWaitEvent", "cudaStreamWaitEvent", "cuStreamWaitValue",
        "cuStreamWriteValue", "cuStreamBatchMemOp"};
    for (const char* prefix : prefixes) if (name.starts_with(prefix)) return true;
    return false;
}
void CUPTIAPI subscriber_callback(void*, CUpti_CallbackDomain domain,
                                 CUpti_CallbackId id, const void* data) {
    if (domain != CUPTI_CB_DOMAIN_RESOURCE || !data) return;
    Timed timed(Site::resource_callback);
    const auto* resource = static_cast<const CUpti_ResourceData*>(data);
    const auto* graph = static_cast<const CUpti_GraphData*>(resource->resourceDescriptor);
    if (!graph) { note_loss(LossReason::graph_metadata_error); return; }
    GraphResource event;
    event.action = id;
    event.timestamp = monotonic_ns();
    event.tid = static_cast<uint64_t>(syscall(SYS_gettid));
    event.node_type = static_cast<uint32_t>(graph->nodeType);
    auto check = [&event](CUptiResult result) {
        if (result != CUPTI_SUCCESS) event.ids_valid = false;
    };
    // These are CUPTI identity getters, as used by NVIDIA's resource callback
    // sample. No CUDA work, application locks, I/O, JSON or allocation here.
    if (resource->context) check(cuptiGetContextId(resource->context, &event.context));
    if (graph->graph) check(cuptiGetGraphId(graph->graph, &event.graph_id));
    if (graph->originalGraph) check(cuptiGetGraphId(graph->originalGraph, &event.original_graph_id));
    if (graph->graphExec) check(cuptiGetGraphExecId(graph->graphExec, &event.graph_exec_id));
    if (graph->node) check(cuptiGetGraphNodeId(graph->node, &event.node_id));
    if (graph->originalNode) check(cuptiGetGraphNodeId(graph->originalNode, &event.original_node_id));
    if (graph->dependency) check(cuptiGetGraphNodeId(graph->dependency, &event.dependency_node_id));
    if (!event.ids_valid) note_loss(LossReason::graph_metadata_error);
    if (!enqueue_graph_resource(event)) note_loss(LossReason::queue_full);
}
// Clock probe. CUPTI timestamps GPU activity in our CLOCK_MONOTONIC domain
// (timestamp_callback below), but nothing measures how far that mapping can be
// off. Every interval, a private thread brackets a 4-byte host-to-device copy
// on its own non-blocking stream between two monotonic reads:
//   t0 = now(); cuMemcpyHtoDAsync(...); cuStreamSynchronize(...); t1 = now();
// The copy's CUPTI [start, end] must then lie inside [t0, t1]. The analyzer
// pairs each ClockProbe record with that copy (same context and stream) and
// takes the worst bracket as the session's clock-error bound. The logic is in
// ClockProbeCore (clock_probe.hpp); this class owns the thread and binds the
// real driver. The collector starts inside CUDA's injection hook, before the
// application has a context, so the driver entry points are bound lazily on
// the thread.
int probe_context_id(void* context, uint32_t* id) {
    return cuptiGetContextId(reinterpret_cast<CUcontext>(context), id) == CUPTI_SUCCESS ? 0 : 1;
}
int probe_stream_id(void* context, void* stream, uint32_t* id) {
    return cuptiGetStreamIdEx(reinterpret_cast<CUcontext>(context), reinterpret_cast<CUstream>(stream), 0, id)
        == CUPTI_SUCCESS ? 0 : 1;
}
class ClockProbe {
public:
    void start(uint64_t interval_ms) {
        if (!interval_ms) return;
        interval = std::chrono::milliseconds(interval_ms);
        worker = std::thread([this] { run(); });
    }
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_all();
        if (worker.joinable()) worker.join();
    }
private:
    template <typename T> bool bind(T& target, const char* name) {
        target = reinterpret_cast<T>(dlsym(driver ? driver : RTLD_DEFAULT, name));
        return target != nullptr;
    }
    bool load(ProbeDriver& d) {
        // The CUDA runtime loads libcuda with RTLD_LOCAL, so its symbols are
        // not in the global scope. Take a handle to the copy already loaded.
        if (!driver) driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        d.context_id = probe_context_id;
        d.stream_id = probe_stream_id;
        return bind(d.device_get, "cuDeviceGet") && bind(d.primary_state, "cuDevicePrimaryCtxGetState") &&
               bind(d.primary_retain, "cuDevicePrimaryCtxRetain") &&
               bind(d.primary_release, "cuDevicePrimaryCtxRelease_v2") &&
               bind(d.set_current, "cuCtxSetCurrent") && bind(d.stream_create, "cuStreamCreate") &&
               bind(d.stream_destroy, "cuStreamDestroy_v2") && bind(d.host_alloc, "cuMemAllocHost_v2") &&
               bind(d.host_free, "cuMemFreeHost") && bind(d.device_alloc, "cuMemAlloc_v2") &&
               bind(d.device_free, "cuMemFree_v2") && bind(d.copy_async, "cuMemcpyHtoDAsync_v2") &&
               bind(d.synchronize, "cuStreamSynchronize") &&
               bind(d.exchange_capture_mode, "cuThreadExchangeStreamCaptureMode");
    }
    void run() {
        pthread_setname_np(pthread_self(), "rqs-probe");
        auto emit = [](const std::string& record) { reqscope_emit_json(record.c_str()); };
        ProbeDriver d;
        const bool loaded = load(d);
        ClockProbeCore core(d, emit, monotonic_ns);
        if (!loaded) core.unavailable();
        std::unique_lock<std::mutex> lock(mutex);
        while (!core.stopped() && !wake.wait_for(lock, interval, [this] { return stopping; })) {
            lock.unlock();
            {
                Timed timed(Site::probe_tick);
                core.tick();
            }
            timespec cpu{};
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu);
            self_stats().probe_cpu_ns.store(uint64_t(cpu.tv_sec) * 1000000000ULL + uint64_t(cpu.tv_nsec),
                                            std::memory_order_relaxed);
            lock.lock();
        }
        lock.unlock();
        core.release();
        if (driver) dlclose(driver);
        driver = nullptr;
    }
    std::chrono::milliseconds interval{500};
    void* driver = nullptr;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable wake;
    bool stopping = false;
};
ClockProbe clock_probe;

uint64_t CUPTIAPI timestamp_callback() {
    self_stats().timestamp_callbacks.fetch_add(1, std::memory_order_relaxed);
    return monotonic_ns();
}
void CUPTIAPI requested(uint8_t** buffer, size_t* size, size_t* max_records) {
    Timed timed(Site::buffer_request);
    *buffer = nullptr; *size = 0; *max_records = 0;
    for (size_t i = 0; i < buffer_count; ++i) {
        auto& candidate = buffers[i];
        bool available = false;
        if (candidate.used.compare_exchange_strong(available, true, std::memory_order_acquire)) {
            *buffer = candidate.data.data(); *size = candidate.data.size(); return;
        }
    }
    note_loss(LossReason::cupti_pool_exhausted);
}
void CUPTIAPI completed(CUcontext context, uint32_t stream, uint8_t* buffer,
                        size_t, size_t valid) {
    if (!buffer) return;
    Timed timed(Site::buffer_complete);
    if (!valid) { backend_release(buffer); return; }
    if (!enqueue_cupti_buffer(buffer, valid, reinterpret_cast<uintptr_t>(context), stream)) {
        note_loss(LossReason::cupti_completed_queue_full);
        backend_release(buffer);
    }
}
std::string result_name(CUptiResult result) {
    const char* name = nullptr;
    if (cuptiGetResultString(result, &name) != CUPTI_SUCCESS || !name) return std::to_string(int(result));
    return name;
}
#endif
}

Json graph_resource_json(const GraphResource& event) {
    // Numeric callback IDs are part of the exact supported 130202 ABI.
    const char* action = "unknown";
    switch (event.action) {
        case 9: action = "created"; break;
        case 10: action = "destroy"; break;
        case 11: action = "clone"; break;
        case 13: action = "node_created"; break;
        case 14: action = "node_destroy"; break;
        case 15: action = "dependency_created"; break;
        case 16: action = "dependency_destroy"; break;
        case 17: action = "exec_create_start"; break;
        case 18: action = "exec_created"; break;
        case 19: action = "exec_destroy"; break;
        case 20: action = "node_clone"; break;
        case 22: action = "node_updated"; break;
        case 23: action = "node_set_params"; break;
    }
    return Json::object({{"type", "GraphResource"}, {"action", action},
        {"graph_id", uint64_t(event.graph_id)}, {"original_graph_id", uint64_t(event.original_graph_id)},
        {"graph_exec_id", uint64_t(event.graph_exec_id)}, {"node_id", event.node_id},
        {"original_node_id", event.original_node_id}, {"dependency_node_id", event.dependency_node_id},
        {"node_type", uint64_t(event.node_type)}, {"ctx", uint64_t(event.context)},
        {"tid", event.tid}, {"timestamp", event.timestamp}, {"ids_valid", event.ids_valid}});
}

BackendInfo backend_start(const std::string& mode, const SamplerConfig& sampling) {
    BackendInfo info;
    graph_mode = mode == "light";
    sampled_mode = mode == "sampled";
    if (sampled_mode) sampler.configure(sampling);
    if (mode == "synthetic" || mode == "manual" || mode == "plugin") {
        info.status = "manual"; return info;
    }
#ifdef REQSCOPE_HAS_CUPTI
    Dl_info loaded{};
    if (dladdr(reinterpret_cast<void*>(&cuptiGetVersion), &loaded) && loaded.dli_fname) {
        std::error_code error;
        auto canonical=std::filesystem::canonical(loaded.dli_fname,error);
        info.path = error ? loaded.dli_fname : canonical.string();
    }
    cuptiGetVersion(&info.version);
    info.v2_symbols = dlsym(RTLD_DEFAULT, "cuptiSubscribe_v2") &&
                      dlsym(RTLD_DEFAULT, "cuptiActivityEnable_v2");
    // The compiled Activity record layouts are those of the selected toolkit.
    // Never label dlsym availability as verified multi-subscriber support.
    // Kernel11/Memcpy6/Memset4/GraphTrace2 below are verified for this exact
    // supported fallback. New headers do not make older record casts safe.
    if (info.version != CUPTI_API_VERSION || CUPTI_API_VERSION != 130202) {
        info.status="unavailable";
        info.error="CUPTI record ABI unsupported (supported=130202): compiled="+std::to_string(CUPTI_API_VERSION)+
                   " loaded="+std::to_string(info.version);
        return info;
    }
    buffer_count = 64;
    if (const char* value = std::getenv("REQSCOPE_CUPTI_BUFFERS")) {
        char* end = nullptr;
        const auto count = std::strtoull(value, &end, 10);
        if (end == value || *end != '\0' || count < 16 || count > 1024) {
            info.status = "unavailable";
            info.error = "REQSCOPE_CUPTI_BUFFERS must be an integer in [16, 1024]";
            return info;
        }
        buffer_count = static_cast<size_t>(count);
    }
    buffers = new Buffer[buffer_count];
    info.host_buffer_count = buffer_count;
    info.host_buffer_bytes = buffer_count * buffer_size;
    auto fail = [&](const std::string& operation, CUptiResult result) {
        info.status = "unavailable"; info.error = operation + ": " + result_name(result);
        disable_api_subset();
        for (auto kind : enabled) cuptiActivityDisable(kind);
        enabled.clear();
        if (subscribed) { cuptiUnsubscribe(subscriber); subscribed = false; }
        info.subscription = "none";
        active.store(false);
        return info;
    };
    CUptiResult result = cuptiSubscribe(&subscriber, subscriber_callback, nullptr);
    if (result != CUPTI_SUCCESS) return fail("cuptiSubscribe (exclusive V1 required)", result);
    subscribed = true;
    info.subscription = "V1-exclusive";
    const std::array<CUpti_CallbackIdResource, 13> graph_callbacks = {
        CUPTI_CBID_RESOURCE_GRAPH_CREATED, CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING,
        CUPTI_CBID_RESOURCE_GRAPH_CLONED, CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED,
        CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING, CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED,
        CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING, CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING,
        CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED, CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING,
        CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED, CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED,
        CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS};
    const char* graph_resources = std::getenv("REQSCOPE_GRAPH_RESOURCES");
    if (graph_resources && std::strcmp(graph_resources, "1") == 0) {
        for (auto callback : graph_callbacks) {
            result = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE, callback);
            if (result != CUPTI_SUCCESS) return fail("cuptiEnableCallback graph resource", result);
        }
        info.graph_resources = true;
    }
    result = cuptiActivityRegisterTimestampCallback(timestamp_callback);
    if (result != CUPTI_SUCCESS) return fail("cuptiActivityRegisterTimestampCallback", result);
    result = cuptiSetThreadIdType(CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM);
    if (result != CUPTI_SUCCESS) return fail("cuptiSetThreadIdType SYSTEM (SYS_gettid)", result);
    result = cuptiActivityRegisterCallbacks(requested, completed);
    if (result != CUPTI_SUCCESS) return fail("cuptiActivityRegisterCallbacks", result);
    // CUPTI's device buffers hold GPU timestamps until its worker thread
    // turns them into activity records. With the default 3.2 MB buffer
    // (about 100,000 records, a few hundred decode steps), moving to the next
    // one stalled the launching thread for 1-2 ms: bench/launch_cost.py saw
    // 21 such stalls in 5,000 steps, and 1 with 32 MB buffers. Four are
    // allocated up front so that none is allocated inside a launch.
    // REQSCOPE_CUPTI_DEVICE_BUFFER_MB (1-1024) changes the size.
    {
        size_t megabytes = 32;
        if (const char* value = std::getenv("REQSCOPE_CUPTI_DEVICE_BUFFER_MB")) {
            char* end = nullptr;
            megabytes = std::strtoull(value, &end, 10);
            if (end == value || *end != '\0' || !megabytes || megabytes > 1024)
                return fail("REQSCOPE_CUPTI_DEVICE_BUFFER_MB must be an integer in [1, 1024]",
                            CUPTI_ERROR_INVALID_PARAMETER);
        }
        size_t bytes = megabytes << 20, preallocated = 4, size = sizeof(size_t);
        // CUPTI requires fewer preallocated buffers than the pool limit.
        if (const char* limit = std::getenv("REQSCOPE_CUPTI_DEVICE_BUFFERS"))
            preallocated = std::min<size_t>(preallocated, std::max<size_t>(std::strtoull(limit, nullptr, 10), 1) - 1);
        result = cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &size, &bytes);
        if (result != CUPTI_SUCCESS) return fail("CUPTI device buffer size", result);
        result = cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE, &size, &preallocated);
        if (result != CUPTI_SUCCESS) return fail("CUPTI device buffer preallocation", result);
        info.device_buffer_bytes = bytes;
        info.device_buffers_preallocated = preallocated;
    }
    // CUPTI collects GPU timestamps into a pool of device buffers per context
    // (default limit 250, allocated on demand). A small pool cannot keep up
    // with vLLM's launch rate: with 4 buffers, 14% of kernel records and most
    // copy records arrived with zero timestamps. Keep CUPTI's default unless
    // REQSCOPE_CUPTI_DEVICE_BUFFERS caps device memory explicitly.
    if (const char* value = std::getenv("REQSCOPE_CUPTI_DEVICE_BUFFERS")) {
        size_t device_pool_limit = std::strtoull(value, nullptr, 10), parameter_size = sizeof(device_pool_limit);
        if (!device_pool_limit) return fail("REQSCOPE_CUPTI_DEVICE_BUFFERS must be a positive integer", CUPTI_ERROR_INVALID_PARAMETER);
        result = cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                                          &parameter_size, &device_pool_limit);
        if (result != CUPTI_SUCCESS) return fail("CUPTI device buffer pool bound", result);
    }
    const std::array<CUpti_ActivityKind, 8> kinds = {
        CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_MEMSET, CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, CUPTI_ACTIVITY_KIND_CUDA_EVENT,
        mode == "trace" ? CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL : CUPTI_ACTIVITY_KIND_GRAPH_TRACE};
    // Both modes record only the APIs that submit work, move data or
    // synchronize, enabled one callback ID at a time. Query APIs that torch
    // and Triton issue around every launch (cudaGetDevice,
    // cudaDeviceGetAttribute, cuOccupancy*) were most of trace mode's records
    // and pushed it past what the writer can drain. REQSCOPE_ALL_APIS=1
    // records every driver and runtime API instead.
    const char* every_api = std::getenv("REQSCOPE_ALL_APIS");
    const bool api_subset = !(every_api && std::strcmp(every_api, "1") == 0);
    for (const auto kind : kinds) {
        if (api_subset && (kind == CUPTI_ACTIVITY_KIND_DRIVER || kind == CUPTI_ACTIVITY_KIND_RUNTIME))
            continue;
        // Kernel tracing in sampled mode is switched per step, and graph
        // envelopes are not recorded at all: once kernel tracing had been on,
        // CUPTI gave vLLM's graph launches no graph records again.
        if (sampled_mode && kind == CUPTI_ACTIVITY_KIND_GRAPH_TRACE) continue;
        result = cuptiActivityEnable(kind);
        if (result != CUPTI_SUCCESS) return fail("cuptiActivityEnable kind=" + std::to_string(int(kind)), result);
        enabled.push_back(kind);
    }
    if (api_subset) {
        info.api_capture_policy = "cupti-native-whitelist-v1";
        constexpr std::array domains = {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_CB_DOMAIN_RUNTIME_API};
        for (auto domain : domains) {
            const uint32_t size = domain == CUPTI_CB_DOMAIN_DRIVER_API
                ? uint32_t(CUPTI_DRIVER_TRACE_CBID_SIZE) : uint32_t(CUPTI_RUNTIME_TRACE_CBID_SIZE);
            bool graph_launch_enabled = false;
            for (uint32_t cbid = 1; cbid < size; ++cbid) {
                const char* name = nullptr;
                if (cuptiGetCallbackName(domain, cbid, &name) != CUPTI_SUCCESS || !name || !light_api_name(name))
                    continue;
                const EnabledApi api{domain, cbid};
                result = enable_api(api, 1);
                if (result != CUPTI_SUCCESS)
                    return fail("CUPTI light API enable " + std::string(name), result);
                enabled_apis.push_back(api);
                info.api_capture_apis.push_back(Json::object({
                    {"domain", domain == CUPTI_CB_DOMAIN_DRIVER_API ? "driver" : "runtime"},
                    {"cbid", uint64_t(cbid)}, {"name", name}}));
                if (std::string_view(name).starts_with(domain == CUPTI_CB_DOMAIN_DRIVER_API
                        ? "cuGraphLaunch" : "cudaGraphLaunch")) graph_launch_enabled = true;
            }
            if (!graph_launch_enabled)
                return fail("CUPTI light API catalog missing graph launch", CUPTI_ERROR_NOT_SUPPORTED);
        }
    } else {
        info.api_capture_policy = "all-driver-runtime";
    }
    active.store(true);
    info.active = true; info.status = "capturing";
    if (const char* value = std::getenv("REQSCOPE_CLOCK_PROBE_MS"); !value || std::strtoull(value, nullptr, 10))
        clock_probe.start(value ? std::strtoull(value, nullptr, 10) : 500);
#else
    info.status = "unavailable";
    info.error = "collector built without CUPTI; only manual/synthetic capture is available";
#endif
    return info;
}
void backend_flush() noexcept {
#ifdef REQSCOPE_HAS_CUPTI
    if (active.load()) {
        if (cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) != CUPTI_SUCCESS)
            note_loss(LossReason::cupti_parse_error);
    }
#endif
}
// Sampled mode, writer thread. With kernel tracing switched on and off at
// run time, CUPTI holds every activity buffer of vLLM's engine until a
// forced flush (a default flush returns none of them), so the writer forces
// one periodically. A forced flush also returns records of GPU work still
// running, without timestamps; to keep traced steps whole, it waits until
// the engine has pushed kSettleSteps steps since the last traced one, or
// until it has waited max_wait_ns.
constexpr uint64_t kSettleSteps = 3;
bool backend_periodic_flush(uint64_t waited_ns, uint64_t max_wait_ns) noexcept {
#ifdef REQSCOPE_HAS_CUPTI
    if (!sampled_mode || !active.load()) return false;
    if (!sampler.settled(kSettleSteps) && waited_ns < max_wait_ns) return false;
    Timed timed(Site::sampled_flush);
    if (cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) != CUPTI_SUCCESS)
        note_loss(LossReason::cupti_parse_error);
    return true;
#else
    (void)waited_ns; (void)max_wait_ns;
    return false;
#endif
}
void backend_stop() noexcept {
#ifdef REQSCOPE_HAS_CUPTI
    clock_probe.stop();      // its last copy must be flushed with everything else
    if (active.load()) {
        backend_flush();
        disable_api_subset();
        for (auto kind : enabled) cuptiActivityDisable(kind);
        if (kernels_on) cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        backend_flush();
        active.store(false);
        enabled.clear();
    }
    if (subscribed) { cuptiUnsubscribe(subscriber); subscribed = false; }
#endif
}
#ifdef REQSCOPE_HAS_CUPTI
// Sampled mode: every kernel, or none, for the launches that follow until
// the next switch. CUPTI decides at launch, so switching between steps on the
// launching thread gives each step one granularity.
bool switch_kernels(bool on) noexcept {
    if (on == kernels_on) return true;
    Timed timed(Site::switch_granularity);
    const auto result = on ? cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
                           : cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (result != CUPTI_SUCCESS) return false;
    kernels_on = on;
    return true;
}
// The first push of each step id decides its granularity. A traced step is
// named by a TracedStep record, written only once tracing is on: a step
// without one was not traced (or its record was lost, and the analyzer
// treats it as not traced).
void sample(uint64_t step) noexcept {
    const auto decision = sampler.on_push(step);
    if (!decision.new_step) return;
    const bool traced = decision.reason != TraceSampler::Reason::none;
    if (!switch_kernels(traced)) { note_loss(LossReason::sampling_switch_error); return; }
    if (!traced) return;
    char text[128];
    std::snprintf(text, sizeof text, "{\"type\":\"TracedStep\",\"step_id\":%llu,\"reason\":\"%s\"}",
                  static_cast<unsigned long long>(step), reason_name(decision.reason));
    reqscope_emit_json(text);
}
#endif
void backend_step_record(uint64_t step, uint64_t t_begin, bool prefill) noexcept {
    if (sampled_mode) sampler.on_step(step, t_begin, prefill);
}
SamplingCounts backend_sampling_counts() noexcept {
    if (!sampled_mode) return {};
    return {sampler.steps(), sampler.traced(), sampler.triggers()};
}
int backend_push(uint64_t step) noexcept {
    mark_engine_thread();
#ifdef REQSCOPE_HAS_CUPTI
    if (sampled_mode && active.load()) sample(step);
#endif
    Timed timed(Site::push_step);
#ifdef REQSCOPE_HAS_CUPTI
    if (active.load() && cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, step) != CUPTI_SUCCESS) {
        note_loss(LossReason::correlation_error); return -1;
    }
#else
    (void)step;
#endif
    ++correlation_depth;
    return 0;
}
int backend_pop() noexcept {
    Timed timed(Site::pop_step);
    if (!correlation_depth) { note_loss(LossReason::correlation_error); return -1; }
#ifdef REQSCOPE_HAS_CUPTI
    if (active.load()) {
        uint64_t previous = 0;
        if (cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &previous) != CUPTI_SUCCESS) {
            note_loss(LossReason::correlation_error); return -1;
        }
    }
#endif
    --correlation_depth;
    return 0;
}
void backend_release(uint8_t* buffer) noexcept {
#ifdef REQSCOPE_HAS_CUPTI
    for (size_t i = 0; i < buffer_count; ++i) {
        auto& candidate = buffers[i];
        if (candidate.data.data() == buffer) {
            candidate.used.store(false, std::memory_order_release); return;
        }
    }
#else
    (void)buffer;
#endif
}
void backend_process(uint8_t* buffer, size_t valid, uintptr_t context, uint32_t stream,
                     const std::function<void(std::string&)>& emit) {
#ifdef REQSCOPE_HAS_CUPTI
    // One string, reused for every record of every buffer on the writer thread.
    static thread_local std::string text;
    auto gpu = [](const char* kind, uint32_t corr, uint32_t context, uint32_t stream, uint64_t start,
                  uint64_t end, uint64_t graph, uint64_t node, uint64_t bytes) {
        RecordText record(text, "GpuActivity");
        record.text("kind", kind).number("corr", corr).number("ctx", context).number("stream", stream)
            .number("start", start).number("end", end).number("graph_id", graph)
            .number("graph_node_id", node).number("bytes", bytes);
        return record;
    };
    CUpti_Activity* activity = nullptr;
    for (;;) {
        const auto result = cuptiActivityGetNextRecord(buffer, valid, &activity);
        if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        if (result != CUPTI_SUCCESS) { note_loss(LossReason::cupti_parse_error); break; }
        self_stats().cupti_records.fetch_add(1, std::memory_order_relaxed);
        switch (activity->kind) {
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
            const auto* api = reinterpret_cast<const CUpti_ActivityAPI*>(activity);
            const bool driver = activity->kind == CUPTI_ACTIVITY_KIND_DRIVER;
            const char* name = nullptr;
            cuptiGetCallbackName(driver ? CUPTI_CB_DOMAIN_DRIVER_API : CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &name);
            RecordText(text, "ApiCall").number("corr", api->correlationId).number("cbid", api->cbid)
                .number("tid", api->threadId).number("start", api->start).number("end", api->end)
                .text("name", name ? name : "unknown").text("tid_domain", "system")
                .text("domain", driver ? "driver" : "runtime").number("return_value", uint64_t(api->returnValue));
            emit(text);
            break;
        }
        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
            const auto* external = reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(activity);
            if (external->externalKind == CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) {
                RecordText(text, "ExtCorr").number("corr", external->correlationId)
                    .number("step_id", external->externalId);
                emit(text);
            }
            break;
        }
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            const auto* kernel = reinterpret_cast<const CUpti_ActivityKernel11*>(activity);
            gpu("kernel", kernel->correlationId, kernel->contextId, kernel->streamId, kernel->start, kernel->end,
                kernel->graphId, kernel->graphNodeId, 0)
                .text("name", kernel->name ? kernel->name : "kernel").number("device", kernel->deviceId)
                .boolean("is_device_launched", kernel->isDeviceLaunched);
            emit(text);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            const auto* copy = reinterpret_cast<const CUpti_ActivityMemcpy6*>(activity);
            if (graph_mode && copy->graphId) break;
            gpu("memcpy", copy->correlationId, copy->contextId, copy->streamId, copy->start, copy->end,
                copy->graphId, copy->graphNodeId, copy->bytes)
                .number("copy_kind", copy->copyKind).number("runtime_corr", copy->runtimeCorrelationId);
            emit(text);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            const auto* set = reinterpret_cast<const CUpti_ActivityMemset4*>(activity);
            if (graph_mode && set->graphId) break;
            gpu("memset", set->correlationId, set->contextId, set->streamId, set->start, set->end,
                set->graphId, set->graphNodeId, set->bytes);
            emit(text);
            break;
        }
        case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
            const auto* sync = reinterpret_cast<const CUpti_ActivitySynchronization2*>(activity);
            RecordText(text, "Sync").number("corr", sync->correlationId).number("sync_type", uint64_t(sync->type))
                .number("ctx", sync->contextId).number("stream", sync->streamId).number("event", sync->cudaEventId)
                .number("event_sync_id", sync->cudaEventSyncId).number("start", sync->start).number("end", sync->end)
                .number("return_value", uint64_t(sync->returnValue));
            emit(text);
            break;
        }
        case CUPTI_ACTIVITY_KIND_CUDA_EVENT: {
            const auto* event = reinterpret_cast<const CUpti_ActivityCudaEvent2*>(activity);
            RecordText(text, "CudaEvent").number("corr", event->correlationId).number("ctx", event->contextId)
                .number("stream", event->streamId).number("event", event->eventId)
                .number("event_sync_id", event->cudaEventSyncId).number("device_timestamp", event->deviceTimestamp);
            emit(text);
            break;
        }
        case CUPTI_ACTIVITY_KIND_GRAPH_TRACE: {
            const auto* graph = reinterpret_cast<const CUpti_ActivityGraphTrace2*>(activity);
            gpu("graph", graph->correlationId, graph->contextId, graph->streamId, graph->start, graph->end,
                graph->graphId, 0, 0)
                .text("granularity", "graph_envelope").boolean("includes_internal_gaps", true);
            emit(text);
            break;
        }
        default: break;
        }
    }
    size_t dropped = 0;
    if (cuptiActivityGetNumDroppedRecords(reinterpret_cast<CUcontext>(context), stream, &dropped) == CUPTI_SUCCESS && dropped)
        note_loss(LossReason::cupti_dropped, dropped);
#else
    (void)buffer; (void)valid; (void)context; (void)stream; (void)emit;
#endif
}
}
