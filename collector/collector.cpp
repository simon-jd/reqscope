#include "collector/reqscope.h"
#include "collector/backend.hpp"
#include "collector/bounded_queue.hpp"
#include "collector/self_stats.hpp"
#include "format/segment.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

namespace reqscope::collector {
namespace fs = std::filesystem;
constexpr size_t max_json_size = 65536;
constexpr size_t reason_count = static_cast<size_t>(LossReason::count);
constexpr const char* reason_names[] = {"queue_full", "oversize_record", "malformed_record",
    "cupti_pool_exhausted", "cupti_completed_queue_full", "cupti_dropped", "cupti_parse_error",
    "correlation_error", "graph_metadata_error"};
constexpr const char* reason_count_units[] = {"records", "records", "records",
    "buffer_requests", "buffers", "CUPTI-reported-activity-records", "errors",
    "errors", "errors"};
static_assert(std::size(reason_names) == reason_count && std::size(reason_count_units) == reason_count);
struct Pending {
    enum Kind { json, cupti, graph_resource } kind = json;
    size_t length = 0;
    uint8_t* buffer = nullptr;
    uintptr_t context = 0;
    uint32_t stream = 0;
    GraphResource graph;
    std::array<char, max_json_size> data{};
};
struct LossCounter {
    std::atomic<uint64_t> count{0}, first{0}, last{0};
};
struct Segment {
    fs::path path;
    uint64_t begin = 0, end = 0, first_seq = 0, last_seq = 0, bytes = 0;
};

uint64_t monotonic_ns() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}
static uint64_t thread_cpu_ns() noexcept {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
}
static SelfStats stats_storage;

static thread_local bool engine_thread = false;
SelfStats& self_stats() noexcept { return stats_storage; }
bool on_engine_thread() noexcept { return engine_thread; }
void mark_engine_thread() noexcept { engine_thread = true; }
static std::string env_string(const char* key, const std::string& fallback) {
    const char* value = std::getenv(key);
    return value && *value ? value : fallback;
}
static uint64_t env_number(const char* key, uint64_t fallback, uint64_t min, uint64_t max) {
    const char* value = std::getenv(key);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    errno = 0;
    auto parsed = std::strtoull(value, &end, 10);
    if (errno || *end || value[0] == '-' || parsed < min || parsed > max)
        throw std::runtime_error(std::string("invalid ") + key);
    return parsed;
}
static std::string read_line(const fs::path& path) {
    std::ifstream input(path);
    std::string line;
    std::getline(input, line);
    return line;
}
static uint64_t process_start() {
    std::string stat = read_line("/proc/self/stat");
    auto close = stat.rfind(')');
    if (close == std::string::npos) throw std::runtime_error("cannot read process identity");
    std::istringstream fields(stat.substr(close + 2));
    std::string field;
    for (unsigned i = 3; i <= 22; ++i) {
        if (!(fields >> field)) throw std::runtime_error("incomplete /proc/self/stat");
        if (i == 22) return std::stoull(field);
    }
    return 0;
}
static std::atomic<int> received_signal{0};
static void termination_handler(int number) { received_signal.store(number, std::memory_order_relaxed); }

class Collector;
static std::atomic<Collector*> instance{nullptr};
class Collector {
public:
    const pid_t owner = getpid();
    BoundedQueue<Pending> queue;
    std::array<LossCounter, reason_count> losses;
    std::atomic<bool> ready{false}, stopping{false}, failed{false};
    std::atomic<uint64_t> flush_requested{0}, flush_completed{0};
    std::thread worker;
    std::mutex flush_mutex;
    std::condition_variable flush_condition;
    fs::path output;
    std::string socket_path, producer, mode, quoted_producer;
    Json header;
    BackendInfo backend;
    uint64_t sequence = 0, segment_number = 0, segment_bytes = 0;
    uint64_t started_ns = 0, last_stats = 0, writer_sleep_ns = 0;
    const uint64_t ring_ns, ring_bytes, target_segment_bytes;
    const uint64_t slow_step_ns, max_triggers;
    uint64_t trigger_count = 0, last_trigger = 0, pending_trigger = 0;
    std::unique_ptr<SegmentWriter> writer;
    Segment current;
    std::deque<Segment> segments;
    uint64_t retained_bytes = 0, overwritten_count = 0, overwritten_first = 0,
             overwritten_last = 0, overwritten_begin = 0, overwritten_end = 0;
    int listener = -1;
    bool signal_installed = false;
    struct sigaction previous_sigterm{};

    Collector() : queue(env_number("REQSCOPE_QUEUE_SLOTS", 128, 2, 4096)),
        ring_ns(env_number("REQSCOPE_RING_SECONDS", 30, 1, 86400) * 1000000000ULL),
        ring_bytes(env_number("REQSCOPE_RING_BYTES", 64 * 1024 * 1024, 65536, 16ULL * 1024 * 1024 * 1024)),
        target_segment_bytes(env_number("REQSCOPE_SEGMENT_BYTES", 16 * 1024 * 1024, 1024, 64 * 1024 * 1024)),
        slow_step_ns(env_number("REQSCOPE_SLOW_STEP_NS", 0, 0, UINT64_MAX)),
        max_triggers(env_number("REQSCOPE_MAX_TRIGGERS", 3, 0, 100)) {
        output = fs::absolute(env_string("REQSCOPE_OUTPUT", "reqscope-run"));
        fs::create_directories(output);
        mode = env_string("REQSCOPE_MODE", "light");
        if (mode != "light" && mode != "trace" && mode != "synthetic" && mode != "manual" && mode != "plugin")
            throw std::runtime_error("REQSCOPE_MODE must be light, trace, plugin, synthetic, or manual");
        std::string boot = read_line("/proc/sys/kernel/random/boot_id");
        const uint64_t ticks = process_start();
        producer = boot + ":" + std::to_string(owner) + ":" + std::to_string(ticks);
        header = Json::object({{"type", "Header"}, {"format_version", 1}, {"producer", producer},
            {"boot_id", boot}, {"pid", uint64_t(owner)}, {"start_time_ticks", ticks},
            {"clock", "CLOCK_MONOTONIC"},
            {"clock_calibrated", false}, {"mode", mode}, {"queue_slots", queue_capacity()},
            {"record_max_bytes", uint64_t(max_json_size)}, {"session_state", "Configured"}});
        quoted_producer = Json(producer).dump();
        socket_path = env_string("REQSCOPE_SOCKET", "/tmp/reqscope-" + std::to_string(owner) + ".sock");
        setup_socket();
    }
    uint64_t queue_capacity() const { return env_number("REQSCOPE_QUEUE_SLOTS", 128, 2, 4096); }
    void setup_socket() {
        if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) throw std::runtime_error("control socket path too long");
        struct stat existing{};
        if (lstat(socket_path.c_str(), &existing) == 0 || errno != ENOENT)
            throw std::runtime_error("refusing existing control socket path: " + socket_path);
        listener = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (listener < 0) throw std::runtime_error("cannot create control socket");
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
        // bind creates a private socket from the beginning, before listen.
        mode_t old_mask = umask(0077);
        int result = bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address));
        umask(old_mask);
        if (result < 0) { close(listener); listener = -1; throw std::runtime_error("cannot bind control socket"); }
        chmod(socket_path.c_str(), 0600);
        if (listen(listener, 8) < 0) { close(listener); listener = -1; unlink(socket_path.c_str()); throw std::runtime_error("cannot listen"); }
    }
    void start() {
        instance.store(this, std::memory_order_release);
        backend = backend_start(env_string("REQSCOPE_BACKEND", "") == "manual" ? "manual" : mode);
        // Manual and synthetic records share one clock by construction. A
        // plugin-only capture has no GPU records at all, so it has no GPU
        // clock bound: clock_error_ns stays absent rather than 0.
        if (backend.status == "manual" && mode != "plugin") {
            header["clock_error_ns"] = 0;
            header["clock_calibrated"] = true;
        }
        header["data_kind"] = mode == "synthetic" ? "synthetic" : mode == "plugin" ? "plugin"
            : backend.status == "manual" ? "manual" : "cupti";
        header["slow_step_threshold_ns"] = slow_step_ns;
        header["max_automatic_snapshots"] = max_triggers;
        header["gpu_granularity"] = mode == "light" ? "graph-envelope; eager kernels not traced"
            : mode == "plugin" ? "none" : "node";
        header["libcupti_path"] = backend.path;
        header["libcupti_version"] = uint64_t(backend.version);
        header["cupti_host_buffer_count"] = backend.host_buffer_count;
        header["cupti_host_buffer_bytes"] = backend.host_buffer_bytes;
        header["graph_resource_callbacks"] = backend.graph_resources;
        header["api_capture_policy"] = backend.api_capture_policy;
        header["api_capture_apis"] = backend.api_capture_apis;
        header["subscription"] = backend.subscription;
        header["thread_id_type"] = backend.active || backend.status == "manual" ? "system" : "unknown";
        header["backend_status"] = backend.status;
        header["backend_error"] = backend.error;
        header["v2_symbols_detected"] = backend.v2_symbols;
        header["v2_verified"] = false;
        header["session_state"] = backend.active || backend.status == "manual" ? "Capturing" : "Partial";
        struct sigaction current_action{};
        if (env_string("REQSCOPE_HANDLE_SIGTERM", "1") == "1" &&
            sigaction(SIGTERM, nullptr, &current_action) == 0 && current_action.sa_handler == SIG_DFL) {
            previous_sigterm = current_action;
            struct sigaction action{};
            action.sa_handler = termination_handler;
            sigemptyset(&action.sa_mask);
            signal_installed = sigaction(SIGTERM, &action, nullptr) == 0;
        }
        header["sigterm_flush_installed"] = signal_installed;
        worker = std::thread([this] { run(); });
        ready.store(true, std::memory_order_release);
    }
    void open_segment() {
        char name[128];
        std::snprintf(name, sizeof(name), "segment-%d-%llu-%06llu.rqs", owner,
            static_cast<unsigned long long>(header.integer("start_time_ticks")),
            static_cast<unsigned long long>(++segment_number));
        current = {output / name, monotonic_ns(), 0, sequence + 1, sequence, 0};
        if (fs::exists(current.path)) throw std::runtime_error("refusing to overwrite segment");
        Json segment_header = header;
        segment_header["segment_index"] = segment_number;
        segment_header["first_seq"] = sequence + 1;
        if (overwritten_count) segment_header["retention_loss"] = retention_loss();
        writer = std::make_unique<SegmentWriter>(current.path, segment_header);
        segment_bytes = segment_header.dump().size() + 16;
    }
    Json retention_loss() const {
        return Json::object({{"type", "Loss"}, {"producer", producer}, {"reason", "ring_overwrite"},
            {"count", overwritten_count}, {"seq_range", Json::array({overwritten_first, overwritten_last})},
            {"count_unit", "sequence_positions"},
            {"time_range", Json::array({uint64_t(0), uint64_t(0)})},
            {"writer_time_range", Json::array({overwritten_begin, overwritten_end})},
            {"range_known", false}});
    }
    void emit(Json record, bool rotate = true) {
        if (!record.is_object() || !record.contains("type") || !record["type"].is_string()) {
            note_loss(LossReason::malformed_record); return;
        }
        if (!writer) open_segment();
        record["producer"] = producer;
        record["seq"] = ++sequence;
        record["pid"] = uint64_t(owner);
        if (!record.contains("timestamp")) record["timestamp"] = monotonic_ns();
        {
            Timed timed(Site::write_record);
            segment_bytes += writer->write(record);
        }
        current.last_seq = sequence;
        current.end = monotonic_ns();
        if (slow_step_ns && record.string("type") == "Step" &&
            record["t_begin"].is_number() && record["t_end"].is_number() && record["step_id"].is_number()) {
            try {
                auto begin = record.integer("t_begin"), end = record.integer("t_end");
                if (end >= begin && end - begin >= slow_step_ns && trigger_count < max_triggers &&
                    (!last_trigger || monotonic_ns() - last_trigger >= 1000000000ULL))
                    pending_trigger = record.integer("step_id") + 1;
            } catch (const std::exception&) { note_loss(LossReason::malformed_record); }
        }
        // A CUPTI buffer can contain thousands of activities. Rotate after
        // each record, rather than after a batch of hundreds of buffers, so
        // one busy drain cannot create an arbitrarily oversized segment.
        // Retention metadata is emitted with rotate=false to avoid a cycle
        // when the configured segment target is smaller than its own header.
        if (rotate && (segment_bytes >= target_segment_bytes ||
                       current.end - current.begin >= 1000000000ULL)) {
            Timed timed(Site::rotate_segment);
            close_segment();
            enforce_retention();
        }
    }
    // A CUPTI record, serialized by the backend and left open: append the
    // fields every record carries and close it.
    void emit_text(std::string& text) {
        if (!writer) open_segment();
        const auto now = monotonic_ns();
        char tail[192];
        const int length = std::snprintf(tail, sizeof tail, ",\"pid\":%llu,\"seq\":%llu,\"timestamp\":%llu,\"producer\":",
            static_cast<unsigned long long>(owner), static_cast<unsigned long long>(++sequence),
            static_cast<unsigned long long>(now));
        text.append(tail, static_cast<size_t>(length));
        text += quoted_producer;
        text += '}';
        {
            Timed timed(Site::write_record);
            segment_bytes += writer->write_text(text);
        }
        current.last_seq = sequence;
        current.end = now;
        if (segment_bytes >= target_segment_bytes || current.end - current.begin >= 1000000000ULL) {
            Timed timed(Site::rotate_segment);
            close_segment();
            enforce_retention();
        }
    }
    // Rotation only writes: the page cache keeps a segment if the profiled
    // process dies, and a per-segment fdatasync on this thread (the only one
    // draining CUPTI buffers) was what let the CUPTI buffer pool run dry.
    // Dumps, explicit flushes and finalization sync.
    void close_segment(bool sync = false) {
        if (!writer) return;
        if (sync) writer->flush(); else writer->drain();
        writer.reset();
        current.bytes = fs::file_size(current.path);
        current.end = std::max(current.end, monotonic_ns());
        retained_bytes += current.bytes;
        segments.push_back(current);
        segment_bytes = 0;
    }
    void enforce_retention() {
        const auto now = monotonic_ns();
        bool changed = false;
        // A single completed segment is retained; max disk budget is therefore
        // ring_bytes + one segment + bounded incoming record size.
        while (segments.size() > 1 &&
               ((now > segments.front().end && now - segments.front().end > ring_ns) || retained_bytes > ring_bytes)) {
            const auto gone = segments.front();
            std::error_code error;
            if (!fs::remove(gone.path, error) || error) throw std::runtime_error("cannot enforce ring retention");
            retained_bytes -= gone.bytes;
            segments.pop_front();
            if (!overwritten_count) { overwritten_first = gone.first_seq; overwritten_begin = gone.begin; }
            overwritten_last = gone.last_seq;
            overwritten_end = gone.end;
            if (gone.last_seq >= gone.first_seq) overwritten_count += gone.last_seq - gone.first_seq + 1;
            changed = true;
        }
        if (changed) emit(retention_loss(), false);
    }
    void report_losses() {
        for (size_t i = 0; i < reason_count; ++i) {
            auto& loss = losses[i];
            const uint64_t count = loss.count.exchange(0);
            if (!count) continue;
            const uint64_t first = loss.first.exchange(0), last = loss.last.load();
            const uint64_t lost_first = sequence + 1;
            sequence += count;
            emit(Json::object({{"type", "Loss"}, {"reason", reason_names[i]}, {"count", count},
                {"count_unit", reason_count_units[i]},
                {"seq_range", Json::array({lost_first, sequence})},
                {"time_range", Json::array({uint64_t(0), uint64_t(0)})},
                {"observed_time_range", Json::array({first ? first : last, last})}, {"range_known", false}}));
        }
    }
    // The writer's busy intervals, from its first record after a sleep to the
    // next sleep: when the collector's own thread was taking a CPU. Shorter
    // ones (a plugin record or two) are only counted in CollectorStats.
    static constexpr uint64_t min_burst_ns = 200000;
    uint64_t burst_begin = 0, burst_records = 0, short_bursts = 0;
    void end_burst() {
        const auto end = monotonic_ns();
        if (end - burst_begin >= min_burst_ns) {
            emit(Json::object({{"type", "CollectorActivity"}, {"thread", "writer"}, {"start", burst_begin},
                {"end", end}, {"cupti_records",
                    self_stats().cupti_records.load(std::memory_order_relaxed) - burst_records}}), false);
        } else {
            ++short_bursts;
        }
        burst_begin = 0;
    }
    // Cumulative totals since the collector started; a reader takes deltas.
    void report_stats() {
        auto& stats = self_stats();
        Json sites = Json::object({});
        for (size_t i = 0; i < static_cast<size_t>(Site::count); ++i) {
            const auto& site = stats.sites[i];
            sites[site_names[i]] = Json::object({{"calls", site.calls.load(std::memory_order_relaxed)},
                {"ns", site.ns.load(std::memory_order_relaxed)},
                {"max_ns", site.max_ns.load(std::memory_order_relaxed)},
                {"engine_calls", site.engine_calls.load(std::memory_order_relaxed)}});
        }
        emit(Json::object({{"type", "CollectorStats"}, {"sites", std::move(sites)},
            {"timestamp_callbacks", stats.timestamp_callbacks.load(std::memory_order_relaxed)},
            {"cupti_records", stats.cupti_records.load(std::memory_order_relaxed)},
            {"cupti_bytes", stats.cupti_bytes.load(std::memory_order_relaxed)},
            {"writer_cpu_ns", thread_cpu_ns()}, {"writer_sleep_ns", writer_sleep_ns},
            {"writer_short_bursts", short_bursts},
            {"probe_cpu_ns", stats.probe_cpu_ns.load(std::memory_order_relaxed)},
            {"since_ns", started_ns}}), false);
        last_stats = monotonic_ns();
    }
    bool drain_one() {
        return queue.try_pop([this](Pending& pending) {
            if (pending.kind == Pending::cupti) {
                Timed timed(Site::process_buffer);
                self_stats().cupti_bytes.fetch_add(pending.length, std::memory_order_relaxed);
                try { backend_process(pending.buffer, pending.length, pending.context, pending.stream,
                                      [this](std::string& text) { emit_text(text); }); }
                catch (...) { backend_release(pending.buffer); throw; }
                backend_release(pending.buffer);
            } else if (pending.kind == Pending::graph_resource) {
                emit(graph_resource_json(pending.graph));
            } else {
                Json record;
                try { record = Json::parse(std::string(pending.data.data(), pending.length)); }
                catch (const std::exception&) { note_loss(LossReason::malformed_record); return; }
                emit(std::move(record));
            }
        });
    }
    Json snapshot(const fs::path& destination, uint64_t last) {
        if (!fs::create_directory(destination))
            throw std::runtime_error("dump output must be a new directory with an existing parent");
        emit(Json::object({{"type", "Session"}, {"state", "Snapshot"}}));
        close_segment(true); enforce_retention(); close_segment(true);
        uint64_t count = 0;
        const auto now = monotonic_ns();
        for (const auto& segment : segments) {
            if (segment.end < now && now - segment.end > last) continue;
            fs::copy_file(segment.path, destination / segment.path.filename(), fs::copy_options::none);
            ++count;
        }
        Json manifest = Json::object({{"producer", producer}, {"created_ns", now},
            {"requested_last_ns", last}, {"segments", count}, {"session_state", "Snapshot"},
            {"backend_status", backend.status}, {"retention_loss", retention_loss()}});
        std::ofstream manifest_file(destination / "manifest.json");
        manifest_file << manifest.dump() << '\n';
        if (!manifest_file) throw std::runtime_error("cannot write dump manifest");
        // The copied files end at Snapshot. The live run resumes with its own
        // capture boundary so a subsequent SIGKILL cannot masquerade as clean.
        emit(Json::object({{"type", "Session"}, {"state", "Capturing"}}));
        return Json::object({{"ok", true}, {"output", destination.string()}, {"segments", count}});
    }
    void automatic_snapshot() {
        if (!pending_trigger || trigger_count >= max_triggers) return;
        auto step = pending_trigger - 1;
        pending_trigger = 0;
        ++trigger_count; last_trigger = monotonic_ns();
        backend_flush();
        unsigned budget = 8192;
        while (budget-- && drain_one()) {}
        report_losses();
        const fs::path root = output / "triggers";
        fs::create_directories(root);
        auto destination = root / ("step-" + std::to_string(step) + "-" + std::to_string(last_trigger));
        (void)snapshot(destination, ring_ns);
    }
    void serve_control() {
        int client = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
        if (client < 0) return;
        Json response;
        try {
            // Same-uid control only; mode 0600 is also enforced by the kernel.
            ucred credentials{}; socklen_t length = sizeof(credentials);
            if (getsockopt(client, SOL_SOCKET, SO_PEERCRED, &credentials, &length) || credentials.uid != geteuid())
                throw std::runtime_error("control peer uid mismatch");
            std::string request;
            auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
            while (request.find('\n') == std::string::npos && request.size() <= 8192) {
                char buffer[1024];
                ssize_t n = recv(client, buffer, sizeof(buffer), 0);
                if (n > 0) request.append(buffer, size_t(n));
                else if (n == 0) break;
                else if (errno != EAGAIN && errno != EINTR) throw std::runtime_error("control read failed");
                else if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("control read timeout");
                else std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (request.size() > 8192) throw std::runtime_error("control request too large");
            Json command = Json::parse(request);
            const auto operation = command.string("command");
            if (operation == "dump" || operation == "flush") {
                backend_flush();
                size_t budget = 8192;
                while (budget-- && drain_one()) {}
                report_losses();
                if (operation == "dump") {
                    if (command.string("output").empty()) throw std::runtime_error("dump output is required");
                    const fs::path destination = fs::absolute(command.string("output"));
                    response = snapshot(destination, command.integer("last_ns", ring_ns));
                } else {
                    close_segment(true); enforce_retention(); close_segment(true);
                    response = Json::object({{"ok", true}});
                }
            } else if (operation == "status") {
                response = Json::object({{"ok", true}, {"producer", producer}, {"sequence", sequence},
                    {"output", output.string()}, {"backend_status", backend.status}, {"subscription", backend.subscription}});
            } else throw std::runtime_error("unknown control command");
        } catch (const std::exception& error) {
            response = Json::object({{"ok", false}, {"error", error.what()}});
        }
        auto data = response.dump() + "\n";
        (void)send(client, data.data(), data.size(), MSG_NOSIGNAL);
        close(client);
    }
    void run() noexcept {
        pthread_setname_np(pthread_self(), "rqs-writer");
        started_ns = last_stats = monotonic_ns();
        try {
            emit(Json::object({{"type", "Session"}, {"state", header.string("session_state")},
                              {"backend_status", backend.status}, {"backend_error", backend.error}}));
            const bool same_clock = backend.status == "manual" && mode != "plugin";
            emit(Json::object({{"type", "ClockCalibration"}, {"clock", "CLOCK_MONOTONIC"},
                {"calibrated", same_clock},
                {"source", same_clock ? "manual same-clock records"
                    : mode == "plugin" ? "plugin records only; no GPU records, GPU bound absent"
                    : "timestamp-callback; GPU bound unmeasured"}}));
            while (!stopping.load()) {
                unsigned budget = 512;
                bool work = false;
                const auto pass_begin = monotonic_ns();
                while (budget-- && drain_one()) work = true;
                if (work && !burst_begin) {
                    burst_begin = pass_begin;
                    burst_records = self_stats().cupti_records.load(std::memory_order_relaxed);
                }
                report_losses();
                automatic_snapshot();
                if (writer && (segment_bytes >= target_segment_bytes || monotonic_ns() - current.begin >= 1000000000ULL)) {
                    Timed timed(Site::rotate_segment);
                    close_segment(); enforce_retention();
                }
                if (monotonic_ns() - last_stats >= 1000000000ULL) report_stats();
                const auto requested = flush_requested.load();
                if (requested > flush_completed.load() && queue.empty()) {
                    report_stats();
                    close_segment(true); enforce_retention(); close_segment(true);
                    flush_completed.store(requested);
                    flush_condition.notify_all();
                }
                serve_control();
                if (received_signal.load()) { stopping.store(true); break; }
                if (!work) {
                    if (burst_begin) end_burst();
                    const auto slept = monotonic_ns();
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    writer_sleep_ns += monotonic_ns() - slept;
                }
            }
            backend_stop();
            while (drain_one()) {}
            report_losses();
            report_stats();
            emit(Json::object({{"type", "Session"}, {"state", "Finalized"}, {"signal", received_signal.load()}}));
            close_segment(true); enforce_retention(); close_segment(true);
        } catch (const std::exception& error) {
            failed.store(true);
            backend_stop();
            // Writer failures must be discoverable even when no segment can be
            // completed. This path is never called from a CUPTI callback.
            try { std::ofstream failure(output / ("collector-error-" + std::to_string(owner) + ".txt")); failure << error.what() << '\n'; } catch (...) {}
        }
        ready.store(false);
        flush_condition.notify_all();
        if (listener >= 0) { close(listener); listener = -1; unlink(socket_path.c_str()); }
        if (received_signal.load() && signal_installed) {
            sigaction(SIGTERM, &previous_sigterm, nullptr);
            kill(owner, SIGTERM);
        }
    }
    int flush() {
        if (getpid() != owner || !ready.load() || failed.load()) return -3;
        backend_flush();
        const auto ticket = flush_requested.fetch_add(1) + 1;
        std::unique_lock lock(flush_mutex);
        bool done = flush_condition.wait_for(lock, std::chrono::seconds(10), [this, ticket] {
            return flush_completed.load() >= ticket || failed.load() || !ready.load();
        });
        return done && !failed.load() && flush_completed.load() >= ticket ? 0 : -3;
    }
    void stop() {
        if (getpid() != owner) return;
        stopping.store(true);
        if (worker.joinable()) worker.join();
        if (signal_installed && !received_signal.load()) sigaction(SIGTERM, &previous_sigterm, nullptr);
    }
};

void note_loss(LossReason reason, uint64_t count) noexcept {
    auto* collector = instance.load(std::memory_order_acquire);
    if (!collector || !count || collector->owner != getpid()) return;
    auto& loss = collector->losses[static_cast<size_t>(reason)];
    const auto now = monotonic_ns();
    uint64_t zero = 0;
    loss.first.compare_exchange_strong(zero, now, std::memory_order_relaxed);
    loss.last.store(now, std::memory_order_relaxed);
    loss.count.fetch_add(count, std::memory_order_release);
}
bool enqueue_cupti_buffer(uint8_t* buffer, size_t valid, uintptr_t context, uint32_t stream) noexcept {
    auto* collector = instance.load(std::memory_order_acquire);
    if (!collector || collector->owner != getpid()) return false;
    return collector->queue.try_push([=](Pending& pending) {
        pending.kind = Pending::cupti; pending.buffer = buffer; pending.length = valid;
        pending.context = context; pending.stream = stream;
    });
}
bool enqueue_graph_resource(const GraphResource& resource) noexcept {
    auto* collector = instance.load(std::memory_order_acquire);
    if (!collector || collector->owner != getpid()) return false;
    return collector->queue.try_push([&resource](Pending& pending) {
        pending.kind = Pending::graph_resource;
        pending.graph = resource;
    });
}
static void finish() { if (auto* collector = instance.load()) collector->stop(); }
static Collector* get_collector() noexcept {
    static std::once_flag once;
    try {
        std::call_once(once, [] {
            // Kept alive until process teardown so late CUDA runtime callbacks
            // cannot touch freed buffer storage.
            auto* collector = new Collector;
            collector->start();
            std::atexit(finish);
        });
    } catch (...) { return nullptr; }
    auto* collector = instance.load(std::memory_order_acquire);
    return collector && collector->owner == getpid() ? collector : nullptr;
}
}

extern "C" int InitializeInjection() {
    auto* collector = reqscope::collector::get_collector();
    return collector && collector->ready.load() &&
        (collector->backend.active || collector->backend.status == "manual") ? 1 : 0;
}
extern "C" int reqscope_emit_json(const char* record) {
    using namespace reqscope::collector;
    auto* collector = get_collector();
    if (!collector || !collector->ready.load() || collector->stopping.load() || collector->failed.load()) return -3;
    if (!record) { note_loss(LossReason::malformed_record); return -1; }
    size_t length = strnlen(record, max_json_size);
    if (length == max_json_size) { note_loss(LossReason::oversize_record); return -1; }
    if (!length) { note_loss(LossReason::malformed_record); return -1; }
    Timed timed(Site::emit_json);
    if (!collector->queue.try_push([=](Pending& pending) {
        pending.kind = Pending::json; pending.length = length;
        std::memcpy(pending.data.data(), record, length);
    })) { note_loss(LossReason::queue_full); return -2; }
    return 0;
}
extern "C" int reqscope_push_step(uint64_t step_id) {
    auto* collector = reqscope::collector::get_collector();
    if (!collector || !collector->ready.load() || collector->stopping.load()) return -3;
    return reqscope::collector::backend_push(step_id);
}
extern "C" int reqscope_pop_step() {
    auto* collector = reqscope::collector::get_collector();
    if (!collector || !collector->ready.load() || collector->stopping.load()) return -3;
    return reqscope::collector::backend_pop();
}
extern "C" int reqscope_flush() {
    try {
        auto* collector = reqscope::collector::get_collector();
        return collector ? collector->flush() : -3;
    } catch (...) { return -3; }
}
