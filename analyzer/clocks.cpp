#include "analyzer/internal.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace reqscope::detail {
namespace {

// A probe's copy is looked for this close to its bracket; farther means it was lost.
constexpr Time kProbeMatchWindowNs = 50'000'000;

}  // namespace

// Session clock bound from ClockProbe records. The collector brackets a
// 4-byte H2D copy on a private stream between host CLOCK_MONOTONIC reads
// t0 and t1. CUPTI reports GPU activity in that same clock (its timestamp
// callback), so the copy's reported [start, end] must lie in [t0, t1]: the
// bracket width bounds the CPU/GPU alignment error at that moment, and a
// reported time outside the bracket adds its distance. The session bound is
// the worst probe. A header-supplied bound takes precedence. A probe that
// stopped (its Diagnostic says "stopped") covered only part of the
// session, so its producer's bound stays unknown.
void Analyzer::derive_probe_bounds() {
    std::map<std::tuple<std::string, Time, Time>, std::vector<const Record*>> copies;
    std::map<std::string, std::vector<const Record*>> probes;
    std::map<std::string, Time> last_gpu_end;
    std::map<std::string, std::string> stopped;
    for (const auto& record : records_) {
        if (record.ambiguous) continue;
        if (record.type == "Diagnostic" && str(*record.data, "source") == "clock_probe" &&
            flag(*record.data, "stopped") && !stopped.contains(record.producer))
            stopped[record.producer] = str(*record.data, "reason");
        if (record.type == "ClockProbe") probes[record.producer].push_back(&record);
        else if (record.type == "GpuActivity" && record.timed) {
            last_gpu_end[record.producer] = std::max(last_gpu_end[record.producer], record.span.end);
            if (str(*record.data, "kind") == "memcpy")
                copies[{record.producer, num(*record.data, "ctx"), num(*record.data, "stream")}].push_back(&record);
        }
    }
    // How far a copy lies outside a probe's [t0, t1]; 0 when it is inside.
    auto outside = [](Time t0, Time t1, const Record& copy) {
        return std::max({Time{0}, t0 > copy.span.start ? t0 - copy.span.start : Time{0},
                         copy.span.end > t1 ? copy.span.end - t1 : Time{0}});
    };
    for (auto& [producer, list] : probes) {
        // Match globally, closest pairs first, so a probe never takes the
        // copy that sits inside a neighbouring probe's bracket.
        std::vector<std::tuple<Time, const Record*, const Record*>> candidates;
        for (const auto* probe : list) {
            const auto t0 = number(*probe->data, "t0"), t1 = number(*probe->data, "t1");
            if (!t0 || !t1 || *t1 < *t0) continue;
            const auto key = std::make_tuple(producer, num(*probe->data, "ctx"), num(*probe->data, "stream"));
            for (const auto* copy : copies[key]) {
                const Time distance = outside(*t0, *t1, *copy);
                if (distance <= kProbeMatchWindowNs) candidates.emplace_back(distance, probe, copy);
            }
        }
        std::sort(candidates.begin(), candidates.end());
        std::map<const Record*, const Record*> matched;
        std::set<const Record*> used;
        for (const auto& [distance, probe, copy] : candidates)
            if (!matched.contains(probe) && !used.contains(copy)) { matched[probe] = copy; used.insert(copy); }
        std::vector<Time> widths;
        Time bound = 0, violation = 0, unpaired = 0, pending = 0;
        for (const auto* probe : list) {
            const auto t0 = number(*probe->data, "t0"), t1 = number(*probe->data, "t1");
            const auto found = matched.find(probe);
            // A probe that completed after the last GPU record CUPTI
            // delivered may still have its copy in a CUPTI buffer at the
            // final flush: it can be checked neither way.
            if (found == matched.end() && t1 && *t1 > last_gpu_end[producer]) { ++pending; continue; }
            if (found == matched.end()) { ++unpaired; continue; }
            probe_copy_ids_.insert(found->second->id);
            const Time width = *t1 - *t0, off = outside(*t0, *t1, *found->second);
            widths.push_back(width);
            violation = std::max(violation, off);
            bound = std::max(bound, width + off);
        }
        std::sort(widths.begin(), widths.end());
        Json stats = Json::object({{"probes", static_cast<Time>(list.size())},
            {"paired", static_cast<Time>(widths.size())}, {"unpaired", unpaired},
            {"pending_at_capture_end", pending},
            {"max_violation_ns", violation},
            {"bracket_width_p50_ns", widths.empty() ? Json() : Json(widths[widths.size() / 2])},
            {"bracket_width_max_ns", widths.empty() ? Json() : Json(widths.back())}});
        const auto halt = stopped.find(producer);
        if (halt != stopped.end()) stats["stopped"] = halt->second;
        probe_stats_[producer] = stats;
        auto& error = clock_errors_[producer];
        if (!error && !widths.empty() && !unpaired && halt == stopped.end()) error = bound;
    }
    for (const auto& [producer, reason] : stopped)
        if (!probes.contains(producer))
            probe_stats_[producer] = Json::object({{"probes", Time{0}}, {"stopped", reason}});
}

void Analyzer::assign_clock_brackets() {
    // A missing header clock_error_ns is not a license to invent one offset.
    // Each activity may cite the host launch entry and a later successful
    // covering synchronization on the same system thread. The bound is the
    // farthest distance from the observed GPU timestamps to that window.
    for (const auto& activity : activities_) {
        const auto& record = *activity.record;
        if (!activity.api || !record.timed || record.ambiguous || activity.api->ambiguous) continue;
        if (!monotonic_producers_.contains(record.producer) || producer_clock_blocked(record.producer)) continue;
        if (!explicit_success(*activity.api) || !system_tid(*activity.api) || !activity.api->timed) continue;
        if (activity.api->span.start >= activity.api->span.end) continue;
        const auto ctx = number(*record.data, "ctx");
        const auto stream = number(*record.data, "stream");
        constexpr Time kInvalid = std::numeric_limits<std::uint32_t>::max();
        if (!ctx || !stream || *ctx == 0 || *ctx == kInvalid || *stream == kInvalid) continue;
        const bool node = num(*record.data, "graph_id") != 0 && str(*record.data, "kind") != "graph";
        const Time tid = num(*activity.api->data, "tid");
        const Record* best_sync = nullptr;
        Time best_upper = std::numeric_limits<Time>::max();
        int best_kind = 0;
        for (const auto* sync : synchronization_) {
            if (sync->producer != record.producer || sync->ambiguous || !explicit_success(*sync)) continue;
            const auto kind = host_completion_kind(*sync);
            if (!kind || (*kind == 3 && node)) continue;
            if (number(*sync->data, "ctx") != ctx) continue;
            if (*kind == 3 && number(*sync->data, "stream") != stream) continue;
            const auto corr = number(*sync->data, "corr");
            if (!corr) continue;
            const auto* api = unique_api({sync->producer, *corr});
            if (!api || api->ambiguous || !api->timed || !explicit_success(*api) || !system_tid(*api)) continue;
            if (num(*api->data, "tid") != tid || api->span.start <= activity.api->span.end) continue;
            if (!sync_api_matches(*kind, *api) || api->span.end >= best_upper) continue;
            best_upper = api->span.end;
            best_sync = sync;
            best_kind = *kind;
        }
        if (!best_sync || best_upper <= activity.api->span.start) continue;
        if (affected(record.producer, {activity.api->span.start, best_upper})) continue;
        auto distance = [](Time left, Time right) { return left > right ? left - right : right - left; };
        ClockBracket bracket;
        bracket.known = true;
        bracket.lower = activity.api->span.start;
        bracket.upper = best_upper;
        bracket.launch_id = activity.api->id;
        bracket.sync_id = best_sync->id;
        bracket.coverage = best_kind == 4 ? "context-synchronize" : "stream-synchronize";
        bracket.error = std::max({distance(record.span.start, bracket.lower), distance(record.span.start, bracket.upper),
                                  distance(record.span.end, bracket.lower), distance(record.span.end, bracket.upper)});
        bracket.inside = bracket.lower <= record.span.start && record.span.end <= bracket.upper;
        brackets_[record.id] = bracket;
    }
}

std::optional<Time> Analyzer::error(const std::string& producer) const {
    const auto found = clock_errors_.find(producer);
    return found == clock_errors_.end() ? std::nullopt : found->second;
}

std::optional<Time> Analyzer::clock_bound_for(const std::string& producer, const std::vector<const Record*>& gpu) const {
    if (const auto header = error(producer)) return header;
    if (gpu.empty()) return std::nullopt;
    Time max_error = 0;
    for (const auto* record : gpu) {
        const auto found = brackets_.find(record->id);
        if (found == brackets_.end() || !found->second.known || !found->second.inside) return std::nullopt;
        max_error = std::max(max_error, found->second.error);
    }
    return max_error;
}

bool Analyzer::reliable(const Record& record) const {
    const auto bound = error(record.producer);
    return record.timed && !record.ambiguous && bound && record.span.end - record.span.start >= *bound;
}

bool Analyzer::scheduler_reliable(const Record& record, const std::string& target, Interval piece) const {
    const auto source_bound = error(record.producer);
    const auto target_bound = error(target);
    return reliable(record) && source_bound && target_bound &&
        piece.end - piece.start >= std::max(*source_bound, *target_bound);
}

bool Analyzer::explicit_success(const Record& record) const {
    return number(*record.data, "return_value") == std::optional<Time>{0};
}

bool Analyzer::system_tid(const Record& api) const {
    if (!number(*api.data, "tid") || num(*api.data, "tid") == 0) return false;
    const Json* header = nullptr;
    for (const auto& item : trace_.headers) {
        if (str(item, "producer") != api.producer) continue;
        if (header && str(*header, "thread_id_type") != str(item, "thread_id_type")) return false;
        header = &item;
    }
    const bool header_names_domain = header && (*header)["thread_id_type"].is_string();
    if (header_names_domain && str(*header, "thread_id_type") != "system") return false;
    const std::string domain = str(*api.data, "tid_domain", header_names_domain ? str(*header, "thread_id_type") : "");
    if (domain != "system") return false;
    if (!header_names_domain && !(*api.data)["tid_domain"].is_string()) return false;
    return true;
}

bool Analyzer::producer_clock_blocked(const std::string& producer) const {
    for (const auto& loss : losses_) {
        if (!loss.producer.empty() && loss.producer != producer) continue;
        if (!loss.timed) return true;
    }
    return false;
}

std::optional<int> Analyzer::host_completion_kind(const Record& sync) const {
    if (const auto kind = number(*sync.data, "sync_type")) {
        if (*kind == 3 || *kind == 4) return static_cast<int>(*kind);
        return std::nullopt;
    }
    const auto name = str(*sync.data, "sync_type");
    if (name == "stream_synchronize" || name == "cudaStreamSynchronize" || name == "cuStreamSynchronize") return 3;
    if (name == "context_synchronize" || name == "device_synchronize" || name == "cudaDeviceSynchronize" ||
        name == "cuCtxSynchronize") return 4;
    return std::nullopt;
}

bool Analyzer::sync_api_matches(int kind, const Record& api) const {
    const auto name = str(*api.data, "name", str(*api.data, "cbid"));
    if (kind == 3) return starts_with(name, "cudaStreamSynchronize") || starts_with(name, "cuStreamSynchronize");
    return starts_with(name, "cudaDeviceSynchronize") || starts_with(name, "cudaThreadSynchronize") ||
           starts_with(name, "cuCtxSynchronize");
}

}  // namespace reqscope::detail
