#include "analyzer/internal.hpp"

#include <algorithm>
#include <cstdint>
#include <map>

namespace reqscope::detail {
namespace {

// Cause rules for slow steps (docs/design.md, "Causes"): this much co-scheduled
// prefill makes a step's slowness interference; a GPU or host component must
// account for this share of a gap's excess to name it; baselines need this
// many comparable steps.
constexpr Tokens kInterferencePrefillTokens{128};
constexpr double kCauseShare = 0.5;
constexpr std::size_t kBaselineSteps = 5;
constexpr Time kBaselineWindow = 200;

}  // namespace

const Analyzer::StepGpu& Analyzer::step_gpu(const Step* step) const {
    if (!step_gpu_built_) {
        step_gpu_built_ = true;
        struct Item { Time start, end, launched; };
        std::map<const Step*, std::vector<Item>> items;
        for (const auto& activity : activities_) {
            if (!activity.step || !activity.record->timed) continue;
            const auto span = activity.record->span;
            const bool launched = activity.api && activity.api->timed;
            items[activity.step].push_back({span.start, span.end, launched ? activity.api->span.end : 0});
            const auto kind = str(*activity.record->data, "kind");
            auto& gpu = step_gpu_[activity.step];
            if (launched && (kind == "kernel" || kind == "graph")) {
                const Time api_end = activity.api->span.end;
                const Time wait = span.start > api_end ? span.start - api_end : 0;
                if (!gpu.has_wait || wait < gpu.wait) { gpu.wait = wait; gpu.launch = api_end; }
                gpu.has_wait = true;
            }
        }
        for (auto& [step, list] : items) {
            std::sort(list.begin(), list.end(), [](const Item& a, const Item& b) { return a.start < b.start; });
            auto& gpu = step_gpu_[step];
            gpu.known = true;
            gpu.first = list.front().start;
            Time cursor = list.front().end;
            gpu.busy = list.front().end - list.front().start;
            for (std::size_t i = 1; i < list.size(); ++i) {
                const auto& item = list[i];
                if (item.start > cursor) {
                    const Time idle = item.start - cursor;
                    const Time host = item.launched > cursor ? std::min(idle, item.launched - cursor) : 0;
                    gpu.inner_host += host;
                    gpu.inner_gpu += idle - host;
                    gpu.busy += item.end - item.start;
                } else if (item.end > cursor) {
                    gpu.busy += item.end - cursor;
                }
                cursor = std::max(cursor, item.end);
            }
            gpu.last = cursor;
        }
    }
    static const StepGpu none;
    const auto found = step_gpu_.find(step);
    return found == step_gpu_.end() ? none : found->second;
}

// Steps are comparable when they have the same producer, member count and
// scheduled tokens and no prefill: in steady decoding, the same batch.
std::string Analyzer::shape(const Step* step) {
    Tokens tokens;
    for (const auto& member : step->members) {
        if (member.phase == "prefill") return "";
        tokens += member.tokens;
    }
    return step->record->producer + '|' + std::to_string(step->members.size()) + '|' + std::to_string(tokens.count());
}

// Medians over the comparable steps within kBaselineWindow step ids of
// this one: local, because GPU work per decode step grows as contexts
// grow, and wide enough that one injected burst cannot dominate it.
Analyzer::Baseline Analyzer::baseline(const Step* step) const {
    if (!baselines_built_) {
        baselines_built_ = true;
        for (const auto& [id, candidate] : steps_) {
            const auto key = shape(&candidate);
            const auto& gpu = step_gpu(&candidate);
            if (!key.empty() && gpu.known) comparable_[key].push_back(&candidate);
        }
        for (auto& [key, list] : comparable_)
            std::sort(list.begin(), list.end(), [](const Step* a, const Step* b) { return a->id < b->id; });
    }
    const auto found = comparable_.find(shape(step));
    if (found == comparable_.end()) return {};
    const auto& list = found->second;
    const Time low = step->id > kBaselineWindow ? step->id - kBaselineWindow : 0;
    auto it = std::lower_bound(list.begin(), list.end(), low, [](const Step* a, Time id) { return a->id < id; });
    std::vector<Time> busy, inner, wait;
    for (; it != list.end() && (*it)->id <= step->id + kBaselineWindow; ++it) {
        const auto& gpu = step_gpu(*it);
        busy.push_back(gpu.busy);
        inner.push_back(gpu.inner_gpu);
        if (gpu.has_wait) wait.push_back(gpu.wait);
    }
    auto median = [](std::vector<Time>& v) {
        if (v.empty()) return Time{0};
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        return v[v.size() / 2];
    };
    return Baseline{busy.size(), median(busy), median(inner), median(wait)};
}

// Host time of a gap [begin, end] around its producing step's GPU work:
// waiting inside the step for its next launch, after its GPU work
// finished, and before its first launch returned.
Time Analyzer::host_part(const Step* step, Time begin, Time end) const {
    const auto& gpu = step_gpu(step);
    if (!gpu.known) return 0;
    const Time launched = gpu.has_wait ? gpu.launch : gpu.first;
    return gpu.inner_host + (end > gpu.last ? end - gpu.last : 0) + (launched > begin ? launched - begin : 0);
}

// The collector's own busy time inside [begin, end] in the step's process:
// the writer's busy intervals (CollectorActivity) and the clock probe's
// brackets. Absent (nullopt) when the capture records no writer activity.
std::optional<Time> Analyzer::collector_active(const Step* step, Time begin, Time end) const {
    if (!collector_busy_built_) {
        collector_busy_built_ = true;
        std::map<std::string, std::vector<Interval>> spans;
        std::set<std::string> with_writer;
        for (const auto& record : records_) {
            if (record.ambiguous) continue;
            if (record.type == "CollectorActivity" && record.timed) {
                spans[record.producer].push_back(record.span);
                with_writer.insert(record.producer);
            } else if (record.type == "ClockProbe") {
                const auto t0 = number(*record.data, "t0"), t1 = number(*record.data, "t1");
                if (t0 && t1 && *t1 > *t0) spans[record.producer].push_back({*t0, *t1});
            }
        }
        for (auto& [producer, list] : spans)
            if (with_writer.contains(producer)) collector_busy_[producer] = unite(std::move(list));
    }
    const auto found = collector_busy_.find(step->record->producer);
    if (found == collector_busy_.end()) return std::nullopt;
    const auto& list = found->second;
    Time total = 0;
    auto it = std::lower_bound(list.begin(), list.end(), begin, [](const Interval& a, Time t) { return a.end <= t; });
    for (; it != list.end() && it->start < end; ++it)
        total += std::min(end, it->end) - std::max(begin, it->start);
    return total;
}

// Why one slow gap [begin, end] was slow, from its producing step.
// host_base is the request's median host_part over all its gaps. Rules
// are in docs/design.md ("Causes").
Json Analyzer::cause(const Step* step, Time begin, Time end, Time excess, Tokens prefill_tokens, Time host_base) const {
    Json evidence = Json::object({{"prefill_tokens", prefill_tokens.count()}});
    if (prefill_tokens >= kInterferencePrefillTokens)
        return Json::object({{"cause", "interference"}, {"evidence", evidence}});
    const auto& gpu = step ? step_gpu(step) : step_gpu(nullptr);
    if (!step || !gpu.known)
        return Json::object({{"cause", "undetermined"}, {"evidence", evidence},
            {"reason", "no GPU records for the producing step; GPU and host delay look alike"}});
    const auto base = baseline(step);
    // GPU: more GPU work than in comparable steps, or more time when the
    // step's next kernel was launched but not running, inside the step or
    // before its first kernel (the GPU was busy elsewhere or slower).
    Time gpu_extra = 0;
    auto above = [](Time value, Time base) { return value > base ? value - base : Time{0}; };
    if (base.steps >= kBaselineSteps) {
        gpu_extra += above(gpu.busy, base.busy) + above(gpu.inner_gpu, base.inner_gpu);
        if (gpu.has_wait) gpu_extra += above(gpu.wait, base.wait);
    }
    // Host: more host time around the step's GPU work than the request's
    // gaps usually have.
    const Time host = host_part(step, begin, end);
    const Time host_extra = above(host, host_base);
    evidence["gpu_busy_ns"] = gpu.busy;
    evidence["gpu_busy_baseline_ns"] = base.busy;
    evidence["gpu_wait_inside_ns"] = gpu.inner_gpu;
    evidence["gpu_wait_inside_baseline_ns"] = base.inner_gpu;
    evidence["host_wait_inside_ns"] = gpu.inner_host;
    evidence["launch_wait_ns"] = gpu.has_wait ? Json(gpu.wait) : Json();
    evidence["launch_wait_baseline_ns"] = base.wait;
    evidence["host_ns"] = host;
    evidence["host_baseline_ns"] = host_base;
    evidence["comparable_steps"] = static_cast<Time>(base.steps);
    evidence["gpu_extra_ns"] = gpu_extra;
    evidence["host_extra_ns"] = host_extra;
    const double needed = kCauseShare * static_cast<double>(excess);
    std::string name = "unexplained";
    if (gpu_extra >= host_extra && static_cast<double>(gpu_extra) >= needed) name = "gpu_slowdown";
    else if (host_extra > gpu_extra && static_cast<double>(host_extra) >= needed) name = "host_delay";
    Json result = Json::object({{"cause", name}, {"evidence", evidence}});
    // A host delay no longer than the time the collector itself was busy
    // in the gap may be the profiler's own doing (docs/design.md, "Causes").
    if (const auto active = collector_active(step, begin, end)) {
        result["evidence"]["collector_active_ns"] = *active;
        if (name == "host_delay" && static_cast<double>(*active) >= kCauseShare * static_cast<double>(host_extra))
            result["flags"] = Json::array({"collector_active"});
    }
    return result;
}

// Why were a request's slow tokens slow? For each inter-token gap well above
// the request's median gap, take the engine step that produced the token
// and report its GPU busy time (union of linked GPU intervals: durations
// in the GPU's own clock, so no CPU/GPU alignment is needed) and the
// requests co-scheduled with it. The gap's excess over the median is
// attributed to co-scheduled prefill work in proportion to its tokens;
// gaps with no co-scheduled prefill are reported as unexplained.
Json Analyzer::blame(const std::string& only, double slow_factor) const {
    Json rows = Json::array();
    std::map<std::string, Time> culprits;
    Time recorded_steps = 0, fallback_steps = 0, flagged_host_delays = 0;
    std::map<std::string, Time> causes;
    for (const auto& [key, request] : requests_) {
        if (request.spans.empty()) continue;
        const Json& raw = *request.spans.front()->data;
        const std::string id = str(raw, "req_id");
        if (!only.empty() && id != only) continue;
        // producers[k] is the step that produced token k. Without it, fall
        // back to the request's steps whose host span ended inside the gap,
        // which is off by one step under async scheduling.
        const auto [token_times, producers, times_valid] = token_series(raw);
        if (!times_valid || token_times.size() < 3 || !std::is_sorted(token_times.begin(), token_times.end())) continue;
        ++(producers.empty() ? fallback_steps : recorded_steps);
        std::vector<Time> gaps;
        for (std::size_t i = 1; i < token_times.size(); ++i) gaps.push_back(token_times[i] - token_times[i - 1]);
        std::vector<Time> sorted = gaps;
        std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        const Time median = sorted[sorted.size() / 2];
        // The request's usual host time per gap, for the cause rules.
        Time host_base = 0;
        if (!producers.empty()) {
            std::map<Time, const Step*> by_id;
            for (const auto* step : request.steps) by_id[step->id] = step;
            std::vector<Time> hosts;
            for (std::size_t i = 0; i < gaps.size(); ++i)
                if (const auto found = by_id.find(producers[i + 1]); found != by_id.end() && step_gpu(found->second).known)
                    hosts.push_back(host_part(found->second, token_times[i], token_times[i + 1]));
            if (!hosts.empty()) {
                std::nth_element(hosts.begin(), hosts.begin() + hosts.size() / 2, hosts.end());
                host_base = hosts[hosts.size() / 2];
            }
        }
        Json slow = Json::array();
        Time excess_total = 0, unexplained = 0;
        for (std::size_t i = 0; i < gaps.size(); ++i) {
            if (static_cast<double>(gaps[i]) <= slow_factor * static_cast<double>(median)) continue;
            const Time begin = token_times[i], end = token_times[i + 1], excess = gaps[i] - median;
            excess_total += excess;
            // Steps of this request that ran inside the gap produced the token.
            Json steps = Json::array();
            std::map<std::string, Tokens> prefill;
            Tokens prefill_tokens;
            const Step* producer = nullptr;
            for (const auto* step : request.steps) {
                const auto span = step->record->span;
                if (!producers.empty() ? step->id != producers[i + 1] : span.end <= begin || span.end > end)
                    continue;
                if (!producer) producer = step;
                std::vector<Interval> busy;
                for (const auto* gpu : step->gpu) busy.push_back(gpu->span);
                Json members = Json::array();
                for (const auto& member : step->members) {
                    if (member.request == key.second) continue;
                    members.push_back(Json::object({{"req_id", member.request}, {"phase", member.phase},
                                                    {"tokens", member.tokens.count()}}));
                    if (member.phase == "prefill") {
                        prefill[member.request] += member.tokens;
                        prefill_tokens += member.tokens;
                    }
                }
                steps.push_back(Json::object({{"step_id", step->id}, {"host_ns", span.end - span.start},
                    {"gpu_busy_ns", union_duration(busy)}, {"co_scheduled", members}}));
            }
            Json blamed = Json::array();
            for (const auto& [culprit, tokens] : prefill) {
                const Time share = static_cast<Time>(static_cast<double>(excess) * static_cast<double>(tokens.count()) /
                                                   static_cast<double>(prefill_tokens.count()));
                culprits[culprit] += share;
                blamed.push_back(Json::object({{"req_id", culprit}, {"prefill_tokens", tokens.count()}, {"excess_ns", share}}));
            }
            if (prefill.empty()) unexplained += excess;
            Json why = cause(producer, begin, end, excess, prefill_tokens, host_base);
            ++causes[str(why, "cause")];
            Json gap = Json::object({{"token_index", static_cast<Time>(i + 1)}, {"begin_ns", begin},
                {"end_ns", end}, {"gap_ns", gaps[i]},
                {"excess_ns", excess}, {"steps", steps}, {"blamed", blamed},
                {"cause", why["cause"]}, {"cause_evidence", why["evidence"]}});
            if (why.contains("flags")) {
                gap["flags"] = why["flags"];
                ++flagged_host_delays;
            }
            slow.push_back(std::move(gap));
        }
        rows.push_back(Json::object({{"req_id", id}, {"tokens", Tokens{token_times.size()}.count()},
            {"producing_steps", producers.empty() ? "host_span_in_gap" : "recorded"},
            {"median_gap_ns", median}, {"slow_gaps", slow}, {"excess_ns", excess_total},
            {"unexplained_ns", unexplained}}));
    }
    Json by = Json::array();
    std::vector<std::pair<Time, std::string>> ranked;
    for (const auto& [culprit, share] : culprits) ranked.push_back({share, culprit});
    std::sort(ranked.rbegin(), ranked.rend());
    for (const auto& [share, culprit] : ranked)
        by.push_back(Json::object({{"req_id", culprit}, {"excess_caused_ns", share}}));
    // Enough context to judge the answer without the full decomposition:
    // how much GPU work was linked to steps, what was lost, and the
    // measured CPU/GPU clock bound.
    Time linked = 0, untimed = 0;
    for (const auto& activity : activities_) linked += activity.step != nullptr;
    // CUPTI delivers a record without timestamps when it could not collect
    // them; such records cannot contribute GPU time anywhere.
    for (const auto& record : records_) untimed += record.type == "GpuActivity" && !record.timed;
    Json clocks = Json::array();
    for (const auto& [producer, error] : clock_errors_) {
        Json row = Json::object({{"producer", producer}, {"clock_error_ns", error ? Json(*error) : Json()}});
        if (const auto found = probe_stats_.find(producer); found != probe_stats_.end()) row["probe"] = found->second;
        clocks.push_back(row);
    }
    Json summary = Json::object({{"gpu_activities", static_cast<Time>(activities_.size())},
        {"linked_gpu_activities", linked}, {"untimed_gpu_activities", untimed},
        {"loss_records", static_cast<Time>(losses_.size())},
        {"loss_count", loss_count_},
        {"requests_with_recorded_steps", recorded_steps},
        {"requests_using_host_span_fallback", fallback_steps}});
    Json cause_counts = Json::object();
    for (const auto& [name, count] : causes) cause_counts[name] = count;
    summary["slow_gaps_by_cause"] = cause_counts;
    summary["host_delay_gaps_with_collector_active"] = flagged_host_delays;
    Json trace = trace_status();
    Json warnings = Json::array();
    if (fallback_steps)
        warnings.push_back(std::to_string(fallback_steps) + " of " + std::to_string(recorded_steps + fallback_steps) +
            " requests have no recorded producing step for every token; their steps are those whose host span "
            "ended inside the gap, which can be one step late under async scheduling");
    if (str(trace, "status") == "Partial") {
        std::string reasons;
        for (const auto& [reason, count] : trace["issues_by_reason"].object_items())
            reasons += (reasons.empty() ? "" : ", ") + reason + " (" + std::to_string(count.as_u64()) + ")";
        warnings.push_back("trace is Partial: " + reasons);
    }
    if (untimed)
        warnings.push_back(std::to_string(untimed) + " GPU activities have no timestamps, so step GPU busy time "
                           "undercounts");
    for (const auto& diagnostic : trace["collector_diagnostics"].array_items())
        if (flag(diagnostic, "stopped"))
            warnings.push_back(str(diagnostic, "source") + " stopped: " + str(diagnostic, "reason"));
    return Json::object({{"schema_version", 1}, {"slow_factor", slow_factor},
        {"method", "token gap above slow_factor x median; excess split over prefill tokens co-scheduled "
                   "in the step that produced the token (recorded by the plugin; without it, the request's "
                   "steps whose host span ended inside the gap)"},
        {"summary", summary}, {"trace", trace}, {"warnings", warnings}, {"clocks", clocks},
        {"requests", rows}, {"culprits", by}});
}

}  // namespace reqscope::detail

namespace reqscope {

Json blame(const Trace& trace, const std::string& request, double slow_factor) {
    return detail::Analyzer(trace, /*edges=*/false).blame(request, slow_factor);
}

}  // namespace reqscope
