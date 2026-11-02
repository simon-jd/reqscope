#include "analyzer/internal.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace reqscope::detail {

// The full decomposition: every GPU activity and request with its evidence,
// host gaps in and between steps, graph replays and topology, synchronization
// records, clock bases, and the losses and diagnostics that limit them.
Json Analyzer::report() const {
    std::size_t linked = 0;
    Json activities = activity_rows(linked);
    Json requests = request_rows();
    Json gaps = gap_rows();
    Json replays = Json::array();
    build_replays(replays);
    Json synchronization = synchronization_rows();
    Json diagnostics = report_diagnostics();
    bool partial = trace_.partial || !losses_.empty() || !diagnostics_.empty();
    for (const auto& request : requests.array_items())
        partial = partial || str(request, "status") == "Partial";
    Json provenance = report_provenance();
    Json summary = report_summary(linked);
    Json clocks = clock_rows();
    return Json::object({
        {"schema_version", 1}, {"status", partial ? "Partial" : "Complete"},
        {"provenance", provenance}, {"clocks", clocks}, {"summary", summary},
        {"edges", edges_}, {"activities", activities}, {"requests", requests},
        {"gaps", gaps}, {"graph_definitions", graph_definitions_}, {"graph_replays", replays},
        {"graph_topology", graph_topology_},
        {"synchronization", synchronization},
        {"diagnostics", diagnostics},
    });
}

// One row per GPU activity; linked counts those attributed to a step.
Json Analyzer::activity_rows(std::size_t& linked) const {
    Json activities = Json::array();
    for (const auto& activity : activities_) {
        if (activity.step) ++linked;
        activities.push_back(activity_row(activity));
    }
    return activities;
}

// A GPU activity with its step and requests, its launch API, and the
// per-activity clock bracket when one was found.
Json Analyzer::activity_row(const Activity& activity) const {
    const auto& record = *activity.record;
    const auto& raw = *record.data;
    std::set<Time> step_ids;
    std::set<std::string> request_ids;
    if (activity.step) {
        step_ids.insert(activity.step->id);
        for (const auto& member : activity.step->members) request_ids.insert(member.request);
    }
    Json row = Json::object({
        {"record_id", record.id}, {"producer", record.producer},
        {"corr", raw.value("corr")}, {"kind", str(raw, "kind", "unknown")},
        {"granularity", str(raw, "granularity", str(raw, "kind") == "graph" ? "graph_envelope" : "activity")},
        {"includes_internal_gaps", flag(raw, "includes_internal_gaps", str(raw, "kind") == "graph")},
        {"start_ns", record.span.start}, {"end_ns", record.span.end},
        {"duration_ns", record.timed ? Json(record.span.end - record.span.start) : Json()},
        {"graph_id", raw.value("graph_id")}, {"graph_node_id", raw.value("graph_node_id")},
        {"step_ids", numbers(step_ids)}, {"request_ids", strings(request_ids)},
        {"source", activity.source.empty() ? "unknown" : activity.source},
        {"reason", activity.reason},
        {"status", affected(record.producer, record.span) || record.ambiguous ? "Partial" : "Complete"},
        {"timing_reliable", reliable(record)},
    });
    if (activity.api) row["api_record_id"] = activity.api->id;
    const auto bracket = brackets_.find(record.id);
    if (bracket != brackets_.end() && bracket->second.known) {
        row["clock_basis"] = Json::object({
            {"method", "per-activity-host-causal-window"},
            {"lower_ns", bracket->second.lower},
            {"upper_ns", bracket->second.upper},
            {"per_activity_timestamp_error_bound_ns", bracket->second.error},
            {"coverage", bracket->second.coverage},
            {"launch_api_record_id", bracket->second.launch_id},
            {"sync_record_id", bracket->second.sync_id},
            {"timestamps_inside_bracket", bracket->second.inside},
            {"global_offset_calibrated", false},
        });
    }
    return row;
}

Json Analyzer::request_rows() const {
    Json requests = Json::array();
    for (const auto& [key, request] : requests_) {
        (void)key;
        requests.push_back(request_report(request));
    }
    return requests;
}

// Idle GPU time inside each step, then between consecutive steps.
Json Analyzer::gap_rows() const {
    Json gaps = Json::array();
    for (const auto& [key, step] : steps_) {
        (void)key;
        append_gaps(step, gaps);
    }
    append_between_step_gaps(gaps);
    return gaps;
}

Json Analyzer::synchronization_rows() const {
    Json synchronization = Json::array();
    for (const auto* sync : synchronization_) {
        Json row = *sync->data;
        row["record_id"] = sync->id;
        row["status"] = affected(sync->producer, sync->span) ? "Partial" : "Complete";
        const auto matched = event_links_.find(sync->id);
        row["event_record_id"] = matched == event_links_.end() ? Json() : Json(matched->second->id);
        if (event_sync(*sync)) row["event_link_reason"] = matched == event_links_.end() ? "missing_invalid_or_ambiguous_event_identity" : "explicit_event_sync_id";
        synchronization.push_back(row);
    }
    return synchronization;
}

// The reader's diagnostics, then the analyzer's.
Json Analyzer::report_diagnostics() const {
    Json diagnostics = Json::array();
    for (const auto& item : trace_.diagnostics) diagnostics.push_back(item);
    for (const auto& item : diagnostics_) diagnostics.push_back(item);
    return diagnostics;
}

Json Analyzer::report_provenance() const {
    return Json::object({
        {"data_kind", data_kind()},
        {"hardware_validated", false},
        {"note", "Reports preserve input evidence; synthetic fixtures do not validate CUPTI, GPU, or eBPF behavior."},
    });
}

Json Analyzer::report_summary(std::size_t linked) const {
    return Json::object({
        {"gpu_activities", static_cast<Time>(activities_.size())},
        {"linked_gpu_activities", static_cast<Time>(linked)},
        {"unlinked_gpu_activities", static_cast<Time>(activities_.size() - linked)},
        {"attribution_completeness", activities_.empty() ? Json() : Json(static_cast<double>(linked) / activities_.size())},
        {"request_count", static_cast<Time>(requests_.size())},
        {"loss_records", static_cast<Time>(losses_.size())},
        {"loss_count", loss_count_},
        {"gpu_activity_duration_ns", gpu_duration_},
        {"gpu_wall_time_ns", union_duration(all_gpu_spans_)},
        {"gpu_duration_semantics", "Observed activity intervals; graph envelopes include internal gaps and do not measure precise GPU busy time."},
    });
}

Json Analyzer::clock_rows() const {
    Json clocks = Json::array();
    for (const auto& [producer, clock_error] : clock_errors_) clocks.push_back(clock_row(producer, clock_error));
    return clocks;
}

// A producer's clock bound and its basis: the session bound (header or
// periodic probe) or, without one, how many of its activities a
// per-activity bracket covers.
Json Analyzer::clock_row(const std::string& producer, const std::optional<Time>& clock_error) const {
    Json row = Json::object({{"producer", producer}, {"clock_error_ns", clock_error ? Json(*clock_error) : Json()}});
    if (const auto found = probe_stats_.find(producer); found != probe_stats_.end()) {
        row["clock_basis_method"] = "periodic-h2d-probe";
        row["probe"] = found->second;
    }
    if (!clock_error) {
        Time covered = 0, total = 0, max_bound = 0;
        for (const auto& activity : activities_) {
            if (activity.record->producer != producer) continue;
            ++total;
            const auto found = brackets_.find(activity.record->id);
            if (found == brackets_.end() || !found->second.known || !found->second.inside) continue;
            ++covered;
            max_bound = std::max(max_bound, found->second.error);
        }
        if (total) {
            row["clock_basis_method"] = "per-activity-host-causal-window";
            row["bracket_activities"] = total;
            row["bracket_covered"] = covered;
            row["bracket_max_error_ns"] = covered ? Json(max_bound) : Json();
            row["global_offset_calibrated"] = false;
        }
    }
    return row;
}

std::string Analyzer::data_kind() const {
    std::set<std::string> kinds;
    for (const auto& header : trace_.headers) {
        std::string kind = str(header, "data_kind", str(header, "source", "unknown"));
        if (flag(header, "synthetic")) kind = "synthetic";
        kinds.insert(kind);
    }
    if (kinds.empty()) return "unknown";
    return kinds.size() == 1 ? *kinds.begin() : "mixed";
}

const std::vector<Interval>& Analyzer::gpu_busy(const std::string& producer) const {
    auto [it, added] = gpu_busy_.try_emplace(producer);
    if (added) {
        std::vector<Interval> spans;
        for (const auto& activity : activities_)
            if (activity.record->producer == producer && activity.record->timed) spans.push_back(activity.record->span);
        it->second = unite(std::move(spans));
    }
    return it->second;
}

const Analyzer::TimeIndex& Analyzer::sync_index(const std::string& producer) const {
    auto [it, added] = sync_index_.try_emplace(producer);
    if (added) {
        auto& index = it->second;
        for (const auto* sync : synchronization_)
            if (sync->producer == producer && sync->timed) index.records.push_back(sync);
        std::stable_sort(index.records.begin(), index.records.end(),
                         [](const Record* a, const Record* b) { return a->span.start < b->span.start; });
        Time latest = 0;
        for (const auto* sync : index.records) index.latest_end.push_back(latest = std::max(latest, sync->span.end));
    }
    return it->second;
}

std::vector<const Record*> Analyzer::sched_for(const std::string& producer, const std::set<Time>& tids) const {
    std::vector<const Record*> result;
    for (const auto* record : scheduling_)
        if (str(*record->data, "target_producer", record->producer) == producer && record->timed &&
            number(*record->data, "tid") && tids.contains(num(*record->data, "tid"))) result.push_back(record);
    return result;
}

bool Analyzer::blocking_sync(const Record& record) const {
    const auto& raw = *record.data;
    if (raw.contains("return_value") && number(raw, "return_value") != std::optional<Time>{0}) return false;
    if (raw.contains("blocking")) return flag(raw, "blocking");
    // CUPTI's public CUpti_ActivitySynchronizationType values: event=1,
    // stream-wait-event=2 (device dependency, not a blocked launch thread),
    // stream=3, context=4. Preserve the numeric wire value from the collector.
    if (const auto kind = number(raw, "sync_type")) return *kind == 1 || *kind == 3 || *kind == 4;
    const auto kind = str(raw, "sync_type", str(raw, "kind"));
    return kind == "stream_synchronize" || kind == "context_synchronize" || kind == "event_synchronize" ||
           kind == "device_synchronize" || kind == "cudaDeviceSynchronize" ||
           kind == "cudaStreamSynchronize" || kind == "cudaEventSynchronize";
}

std::optional<Time> Analyzer::sync_tid(const Record& record) const {
    if (const auto tid = number(*record.data, "tid")) return tid;
    if (const auto corr = number(*record.data, "corr"))
        if (const auto* api = unique_api({record.producer, *corr})) return number(*api->data, "tid");
    return std::nullopt;
}

void Analyzer::append_gaps(const Step& step, Json& output, std::optional<Interval> override_span,
                           const std::string& preceding_step) const {
    if (!step.record->timed || step.record->span.start == step.record->span.end) return;
    const auto& producer = step.record->producer;
    const auto step_span = override_span.value_or(step.record->span);
    // The producer's GPU busy time clipped to the step: the same intervals
    // as uniting the clipped activities, without scanning them per step.
    std::vector<Interval> busy;
    const auto& all_busy = gpu_busy(producer);
    for (auto it = std::upper_bound(all_busy.begin(), all_busy.end(), step_span.start,
                                    [](Time t, const Interval& busy) { return t < busy.end; });
         it != all_busy.end() && it->start < step_span.end; ++it) {
        const Interval clipped{std::max(step_span.start, it->start), std::min(step_span.end, it->end)};
        if (clipped.end > clipped.start) busy.push_back(clipped);
    }
    std::vector<Interval> idle;
    Time cursor = step_span.start;
    for (const auto interval : busy) {
        if (interval.start > cursor) idle.push_back({cursor, interval.start});
        cursor = std::max(cursor, interval.end);
    }
    if (cursor < step_span.end) idle.push_back({cursor, step_span.end});
    const auto scheduler = sched_for(producer, step.tids);
    const auto bound = clock_bound_for(producer, step.gpu);
    const auto& syncs = sync_index(producer);
    for (const auto gap : idle) {
        std::vector<Time> boundaries{gap.start, gap.end};
        auto add = [&](Interval interval) {
            if (!overlaps(interval, gap)) return;
            boundaries.push_back(std::max(gap.start, interval.start));
            boundaries.push_back(std::min(gap.end, interval.end));
        };
        for (const auto* sched : scheduler) add(sched->span);
        const auto gap_syncs = syncs.overlapping(gap);
        for (const auto* sync : gap_syncs) add(sync->span);
        for (const auto& loss : losses_)
            if (loss.timed && (loss.producer.empty() || loss.producer == producer)) add(loss.span);
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
        for (std::size_t i = 1; i < boundaries.size(); ++i) {
            const Interval piece{boundaries[i - 1], boundaries[i]};
            const bool lost = affected(producer, piece);
            std::string category = "unknown";
            std::string reason;
            std::set<std::string> evidence{step.record->id};
            if (!preceding_step.empty()) evidence.insert(preceding_step);
            if (lost) reason = "overlapping_loss";
            else if (step.ambiguous) reason = "ambiguous_step";
            else if (!bound) reason = "missing_clock_calibration";
            else if (piece.end - piece.start < *bound) reason = "shorter_than_clock_error";
            else if (step.members.empty() && !override_span) category = "no_work_scheduled";
            else if (step.tids.size() != 1) reason = "unknown_or_multiple_launch_threads";
            else {
                const Time tid = *step.tids.begin();
                bool sync_evidence = false;
                for (const auto* sync : gap_syncs)
                    if (contains(sync->span, piece) && reliable(*sync) &&
                        sync_tid(*sync) == std::optional<Time>{tid} && blocking_sync(*sync)) {
                        sync_evidence = true;
                        evidence.insert(sync->id);
                    }
                std::set<std::string> states;
                for (const auto* sched : scheduler)
                    if (contains(sched->span, piece) && scheduler_reliable(*sched, producer, piece)) {
                        states.insert(str(*sched->data, "state", "unknown"));
                        evidence.insert(sched->id);
                    }
                if (states.size() > 1) reason = "conflicting_scheduler_evidence";
                else if (sync_evidence) category = "blocked_on_sync";
                else if (states == std::set<std::string>{"runnable"}) category = "launch_thread_runnable";
                else if (states == std::set<std::string>{"sleeping"}) category = "launch_thread_sleeping";
                else if (states == std::set<std::string>{"on_cpu"}) category = "launch_thread_on_cpu";
                else reason = "missing_scheduler_or_sync_evidence";
            }
            output.push_back(Json::object({
                {"producer", producer}, {"step_id", step.id},
                {"scope", override_span ? "between_steps" : "in_step"},
                {"start_ns", piece.start}, {"end_ns", piece.end}, {"duration_ns", piece.end - piece.start},
                {"category", category}, {"reason", reason},
                {"status", lost || step.ambiguous ? "Partial" : "Complete"},
                {"evidence_record_ids", strings(evidence)},
                {"clock_error_ns", bound ? Json(*bound) : Json()},
                {"causal_claim", "Observed launch-thread state only; this does not establish CPU throttling or a root cause."},
            }));
        }
    }
}

void Analyzer::append_between_step_gaps(Json& output) const {
    std::map<std::string, std::vector<const Step*>> producers;
    for (const auto& [key, step] : steps_)
        if (step.record->timed && !step.ambiguous) producers[key.first].push_back(&step);
    for (auto& [producer, steps] : producers) {
        (void)producer;
        std::sort(steps.begin(), steps.end(), [](const auto* a, const auto* b) {
            return std::tie(a->record->span.start, a->record->span.end) < std::tie(b->record->span.start, b->record->span.end);
        });
        const Step* latest = nullptr;
        for (const auto* step : steps) {
            if (latest && latest->record->span.end < step->record->span.start)
                append_gaps(*step, output, Interval{latest->record->span.end, step->record->span.start}, latest->record->id);
            if (!latest || latest->record->span.end < step->record->span.end) latest = step;
        }
    }
}

void Analyzer::build_replays(Json& output) const {
    std::map<std::string, std::vector<const Activity*>> replays;
    for (const auto& activity : activities_) {
        if (!activity.api || !number(*activity.record->data, "graph_id") || num(*activity.record->data, "graph_id") == 0) continue;
        replays[activity.api->id].push_back(&activity);
    }
    for (const auto& [launch_id, activities] : replays) {
        const auto& first = *activities.front();
        std::set<Time> graph_ids;
        std::set<Time> step_ids;
        std::set<std::string> request_ids;
        std::set<std::string> activity_ids;
        bool partial = false;
        for (const auto* activity : activities) {
            graph_ids.insert(num(*activity->record->data, "graph_id"));
            activity_ids.insert(activity->record->id);
            partial = partial || affected(activity->record->producer, activity->record->span);
            if (activity->step) {
                step_ids.insert(activity->step->id);
                for (const auto& member : activity->step->members) request_ids.insert(member.request);
            }
        }
        output.push_back(Json::object({
            {"producer", first.record->producer}, {"launch_record_id", launch_id},
            {"corr", first.record->data->value("corr")}, {"graph_ids", numbers(graph_ids)},
            {"step_ids", numbers(step_ids)}, {"request_ids", strings(request_ids)},
            {"activity_record_ids", strings(activity_ids)},
            {"membership_source", step_ids.empty() ? "unknown" : "engine-membership"},
            {"status", partial ? "Partial" : "Complete"},
        }));
    }
}

}  // namespace reqscope::detail

namespace reqscope {

Json analyze(const Trace& trace) { return detail::Analyzer(trace).report(); }

}  // namespace reqscope
