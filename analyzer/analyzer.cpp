#include "analyzer/analyzer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace reqscope {
namespace {

using Time = std::uint64_t;
using Key = std::pair<std::string, std::uint64_t>;
using RequestKey = std::pair<std::string, std::string>;
constexpr std::array<const char*, 6> kComponents{
    "gpu_exec", "gpu_idle_in_step", "between_steps_cpu", "between_steps", "not_scheduled", "residual"};

std::string str(const Json& value, const std::string& key, const std::string& fallback = "") {
    const auto& field = value[key];
    return field.is_string() ? field.as_string() : fallback;
}

std::optional<Time> number(const Json& value, const std::string& key) {
    const auto& field = value[key];
    if (!field.is_number()) return std::nullopt;
    try {
        return field.as_u64();
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

Time num(const Json& value, const std::string& key, Time fallback = 0) {
    return number(value, key).value_or(fallback);
}

bool flag(const Json& value, const std::string& key, bool fallback = false) {
    const auto& field = value[key];
    return field.is_bool() ? field.as_bool() : fallback;
}

Json strings(const std::set<std::string>& values) {
    Json result = Json::array();
    for (const auto& value : values) result.push_back(value);
    return result;
}

Json numbers(const std::set<Time>& values) {
    Json result = Json::array();
    for (const auto value : values) result.push_back(value);
    return result;
}

struct Interval {
    Time start = 0;
    Time end = 0;
};

bool overlaps(Interval a, Interval b) { return a.start < b.end && b.start < a.end; }
bool contains(Interval a, Interval b) { return a.start <= b.start && a.end >= b.end; }

std::vector<Interval> unite(std::vector<Interval> spans) {
    std::sort(spans.begin(), spans.end(), [](const auto& a, const auto& b) {
        return std::tie(a.start, a.end) < std::tie(b.start, b.end);
    });
    std::vector<Interval> result;
    for (const auto span : spans) {
        if (span.end <= span.start) continue;
        if (!result.empty() && span.start <= result.back().end)
            result.back().end = std::max(result.back().end, span.end);
        else
            result.push_back(span);
    }
    return result;
}

Time union_duration(std::vector<Interval> spans) {
    Time result = 0;
    for (const auto span : unite(std::move(spans))) {
        if (std::numeric_limits<Time>::max() - result < span.end - span.start)
            throw std::runtime_error("GPU duration sum exceeds the trace time range");
        result += span.end - span.start;
    }
    return result;
}

// A probe's copy is looked for this close to its bracket; farther means it was lost.
constexpr Time kProbeMatchWindowNs = 50'000'000;

Json empty_components() {
    Json result = Json::object();
    for (const auto* name : kComponents) result[name] = Time{0};
    return result;
}

struct Record {
    const Json* data = nullptr;
    std::string id;
    std::string producer;
    std::string type;
    Interval span;
    bool timed = false;
    bool ambiguous = false;
};

struct Loss {
    std::string producer;
    std::string target_producer;
    std::string reason;
    std::string record_id;
    Interval span;
    bool timed = false;
};

struct Member {
    std::string request;
    Time tokens = 0;
    std::string phase;
};

struct Step {
    const Record* record = nullptr;
    Time id = 0;
    std::vector<Member> members;
    std::vector<const Record*> apis;
    std::vector<const Record*> gpu;
    std::set<Time> tids;
    bool ambiguous = false;
};

struct Activity {
    const Record* record = nullptr;
    const Record* api = nullptr;
    Step* step = nullptr;
    std::string reason;
    std::string source;
};

struct ClockBracket {
    bool known = false;
    bool inside = false;
    Time lower = 0;
    Time upper = 0;
    Time error = 0;
    std::string coverage;
    std::string launch_id;
    std::string sync_id;
};

bool starts_with(const std::string& value, const char* prefix) {
    const auto needle = std::string(prefix);
    return value.size() >= needle.size() && value.compare(0, needle.size(), needle) == 0;
}

struct Request {
    RequestKey key;
    std::vector<const Record*> spans;
    std::vector<Step*> steps;
    std::vector<const Record*> gpu;
    double share = 0;
    bool share_known = true;
};

class Analyzer {
public:
    // Without edges the dependency graph is not recorded: blame needs none of
    // it, and on a 2M-record trace it was a quarter of the analyzer's memory.
    explicit Analyzer(const Trace& input, bool edges = true) : trace_(input), record_edges_(edges) {
        read_records();
        build_indices();
        derive_probe_bounds();     // before link(): probe copies are not application work
        link();
        assign_clock_brackets();
    }

    // Why were a request's slow tokens slow? For each inter-token gap well above
    // the request's median gap, take the engine step that produced the token
    // and report its GPU busy time (union of linked GPU intervals: durations
    // in the GPU's own clock, so no CPU/GPU alignment is needed) and the
    // requests co-scheduled with it. The gap's excess over the median is
    // attributed to co-scheduled prefill work in proportion to its tokens;
    // gaps with no co-scheduled prefill are reported as unexplained.
    // One request's token times and the steps that produced them, reassembled
    // from the RequestTokens chunks the adapter emits every 1024 tokens plus
    // the final Request record. Steps are returned only if every part has
    // them; any non-integer time makes the series invalid.
    std::tuple<std::vector<Time>, std::vector<Time>, bool> token_series(const Json& raw) const {
        std::vector<const Json*> parts;
        if (auto it = token_chunks_.find(str(raw, "producer") + '\n' + str(raw, "req_id")); it != token_chunks_.end())
            parts = it->second;
        parts.push_back(&raw);
        std::vector<Time> times, steps;
        bool valid = true, have_steps = true;
        for (const Json* part : parts) {
            const Json& t = (*part)["token_times"];
            const Json& s = (*part)["token_steps"];
            if (!t.is_array()) { valid = valid && !part->contains("token_times"); continue; }
            have_steps = have_steps && s.is_array() && s.array_items().size() == t.array_items().size();
            for (const auto& value : t.array_items()) {
                try { times.push_back(static_cast<Time>(value.as_u64())); }
                catch (const std::exception&) { valid = false; }
            }
            if (have_steps)
                for (const auto& value : s.array_items())
                    steps.push_back(value.is_number() ? static_cast<Time>(value.as_double()) : -1);
        }
        if (!have_steps || steps.size() != times.size()) steps.clear();
        return {times, steps, valid};
    }

    Json blame(const std::string& only, double slow_factor) const {
        Json rows = Json::array();
        std::map<std::string, Time> culprits;
        for (const auto& [key, request] : requests_) {
            if (request.spans.empty()) continue;
            const Json& raw = *request.spans.front()->data;
            const std::string id = str(raw, "req_id");
            if (!only.empty() && id != only) continue;
            // producers[k] is the step that produced token k. Without it, fall
            // back to the request's steps whose host span ended inside the gap,
            // which is off by one step under async scheduling.
            const auto [tokens, producers, tokens_valid] = token_series(raw);
            if (!tokens_valid || tokens.size() < 3 || !std::is_sorted(tokens.begin(), tokens.end())) continue;
            std::vector<Time> gaps;
            for (std::size_t i = 1; i < tokens.size(); ++i) gaps.push_back(tokens[i] - tokens[i - 1]);
            std::vector<Time> sorted = gaps;
            std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
            const Time median = sorted[sorted.size() / 2];
            Json slow = Json::array();
            Time excess_total = 0, unexplained = 0;
            for (std::size_t i = 0; i < gaps.size(); ++i) {
                if (static_cast<double>(gaps[i]) <= slow_factor * static_cast<double>(median)) continue;
                const Time begin = tokens[i], end = tokens[i + 1], excess = gaps[i] - median;
                excess_total += excess;
                // Steps of this request that ran inside the gap produced the token.
                Json steps = Json::array();
                std::map<std::string, Time> prefill;
                Time prefill_tokens = 0;
                for (const auto* step : request.steps) {
                    const auto span = step->record->span;
                    if (!producers.empty() ? step->id != producers[i + 1] : span.end <= begin || span.end > end)
                        continue;
                    std::vector<Interval> busy;
                    for (const auto* gpu : step->gpu) busy.push_back(gpu->span);
                    Json members = Json::array();
                    for (const auto& member : step->members) {
                        if (member.request == key.second) continue;
                        members.push_back(Json::object({{"req_id", member.request}, {"phase", member.phase},
                                                        {"tokens", member.tokens}}));
                        if (member.phase == "prefill") {
                            prefill[member.request] += member.tokens;
                            prefill_tokens += member.tokens;
                        }
                    }
                    steps.push_back(Json::object({{"step_id", step->id}, {"host_ns", span.end - span.start},
                        {"gpu_busy_ns", union_duration(busy)}, {"co_scheduled", members}}));
                }
                Json blamed = Json::array();
                for (const auto& [culprit, count] : prefill) {
                    const Time share = static_cast<Time>(static_cast<double>(excess) * count / prefill_tokens);
                    culprits[culprit] += share;
                    blamed.push_back(Json::object({{"req_id", culprit}, {"prefill_tokens", count}, {"excess_ns", share}}));
                }
                if (prefill.empty()) unexplained += excess;
                slow.push_back(Json::object({{"token_index", static_cast<Time>(i + 1)}, {"begin_ns", begin},
                    {"end_ns", end}, {"gap_ns", gaps[i]},
                    {"excess_ns", excess}, {"steps", steps}, {"blamed", blamed}}));
            }
            rows.push_back(Json::object({{"req_id", id}, {"tokens", static_cast<Time>(tokens.size())},
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
            {"loss_count", loss_count_}});
        return Json::object({{"schema_version", 1}, {"slow_factor", slow_factor},
            {"method", "token gap above slow_factor x median; excess split over prefill tokens co-scheduled "
                       "in the step that produced the token"},
            {"summary", summary}, {"clocks", clocks}, {"requests", rows}, {"culprits", by}});
    }

    Json report() {
        Json activities = Json::array();
        Json requests = Json::array();
        Json gaps = Json::array();
        Json replays = Json::array();
        Json synchronization = Json::array();
        std::size_t linked = 0;
        for (const auto& activity : activities_) {
            const auto& record = *activity.record;
            const auto& raw = *record.data;
            std::set<Time> step_ids;
            std::set<std::string> request_ids;
            if (activity.step) {
                step_ids.insert(activity.step->id);
                for (const auto& member : activity.step->members) request_ids.insert(member.request);
                ++linked;
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
            activities.push_back(row);
        }
        for (const auto& [key, request] : requests_) {
            (void)key;
            requests.push_back(request_report(request));
        }
        for (const auto& [key, step] : steps_) {
            (void)key;
            append_gaps(step, gaps);
        }
        append_between_step_gaps(gaps);
        build_replays(replays);
        for (const auto* sync : synchronization_) {
            Json row = *sync->data;
            row["record_id"] = sync->id;
            row["status"] = affected(sync->producer, sync->span) ? "Partial" : "Complete";
            const auto matched = event_links_.find(sync->id);
            row["event_record_id"] = matched == event_links_.end() ? Json() : Json(matched->second->id);
            if (event_sync(*sync)) row["event_link_reason"] = matched == event_links_.end() ? "missing_invalid_or_ambiguous_event_identity" : "explicit_event_sync_id";
            synchronization.push_back(row);
        }
        Json diagnostics = Json::array();
        for (const auto& item : trace_.diagnostics) diagnostics.push_back(item);
        for (const auto& item : diagnostics_) diagnostics.push_back(item);
        bool partial = trace_.partial || !losses_.empty() || !diagnostics_.empty();
        for (const auto& request : requests.array_items())
            partial = partial || str(request, "status") == "Partial";
        Json provenance = Json::object({
            {"data_kind", data_kind()},
            {"hardware_validated", false},
            {"note", "Reports preserve input evidence; synthetic fixtures do not validate CUPTI, GPU, or eBPF behavior."},
        });
        Json summary = Json::object({
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
        Json clocks = Json::array();
        for (const auto& [producer, error] : clock_errors_) {
            Json row = Json::object({{"producer", producer}, {"clock_error_ns", error ? Json(*error) : Json()}});
            if (const auto found = probe_stats_.find(producer); found != probe_stats_.end()) {
                row["clock_basis_method"] = "periodic-h2d-probe";
                row["probe"] = found->second;
            }
            if (!error) {
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
            clocks.push_back(row);
        }
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

private:
    const Trace& trace_;
    const bool record_edges_;
    std::vector<Record> records_;
    // RequestTokens chunks by producer and request id, ordered by "first".
    std::map<std::string, std::vector<const Json*>> token_chunks_;
    std::map<std::string, std::optional<Time>> clock_errors_;
    std::map<std::string, Json> probe_stats_;
    std::set<std::string> probe_copy_ids_;
    std::map<std::string, ClockBracket> brackets_;
    std::set<std::string> monotonic_producers_;
    std::map<std::string, std::set<std::string>> scheduler_targets_;
    std::vector<Loss> losses_;
    std::set<std::string> loss_keys_;
    std::map<std::string, Time> retention_counts_;
    std::vector<Json> diagnostics_;
    std::map<Key, Step> steps_;
    std::map<Key, std::vector<const Record*>> apis_;
    std::map<Key, std::vector<const Record*>> correlations_;
    std::map<RequestKey, Request> requests_;
    std::vector<Activity> activities_;
    std::vector<const Record*> scheduling_;
    std::vector<const Record*> synchronization_;
    std::vector<const Record*> cuda_events_;
    std::vector<const Record*> graph_resources_;
    std::map<std::string, bool> graph_resource_capture_;
    std::map<std::string, const Record*> event_links_;
    std::vector<Interval> all_gpu_spans_;
    // Per producer, built on first use: the union of its timed GPU activity,
    // and its timed synchronization records ordered by start.
    struct TimeIndex {
        std::vector<const Record*> records;
        std::vector<Time> latest_end;  // the latest end among records[0..i]
        std::vector<const Record*> overlapping(Interval window) const {
            std::vector<const Record*> result;
            // Every record before `i` ends at or before the window starts.
            auto i = std::size_t(std::upper_bound(latest_end.begin(), latest_end.end(), window.start) - latest_end.begin());
            for (; i < records.size() && records[i]->span.start < window.end; ++i)
                if (records[i]->span.end > window.start) result.push_back(records[i]);
            return result;
        }
    };
    mutable std::map<std::string, std::vector<Interval>> gpu_busy_;
    mutable std::map<std::string, TimeIndex> sync_index_;
    Json edges_ = Json::array();
    Json graph_definitions_ = Json::array();
    Json graph_topology_ = Json::array();
    std::set<std::tuple<std::string, std::string, std::string>> edge_keys_;
    Time loss_count_ = 0;
    double gpu_duration_ = 0;

    std::string data_kind() const {
        std::set<std::string> kinds;
        for (const auto& header : trace_.headers) {
            std::string kind = str(header, "data_kind", str(header, "source", "unknown"));
            if (flag(header, "synthetic")) kind = "synthetic";
            kinds.insert(kind);
        }
        if (kinds.empty()) return "unknown";
        return kinds.size() == 1 ? *kinds.begin() : "mixed";
    }

    void diagnostic(const std::string& reason, const Record* record = nullptr) {
        Json item = Json::object({{"reason", reason}, {"status", "Partial"}});
        if (record) {
            item["record_id"] = record->id;
            item["producer"] = record->producer;
        }
        diagnostics_.push_back(item);
    }

    void add_loss(const Json& raw, const std::string& record_id = "") {
        const std::string loss_key = str(raw, "producer") + "|" + str(raw, "reason", "missing_evidence") + "|" +
            raw.value("seq_range").dump() + "|" + raw.value("time_range").dump() +
            (raw["seq_range"].is_array() ? "" : "|" + record_id);
        if (!loss_keys_.insert(loss_key).second) return;
        Loss loss;
        loss.producer = str(raw, "producer");
        loss.target_producer = str(raw, "target_producer");
        loss.reason = str(raw, "reason", "missing_evidence");
        loss.record_id = record_id;
        const auto& range = raw["time_range"];
        if (flag(raw, "range_known", true) && range.is_array() && range.size() == 2 && range[0].is_number() && range[1].is_number()) {
            try {
                loss.span = {range[0].as_u64(), range[1].as_u64()};
                loss.timed = loss.span.end > loss.span.start;
            } catch (const std::exception&) {
                loss.timed = false;
            }
        }
        losses_.push_back(loss);
        Time count = num(raw, "count", 0);
        if (str(raw, "reason") == "ring_overwrite") {
            auto& previous = retention_counts_[str(raw, "producer")];
            const auto maximum = std::max(previous, count);
            count = maximum - previous;
            previous = maximum;
        }
        loss_count_ = count > std::numeric_limits<Time>::max() - loss_count_
            ? std::numeric_limits<Time>::max() : loss_count_ + count;
    }

    void read_records() {
        std::set<std::string> sessions_requiring_boundary;
        std::map<std::string, Time> terminal_sequence;
        std::map<std::string, Time> last_work_sequence;
        auto session_loss = [&](const std::string& producer, const std::string& reason) {
            Json issue = Json::object({{"producer", producer}, {"reason", reason}, {"status", "Partial"}});
            diagnostics_.push_back(issue);
            add_loss(issue);
        };
        for (const auto& header : trace_.headers) {
            const auto producer = str(header, "producer");
            if (str(header, "clock") == "CLOCK_MONOTONIC") monotonic_producers_.insert(producer);
            const bool graph_capture = flag(header, "graph_resource_callbacks", false);
            auto [graph_capture_it, inserted_graph_capture] = graph_resource_capture_.emplace(producer, graph_capture);
            if (!inserted_graph_capture) graph_capture_it->second = graph_capture_it->second && graph_capture;
            const auto state = str(header, "session_state");
            const auto backend = str(header, "backend_status");
            if (state == "Configured" || state == "Attached" || state == "Capturing" || state == "Draining")
                sessions_requiring_boundary.insert(producer);
            if (state == "Partial" || backend == "unavailable" || backend == "failed")
                session_loss(producer, "capture_backend_unavailable_or_partial");
            if (flag(header, "retention_partial")) session_loss(producer, "retention_partial");
            if (header["retention_loss"].is_object() && num(header["retention_loss"], "count") > 0) {
                Json retention = header["retention_loss"];
                retention["producer"] = producer;
                add_loss(retention);
            }
            auto error = number(header, "clock_error_ns");
            if (str(header, "clock", "unknown") != "CLOCK_MONOTONIC") error.reset();
            auto it = clock_errors_.find(producer);
            if (it == clock_errors_.end())
                clock_errors_[producer] = error;
            else if (!it->second || !error)
                it->second.reset();
            else
                it->second = std::max(*it->second, *error);
        }
        records_.reserve(trace_.records.size());
        std::map<std::string, std::size_t> ids;
        for (const auto& raw : trace_.records) {
            Record record;
            record.data = &raw;
            record.type = str(raw, "type");
            record.producer = str(raw, "producer");
            if (record.type == "Sched") {
                const auto target = str(raw, "target_producer", record.producer);
                scheduler_targets_[record.producer].insert(target);
            }
            record.id = record.producer + ":" + std::to_string(num(raw, "seq"));
            if (record.type == "Session") {
                const auto state = str(raw, "state");
                if (state == "Configured" || state == "Attached" || state == "Capturing" || state == "Draining") {
                    sessions_requiring_boundary.insert(record.producer);
                    last_work_sequence[record.producer] = std::max(last_work_sequence[record.producer], num(raw, "seq"));
                }
                if (state == "Finalized" || state == "Snapshot")
                    terminal_sequence[record.producer] = std::max(terminal_sequence[record.producer], num(raw, "seq"));
                if (state == "Partial") session_loss(record.producer, "partial_session");
            } else if (record.type != "Loss" && record.type != "ClockCalibration") {
                last_work_sequence[record.producer] = std::max(last_work_sequence[record.producer], num(raw, "seq"));
            }
            const auto start = number(raw, record.type == "Step" ? "t_begin" : "start");
            const auto end = number(raw, record.type == "Step" ? "t_end" : "end");
            // CUPTI reports 0 for a GPU timestamp it could not collect.
            if (start && end && *end >= *start && (record.type != "GpuActivity" || *start > 0)) {
                record.span = {*start, *end};
                record.timed = true;
            }
            if (record.producer.empty() || !number(raw, "seq") || num(raw, "seq") == 0) {
                record.ambiguous = true;
                diagnostic("invalid_record_identity", &record);
            }
            if (const auto found = ids.find(record.id); found != ids.end()) {
                record.ambiguous = true;
                records_[found->second].ambiguous = true;
                diagnostic("duplicate_record_identity", &record);
            } else {
                ids[record.id] = records_.size();
            }
            if ((record.type == "Step" || record.type == "ApiCall" || record.type == "GraphLaunch" ||
                 record.type == "GpuActivity" || record.type == "Request" || record.type == "Sync" ||
                 record.type == "Sched") && !record.timed) {
                record.ambiguous = true;
                diagnostic("invalid_time_interval", &record);
            }
            if (record.type == "Loss" && num(raw, "count", 1) > 0) add_loss(raw, record.id);
            if (record.type == "RequestTokens") token_chunks_[str(raw, "producer") + '\n' + str(raw, "req_id")].push_back(&raw);
            records_.push_back(record);
        }
        for (auto& [key, chunks] : token_chunks_)
            std::sort(chunks.begin(), chunks.end(), [](const Json* a, const Json* b) {
                return num(*a, "first") < num(*b, "first");
            });
        for (const auto& producer : sessions_requiring_boundary)
            if (!terminal_sequence.contains(producer) || terminal_sequence[producer] < last_work_sequence[producer])
                session_loss(producer, "missing_finalized_or_snapshot_boundary");
        for (const auto& diagnostic : trace_.diagnostics) add_loss(diagnostic);
        if (trace_.partial && trace_.diagnostics.empty() && losses_.empty()) add_loss(Json::object({{"reason", "partial_input"}}));
    }

    bool affected(const std::string& producer, Interval span) const {
        for (const auto& loss : losses_) {
            const auto observed = scheduler_targets_.find(loss.producer);
            const bool scheduler_loss = observed != scheduler_targets_.end() && observed->second.contains(producer);
            if (!loss.target_producer.empty() && loss.target_producer != producer) continue;
            if (!loss.producer.empty() && loss.producer != producer && !scheduler_loss && loss.target_producer != producer) continue;
            if (!loss.timed || overlaps(loss.span, span)) return true;
        }
        return false;
    }

    std::optional<Time> error(const std::string& producer) const {
        const auto found = clock_errors_.find(producer);
        return found == clock_errors_.end() ? std::nullopt : found->second;
    }

    // Session clock bound from ClockProbe records. The collector brackets a
    // 4-byte H2D copy on a private stream between host CLOCK_MONOTONIC reads
    // t0 and t1. CUPTI reports GPU activity in that same clock (its timestamp
    // callback), so the copy's reported [start, end] must lie in [t0, t1]: the
    // bracket width bounds the CPU/GPU alignment error at that moment, and a
    // reported time outside the bracket adds its distance. The session bound is
    // the worst probe. A header-supplied bound takes precedence.
    void derive_probe_bounds() {
        std::map<std::tuple<std::string, Time, Time>, std::vector<const Record*>> copies;
        std::map<std::string, std::vector<const Record*>> probes;
        std::map<std::string, Time> last_gpu_end;
        for (const auto& record : records_) {
            if (record.ambiguous) continue;
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
            probe_stats_[producer] = stats;
            auto& error = clock_errors_[producer];
            if (!error && !widths.empty() && !unpaired) error = bound;
        }
    }

    const std::vector<Interval>& gpu_busy(const std::string& producer) const {
        auto [it, added] = gpu_busy_.try_emplace(producer);
        if (added) {
            std::vector<Interval> spans;
            for (const auto& activity : activities_)
                if (activity.record->producer == producer && activity.record->timed) spans.push_back(activity.record->span);
            it->second = unite(std::move(spans));
        }
        return it->second;
    }

    const TimeIndex& sync_index(const std::string& producer) const {
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

    std::optional<Time> clock_bound_for(const std::string& producer, const std::vector<const Record*>& gpu) const {
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

    bool reliable(const Record& record) const {
        const auto bound = error(record.producer);
        return record.timed && !record.ambiguous && bound && record.span.end - record.span.start >= *bound;
    }

    void edge(const std::string& from, const std::string& to, const std::string& source,
              const std::vector<std::string>& evidence = {}, const Json& attributes = Json()) {
        if (!record_edges_ || !edge_keys_.emplace(from, to, source).second) return;
        Json ids = Json::array();
        for (const auto& id : evidence) ids.push_back(id);
        Json record = Json::object({{"from", from}, {"to", to}, {"source", source}, {"evidence_record_ids", ids}});
        if (attributes.is_object())
            for (const auto& [key, value] : attributes.object_items()) record[key] = value;
        edges_.push_back(std::move(record));
    }

    Request& request(const RequestKey& key) {
        auto [it, inserted] = requests_.try_emplace(key);
        if (inserted) it->second.key = key;
        return it->second;
    }

    void build_indices() {
        for (const auto& record : records_) {
            const auto& raw = *record.data;
            const auto corr = number(raw, "corr");
            if ((record.type == "ApiCall" || record.type == "GraphLaunch") && corr)
                apis_[{record.producer, *corr}].push_back(&record);
            else if (record.type == "ExtCorr" && corr)
                correlations_[{record.producer, *corr}].push_back(&record);
            else if (record.type == "Sched")
                scheduling_.push_back(&record);
            else if (record.type == "Sync")
                synchronization_.push_back(&record);
            else if (record.type == "CudaEvent")
                cuda_events_.push_back(&record);
            else if (record.type == "GraphResource")
                graph_resources_.push_back(&record);
            else if (record.type == "GraphDefinition")
                graph_definitions_.push_back(raw);
            if (record.type == "Request") {
                const auto id = str(raw, "req_id");
                if (id.empty()) diagnostic("request_missing_req_id", &record);
                else request({record.producer, id}).spans.push_back(&record);
            }
            if (record.type != "Step") continue;
            const auto step_id = number(raw, "step_id");
            if (!step_id) {
                diagnostic("step_missing_step_id", &record);
                continue;
            }
            auto [it, inserted] = steps_.try_emplace(Key{record.producer, *step_id});
            if (!inserted) {
                it->second.ambiguous = true;
                diagnostic("duplicate_step_id", &record);
                continue;
            }
            auto& step = it->second;
            step.record = &record;
            step.id = *step_id;
            step.ambiguous = record.ambiguous;
            if (const auto tid = number(raw, "tid")) step.tids.insert(*tid);
            if (!raw["members"].is_array()) {
                step.ambiguous = true;
                diagnostic("step_missing_members", &record);
                continue;
            }
            std::set<std::string> seen;
            for (const auto& member : raw["members"].array_items()) {
                const auto id = str(member, "req_id");
                if (id.empty() || !seen.insert(id).second) {
                    step.ambiguous = true;
                    diagnostic("invalid_or_duplicate_step_member", &record);
                    continue;
                }
                step.members.push_back({id, num(member, "tokens"), str(member, "phase", "unknown")});
            }
        }
        for (auto& [key, step] : steps_) {
            for (const auto& member : step.members) {
                request({key.first, member.request}).steps.push_back(&step);
                if (!step.ambiguous)
                    edge("request:" + key.first + ":" + member.request, step.record->id,
                         "engine-membership", {step.record->id});
            }
        }
    }

    const Record* unique_api(const Key& key) const {
        const auto it = apis_.find(key);
        if (it == apis_.end()) return nullptr;
        // GraphLaunch may be an additional view of the same API. Only collapse
        // the pair if its exact correlation, thread, and interval agree.
        const Record* chosen = nullptr;
        for (const auto* candidate : it->second) {
            if (candidate->ambiguous) return nullptr;
            if (!chosen) {
                chosen = candidate;
                continue;
            }
            if (candidate->type == chosen->type || candidate->span.start != chosen->span.start ||
                candidate->span.end != chosen->span.end ||
                number(*candidate->data, "tid") != number(*chosen->data, "tid")) return nullptr;
            if (candidate->type == "ApiCall") chosen = candidate;
        }
        return chosen;
    }

    Step* linked_step(const Key& key, std::vector<std::string>& evidence, std::string& reason) {
        const auto it = correlations_.find(key);
        if (it == correlations_.end()) {
            reason = "missing_external_correlation";
            return nullptr;
        }
        std::set<Time> ids;
        for (const auto* correlation : it->second) {
            const auto id = number(*correlation->data, "step_id");
            if (!id || correlation->ambiguous) {
                reason = "invalid_external_correlation";
                return nullptr;
            }
            ids.insert(*id);
            evidence.push_back(correlation->id);
        }
        if (ids.size() != 1) {
            reason = "ambiguous_external_correlation";
            return nullptr;
        }
        auto found = steps_.find({key.first, *ids.begin()});
        if (found == steps_.end() || found->second.ambiguous) {
            reason = found == steps_.end() ? "missing_step" : "ambiguous_step";
            return nullptr;
        }
        return &found->second;
    }

    void link() {
        for (const auto& [key, entries] : apis_) {
            (void)entries;
            const auto* api = unique_api(key);
            if (!api) continue;
            std::vector<std::string> evidence;
            std::string reason;
            if (auto* step = linked_step(key, evidence, reason)) {
                step->apis.push_back(api);
                if (const auto tid = number(*api->data, "tid")) step->tids.insert(*tid);
                edge(step->record->id, api->id, "external-correlation", evidence);
            }
        }
        for (const auto& record : records_) {
            if (record.type != "GpuActivity" || probe_copy_ids_.contains(record.id)) continue;
            const auto& raw = *record.data;
            Activity activity;
            activity.record = &record;
            if (record.timed) {
                gpu_duration_ += static_cast<double>(record.span.end - record.span.start);
                all_gpu_spans_.push_back(record.span);
            }
            const auto corr = number(raw, "corr");
            if (record.ambiguous)
                activity.reason = "invalid_record";
            else if (!corr)
                activity.reason = "missing_correlation_id";
            else {
                const Key key{record.producer, *corr};
                activity.api = unique_api(key);
                if (!activity.api)
                    activity.reason = apis_.contains(key) ? "ambiguous_api_correlation" : "missing_api";
                else {
                    edge(activity.api->id, record.id, "cupti-correlation", {activity.api->id, record.id});
                    std::vector<std::string> evidence;
                    activity.step = linked_step(key, evidence, activity.reason);
                    if (activity.step) {
                        activity.source = "external-correlation";
                        activity.reason = "";
                    }
                }
            }
            if (activity.step) {
                auto& step = *activity.step;
                step.gpu.push_back(&record);
                double total_tokens = 0;
                for (const auto& member : step.members) total_tokens += static_cast<double>(member.tokens);
                for (const auto& member : step.members) {
                    auto& target = request({record.producer, member.request});
                    target.gpu.push_back(&record);
                    if (total_tokens > 0 && reliable(record))
                        target.share += static_cast<double>(record.span.end - record.span.start) * member.tokens / total_tokens;
                    else
                        target.share_known = false;
                }
            } else {
                edge("unknown:" + record.producer, record.id, "unknown", {record.id});
            }
            activities_.push_back(activity);
        }
        link_streams();
        link_graph_topology();
        using EventKey = std::tuple<std::string, Time, Time>;
        std::map<EventKey, std::vector<const Record*>> events;
        for (const auto* event : cuda_events_) {
            if (event->ambiguous) continue;
            if (const auto corr = number(*event->data, "corr"))
                if (const auto* api = unique_api({event->producer, *corr}))
                    edge(api->id, event->id, "cupti-correlation", {api->id, event->id});
            const auto ctx = number(*event->data, "ctx");
            const auto id = number(*event->data, "event_sync_id");
            if (ctx && id && valid_event_sync_id(*id)) events[{event->producer, *ctx, *id}].push_back(event);
        }
        for (const auto* sync : synchronization_) {
            if (const auto corr = number(*sync->data, "corr"))
                if (const auto* api = unique_api({sync->producer, *corr}))
                    edge(api->id, sync->id, "sync-record", {api->id, sync->id});
            const auto ctx = number(*sync->data, "ctx");
            const auto id = number(*sync->data, "event_sync_id");
            if (!sync->ambiguous && event_sync(*sync) && ctx && id && valid_event_sync_id(*id)) {
                const auto found = events.find({sync->producer, *ctx, *id});
                if (found != events.end() && found->second.size() == 1) {
                    const auto* event = found->second.front();
                    const auto recorded_id = number(*event->data, "event");
                    const auto waited_id = number(*sync->data, "event");
                    if ((!recorded_id || !waited_id || recorded_id == waited_id) && num(*sync->data, "return_value") == 0) {
                        edge(event->id, sync->id, "sync-record", {event->id, sync->id});
                        event_links_[sync->id] = event;
                    }
                }
            }
            if (const auto dependency = number(*sync->data, "wait_for_corr"))
                for (const auto& activity : activities_)
                    if (activity.record->producer == sync->producer &&
                        number(*activity.record->data, "corr") == dependency)
                        edge(activity.record->id, sync->id, "sync-record", {sync->id});
        }
    }

    void link_graph_topology() {
        using ReplayKey = std::tuple<std::string, Time, Time>;
        using NodeEdge = std::pair<Time, Time>;
        std::map<ReplayKey, std::vector<const Activity*>> replays;
        std::map<Key, std::vector<const Record*>> resources, executions;
        std::map<std::string, std::vector<const Record*>> unsupported_updates;
        for (const auto* record : graph_resources_) {
            const auto& raw = *record->data;
            if (const auto graph = number(raw, "graph_id"); graph && *graph)
                resources[{record->producer, *graph}].push_back(record);
            if (const auto exec = number(raw, "graph_exec_id"); exec && *exec)
                executions[{record->producer, *exec}].push_back(record);
            const auto action = str(raw, "action");
            if (action == "node_updated" || action == "node_set_params")
                unsupported_updates[record->producer].push_back(record);
        }
        // Activity API records do not carry the executable handle for updates.
        // Without a reliable targeted invalidation, conservatively reject later
        // topology projections in that producer instead of retaining stale edges.
        for (const auto& record : records_)
            if (record.type == "ApiCall" && str(*record.data, "name").find("GraphExecUpdate") != std::string::npos)
                unsupported_updates[record.producer].push_back(&record);
        for (const auto& activity : activities_) {
            const auto graph = number(*activity.record->data, "graph_id");
            const auto corr = number(*activity.record->data, "corr");
            if (graph && *graph && corr) replays[{activity.record->producer, *graph, *corr}].push_back(&activity);
        }
        auto valid = [](const Record* record) {
            return !record->ambiguous && flag(*record->data, "ids_valid", false) &&
                   number(*record->data, "timestamp").has_value();
        };
        for (const auto& [key, activities] : replays) {
            const auto& [producer, graph, corr] = key;
            const auto* api = unique_api({producer, corr});
            Json row = Json::object({{"producer", producer}, {"graph_id", graph}, {"corr", corr},
                {"launch_record_id", api ? Json(api->id) : Json()}, {"status", "Partial"},
                {"dependency_edges", 0}, {"reason", "missing_graph_resource_evidence"}});
            auto project = [&]() -> std::string {
                if (!api || !api->timed || api->ambiguous || !monotonic_producers_.contains(producer)) return "ambiguous_graph_launch";
                if (!graph_resource_capture_[producer]) return "missing_graph_resource_capture_contract";
                if (api->data->contains("return_value") && number(*api->data, "return_value") != std::optional<Time>{0})
                    return "failed_graph_launch";
                const auto found = resources.find({producer, graph});
                const auto exec_found = executions.find({producer, graph});
                if (found == resources.end() || exec_found == executions.end()) return "missing_graph_resource_evidence";
                const Record* created = nullptr;
                const Record* ready = nullptr;
                for (const auto* record : found->second)
                    if (str(*record->data, "action") == "created") {
                        if (created || !valid(record)) return "ambiguous_graph_lifecycle";
                        created = record;
                    }
                for (const auto* record : exec_found->second)
                    if (str(*record->data, "action") == "exec_created") {
                        if (ready || !valid(record)) return "ambiguous_graph_executable";
                        ready = record;
                    }
                if (!created || !ready) return "missing_graph_lifecycle";
                const Time began = num(*created->data, "timestamp");
                const Time instantiated = num(*ready->data, "timestamp");
                if (began >= instantiated || instantiated >= api->span.start) return "unordered_graph_lifecycle";
                if (affected(producer, {began, api->span.end})) return "loss_during_graph_lifecycle";
                for (const auto* update : unsupported_updates[producer]) {
                    if (update->type == "GraphResource") {
                        const auto target = number(*update->data, "graph_exec_id");
                        if (target && *target && *target != graph) continue;
                    }
                    const auto timestamp = update->type == "ApiCall" ? number(*update->data, "start") : number(*update->data, "timestamp");
                    if (!timestamp || *timestamp <= api->span.end) return "unsupported_graph_update";
                }
                for (const auto* record : exec_found->second) {
                    if (!valid(record)) return "invalid_graph_resource";
                    if (str(*record->data, "action") == "exec_destroy" && num(*record->data, "timestamp") <= api->span.end)
                        return "graph_executable_destroyed_or_racing_launch";
                }
                std::vector<const Record*> history = found->second;
                std::sort(history.begin(), history.end(), [](const auto* a, const auto* b) {
                    return num(*a->data, "timestamp") < num(*b->data, "timestamp");
                });
                std::map<Time, const Record*> nodes;
                std::map<NodeEdge, const Record*> dependencies;
                Time last_timestamp = 0;
                for (const auto* record : history) {
                    if (!valid(record)) return "invalid_graph_resource";
                    const auto& raw = *record->data;
                    const Time timestamp = num(raw, "timestamp");
                    if (timestamp > api->span.end) continue;
                    if (timestamp < began || (last_timestamp && timestamp == last_timestamp)) return "ambiguous_graph_resource_order";
                    last_timestamp = timestamp;
                    const auto action = str(raw, "action");
                    if (action == "destroy" || action == "node_destroy") return "graph_destroyed_or_racing_launch";
                    if (timestamp >= instantiated && action != "created") return "unsupported_post_instantiation_graph_change";
                    if (action == "node_created") {
                        const auto node = number(raw, "node_id");
                        if (!node || !*node || !nodes.emplace(*node, record).second) return "ambiguous_graph_node_identity";
                    } else if (action == "dependency_created" || action == "dependency_destroy") {
                        const auto from = number(raw, "dependency_node_id");
                        const auto to = number(raw, "node_id");
                        if (!from || !*from || !to || !*to || from == to) return "invalid_graph_dependency";
                        const NodeEdge endpoints{*from, *to};
                        if (action == "dependency_created") {
                            if (!dependencies.emplace(endpoints, record).second) return "ambiguous_graph_dependency";
                        } else if (!dependencies.erase(endpoints)) return "missing_graph_dependency_creation";
                    } else if (action != "created" && action != "node_clone") return "unsupported_graph_resource_action";
                }
                std::map<Time, const Record*> observed;
                const Time context = num(*ready->data, "ctx");
                for (const auto* activity : activities) {
                    const auto& record = *activity->record;
                    const auto node = number(*record.data, "graph_node_id");
                    if (record.ambiguous || !node || !*node || flag(*record.data, "is_device_launched") ||
                        (context && num(*record.data, "ctx") != context)) return "unresolved_graph_activity_identity";
                    if (!observed.emplace(*node, &record).second) return "repeated_graph_node_without_iteration_identity";
                }
                // Do not invent transitive edges through unobserved host,
                // conditional, empty or nested nodes. The callback node_type
                // field is not assumed populated on this CUPTI version.
                if (observed.size() != nodes.size()) return "unobserved_or_missing_graph_nodes";
                for (const auto& [node, record] : observed) {
                    (void)record;
                    if (!nodes.contains(node)) return "missing_graph_node_creation";
                }
                // A cycle means the metadata is inconsistent with a static DAG.
                std::map<Time, std::size_t> indegree;
                std::map<Time, std::vector<Time>> successors;
                for (const auto& [node, record] : nodes) { (void)record; indegree[node] = 0; }
                for (const auto& [endpoints, dependency] : dependencies) {
                    (void)dependency;
                    if (!nodes.contains(endpoints.first) || !nodes.contains(endpoints.second)) return "missing_graph_dependency_endpoint";
                    ++indegree[endpoints.second];
                    successors[endpoints.first].push_back(endpoints.second);
                }
                std::vector<Time> ready_nodes;
                for (const auto& [node, degree] : indegree) if (!degree) ready_nodes.push_back(node);
                for (std::size_t i = 0; i < ready_nodes.size(); ++i)
                    for (const auto next : successors[ready_nodes[i]]) if (--indegree[next] == 0) ready_nodes.push_back(next);
                if (ready_nodes.size() != nodes.size()) return "cyclic_graph_resource_dependencies";
                for (const auto& [endpoints, dependency] : dependencies) {
                    const auto* from = observed.at(endpoints.first);
                    const auto* to = observed.at(endpoints.second);
                    edge(from->id, to->id, "cupti-resource",
                         {created->id, ready->id, nodes.at(endpoints.first)->id, nodes.at(endpoints.second)->id,
                          dependency->id, api->id, from->id, to->id},
                         Json::object({{"relation", "graph-dependency"}, {"graph_id", graph}, {"corr", corr},
                             {"from_graph_node_id", endpoints.first}, {"to_graph_node_id", endpoints.second},
                             {"gpu_timestamps_used", false},
                             {"dependency_semantics", "Explicit CUPTI graph dependency; edge ports and dependency type are not recorded."}}));
                }
                row["dependency_edges"] = static_cast<Time>(dependencies.size());
                row["status"] = "Complete";
                return "explicit_executable_graph_resources";
            };
            row["reason"] = project();
            graph_topology_.push_back(row);
        }
    }

    static bool valid_event_sync_id(Time id) {
        return id != 0 && id != std::numeric_limits<std::uint32_t>::max() && id != std::numeric_limits<Time>::max();
    }

    static bool event_sync(const Record& record) {
        if (const auto kind = number(*record.data, "sync_type")) return *kind == 1 || *kind == 2;
        const auto kind = str(*record.data, "sync_type");
        return kind == "event_synchronize" || kind == "stream_wait_event" ||
               kind == "cudaEventSynchronize" || kind == "cudaStreamWaitEvent";
    }

    void link_streams() {
        using StreamKey = std::tuple<std::string, Time, Time>;
        struct Submission { const Record* operation; const Record* api; };
        std::map<StreamKey, std::vector<Submission>> streams;
        auto add = [&](const Record& record, const Record* api) {
            const auto ctx = number(*record.data, "ctx");
            const auto stream = number(*record.data, "stream");
            if (!api || record.ambiguous || !api->timed || api->ambiguous ||
                api->span.start >= api->span.end || !number(*api->data, "tid") ||
                !ctx || !stream || *ctx == std::numeric_limits<std::uint32_t>::max() ||
                *stream == std::numeric_limits<std::uint32_t>::max() ||
                !monotonic_producers_.contains(record.producer)) return;
            if (api->data->contains("return_value") && number(*api->data, "return_value") != std::optional<Time>{0}) return;
            streams[{record.producer, *ctx, *stream}].push_back({&record, api});
        };
        for (const auto& activity : activities_) {
            const auto& record = *activity.record;
            // Graph nodes can execute independently while reporting the launch
            // stream. Their observed order is not graph dependency evidence.
            if (num(*record.data, "graph_id") != 0 || num(*record.data, "graph_node_id") != 0) continue;
            add(record, activity.api);
        }
        for (const auto* event : cuda_events_)
            if (const auto corr = number(*event->data, "corr")) add(*event, unique_api({event->producer, *corr}));
        for (const auto* sync : synchronization_) {
            const auto kind = number(*sync->data, "sync_type");
            const auto kind_name = str(*sync->data, "sync_type");
            if (!(kind == std::optional<Time>{2} || kind_name == "stream_wait_event" || kind_name == "cudaStreamWaitEvent")) continue;
            if (sync->data->contains("return_value") && number(*sync->data, "return_value") != std::optional<Time>{0}) continue;
            if (const auto corr = number(*sync->data, "corr")) add(*sync, unique_api({sync->producer, *corr}));
        }
        for (auto& [key, stream] : streams) {
            std::sort(stream.begin(), stream.end(), [](const auto& a, const auto& b) {
                return std::tie(a.api->span.start, a.api->span.end, a.operation->id) <
                       std::tie(b.api->span.start, b.api->span.end, b.operation->id);
            });
            for (std::size_t i = 1; i < stream.size(); ++i) {
                const auto& previous = stream[i - 1];
                const auto& current = stream[i];
                if (previous.api == current.api || num(*previous.api->data, "tid") != num(*current.api->data, "tid") ||
                    previous.api->span.end >= current.api->span.start ||
                    str(*previous.api->data, "tid_domain") != str(*current.api->data, "tid_domain") ||
                    affected(current.operation->producer, {previous.api->span.start, current.api->span.end})) continue;
                // The dependence comes from CUDA stream semantics and serial
                // host submission on one identified thread. These API clocks
                // are already CLOCK_MONOTONIC; no GPU timestamp is used to
                // decide their order and no CPU/GPU calibration is fabricated.
                edge(previous.operation->id, current.operation->id, "stream-order",
                     {previous.operation->id, current.operation->id, previous.api->id, current.api->id},
                     Json::object({{"ordering_basis", "serialized-host-api"},
                         {"ctx", std::get<1>(key)}, {"stream", std::get<2>(key)},
                         {"tid", num(*current.api->data, "tid")},
                         {"gpu_timestamps_used", false}}));
            }
        }
    }

    std::vector<const Record*> sched_for(const std::string& producer, const std::set<Time>& tids) const {
        std::vector<const Record*> result;
        for (const auto* record : scheduling_)
            if (str(*record->data, "target_producer", record->producer) == producer && record->timed &&
                number(*record->data, "tid") && tids.contains(num(*record->data, "tid"))) result.push_back(record);
        return result;
    }

    bool scheduler_reliable(const Record& record, const std::string& target, Interval piece) const {
        const auto source_bound = error(record.producer);
        const auto target_bound = error(target);
        return reliable(record) && source_bound && target_bound &&
            piece.end - piece.start >= std::max(*source_bound, *target_bound);
    }

    bool explicit_success(const Record& record) const {
        return number(*record.data, "return_value") == std::optional<Time>{0};
    }

    bool system_tid(const Record& api) const {
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

    bool producer_clock_blocked(const std::string& producer) const {
        for (const auto& loss : losses_) {
            if (!loss.producer.empty() && loss.producer != producer) continue;
            if (!loss.timed) return true;
        }
        return false;
    }

    std::optional<int> host_completion_kind(const Record& sync) const {
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

    bool sync_api_matches(int kind, const Record& api) const {
        const auto name = str(*api.data, "name", str(*api.data, "cbid"));
        if (kind == 3) return starts_with(name, "cudaStreamSynchronize") || starts_with(name, "cuStreamSynchronize");
        return starts_with(name, "cudaDeviceSynchronize") || starts_with(name, "cudaThreadSynchronize") ||
               starts_with(name, "cuCtxSynchronize");
    }

    void assign_clock_brackets() {
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

    Json request_report(const Request& request) const {
        const auto& [producer, id] = request.key;
        std::set<Time> step_ids;
        std::set<std::string> gpu_ids;
        std::set<Time> tids;
        Json phases = Json::array();
        bool incomplete = false;
        bool graph_envelopes = false;
        Time first_step = std::numeric_limits<Time>::max();
        Time last_step = 0;
        for (const auto* step : request.steps) {
            step_ids.insert(step->id);
            tids.insert(step->tids.begin(), step->tids.end());
            incomplete = incomplete || step->ambiguous;
            if (step->record->timed) {
                first_step = std::min(first_step, step->record->span.start);
                last_step = std::max(last_step, step->record->span.end);
            }
            for (const auto& member : step->members)
                if (member.request == id)
                    phases.push_back(Json::object({{"step_id", step->id}, {"tokens", member.tokens}, {"phase", member.phase}}));
        }
        for (const auto* gpu : request.gpu) {
            gpu_ids.insert(gpu->id);
            graph_envelopes = graph_envelopes || str(*gpu->data, "kind") == "graph" || flag(*gpu->data, "includes_internal_gaps");
        }
        Json reasons = Json::array();
        Json row = Json::object({
            {"request_id", id}, {"producer", producer},
            {"record_id", "request:" + producer + ":" + id},
            {"step_ids", numbers(step_ids)}, {"activity_record_ids", strings(gpu_ids)},
            {"memberships", phases}, {"estimate", true},
            {"gpu_share_estimate_ns", request.share_known ? Json(request.share) : Json()},
            {"gpu_share_estimate_method", "activity duration * request scheduled tokens / total real scheduled tokens; excludes padding"},
            {"gpu_exec_semantics", graph_envelopes
                ? "Request wall-clock overlap with GPU graph envelopes, including unobserved internal gaps; this is not precise GPU busy time or exclusive ownership."
                : "Request wall-clock GPU activity overlap; shared intervals are not exclusive GPU ownership."},
            {"gpu_exec_includes_graph_envelopes", graph_envelopes},
            {"intervals", Json::array()}, {"token_intervals", Json::array()},
        });
        // Aliases are evidence only when supplied by one unambiguous Request
        // span. They never alter the producer-scoped internal membership key.
        if (request.spans.size() == 1 && !request.spans.front()->ambiguous) {
            const auto& identity = *request.spans.front()->data;
            if (identity["external_request_id"].is_string())
                row["external_request_id"] = identity["external_request_id"];
            if (identity["identity_source"].is_string())
                row["identity_source"] = identity["identity_source"];
        }
        if (request.spans.size() != 1 || !request.spans.front()->timed || request.spans.front()->ambiguous) {
            reasons.push_back(request.spans.empty() ? "missing_request_span" : "ambiguous_request_span");
            row["start_ns"] = Json();
            row["end_ns"] = Json();
            row["e2e_ns"] = Json();
            row["components_ns"] = Json();
            row["status"] = "Partial";
            row["reasons"] = reasons;
            return row;
        }
        const auto& span_record = *request.spans.front();
        const auto& raw = *span_record.data;
        const auto span = span_record.span;
        const auto header_bound = error(producer);
        const auto bound = clock_bound_for(producer, request.gpu);
        const bool bracket_only = !header_bound && bound.has_value();
        incomplete = incomplete || affected(producer, span) || !bound;
        if (affected(producer, span)) reasons.push_back("overlapping_loss");
        if (!bound) reasons.push_back("missing_clock_calibration");
        else if (bracket_only && span.end > span.start && span.end - span.start < *bound) {
            reasons.push_back("clock_window_wider_than_request");
            incomplete = true;
        }
        if (bracket_only) {
            row["clock_error_basis_ns"] = *bound;
            row["clock_basis_method"] = "per-activity-host-causal-window";
        }
        if (request.steps.empty()) {
            reasons.push_back("missing_engine_membership");
            incomplete = true;
        }
        if (!request.share_known) reasons.push_back("gpu_share_estimate_unknown");
        row["span_record_id"] = span_record.id;
        row["start_ns"] = span.start;
        row["end_ns"] = span.end;
        row["e2e_ns"] = span.end - span.start;
        row["prompt_tokens"] = raw.value("prompt_tokens");
        row["prompt_bucket"] = prompt_bucket(raw);
        if (raw.contains("token_time_semantics")) row["token_time_semantics"] = raw["token_time_semantics"];
        if (raw.contains("class")) row["class"] = raw["class"];
        std::vector<Time> boundaries{span.start, span.end};
        auto add_boundaries = [&](Interval interval) {
            if (overlaps(interval, span)) {
                boundaries.push_back(std::max(span.start, interval.start));
                boundaries.push_back(std::min(span.end, interval.end));
            }
        };
        for (const auto* step : request.steps) add_boundaries(step->record->span);
        for (const auto* gpu : request.gpu) add_boundaries(gpu->span);
        std::vector<const Record*> other_gpu;
        for (const auto& activity : activities_) {
            const auto* gpu = activity.record;
            if (gpu->producer == producer && gpu->timed && overlaps(gpu->span, span)) {
                add_boundaries(gpu->span);
                other_gpu.push_back(gpu);
            }
        }
        const auto scheduler = sched_for(producer, tids);
        for (const auto* sched : scheduler) add_boundaries(sched->span);
        for (const auto& loss : losses_)
            if (loss.timed && (loss.producer.empty() || loss.producer == producer)) add_boundaries(loss.span);
        std::vector<Interval> not_scheduled;
        if (const auto queued = number(raw, "queued_until")) {
            if (*queued >= span.start && *queued <= span.end) {
                not_scheduled.push_back({span.start, *queued});
                add_boundaries(not_scheduled.back());
            }
        }
        if (raw["not_scheduled_intervals"].is_array()) {
            for (const auto& interval : raw["not_scheduled_intervals"].array_items()) {
                if (!interval.is_array() || interval.size() != 2 || !interval[0].is_number() || !interval[1].is_number()) continue;
                try {
                    const Interval parsed{interval[0].as_u64(), interval[1].as_u64()};
                    if (parsed.end > parsed.start) {
                        not_scheduled.push_back(parsed);
                        add_boundaries(parsed);
                    }
                } catch (const std::exception&) {
                    // Invalid optional intervals cannot become evidence.
                }
            }
        }
        // A preemption record is positive evidence of being unscheduled until
        // the next membership. Absence from an incomplete trace is not evidence.
        for (const auto& [key, step] : steps_) {
            if (key.first != producer || !step.record->timed || step.ambiguous) continue;
            const auto& preempted = (*step.record->data)["preempted"];
            if (!preempted.is_array()) continue;
            bool found = false;
            for (const auto& entry : preempted.array_items())
                found = found || (entry.is_string() && entry.as_string() == id);
            if (!found) continue;
            Time resumed = span.end;
            for (const auto* member_step : request.steps)
                if (member_step->record->span.start >= step.record->span.start)
                    resumed = std::min(resumed, member_step->record->span.start);
            if (resumed > step.record->span.start) {
                not_scheduled.push_back({step.record->span.start, resumed});
                add_boundaries(not_scheduled.back());
            }
        }
        std::vector<Time> token_times;
        if (raw.contains("token_times")) {
            auto [times, steps, valid] = token_series(raw);
            for (const auto time : times)
                valid = valid && time >= span.start && time <= span.end;
            token_times = std::move(times);
            // A malformed entry invalidates the sequence rather than silently
            // changing token count. Equal arrival times remain separate tokens:
            // multiple tokens can arrive together in one client SSE chunk.
            if (!valid || !std::is_sorted(token_times.begin(), token_times.end())) {
                token_times.clear();
                reasons.push_back("invalid_token_times");
                incomplete = true;
            }
            boundaries.insert(boundaries.end(), token_times.begin(), token_times.end());
        }
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
        // Each interval whose ends were added above as boundaries covers every
        // piece between consecutive boundaries entirely or not at all. One sweep
        // over the pieces therefore keeps the intervals covering the current one,
        // instead of rescanning them all for every piece: a request that spans a
        // capture has as many pieces as there are GPU activities around it.
        enum Kind : unsigned char { kGpu, kUncertainGpu, kStep, kUncertainStep, kOtherGpu, kSched, kNotScheduled };
        struct Change {
            std::size_t piece;  // the first piece covered, or the first one past the end
            bool enter;
            Kind kind;
            const Record* record;
        };
        std::vector<Change> changes;
        const auto piece_after = [&](Time t) {
            return std::size_t(std::lower_bound(boundaries.begin(), boundaries.end(), t) - boundaries.begin()) + 1;
        };
        const auto cover = [&](Interval interval, Kind kind, const Record* record) {
            if (!overlaps(interval, span)) return;
            changes.push_back({piece_after(std::max(span.start, interval.start)), true, kind, record});
            changes.push_back({piece_after(std::min(span.end, interval.end)), false, kind, record});
        };
        for (const auto* gpu : request.gpu) cover(gpu->span, reliable(*gpu) ? kGpu : kUncertainGpu, gpu);
        for (const auto* step : request.steps)
            cover(step->record->span, !step->ambiguous && reliable(*step->record) ? kStep : kUncertainStep, step->record);
        for (const auto* gpu : other_gpu) cover(gpu->span, kOtherGpu, gpu);
        for (const auto* sched : scheduler) cover(sched->span, kSched, sched);
        for (const auto interval : not_scheduled) cover(interval, kNotScheduled, nullptr);
        std::stable_sort(changes.begin(), changes.end(), [](const Change& a, const Change& b) { return a.piece < b.piece; });
        std::multiset<std::string_view> covering_gpus, covering_steps;
        std::vector<const Record*> covering_scheds;
        std::size_t uncertain_gpus = 0, uncertain_steps = 0, other_gpus = 0, not_scheduled_cover = 0;
        auto change = changes.begin();
        struct Piece {
            Interval span;
            std::string_view category;
            bool lost;
        };
        std::vector<Piece> pieces;
        pieces.reserve(boundaries.size());
        Json intervals = Json::array();
        Json components = empty_components();
        for (std::size_t i = 1; i < boundaries.size(); ++i) {
            for (; change != changes.end() && change->piece == i; ++change) {
                const auto apply = [&](std::size_t& count) { count = change->enter ? count + 1 : count - 1; };
                switch (change->kind) {
                case kGpu:
                    if (change->enter) covering_gpus.insert(change->record->id);
                    else covering_gpus.erase(covering_gpus.find(change->record->id));
                    break;
                case kStep:
                    if (change->enter) covering_steps.insert(change->record->id);
                    else covering_steps.erase(covering_steps.find(change->record->id));
                    break;
                case kSched:
                    if (change->enter) covering_scheds.push_back(change->record);
                    else covering_scheds.erase(std::find(covering_scheds.begin(), covering_scheds.end(), change->record));
                    break;
                case kUncertainGpu: apply(uncertain_gpus); break;
                case kUncertainStep: apply(uncertain_steps); break;
                case kOtherGpu: apply(other_gpus); break;
                case kNotScheduled: apply(not_scheduled_cover); break;
                }
            }
            const Interval piece{boundaries[i - 1], boundaries[i]};
            if (piece.start == piece.end) continue;
            const bool lost = affected(producer, piece);
            const bool measurable = bound && piece.end - piece.start >= *bound;
            std::string_view category = "residual";
            std::set<std::string> evidence;
            if (!lost && measurable) {
                for (const auto id : covering_gpus) evidence.emplace(id);
                for (const auto id : covering_steps) evidence.emplace(id);
                const bool on_gpu = !covering_gpus.empty(), in_step = !covering_steps.empty();
                if (on_gpu) category = "gpu_exec";
                else if (!uncertain_gpus && !uncertain_steps) {
                    if (in_step && !other_gpus) category = "gpu_idle_in_step";
                    else if (!in_step && !other_gpus) {
                        std::set<std::string> states;
                        for (const auto* sched : covering_scheds)
                            if (scheduler_reliable(*sched, producer, piece)) {
                                states.insert(str(*sched->data, "state", "unknown"));
                                evidence.insert(sched->id);
                            }
                        const bool inside_steps = first_step <= piece.start && last_step >= piece.end;
                        if (inside_steps && states == std::set<std::string>{"on_cpu"})
                            category = "between_steps_cpu";
                        else if (inside_steps && states.empty() && scheduler.empty())
                            // Between this request's steps with no GPU work; without
                            // scheduler (BPF) evidence the CPU state is unknown.
                            category = "between_steps";
                        else if (not_scheduled_cover)
                            category = "not_scheduled";
                    }
                }
            }
            const std::string name(category);
            components[name] = num(components, name) + piece.end - piece.start;
            intervals.push_back(Json::object({
                {"start_ns", piece.start}, {"end_ns", piece.end}, {"duration_ns", piece.end - piece.start},
                {"category", name}, {"status", lost ? "Partial" : "Complete"},
                {"evidence_record_ids", strings(evidence)},
                {"reason", lost ? "overlapping_loss" : (!measurable ? "clock_uncertainty" : (category == "residual" ? "insufficient_evidence" : ""))},
            }));
            pieces.push_back({piece, category, lost});
        }
        row["components_ns"] = components;
        row["intervals"] = intervals;
        row["status"] = incomplete ? "Partial" : "Complete";
        row["reasons"] = reasons;
        // Tokens and pieces are both in time order, so each token's range is a
        // run of consecutive pieces.
        Json tokens = Json::array();
        Time previous = span.start;
        Time token_index = 0;
        std::size_t first = 0;
        for (const auto time : token_times) {
            Json token_components = empty_components();
            bool token_partial = false;
            while (first < pieces.size() && pieces[first].span.end <= previous) ++first;
            for (auto k = first; k < pieces.size() && pieces[k].span.start < time; ++k) {
                const Time begin = std::max(previous, pieces[k].span.start);
                const Time end = std::min(time, pieces[k].span.end);
                if (end <= begin) continue;
                const std::string name(pieces[k].category);
                token_components[name] = num(token_components, name) + end - begin;
                token_partial = token_partial || pieces[k].lost;
            }
            tokens.push_back(Json::object({
                {"token_index", token_index++}, {"start_ns", previous}, {"end_ns", time},
                {"duration_ns", time - previous}, {"components_ns", token_components},
                {"status", token_partial ? "Partial" : "Complete"},
            }));
            previous = time;
        }
        row["token_intervals"] = tokens;
        row["after_last_token_ns"] = token_times.empty() ? Json() : Json(span.end - token_times.back());
        return row;
    }

    static std::string prompt_bucket(const Json& request) {
        const auto explicit_bucket = str(request, "prompt_bucket");
        if (!explicit_bucket.empty()) return explicit_bucket;
        const auto tokens = number(request, "prompt_tokens");
        if (!tokens) return "unknown";
        if (*tokens <= 128) return "0-128";
        if (*tokens <= 512) return "129-512";
        if (*tokens <= 2048) return "513-2048";
        return "2049+";
    }

    bool blocking_sync(const Record& record) const {
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

    std::optional<Time> sync_tid(const Record& record) const {
        if (const auto tid = number(*record.data, "tid")) return tid;
        if (const auto corr = number(*record.data, "corr"))
            if (const auto* api = unique_api({record.producer, *corr})) return number(*api->data, "tid");
        return std::nullopt;
    }

    void append_gaps(const Step& step, Json& output, std::optional<Interval> override_span = std::nullopt,
                     const std::string& preceding_step = "") const {
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

    void append_between_step_gaps(Json& output) const {
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

    void build_replays(Json& output) const {
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
};

}  // namespace

Json analyze(const Trace& trace) { return Analyzer(trace).report(); }

Json blame(const Trace& trace, const std::string& request, double slow_factor) {
    return Analyzer(trace, /*edges=*/false).blame(request, slow_factor);
}

Json explain(const Trace& trace, const std::string& requested) {
    Json result = analyze(trace);
    std::vector<Json> matching;
    for (const auto& request : result["requests"].array_items())
        if (str(request, "record_id") == requested) matching.push_back(request);
    if (matching.empty())
        for (const auto& request : result["requests"].array_items())
            if (str(request, "request_id") == requested ||
                (!requested.empty() && str(request, "external_request_id") == requested)) matching.push_back(request);
    if (matching.empty()) throw std::runtime_error("request not found: " + requested);
    if (matching.size() > 1)
        throw std::runtime_error("request id or external alias is ambiguous; use request:<producer>:<req_id>: " + requested);
    const Json selected = matching.front();
    const std::string producer = str(selected, "producer");
    std::set<Time> steps;
    for (const auto& id : selected["step_ids"].array_items()) steps.insert(id.as_u64());
    std::set<std::string> records;
    records.insert(str(selected, "record_id"));
    if (selected.contains("span_record_id")) records.insert(str(selected, "span_record_id"));
    for (const auto& id : selected["activity_record_ids"].array_items()) records.insert(id.as_string());
    for (const auto& record : trace.records)
        if (str(record, "producer") == producer && str(record, "type") == "Step" && steps.contains(num(record, "step_id")))
            records.insert(producer + ":" + std::to_string(num(record, "seq")));
    // Host-only observation: deliberately separate from GPU-clock partition.
    Json host_waits = Json::array();
    for (const auto& record : trace.records) {
        if (str(record, "producer") != producer || str(record, "type") != "HostSubmitWait" ||
            !steps.contains(num(record, "step_id"))) continue;
        const auto start = number(record, "start"), end = number(record, "end");
        if (!start || !end || *end < *start || str(record, "clock") != "CLOCK_MONOTONIC") continue;
        const Json* boundary_step = nullptr;
        unsigned step_count = 0, boundary_count = 0;
        for (const auto& candidate : trace.records) {
            if (str(candidate, "producer") != producer || num(candidate, "step_id") != num(record, "step_id")) continue;
            if (str(candidate, "type") == "Step") { boundary_step = &candidate; ++step_count; }
            if (str(candidate, "type") == "HostSubmitWait") ++boundary_count;
        }
        const bool valid = step_count == 1 && boundary_count == 1 &&
            number(*boundary_step, "t_begin") && number(*boundary_step, "t_end") &&
            num(*boundary_step, "t_begin") <= *start && *end <= num(*boundary_step, "t_end") &&
            num(record, "tid") != 0 && num(*boundary_step, "tid") == num(record, "tid");
        records.insert(producer + ":" + std::to_string(num(record, "seq")));
        host_waits.push_back(Json::object({{"start_ns", *start}, {"end_ns", *end},
            {"duration_ns", *end - *start}, {"step_id", num(record, "step_id")},
            {"evidence_record_id", producer + ":" + std::to_string(num(record, "seq"))},
            {"classification", "host_pre_submit_wait"},
            {"status", valid && !trace.partial && num(result["summary"], "loss_records") == 0 &&
                str(*boundary_step, "status") != "Partial" ? "Observed" : "Partial"},
            {"measurement", "enclosing host boundary; no injected-fault label or GPU-idle inference"},
            {"scope", "shared step; GPU overlap and remaining request time unresolved"}}));
    }
    Json gaps = Json::array();
    for (const auto& gap : result["gaps"].array_items())
        if (str(gap, "producer") == producer && steps.contains(num(gap, "step_id"))) {
            gaps.push_back(gap);
            for (const auto& id : gap["evidence_record_ids"].array_items()) records.insert(id.as_string());
        }
    // Include the provenance chain, not arbitrary neighbors in timestamp order.
    std::map<std::string, std::vector<const Json*>> incoming;
    for (const auto& edge : result["edges"].array_items()) {
        if (str(edge, "source") == "stream-order" && str(edge, "ordering_basis") != "serialized-host-api") continue;
        incoming[str(edge, "to")].push_back(&edge);
    }
    std::vector<std::string> pending(records.begin(), records.end());
    auto discover = [&](const std::string& id) {
        if (records.insert(id).second) pending.push_back(id);
    };
    for (std::size_t cursor = 0; cursor < pending.size(); ++cursor) {
        const auto found = incoming.find(pending[cursor]);
        if (found == incoming.end()) continue;
        for (const auto* edge : found->second) {
            discover(str(*edge, "from"));
            for (const auto& id : (*edge)["evidence_record_ids"].array_items()) discover(id.as_string());
        }
    }
    Json edges = Json::array();
    for (const auto& edge : result["edges"].array_items())
        if (records.contains(str(edge, "from")) && records.contains(str(edge, "to"))) edges.push_back(edge);
    Json raw_records = Json::array();
    for (const auto& record : trace.records)
        if (records.contains(str(record, "producer") + ":" + std::to_string(num(record, "seq")))) raw_records.push_back(record);
    Json activities = Json::array();
    for (const auto& activity : result["activities"].array_items())
        if (records.contains(str(activity, "record_id"))) activities.push_back(activity);
    return Json::object({
        {"schema_version", 1}, {"status", selected["status"]}, {"trace_status", result["status"]},
        {"provenance", result["provenance"]}, {"request", selected}, {"gaps", gaps}, {"host_submit_waits", host_waits},
        {"activities", activities}, {"edges", edges}, {"raw_records", raw_records},
        {"summary", result["summary"]}, {"diagnostics", result["diagnostics"]},
    });
}

namespace {

struct Group {
    std::vector<const Json*> requests;
    bool partial = false;
};

Json aggregate(const Group& group) {
    Json result = Json::object({{"count", static_cast<Time>(group.requests.size())}, {"status", group.partial ? "Partial" : "Complete"}});
    Json means = Json::object();
    std::size_t measured = 0;
    std::size_t estimates = 0;
    double e2e = 0;
    double share = 0;
    std::map<std::string, double> sums;
    for (const auto* request : group.requests) {
        if ((*request)["e2e_ns"].is_number() && (*request)["components_ns"].is_object()) {
            ++measured;
            e2e += (*request)["e2e_ns"].as_double();
            for (const auto* component : kComponents) sums[component] += (*request)["components_ns"][component].as_double();
        }
        if ((*request)["gpu_share_estimate_ns"].is_number()) {
            ++estimates;
            share += (*request)["gpu_share_estimate_ns"].as_double();
        }
    }
    for (const auto* component : kComponents)
        means[component] = measured ? Json(sums[component] / measured) : Json();
    result["measured_count"] = static_cast<Time>(measured);
    result["mean_e2e_ns"] = measured ? Json(e2e / measured) : Json();
    result["mean_components_ns"] = means;
    result["mean_gpu_share_estimate_ns"] = estimates ? Json(share / estimates) : Json();
    result["estimate_count"] = static_cast<Time>(estimates);
    return result;
}

// Stable, local PRNG so bootstrap intervals are reproducible across platforms.
std::uint64_t next_random(std::uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

Json confidence_interval(const Group& a, const Group& b, const std::string& metric) {
    std::vector<double> left;
    std::vector<double> right;
    auto values = [&](const Group& group, std::vector<double>& target) {
        for (const auto* request : group.requests) {
            const auto value = metric == "e2e_ns" ? (*request)["e2e_ns"] : (*request)["components_ns"][metric];
            if (value.is_number()) target.push_back(value.as_double());
        }
    };
    values(a, left);
    values(b, right);
    if (left.size() < 2 || right.size() < 2 || a.partial || b.partial) return Json();
    std::uint64_t state = 0x62d32531cafe0091ULL;
    std::vector<double> samples;
    constexpr std::size_t repetitions = 1000;
    samples.reserve(repetitions);
    for (std::size_t i = 0; i < repetitions; ++i) {
        double sum_a = 0;
        double sum_b = 0;
        for (std::size_t j = 0; j < left.size(); ++j) sum_a += left[next_random(state) % left.size()];
        for (std::size_t j = 0; j < right.size(); ++j) sum_b += right[next_random(state) % right.size()];
        samples.push_back(sum_b / right.size() - sum_a / left.size());
    }
    std::sort(samples.begin(), samples.end());
    return Json::array({samples[25], samples[974]});
}

}  // namespace

Json diff(const Trace& before, const Trace& after, const std::string& by) {
    if (by != "prompt-bucket" && by != "request" && by != "class" && by != "all")
        throw std::runtime_error("unsupported diff grouping: " + by + " (expected prompt-bucket, request, class, or all)");
    const Json left = analyze(before);
    const Json right = analyze(after);
    std::map<std::string, std::pair<Group, Group>> groups;
    auto collect = [&](const Json& report, bool after_side) {
        for (const auto& request : report["requests"].array_items()) {
            const auto key = by == "all" ? "all" : str(request, by == "prompt-bucket" ? "prompt_bucket" :
                (by == "request" ? "request_id" : "class"), "unknown");
            auto& group = after_side ? groups[key].second : groups[key].first;
            group.requests.push_back(&request);
            group.partial = group.partial || str(request, "status") == "Partial" || str(report, "status") == "Partial";
        }
    };
    collect(left, false);
    collect(right, true);
    Json rows = Json::array();
    for (const auto& [key, sides] : groups) {
        const auto& [a, b] = sides;
        const Json aggregate_a = aggregate(a);
        const Json aggregate_b = aggregate(b);
        Json delta_components = Json::object();
        Json confidence = Json::object();
        for (const auto* component : kComponents) {
            const auto& va = aggregate_a["mean_components_ns"][component];
            const auto& vb = aggregate_b["mean_components_ns"][component];
            delta_components[component] = va.is_number() && vb.is_number() ? Json(vb.as_double() - va.as_double()) : Json();
            confidence[component] = confidence_interval(a, b, component);
        }
        const auto& e2e_a = aggregate_a["mean_e2e_ns"];
        const auto& e2e_b = aggregate_b["mean_e2e_ns"];
        const auto& share_a = aggregate_a["mean_gpu_share_estimate_ns"];
        const auto& share_b = aggregate_b["mean_gpu_share_estimate_ns"];
        confidence["e2e_ns"] = confidence_interval(a, b, "e2e_ns");
        rows.push_back(Json::object({
            {"group", key}, {"before", aggregate_a}, {"after", aggregate_b},
            {"status", a.partial || b.partial ? "Partial" : "Complete"},
            {"delta_mean_e2e_ns", e2e_a.is_number() && e2e_b.is_number() ? Json(e2e_b.as_double() - e2e_a.as_double()) : Json()},
            {"delta_mean_components_ns", delta_components},
            {"delta_mean_gpu_share_estimate_ns", share_a.is_number() && share_b.is_number() ? Json(share_b.as_double() - share_a.as_double()) : Json()},
            {"bootstrap_95pct_ci_ns", confidence},
        }));
    }
    return Json::object({
        {"schema_version", 1}, {"by", by},
        {"status", str(left, "status") == "Partial" || str(right, "status") == "Partial" ? "Partial" : "Complete"},
        {"before_provenance", left["provenance"]}, {"after_provenance", right["provenance"]},
        {"groups", rows}, {"delta_direction", "after-minus-before"},
        {"confidence_method", "independent request bootstrap; 1000 deterministic resamples, percentile 95%; null for Partial or fewer than 2 measured requests per run"},
        {"limitations", "Request bootstrap describes observed requests, not independent repeated-run uncertainty or causal effects."},
    });
}

Json perfetto(const Trace& trace) {
    const Json report = analyze(trace);
    Json events = Json::array();
    struct Position { Time timestamp; Time pid; Time tid; };
    std::map<std::string, Position> positions;
    std::map<std::string, Time> producer_ids;
    std::map<std::pair<std::string, std::string>, Time> track_ids;
    Time next_pid = 1;
    Time next_tid = 1;
    std::map<Key, std::vector<const Json*>> api_records;
    for (const auto& record : trace.records)
        if (str(record, "type") == "ApiCall" && number(record, "corr"))
            api_records[{str(record, "producer"), num(record, "corr")}].push_back(&record);
    auto pid_for = [&](const std::string& producer) {
        auto [it, inserted] = producer_ids.try_emplace(producer, next_pid);
        if (inserted) {
            ++next_pid;
            events.push_back(Json::object({{"name", "process_name"}, {"ph", "M"}, {"pid", it->second},
                {"tid", 0}, {"args", Json::object({{"name", producer}})}}));
        }
        return it->second;
    };
    auto tid_for = [&](const std::string& producer, const std::string& track) {
        auto [it, inserted] = track_ids.try_emplace(std::make_pair(producer, track), next_tid);
        if (inserted) {
            ++next_tid;
            events.push_back(Json::object({{"name", "thread_name"}, {"ph", "M"}, {"pid", pid_for(producer)},
                {"tid", it->second}, {"args", Json::object({{"name", track}})}}));
        }
        return it->second;
    };
    for (const auto& raw : trace.records) {
        const auto type = str(raw, "type");
        const auto producer = str(raw, "producer");
        if (type == "CudaEvent" && number(raw, "corr")) {
            const auto found = api_records.find({producer, num(raw, "corr")});
            if (found != api_records.end() && found->second.size() == 1) {
                const auto& api = *found->second.front();
                if (const auto end = number(api, "end")) {
                    const auto id = producer + ":" + std::to_string(num(raw, "seq"));
                    const auto pid = pid_for(producer);
                    const auto tid = tid_for(producer, "CUDA event host submissions");
                    Json args = raw;
                    args["record_id"] = id;
                    args["timestamp_semantics"] = "Associated host event-record API return; not device event completion.";
                    events.push_back(Json::object({
                        {"name", "CUDA event record (host submission)"}, {"cat", "CudaEvent"},
                        {"ph", "i"}, {"s", "t"}, {"pid", pid}, {"tid", tid},
                        {"ts", static_cast<double>(*end) / 1000.0}, {"args", args},
                    }));
                    positions[id] = {*end, pid, tid};
                }
            }
            continue;
        }
        const auto start = number(raw, type == "Step" ? "t_begin" : "start");
        const auto end = number(raw, type == "Step" ? "t_end" : "end");
        if (!start || !end || *end < *start) continue;
        const auto id = producer + ":" + std::to_string(num(raw, "seq"));
        std::string track;
        if (type == "GpuActivity")
            track = "GPU ctx " + std::to_string(num(raw, "ctx")) + " stream " + std::to_string(num(raw, "stream"));
        else if (type == "Request") track = "request " + str(raw, "req_id");
        else if (type == "Step") track = "engine steps";
        else if (type == "Sched") track = "scheduler tid " + std::to_string(num(raw, "tid"));
        else track = "CPU tid " + std::to_string(num(raw, "tid"));
        const auto pid = pid_for(producer);
        const auto tid = tid_for(producer, track);
        std::string name = str(raw, "name", type);
        if (type == "Request") name = str(raw, "req_id");
        else if (type == "Step") name = "step " + std::to_string(num(raw, "step_id"));
        else if (type == "GpuActivity") name = str(raw, "name", str(raw, "kind", "GPU activity"));
        else if (type == "Sched") name = str(raw, "state", "unknown");
        Json args = raw;
        args["record_id"] = id;
        events.push_back(Json::object({
            {"name", name}, {"cat", type}, {"ph", "X"}, {"pid", pid}, {"tid", tid},
            {"ts", static_cast<double>(*start) / 1000.0}, {"dur", static_cast<double>(*end - *start) / 1000.0},
            {"args", args},
        }));
        positions[id] = {*start, pid, tid};
        if (type == "Request") positions["request:" + producer + ":" + str(raw, "req_id")] = {*start, pid, tid};
    }
    Time flow_id = 1;
    for (const auto& edge : report["edges"].array_items()) {
        const auto from = positions.find(str(edge, "from"));
        const auto to = positions.find(str(edge, "to"));
        if (from == positions.end() || to == positions.end()) continue;
        const auto& a = from->second;
        const auto& b = to->second;
        // A correlation can survive timestamp uncertainty, but a backwards flow
        // in Perfetto would pretend to establish a temporal order we do not know.
        if (a.timestamp > b.timestamp) continue;
        for (const auto& [phase, position] : std::array<std::pair<const char*, Position>, 2>{{{"s", a}, {"f", b}}}) {
            Json event = Json::object({
                {"name", str(edge, "source")}, {"cat", "provenance"}, {"ph", phase}, {"id", flow_id},
                {"pid", position.pid}, {"tid", position.tid}, {"ts", static_cast<double>(position.timestamp) / 1000.0},
                {"args", edge},
            });
            if (std::string(phase) == "f") event["bp"] = "e";
            events.push_back(event);
        }
        ++flow_id;
    }
    return Json::object({
        {"traceEvents", events}, {"displayTimeUnit", "ns"},
        {"metadata", Json::object({{"status", report["status"]}, {"provenance", report["provenance"]},
            {"time_encoding", "Trace Event JSON ts/dur are microseconds, converted from monotonic nanoseconds"},
            {"diagnostics", report["diagnostics"]}})},
    });
}

}  // namespace reqscope
