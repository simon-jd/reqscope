#include "analyzer/internal.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace reqscope::detail {

void Analyzer::edge(const std::string& from, const std::string& to, const std::string& source,
                    const std::vector<std::string>& evidence, const Json& attributes) {
    if (!record_edges_ || !edge_keys_.emplace(from, to, source).second) return;
    Json ids = Json::array();
    for (const auto& id : evidence) ids.push_back(id);
    Json record = Json::object({{"from", from}, {"to", to}, {"source", source}, {"evidence_record_ids", ids}});
    if (attributes.is_object())
        for (const auto& [key, value] : attributes.object_items()) record[key] = value;
    edges_.push_back(std::move(record));
}

const Record* Analyzer::unique_api(const Key& key) const {
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

Step* Analyzer::linked_step(const Key& key, std::vector<std::string>& evidence, std::string& reason) {
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

void Analyzer::link() {
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
            for (const auto& member : step.members) total_tokens += static_cast<double>(member.tokens.count());
            for (const auto& member : step.members) {
                auto& target = request({record.producer, member.request});
                target.gpu.push_back(&record);
                if (total_tokens > 0 && reliable(record))
                    target.share += static_cast<double>(record.span.end - record.span.start) *
                                    static_cast<double>(member.tokens.count()) / total_tokens;
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

void Analyzer::link_streams() {
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

void Analyzer::link_graph_topology() {
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

bool Analyzer::valid_event_sync_id(Time id) {
    return id != 0 && id != std::numeric_limits<std::uint32_t>::max() && id != std::numeric_limits<Time>::max();
}

bool Analyzer::event_sync(const Record& record) {
    if (const auto kind = number(*record.data, "sync_type")) return *kind == 1 || *kind == 2;
    const auto kind = str(*record.data, "sync_type");
    return kind == "event_synchronize" || kind == "stream_wait_event" ||
           kind == "cudaEventSynchronize" || kind == "cudaStreamWaitEvent";
}

}  // namespace reqscope::detail
