#include "analyzer/internal.hpp"

#include <algorithm>
#include <exception>
#include <limits>

namespace reqscope::detail {

void Analyzer::diagnostic(const std::string& reason, const Record* record) {
    Json item = Json::object({{"reason", reason}, {"status", "Partial"}});
    if (record) {
        item["record_id"] = record->id;
        item["producer"] = record->producer;
    }
    diagnostics_.push_back(item);
}

void Analyzer::add_loss(const Json& raw, const std::string& record_id) {
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

void Analyzer::read_records() {
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
        if (str(header, "mode") == "sampled") sampled_producers_.insert(producer);
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

Request& Analyzer::request(const RequestKey& key) {
    auto [it, inserted] = requests_.try_emplace(key);
    if (inserted) it->second.key = key;
    return it->second;
}

// Sampled mode keeps kernel tracing on for one step after each run of traced
// steps (the guard step), and CUPTI gives that step's node records the end
// times of later launches of the same graphs (docs/design.md, "Sampled
// tracing"). Node records of a step that was not traced are therefore left
// out before anything reads them. Runs before build_indices takes pointers
// into records_.
void Analyzer::drop_untraced_node_records() {
    if (sampled_producers_.empty()) return;
    std::set<Key> traced;
    std::map<Key, std::uint64_t> step_of;   // (producer, corr) -> step id
    for (const auto& record : records_) {
        if (record.ambiguous || !sampled_producers_.contains(record.producer)) continue;
        const auto& raw = *record.data;
        if (record.type == "TracedStep") {
            if (const auto step_id = number(raw, "step_id")) traced.insert(Key{record.producer, *step_id});
        } else if (record.type == "ExtCorr") {
            const auto corr = number(raw, "corr");
            const auto step_id = number(raw, "step_id");
            if (corr && step_id) step_of[Key{record.producer, *corr}] = *step_id;
        }
    }
    std::erase_if(records_, [&](const Record& record) {
        if (record.type != "GpuActivity" || !sampled_producers_.contains(record.producer)) return false;
        const auto& raw = *record.data;
        if (str(raw, "kind") != "kernel" && num(raw, "graph_node_id") == 0) return false;
        const auto corr = number(raw, "corr");
        const auto found = corr ? step_of.find(Key{record.producer, *corr}) : step_of.end();
        if (found == step_of.end() || traced.contains(Key{record.producer, found->second})) return false;
        ++untraced_node_records_;
        return true;
    });
}

void Analyzer::build_indices() {
    std::set<Key> traced;
    for (const auto& record : records_) {
        const auto& raw = *record.data;
        const auto corr = number(raw, "corr");
        if (record.type == "TracedStep") {
            if (const auto step_id = number(raw, "step_id")) traced.insert(Key{record.producer, *step_id});
            continue;
        }
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
            step.members.push_back({id, tokens_field(member, "tokens").value_or(Tokens{}), str(member, "phase", "unknown")});
        }
    }
    // In a sampled capture only the steps named by TracedStep records have
    // every kernel; the others have copies only, and the guard steps also
    // node records that link() leaves out.
    for (auto& [key, step] : steps_)
        if (sampled_producers_.contains(key.first)) step.traced = traced.contains(key);
    for (auto& [key, step] : steps_) {
        for (const auto& member : step.members) {
            request({key.first, member.request}).steps.push_back(&step);
            if (!step.ambiguous)
                edge("request:" + key.first + ":" + member.request, step.record->id,
                     "engine-membership", {step.record->id});
        }
    }
}

bool Analyzer::affected(const std::string& producer, Interval span) const {
    for (const auto& loss : losses_) {
        const auto observed = scheduler_targets_.find(loss.producer);
        const bool scheduler_loss = observed != scheduler_targets_.end() && observed->second.contains(producer);
        if (!loss.target_producer.empty() && loss.target_producer != producer) continue;
        if (!loss.producer.empty() && loss.producer != producer && !scheduler_loss && loss.target_producer != producer) continue;
        if (!loss.timed || overlaps(loss.span, span)) return true;
    }
    return false;
}

// How complete the input is: reader and analyzer diagnostics, losses by
// reason, and the Diagnostic records the collector or plugin wrote.
Json Analyzer::trace_status() const {
    std::map<std::string, Time> issues;
    Json first = Json::array();
    auto note = [&](const Json& item) {
        ++issues[str(item, "reason", "unknown")];
        if (first.size() < 20) first.push_back(item);
    };
    for (const auto& item : trace_.diagnostics) note(item);
    for (const auto& item : diagnostics_) note(item);
    std::map<std::string, Time> by_reason;
    for (const auto& loss : losses_) ++by_reason[loss.reason];
    Json issue_counts = Json::object(), loss_counts = Json::object(), collector = Json::array();
    for (const auto& [reason, count] : issues) issue_counts[reason] = count;
    for (const auto& [reason, count] : by_reason) loss_counts[reason] = count;
    for (const auto& record : records_)
        if (record.type == "Diagnostic")
            collector.push_back(Json::object({{"producer", record.producer},
                {"source", str(*record.data, "source")}, {"reason", str(*record.data, "reason")},
                {"stopped", flag(*record.data, "stopped")}}));
    const bool partial = trace_.partial || !losses_.empty() || !diagnostics_.empty();
    return Json::object({{"status", partial ? "Partial" : "Complete"},
        {"issues_by_reason", issue_counts}, {"issues", first},
        {"loss_records_by_reason", loss_counts}, {"loss_count", loss_count_},
        {"collector_diagnostics", collector}});
}

// One request's token times and the steps that produced them, reassembled
// from the RequestTokens chunks the adapter emits every 1024 tokens plus
// the final Request record. Steps are returned only if every part has
// them; any non-integer time makes the series invalid.
std::tuple<std::vector<Time>, std::vector<Time>, bool> Analyzer::token_series(const Json& raw) const {
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

}  // namespace reqscope::detail
