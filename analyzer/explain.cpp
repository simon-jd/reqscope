#include "analyzer/internal.hpp"

#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace reqscope {

using namespace detail;

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

}  // namespace reqscope
