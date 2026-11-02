#include "analyzer/internal.hpp"

#include <array>
#include <map>
#include <string>
#include <utility>

namespace reqscope {

using namespace detail;

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
