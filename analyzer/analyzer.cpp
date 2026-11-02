#include "analyzer/internal.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace reqscope::detail {

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

Json empty_components() {
    Json result = Json::object();
    for (const auto* name : kComponents) result[name] = Time{0};
    return result;
}

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

Analyzer::Analyzer(const Trace& input, bool edges) : trace_(input), record_edges_(edges) {
    read_records();
    build_indices();
    derive_probe_bounds();     // before link(): probe copies are not application work
    link();
    assign_clock_brackets();
}

std::vector<const Record*> Analyzer::TimeIndex::overlapping(Interval window) const {
    std::vector<const Record*> result;
    // Every record before `i` ends at or before the window starts.
    auto i = std::size_t(std::upper_bound(latest_end.begin(), latest_end.end(), window.start) - latest_end.begin());
    for (; i < records.size() && records[i]->span.start < window.end; ++i)
        if (records[i]->span.end > window.start) result.push_back(records[i]);
    return result;
}

}  // namespace reqscope::detail
