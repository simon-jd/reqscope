#include "analyzer/internal.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace reqscope {

using namespace detail;

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

}  // namespace reqscope
