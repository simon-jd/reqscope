#include "analyzer/internal.hpp"

#include <algorithm>
#include <exception>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace reqscope::detail {

// One request's row in analyze(): its memberships and GPU share, the reasons
// it is Partial, and its span cut at every evidence boundary into pieces,
// each classified as one of kComponents, then summed per token.
Json Analyzer::request_report(const Request& request) const {
    RequestScan scan;
    Json row = request_row(request, scan);
    if (request.spans.size() != 1 || !request.spans.front()->timed || request.spans.front()->ambiguous)
        return unmeasured_request(std::move(row), request);
    const auto& span_record = *request.spans.front();
    const auto& raw = *span_record.data;
    const auto span = span_record.span;
    Json reasons = Json::array();
    const auto bound = request_clock_bound(request, span, row, reasons, scan.incomplete);
    if (request.steps.empty()) {
        reasons.push_back("missing_engine_membership");
        scan.incomplete = true;
    }
    if (!request.share_known) reasons.push_back("gpu_share_estimate_unknown");
    describe_request_span(span_record, row);
    auto cover = request_cover(request, span, raw, scan.tids);
    const auto token_times = request_token_times(raw, span, reasons, scan.incomplete);
    auto& boundaries = cover.boundaries;
    boundaries.insert(boundaries.end(), token_times.begin(), token_times.end());
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    auto partition = partition_request(request, span, cover, scan, bound);
    row["components_ns"] = std::move(partition.components);
    row["intervals"] = std::move(partition.intervals);
    row["status"] = scan.incomplete ? "Partial" : "Complete";
    row["reasons"] = reasons;
    row["token_intervals"] = token_intervals(token_times, partition.pieces, span.start);
    row["after_last_token_ns"] = token_times.empty() ? Json() : Json(span.end - token_times.back());
    return row;
}

// The request's identity, memberships and GPU share estimate. scan collects
// what its steps say for the stages that follow.
Json Analyzer::request_row(const Request& request, RequestScan& scan) const {
    const auto& [producer, id] = request.key;
    std::set<Time> step_ids;
    std::set<std::string> gpu_ids;
    Json phases = Json::array();
    bool graph_envelopes = false;
    for (const auto* step : request.steps) {
        step_ids.insert(step->id);
        scan.tids.insert(step->tids.begin(), step->tids.end());
        scan.incomplete = scan.incomplete || step->ambiguous;
        if (step->record->timed) {
            scan.first_step = std::min(scan.first_step, step->record->span.start);
            scan.last_step = std::max(scan.last_step, step->record->span.end);
        }
        for (const auto& member : step->members)
            if (member.request == id)
                phases.push_back(Json::object({{"step_id", step->id}, {"tokens", member.tokens}, {"phase", member.phase}}));
    }
    for (const auto* gpu : request.gpu) {
        gpu_ids.insert(gpu->id);
        graph_envelopes = graph_envelopes || str(*gpu->data, "kind") == "graph" || flag(*gpu->data, "includes_internal_gaps");
    }
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
    return row;
}

// Without one unambiguous, timed Request span there is nothing to partition.
Json Analyzer::unmeasured_request(Json row, const Request& request) {
    Json reasons = Json::array();
    reasons.push_back(request.spans.empty() ? "missing_request_span" : "ambiguous_request_span");
    row["start_ns"] = Json();
    row["end_ns"] = Json();
    row["e2e_ns"] = Json();
    row["components_ns"] = Json();
    row["status"] = "Partial";
    row["reasons"] = reasons;
    return row;
}

// The clock bound a piece of the span must reach to be measured: the
// producer's session bound, or else the widest per-activity bracket of the
// request's GPU records. Losses over the span and a missing or too-wide
// bound make the request Partial.
std::optional<Time> Analyzer::request_clock_bound(const Request& request, Interval span, Json& row, Json& reasons,
                                                  bool& incomplete) const {
    const auto& producer = request.key.first;
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
    return bound;
}

// The Request span record's own fields.
void Analyzer::describe_request_span(const Record& span_record, Json& row) {
    const auto& raw = *span_record.data;
    const auto span = span_record.span;
    row["span_record_id"] = span_record.id;
    row["start_ns"] = span.start;
    row["end_ns"] = span.end;
    row["e2e_ns"] = span.end - span.start;
    row["prompt_tokens"] = raw.value("prompt_tokens");
    row["prompt_bucket"] = prompt_bucket(raw);
    if (raw.contains("token_time_semantics")) row["token_time_semantics"] = raw["token_time_semantics"];
    if (raw.contains("class")) row["class"] = raw["class"];
}

// Positive evidence that the request was waiting to be scheduled: queued
// until a time, explicit intervals, and preemption until its next membership.
std::vector<Interval> Analyzer::not_scheduled_intervals(const Request& request, Interval span, const Json& raw) const {
    const auto& [producer, id] = request.key;
    std::vector<Interval> not_scheduled;
    if (const auto queued = number(raw, "queued_until")) {
        if (*queued >= span.start && *queued <= span.end) not_scheduled.push_back({span.start, *queued});
    }
    if (raw["not_scheduled_intervals"].is_array()) {
        for (const auto& interval : raw["not_scheduled_intervals"].array_items()) {
            if (!interval.is_array() || interval.size() != 2 || !interval[0].is_number() || !interval[1].is_number()) continue;
            try {
                const Interval parsed{interval[0].as_u64(), interval[1].as_u64()};
                if (parsed.end > parsed.start) not_scheduled.push_back(parsed);
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
        if (resumed > step.record->span.start) not_scheduled.push_back({step.record->span.start, resumed});
    }
    return not_scheduled;
}

// Every interval that can decide a piece of the span: the request's steps and
// GPU records, its producer's other GPU work, its launch threads' scheduler
// records, losses, and not-scheduled evidence. Their ends inside the span are
// the piece boundaries; the caller adds token times.
Analyzer::RequestCover Analyzer::request_cover(const Request& request, Interval span, const Json& raw,
                                               const std::set<Time>& tids) const {
    const auto& producer = request.key.first;
    RequestCover cover;
    cover.boundaries = {span.start, span.end};
    auto add_boundaries = [&](Interval interval) {
        if (overlaps(interval, span)) {
            cover.boundaries.push_back(std::max(span.start, interval.start));
            cover.boundaries.push_back(std::min(span.end, interval.end));
        }
    };
    for (const auto* step : request.steps) add_boundaries(step->record->span);
    for (const auto* gpu : request.gpu) add_boundaries(gpu->span);
    for (const auto& activity : activities_) {
        const auto* gpu = activity.record;
        if (gpu->producer == producer && gpu->timed && overlaps(gpu->span, span)) {
            add_boundaries(gpu->span);
            cover.other_gpu.push_back(gpu);
        }
    }
    cover.scheduler = sched_for(producer, tids);
    for (const auto* sched : cover.scheduler) add_boundaries(sched->span);
    for (const auto& loss : losses_)
        if (loss.timed && (loss.producer.empty() || loss.producer == producer)) add_boundaries(loss.span);
    cover.not_scheduled = not_scheduled_intervals(request, span, raw);
    for (const auto interval : cover.not_scheduled) add_boundaries(interval);
    return cover;
}

// The request's token arrival times, if it reports them. A malformed entry
// invalidates the sequence rather than silently changing token count. Equal
// arrival times remain separate tokens: multiple tokens can arrive together
// in one client SSE chunk.
std::vector<Time> Analyzer::request_token_times(const Json& raw, Interval span, Json& reasons, bool& incomplete) const {
    std::vector<Time> token_times;
    if (!raw.contains("token_times")) return token_times;
    auto [times, steps, valid] = token_series(raw);
    for (const auto time : times)
        valid = valid && time >= span.start && time <= span.end;
    token_times = std::move(times);
    if (!valid || !std::is_sorted(token_times.begin(), token_times.end())) {
        token_times.clear();
        reasons.push_back("invalid_token_times");
        incomplete = true;
    }
    return token_times;
}

// Each interval whose ends were added as boundaries covers every piece
// between consecutive boundaries entirely or not at all. One sweep over the
// pieces therefore keeps the intervals covering the current one, instead of
// rescanning them all for every piece: a request that spans a capture has as
// many pieces as there are GPU activities around it. These are the sweep's
// enter and leave events, in piece order.
std::vector<Analyzer::CoverChange> Analyzer::cover_changes(const Request& request, Interval span,
                                                           const RequestCover& cover) const {
    const auto& boundaries = cover.boundaries;
    std::vector<CoverChange> changes;
    const auto piece_after = [&](Time t) {
        return std::size_t(std::lower_bound(boundaries.begin(), boundaries.end(), t) - boundaries.begin()) + 1;
    };
    const auto add = [&](Interval interval, CoverKind kind, const Record* record) {
        if (!overlaps(interval, span)) return;
        changes.push_back({piece_after(std::max(span.start, interval.start)), true, kind, record});
        changes.push_back({piece_after(std::min(span.end, interval.end)), false, kind, record});
    };
    for (const auto* gpu : request.gpu) add(gpu->span, reliable(*gpu) ? kGpu : kUncertainGpu, gpu);
    for (const auto* step : request.steps)
        add(step->record->span, !step->ambiguous && reliable(*step->record) ? kStep : kUncertainStep, step->record);
    for (const auto* gpu : cover.other_gpu) add(gpu->span, kOtherGpu, gpu);
    for (const auto* sched : cover.scheduler) add(sched->span, kSched, sched);
    for (const auto interval : cover.not_scheduled) add(interval, kNotScheduled, nullptr);
    std::stable_sort(changes.begin(), changes.end(),
                     [](const CoverChange& a, const CoverChange& b) { return a.piece < b.piece; });
    return changes;
}

void Analyzer::Covering::apply(const CoverChange& change) {
    const auto count = [&](std::size_t& value) { value = change.enter ? value + 1 : value - 1; };
    switch (change.kind) {
    case kGpu:
        if (change.enter) gpus.insert(change.record->id);
        else gpus.erase(gpus.find(change.record->id));
        break;
    case kStep:
        if (change.enter) steps.insert(change.record->id);
        else steps.erase(steps.find(change.record->id));
        break;
    case kSched:
        if (change.enter) scheds.push_back(change.record);
        else scheds.erase(std::find(scheds.begin(), scheds.end(), change.record));
        break;
    case kUncertainGpu: count(uncertain_gpus); break;
    case kUncertainStep: count(uncertain_steps); break;
    case kOtherGpu: count(other_gpus); break;
    case kNotScheduled: count(not_scheduled); break;
    }
}

// The component of a measurable piece with no overlapping loss, adding the
// records that decide it to evidence. no_scheduler: the request's launch
// threads have no scheduler records at all.
std::string_view Analyzer::piece_category(const Covering& covering, Interval piece, const std::string& producer,
                                          const RequestScan& scan, bool no_scheduler,
                                          std::set<std::string>& evidence) const {
    for (const auto id : covering.gpus) evidence.emplace(id);
    for (const auto id : covering.steps) evidence.emplace(id);
    if (!covering.gpus.empty()) return "gpu_exec";
    if (covering.uncertain_gpus || covering.uncertain_steps || covering.other_gpus) return "residual";
    if (!covering.steps.empty()) return "gpu_idle_in_step";
    std::set<std::string> states;
    for (const auto* sched : covering.scheds)
        if (scheduler_reliable(*sched, producer, piece)) {
            states.insert(str(*sched->data, "state", "unknown"));
            evidence.insert(sched->id);
        }
    const bool inside_steps = scan.first_step <= piece.start && scan.last_step >= piece.end;
    if (inside_steps && states == std::set<std::string>{"on_cpu"}) return "between_steps_cpu";
    // Between this request's steps with no GPU work; without scheduler (BPF)
    // evidence the CPU state is unknown.
    if (inside_steps && states.empty() && no_scheduler) return "between_steps";
    if (covering.not_scheduled) return "not_scheduled";
    return "residual";
}

// Classify every piece of the span and sum the components. A piece is
// residual when it overlaps a loss, is shorter than the clock bound, or has
// no deciding evidence.
Analyzer::Partition Analyzer::partition_request(const Request& request, Interval span, const RequestCover& cover,
                                                const RequestScan& scan, std::optional<Time> bound) const {
    const auto& producer = request.key.first;
    const auto& boundaries = cover.boundaries;
    const auto changes = cover_changes(request, span, cover);
    auto change = changes.begin();
    Covering covering;
    Partition result{Json::array(), empty_components(), {}};
    result.pieces.reserve(boundaries.size());
    for (std::size_t i = 1; i < boundaries.size(); ++i) {
        for (; change != changes.end() && change->piece == i; ++change) covering.apply(*change);
        const Interval piece{boundaries[i - 1], boundaries[i]};
        if (piece.start == piece.end) continue;
        const bool lost = affected(producer, piece);
        const bool measurable = bound && piece.end - piece.start >= *bound;
        std::set<std::string> evidence;
        const std::string_view category = !lost && measurable
            ? piece_category(covering, piece, producer, scan, cover.scheduler.empty(), evidence)
            : std::string_view("residual");
        const std::string name(category);
        result.components[name] = num(result.components, name) + piece.end - piece.start;
        result.intervals.push_back(Json::object({
            {"start_ns", piece.start}, {"end_ns", piece.end}, {"duration_ns", piece.end - piece.start},
            {"category", name}, {"status", lost ? "Partial" : "Complete"},
            {"evidence_record_ids", strings(evidence)},
            {"reason", lost ? "overlapping_loss" : (!measurable ? "clock_uncertainty" : (category == "residual" ? "insufficient_evidence" : ""))},
        }));
        result.pieces.push_back({piece, category, lost});
    }
    return result;
}

// Each token's components, from the previous token (or the span start) to
// its arrival. Tokens and pieces are both in time order, so each token's
// range is a run of consecutive pieces.
Json Analyzer::token_intervals(const std::vector<Time>& token_times, const std::vector<Piece>& pieces, Time start) {
    Json tokens = Json::array();
    Time previous = start;
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
    return tokens;
}

std::string Analyzer::prompt_bucket(const Json& request) {
    const auto explicit_bucket = str(request, "prompt_bucket");
    if (!explicit_bucket.empty()) return explicit_bucket;
    const auto tokens = number(request, "prompt_tokens");
    if (!tokens) return "unknown";
    if (*tokens <= 128) return "0-128";
    if (*tokens <= 512) return "129-512";
    if (*tokens <= 2048) return "513-2048";
    return "2049+";
}

}  // namespace reqscope::detail
