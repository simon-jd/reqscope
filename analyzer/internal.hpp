#pragma once

// Shared by the analyzer's translation units; not part of the public API
// (analyzer.hpp). The Analyzer reads a Trace once into records, indices and
// links, and each report is built from that state:
//   analyzer.cpp        shared helpers, the Analyzer's construction order
//   records.cpp         reading records, losses, indices, token series
//   clocks.cpp          clock bounds: probes, per-activity brackets
//   linking.cpp         API, step, stream, event and graph-topology links
//   blame.cpp           slow-token blame and its cause rules
//   report.cpp          analyze(): activities, gaps, replays, clocks
//   request_report.cpp  one request's partition of its span
//   explain.cpp, diff.cpp, perfetto.cpp  reports built from analyze()

#include "analyzer/analyzer.hpp"

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace reqscope::detail {

using Time = std::uint64_t;
using Key = std::pair<std::string, std::uint64_t>;
using RequestKey = std::pair<std::string, std::string>;

// A count of tokens: scheduled for a request in a step, co-scheduled prefill,
// in a prompt, or generated. Its own type because Time is also an unsigned
// 64-bit integer: a time or duration cannot be passed where tokens are
// expected, or the reverse, and Json takes only count().
class Tokens {
public:
    constexpr Tokens() = default;
    constexpr explicit Tokens(std::uint64_t count) : count_(count) {}
    constexpr std::uint64_t count() const { return count_; }
    constexpr Tokens& operator+=(Tokens other) {
        count_ += other.count_;
        return *this;
    }
    friend constexpr auto operator<=>(const Tokens&, const Tokens&) = default;

private:
    std::uint64_t count_ = 0;
};
static_assert(!std::is_convertible_v<Time, Tokens> && !std::is_convertible_v<Tokens, Time>);
inline constexpr std::array<const char*, 6> kComponents{
    "gpu_exec", "gpu_idle_in_step", "between_steps_cpu", "between_steps", "not_scheduled", "residual"};

inline std::string str(const Json& value, const std::string& key, const std::string& fallback = "") {
    const auto& field = value[key];
    return field.is_string() ? field.as_string() : fallback;
}

inline std::optional<Time> number(const Json& value, const std::string& key) {
    const auto& field = value[key];
    if (!field.is_number()) return std::nullopt;
    try {
        return field.as_u64();
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

inline Time num(const Json& value, const std::string& key, Time fallback = 0) {
    return number(value, key).value_or(fallback);
}

inline std::optional<Tokens> tokens_field(const Json& value, const std::string& key) {
    if (const auto count = number(value, key)) return Tokens{*count};
    return std::nullopt;
}

inline bool flag(const Json& value, const std::string& key, bool fallback = false) {
    const auto& field = value[key];
    return field.is_bool() ? field.as_bool() : fallback;
}

inline bool starts_with(const std::string& value, const char* prefix) {
    const auto needle = std::string(prefix);
    return value.size() >= needle.size() && value.compare(0, needle.size(), needle) == 0;
}

Json strings(const std::set<std::string>& values);
Json numbers(const std::set<Time>& values);
Json empty_components();

struct Interval {
    Time start = 0;
    Time end = 0;
};

inline bool overlaps(Interval a, Interval b) { return a.start < b.end && b.start < a.end; }
inline bool contains(Interval a, Interval b) { return a.start <= b.start && a.end >= b.end; }
std::vector<Interval> unite(std::vector<Interval> spans);
Time union_duration(std::vector<Interval> spans);

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
    Tokens tokens;
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
    // False for a step that sampled mode did not trace kernel by kernel.
    bool traced = true;
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
    explicit Analyzer(const Trace& input, bool edges = true);

    // The full decomposition analyze() returns (report.cpp).
    Json report() const;
    // Why were a request's slow tokens slow? (blame.cpp)
    Json blame(const std::string& only, double slow_factor) const;

private:
    // A step's own GPU work. busy is the union of its GPU intervals, first and
    // last bound them, and wait is the shortest time from a launch API's
    // return to the start of the kernel it launched. Idle time between the
    // step's own intervals is split by who was waiting: inner_host while the
    // next kernel had not been launched yet, inner_gpu once it had (the GPU
    // was running something else, or was slower).
    struct StepGpu {
        bool known = false, has_wait = false;
        Time busy = 0, first = 0, last = 0, wait = 0, launch = 0, inner_gpu = 0, inner_host = 0;
    };
    struct Baseline {
        std::size_t steps = 0;
        Time busy = 0, inner_gpu = 0, wait = 0;
    };
    // Per producer, built on first use: the union of its timed GPU activity,
    // and its timed synchronization records ordered by start.
    struct TimeIndex {
        std::vector<const Record*> records;
        std::vector<Time> latest_end;  // the latest end among records[0..i]
        std::vector<const Record*> overlapping(Interval window) const;
    };

    // request_report() stages (request_report.cpp).
    // What a request's steps say before its span is read.
    struct RequestScan {
        std::set<Time> tids;
        bool incomplete = false;
        Time first_step = std::numeric_limits<Time>::max();
        Time last_step = 0;
    };
    // The records that partition a measured request's span, and the
    // boundaries of the pieces they cut it into.
    struct RequestCover {
        std::vector<Time> boundaries;
        std::vector<const Record*> other_gpu;
        std::vector<const Record*> scheduler;
        std::vector<Interval> not_scheduled;
    };
    // One piece between consecutive boundaries, as classified.
    struct Piece {
        Interval span;
        std::string_view category;
        bool lost;
    };
    struct Partition {
        Json intervals;
        Json components;
        std::vector<Piece> pieces;
    };
    // An interval of a request's span starting or ending its cover of the
    // pieces: piece is the first one covered, or the first one past the end.
    enum CoverKind : unsigned char { kGpu, kUncertainGpu, kStep, kUncertainStep, kOtherGpu, kSched, kNotScheduled };
    struct CoverChange {
        std::size_t piece;
        bool enter;
        CoverKind kind;
        const Record* record;
    };
    // The intervals covering the current piece during the sweep.
    struct Covering {
        std::multiset<std::string_view> gpus, steps;
        std::vector<const Record*> scheds;
        std::size_t uncertain_gpus = 0, uncertain_steps = 0, other_gpus = 0, not_scheduled = 0;
        void apply(const CoverChange& change);
    };

    // records.cpp
    void diagnostic(const std::string& reason, const Record* record = nullptr);
    void add_loss(const Json& raw, const std::string& record_id = "");
    void read_records();
    void build_indices();
    Request& request(const RequestKey& key);
    bool affected(const std::string& producer, Interval span) const;
    Json trace_status() const;
    std::tuple<std::vector<Time>, std::vector<Time>, bool> token_series(const Json& raw) const;

    // clocks.cpp
    void derive_probe_bounds();
    void assign_clock_brackets();
    std::optional<Time> error(const std::string& producer) const;
    std::optional<Time> clock_bound_for(const std::string& producer, const std::vector<const Record*>& gpu) const;
    bool reliable(const Record& record) const;
    bool scheduler_reliable(const Record& record, const std::string& target, Interval piece) const;
    bool explicit_success(const Record& record) const;
    bool system_tid(const Record& api) const;
    bool producer_clock_blocked(const std::string& producer) const;
    std::optional<int> host_completion_kind(const Record& sync) const;
    bool sync_api_matches(int kind, const Record& api) const;

    // linking.cpp
    void edge(const std::string& from, const std::string& to, const std::string& source,
              const std::vector<std::string>& evidence = {}, const Json& attributes = Json());
    const Record* unique_api(const Key& key) const;
    Step* linked_step(const Key& key, std::vector<std::string>& evidence, std::string& reason);
    void link();
    void link_graph_topology();
    void link_streams();
    static bool valid_event_sync_id(Time id);
    static bool event_sync(const Record& record);

    // blame.cpp
    const StepGpu& step_gpu(const Step* step) const;
    static std::string shape(const Step* step);
    Baseline baseline(const Step* step) const;
    Time host_part(const Step* step, Time begin, Time end) const;
    std::optional<Time> collector_active(const Step* step, Time begin, Time end) const;
    Json cause(const Step* step, Time begin, Time end, Time excess, Tokens prefill_tokens, Time host_base) const;

    // report.cpp
    std::string data_kind() const;
    Json activity_rows(std::size_t& linked) const;
    Json activity_row(const Activity& activity) const;
    Json request_rows() const;
    Json gap_rows() const;
    Json synchronization_rows() const;
    Json report_diagnostics() const;
    Json report_provenance() const;
    Json report_summary(std::size_t linked) const;
    Json clock_rows() const;
    Json clock_row(const std::string& producer, const std::optional<Time>& clock_error) const;
    const std::vector<Interval>& gpu_busy(const std::string& producer) const;
    const TimeIndex& sync_index(const std::string& producer) const;
    std::vector<const Record*> sched_for(const std::string& producer, const std::set<Time>& tids) const;
    bool blocking_sync(const Record& record) const;
    std::optional<Time> sync_tid(const Record& record) const;
    void append_gaps(const Step& step, Json& output, std::optional<Interval> override_span = std::nullopt,
                     const std::string& preceding_step = "") const;
    void append_between_step_gaps(Json& output) const;
    void build_replays(Json& output) const;

    // request_report.cpp
    Json request_report(const Request& request) const;
    Json request_row(const Request& request, RequestScan& scan) const;
    static Json unmeasured_request(Json row, const Request& request);
    std::optional<Time> request_clock_bound(const Request& request, Interval span, Json& row, Json& reasons,
                                            bool& incomplete) const;
    static void describe_request_span(const Record& span_record, Json& row);
    std::vector<Interval> not_scheduled_intervals(const Request& request, Interval span, const Json& raw) const;
    RequestCover request_cover(const Request& request, Interval span, const Json& raw,
                               const std::set<Time>& tids) const;
    std::vector<Time> request_token_times(const Json& raw, Interval span, Json& reasons, bool& incomplete) const;
    std::vector<CoverChange> cover_changes(const Request& request, Interval span, const RequestCover& cover) const;
    std::string_view piece_category(const Covering& covering, Interval piece, const std::string& producer,
                                    const RequestScan& scan, bool no_scheduler,
                                    std::set<std::string>& evidence) const;
    Partition partition_request(const Request& request, Interval span, const RequestCover& cover,
                                const RequestScan& scan, std::optional<Time> bound) const;
    static Json token_intervals(const std::vector<Time>& token_times, const std::vector<Piece>& pieces, Time start);
    static std::string prompt_bucket(const Json& request);

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
    std::set<std::string> sampled_producers_;
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
    mutable std::map<std::string, std::vector<Interval>> gpu_busy_;
    mutable std::map<const Step*, StepGpu> step_gpu_;
    mutable std::map<std::string, std::vector<Interval>> collector_busy_;
    mutable bool collector_busy_built_ = false;
    mutable bool step_gpu_built_ = false;
    mutable std::map<std::string, std::vector<const Step*>> comparable_;
    mutable bool baselines_built_ = false;
    mutable std::map<std::string, TimeIndex> sync_index_;
    Json edges_ = Json::array();
    Json graph_definitions_ = Json::array();
    Json graph_topology_ = Json::array();
    std::set<std::tuple<std::string, std::string, std::string>> edge_keys_;
    Time loss_count_ = 0;
    double gpu_duration_ = 0;
};

}  // namespace reqscope::detail
