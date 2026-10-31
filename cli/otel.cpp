#include "cli/otel.hpp"
#include "cli/client.hpp"
#include "format/segment.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

using namespace reqscope;
namespace fs = std::filesystem;
namespace {
uint64_t integer(const Json &v) {
  if (!v.is_string())
    return v.as_u64();
  uint64_t n = 0;
  const auto &s = v.as_string();
  auto [p, e] = std::from_chars(s.data(), s.data() + s.size(), n);
  if (e != std::errc{} || p != s.data() + s.size())
    throw std::runtime_error("invalid OTLP integer");
  return n;
}
Json attributes(const Json &span) {
  Json result = Json::object();
  const Json attrs = span.value("attributes", Json::array());
  std::set<std::string> keys;
  for (const auto &a : attrs.array_items()) {
    if (!keys.insert(a.string("key")).second)
      throw std::runtime_error("duplicate OTLP attribute: " + a.string("key"));
    const auto &v = a.at("value");
    for (const auto *type :
         {"stringValue", "intValue", "doubleValue", "boolValue"})
      if (v.contains(type)) {
        result[a.string("key")] = v[type];
        break;
      }
  }
  return result;
}
void spans(const Json &j, std::vector<Json> &out) {
  if (j.is_array()) {
    for (const auto &x : j.array_items())
      spans(x, out);
    return;
  }
  if (!j.is_object())
    return;
  if (j.contains("startTimeUnixNano") && j.contains("endTimeUnixNano")) {
    out.push_back(j);
    return;
  }
  for (const auto *key :
       {"resourceSpans", "scopeSpans", "instrumentationLibrarySpans", "spans"})
    if (j.contains(key))
      spans(j[key], out);
}
} // namespace
int import_otel(int argc, char **argv) {
  if (argc < 3)
    throw std::runtime_error("import-otel requires RUN");
  std::string input, output, producer, offset_text, error_text, client_path;
  for (int i = 3; i < argc; i++) {
    std::string key = argv[i];
    if (++i >= argc)
      throw std::runtime_error("missing import option value");
    if (key == "--input")
      input = argv[i];
    else if (key == "--output")
      output = argv[i];
    else if (key == "--producer")
      producer = argv[i];
    else if (key == "--unix-minus-monotonic-ns")
      offset_text = argv[i];
    else if (key == "--clock-error-ns")
      error_text = argv[i];
    else if (key == "--client-log")
      client_path = argv[i];
    else
      throw std::runtime_error("unknown import option: " + key);
  }
  if (input.empty() || output.empty() || producer.empty() ||
      offset_text.empty() || error_text.empty())
    throw std::runtime_error(
        "import-otel requires --input, --output, --producer, "
        "--unix-minus-monotonic-ns and --clock-error-ns");
  int64_t offset;
  auto [p, e] = std::from_chars(
      offset_text.data(), offset_text.data() + offset_text.size(), offset);
  if (e != std::errc{} || p != offset_text.data() + offset_text.size())
    throw std::runtime_error("invalid clock offset");
  uint64_t uncertainty = integer(Json(error_text));
  auto trace = read_trace(argv[2]);
  std::map<std::string, Json> headers;
  for (const auto &h : trace.headers)
    headers.try_emplace(h.string("producer"), h);
  if (!headers.contains(producer))
    throw std::runtime_error("OTel target producer not found in RUN");
  auto clients = read_client_log(client_path, producer);
  const size_t client_count = clients.size();
  std::set<std::string> client_identities, bound_clients;
  for (const auto &[id, client] : clients) client_identities.insert(id);
  if (fs::file_size(input) > 64 * 1024 * 1024)
    throw std::runtime_error("OTLP input exceeds 64 MiB");
  std::ifstream f(input);
  std::string text((std::istreambuf_iterator<char>(f)), {});
  std::vector<Json> imported;
  try {
    spans(Json::parse(text), imported);
  } catch (const std::exception &) {
    imported.clear();
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line))
      if (!line.empty())
        spans(Json::parse(line), imported);
  }
  std::set<std::string> existing;
  std::map<std::string, bool> closed_source;
  uint64_t seq = 0;
  for (const auto &r : trace.records) {
    const auto id = r.string("producer");
    if (r.string("type") == "Session")
      closed_source[id] =
          r.string("state") == "Finalized" || r.string("state") == "Snapshot";
    else if (r.string("type") != "Loss")
      closed_source[id] = false;
    if (r.string("producer") == producer) {
      seq = std::max(seq, r.integer("seq"));
      if (r.string("type") == "Request")
        existing.insert(r.string("req_id"));
    }
  }
  size_t count = 0;
  auto monotonic = [&](uint64_t unix_ns) {
    __int128 n = __int128(unix_ns) - offset;
    if (n < 0 || n > UINT64_MAX)
      throw std::runtime_error("OTLP clock conversion out of range");
    return uint64_t(n);
  };
  for (const auto &span : imported) {
    auto attrs = attributes(span);
    auto external_request = attrs.string("gen_ai.request.id");
    if (external_request.empty())
      continue;
    if (client_identities.contains(external_request) &&
        !bound_clients.insert(external_request).second)
      throw std::runtime_error("ambiguous external request identity in client binding: " + external_request);
    auto request = attrs.string("reqscope.internal_request_id", external_request);
    if (request.empty())
      throw std::runtime_error("empty OTel internal request identity");
    if (!existing.insert(request).second)
      throw std::runtime_error("duplicate request span: " + request);
    auto start = monotonic(integer(span.at("startTimeUnixNano"))),
         end = monotonic(integer(span.at("endTimeUnixNano")));
    if (end < start)
      throw std::runtime_error("negative OTLP duration");
    if (seq == UINT64_MAX)
      throw std::runtime_error("producer sequence exhausted");
    Json r = Json::object({{"type", "Request"},
                           {"producer", producer},
                           {"seq", ++seq},
                           {"req_id", request},
                           {"external_request_id", external_request},
                           {"identity_source", attrs.contains("reqscope.internal_request_id")
                               ? "otel-explicit-internal-request-id" : "otel-request-id"},
                           {"start", start},
                           {"end", end},
                           {"source", "otel-request-span"},
                           {"trace_id", span.string("traceId")},
                           {"span_id", span.string("spanId")},
                           {"external_clock_error_ns", uncertainty}});
    if (attrs.contains("gen_ai.usage.prompt_tokens"))
      r["prompt_tokens"] = integer(attrs["gen_ai.usage.prompt_tokens"]);
    if (attrs.contains("gen_ai.latency.time_in_queue")) {
      double seconds = attrs["gen_ai.latency.time_in_queue"].as_double();
      if (!std::isfinite(seconds) || seconds < 0 ||
          seconds > double(end - start) / 1e9)
        throw std::runtime_error("invalid OTLP queue duration");
      r["queued_until"] = start + uint64_t(seconds * 1e9);
    }
    if (auto found = clients.find(external_request); found != clients.end()) {
      merge_client_request(r, found->second, uncertainty);
      clients.erase(found);
    }
    trace.records.push_back(std::move(r));
    ++count;
  }
  if (!count)
    throw std::runtime_error("no gen_ai.request.id spans found");
  if (!clients.empty())
    throw std::runtime_error("client engine_request_id has no matching OTel request span");
  if (headers[producer].contains("clock_error_ns") &&
      !headers[producer]["clock_error_ns"].is_null())
    headers[producer]["clock_error_ns"] =
        std::max(headers[producer].integer("clock_error_ns"), uncertainty);
  // Reserve room for the explicit import/lifecycle records before creating any output.
  std::map<std::string, uint64_t> last_sequences;
  for (const auto &r : trace.records)
    last_sequences[r.string("producer")] = std::max(last_sequences[r.string("producer")], r.integer("seq"));
  for (const auto &[id, last] : last_sequences) {
    const uint64_t extra = uint64_t(trace.partial) + uint64_t(id == producer && closed_source[id]);
    if (last > UINT64_MAX - extra)
      throw std::runtime_error("producer sequence exhausted by import boundary");
  }
  if (fs::exists(output) || !fs::create_directories(output))
    throw std::runtime_error("import output must be a new directory");
  size_t i = 0;
  for (auto &[id, header] : headers) {
    SegmentWriter writer(fs::path(output) /
                             ("producer-" + std::to_string(i++) + ".rqs"),
                         header);
    uint64_t last = 0;
    for (const auto &r : trace.records)
      if (r.string("producer") == id) {
        writer.write(r);
        last = std::max(last, r.integer("seq"));
      }
    if (trace.partial) {
      writer.write(Json::object({{"type", "Loss"},
                                 {"producer", id},
                                 {"seq", ++last},
                                 {"reason", "imported_partial_source"},
                                 {"count", 1}}));
    }
    if (id == producer && closed_source[id])
      writer.write(Json::object({{"type", "Session"},
                                 {"producer", id},
                                 {"seq", ++last},
                                 {"state", "Snapshot"},
                                 {"source", "otel-offline-import"}}));
    writer.flush();
  }
  Json summary = Json::object({{"requests_imported", uint64_t(count)},
                               {"client_requests_imported", uint64_t(client_count)},
                               {"output", output},
                               {"clock_offset_ns", offset},
                               {"clock_error_ns", uncertainty},
                               {"source_partial", trace.partial}});
  std::ofstream manifest(fs::path(output) / "otel-import.json");
  manifest << summary.dump() << '\n';
  std::cout << summary.dump() << '\n';
  return 0;
}
