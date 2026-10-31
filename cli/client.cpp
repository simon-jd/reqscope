#include "cli/client.hpp"
#include <filesystem>
#include <fstream>
#include <set>

namespace reqscope {
std::map<std::string, Json> read_client_log(const std::string &path,
                                           const std::string &producer) {
  std::map<std::string, Json> result;
  if (path.empty()) return result;
  if (std::filesystem::file_size(path) > 16 * 1024 * 1024)
    throw std::runtime_error("client log exceeds 16 MiB");
  std::ifstream input(path);
  Json rows = Json::parse(std::string(std::istreambuf_iterator<char>(input), {}));
  if (rows.is_object()) rows = Json::array({rows});
  for (const auto &row : rows.array_items()) {
    if (row.string("clock") != "CLOCK_MONOTONIC" ||
        row.string("boot_id") != producer.substr(0, producer.find(':')))
      throw std::runtime_error("client log must use the target host boot and CLOCK_MONOTONIC");
    const auto id = row.string("engine_request_id");
    if (id.empty() || !result.emplace(id, row).second)
      throw std::runtime_error("client log requires unique explicit engine_request_id");
  }
  if (result.empty()) throw std::runtime_error("empty client log");
  return result;
}

void merge_client_request(Json &request, const Json &client, uint64_t error) {
  if (client.integer("http_status") != 200)
    throw std::runtime_error("client request did not succeed");
  const auto start = client.at("request_start_monotonic_ns").as_u64();
  const auto end = client.at("request_end_monotonic_ns").as_u64();
  const auto otel_start = request.integer("start"), otel_end = request.integer("end");
  if (end < start || __int128(start) > __int128(otel_start) + error ||
      __int128(end) + error < __int128(otel_end))
    throw std::runtime_error("client interval does not contain the OTel request span");
  const auto response_id = client.string("response_id");
  if (response_id.empty()) throw std::runtime_error("client response_id missing");
  Json arrivals = Json::array(), token_ids = Json::array();
  uint64_t previous = start;
  bool token_identity_complete = true;
  for (const auto &chunk : client.at("chunks").array_items()) {
    const auto arrival = chunk.at("arrival_monotonic_ns").as_u64();
    if (arrival < previous || arrival > end)
      throw std::runtime_error("invalid client chunk arrival order or interval");
    previous = arrival;
    const auto &data = chunk.at("data");
    if (data.string("id") != response_id)
      throw std::runtime_error("mixed response IDs in client SSE log");
    for (const auto &choice : data.at("choices").array_items()) {
      if (choice.integer("index") != 0)
        throw std::runtime_error("client import currently requires one completion choice");
      if (!choice["token_ids"].is_array()) {
        if (!choice.string("text").empty()) token_identity_complete = false;
        continue;
      }
      for (const auto &token : choice["token_ids"].array_items()) {
        token_ids.push_back(token.as_u64());
        arrivals.push_back(arrival);
      }
    }
  }
  if (token_identity_complete && !arrivals.array_items().empty()) {
    if (!client["usage"].contains("completion_tokens") ||
        client["usage"].integer("completion_tokens") != arrivals.size())
      throw std::runtime_error("client token IDs do not match completion_tokens");
    request["token_times"] = arrivals;
    request["token_ids"] = token_ids;
    request["token_time_semantics"] =
        "Client receipt of explicit SSE token IDs; tokens in one chunk share an arrival time. This is not GPU generation time.";
  }
  if (request.contains("queued_until")) {
    request["not_scheduled_intervals"] = Json::array({
        Json::array({otel_start, request["queued_until"]})});
    request.erase("queued_until");
  }
  request["otel_start"] = otel_start;
  request["otel_end"] = otel_end;
  request["start"] = start;
  request["end"] = end;
  request["response_id"] = response_id;
  request["output_tokens"] = client["usage"].value("completion_tokens");
  request["source"] = "client-monotonic-with-otel-identity";
  request["client_token_times_available"] = request.contains("token_times");
}
} // namespace reqscope
