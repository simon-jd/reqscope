#pragma once
#include "format/json.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace reqscope::collector {
uint64_t monotonic_ns() noexcept;
enum class LossReason : unsigned { queue_full, oversize_record, malformed_record,
    cupti_pool_exhausted, cupti_completed_queue_full, cupti_dropped, cupti_parse_error,
    correlation_error, graph_metadata_error, count };
struct GraphResource {
    uint32_t action = 0, graph_id = 0, original_graph_id = 0, graph_exec_id = 0;
    uint64_t node_id = 0, original_node_id = 0, dependency_node_id = 0;
    uint32_t node_type = 0, context = 0;
    uint64_t tid = 0, timestamp = 0;
    bool ids_valid = true;
};
// Both functions are bounded and allocation-free; safe from CUPTI callbacks.
void note_loss(LossReason reason, uint64_t count = 1) noexcept;
bool enqueue_cupti_buffer(uint8_t* buffer, size_t valid, uintptr_t context,
                          uint32_t stream) noexcept;
bool enqueue_graph_resource(const GraphResource& resource) noexcept;
Json graph_resource_json(const GraphResource& resource);
struct BackendInfo {
    std::string path, subscription = "none", status = "disabled", error;
    std::string api_capture_policy = "none";
    Json api_capture_apis = Json::array({});
    uint32_t version = 0;
    uint64_t host_buffer_count = 0, host_buffer_bytes = 0;
    bool active = false, v2_symbols = false, graph_resources = false;
};
BackendInfo backend_start(const std::string& mode);
void backend_flush() noexcept;
void backend_stop() noexcept;
int backend_push(uint64_t step) noexcept;
int backend_pop() noexcept;
void backend_process(uint8_t* buffer, size_t valid, uintptr_t context,
                     uint32_t stream, const std::function<void(Json)>& emit);
void backend_release(uint8_t* buffer) noexcept;
}
