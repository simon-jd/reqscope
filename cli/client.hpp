#pragma once
#include "format/json.hpp"
#include <map>
#include <string>

namespace reqscope {
std::map<std::string, Json> read_client_log(const std::string &path,
                                           const std::string &producer);
void merge_client_request(Json &request, const Json &client, uint64_t clock_error);
}
