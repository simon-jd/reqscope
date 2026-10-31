#pragma once

#include "format/segment.hpp"

#include <string>

namespace reqscope {

// These reports describe observed evidence. Synthetic input remains synthetic;
// missing clocks, membership, or scheduling evidence is never fabricated.
Json analyze(const Trace& trace);
Json explain(const Trace& trace, const std::string& request);
// Attribute each request's slow inter-token gaps to co-scheduled prefill work.
Json blame(const Trace& trace, const std::string& request = "", double slow_factor = 1.5);
Json diff(const Trace& before, const Trace& after, const std::string& by);
Json perfetto(const Trace& trace);

}  // namespace reqscope
