#pragma once
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace reqscope::sched {
inline uint64_t unsigned_option(std::string_view text, uint64_t maximum, const char* name) {
    uint64_t value = 0;
    auto parsed = std::from_chars(text.data(), text.data() + text.size(), value, 10);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value > maximum)
        throw std::runtime_error(std::string(name) + ": expected an unsigned decimal integer in range");
    return value;
}
inline uint64_t deadline_after_seconds(uint64_t now, uint64_t seconds) {
    constexpr auto maximum = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t nanoseconds = 1000000000ULL;
    if (!seconds) return maximum;
    if (seconds > (maximum - now) / nanoseconds)
        throw std::runtime_error("duration-seconds: monotonic deadline would overflow");
    return now + seconds * nanoseconds;
}
}
