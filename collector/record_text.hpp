#pragma once
#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>

namespace reqscope::collector {
// Writes one JSON record straight into a reused string, for the writer's
// CUPTI records: building a Json object per record and dumping it cost about
// 2 us per kernel record on this path, formatting the text directly 0.3 us.
// The text is left open (no closing brace) so the collector can append the
// fields it adds to every record (pid, producer, seq, timestamp).
class RecordText {
public:
    RecordText(std::string& out, std::string_view type) : out_(out) {
        out_.clear();
        out_ += "{\"type\":";
        quoted(type);
    }
    RecordText& number(std::string_view key, uint64_t value) {
        name(key);
        char digits[24];
        const auto end = std::to_chars(digits, digits + sizeof digits, value).ptr;
        out_.append(digits, end);
        return *this;
    }
    RecordText& text(std::string_view key, std::string_view value) {
        name(key);
        quoted(value);
        return *this;
    }
    RecordText& boolean(std::string_view key, bool value) {
        name(key);
        out_ += value ? "true" : "false";
        return *this;
    }

private:
    void name(std::string_view key) {
        out_ += ',';
        quoted(key);
        out_ += ':';
    }
    // JSON string escaping, as Json::dump does: quote, backslash and control
    // characters are escaped; other bytes are copied.
    void quoted(std::string_view value) {
        out_ += '"';
        for (const char c : value) {
            const auto byte = static_cast<unsigned char>(c);
            if (c == '"' || c == '\\') {
                out_ += '\\';
                out_ += c;
            } else if (byte < 0x20) {
                constexpr char hex[] = "0123456789abcdef";
                out_ += "\\u00";
                out_ += hex[byte >> 4];
                out_ += hex[byte & 15];
            } else {
                out_ += c;
            }
        }
        out_ += '"';
    }
    std::string& out_;
};
}
