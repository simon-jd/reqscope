// The writer's direct record text must parse to the record Json would build.
#include "collector/record_text.hpp"
#include "format/json.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using reqscope::Json;
using reqscope::collector::RecordText;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Json closed(std::string text) { return Json::parse(text + "}"); }

void fields_parse_back_with_their_types() {
    std::string text;
    RecordText(text, "GpuActivity").text("kind", "kernel").number("corr", 604384)
        .number("start", UINT64_MAX).number("bytes", 0).boolean("is_device_launched", false)
        .boolean("includes_internal_gaps", true);
    const Json record = closed(text);
    const Json expected = Json::object({{"type", "GpuActivity"}, {"kind", "kernel"}, {"corr", uint64_t(604384)},
        {"start", UINT64_MAX}, {"bytes", uint64_t(0)}, {"is_device_launched", false},
        {"includes_internal_gaps", true}});
    check(record.dump() == expected.dump(), "record differs: " + record.dump());
}

void strings_are_escaped() {
    std::string text;
    const std::string name = std::string("a\"b\\c\nd\te") + char(1) + "_ZN7cutlass$";
    RecordText(text, "ApiCall").text("name", name);
    check(closed(text).string("name") == name, "escaped name did not round-trip: " + text);
}

void the_string_is_reused_from_the_start() {
    std::string text = "left over from the previous record";
    RecordText(text, "ExtCorr").number("step_id", 7);
    check(text == "{\"type\":\"ExtCorr\",\"step_id\":7", "unexpected text: " + text);
}
}

int main() {
    try {
        fields_parse_back_with_their_types();
        strings_are_escaped();
        the_string_is_reused_from_the_start();
    } catch (const std::exception& error) {
        std::cerr << "record text test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "record text tests passed\n";
    return 0;
}
