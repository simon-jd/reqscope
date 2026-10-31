#pragma once
#include "format/json.hpp"
#include <filesystem>
#include <fstream>

namespace reqscope {
inline constexpr size_t max_record_bytes = 1024 * 1024;
inline constexpr size_t max_trace_bytes = 2048ULL * 1024 * 1024;
inline constexpr size_t max_trace_records = 20'000'000;
struct Trace {
  std::vector<Json> headers, records, diagnostics;
  bool partial = false;
};
uint32_t crc32(std::string_view data);
class SegmentWriter {
public:
  SegmentWriter(const std::filesystem::path &path, Json header);
  ~SegmentWriter();
  SegmentWriter(const SegmentWriter &) = delete;
  SegmentWriter &operator=(const SegmentWriter &) = delete;
  // Appends one framed record to an in-memory buffer and returns its framed
  // size. The buffer goes to the file in one write once it reaches 1 MiB,
  // and on drain() or flush(), so a record costs no syscall of its own.
  size_t write(const Json &record);
  // Writes the buffer to the file.
  void drain();
  // drain(), then fdatasync.
  void flush();

private:
  int fd_ = -1;
  std::string buffer_;
};
// Throws when the trace exceeds the parser's memory budget or record limit:
// a resource limit is not data damage, so it never yields a silently partial trace.
Trace read_trace(const std::filesystem::path &path);
} // namespace reqscope
