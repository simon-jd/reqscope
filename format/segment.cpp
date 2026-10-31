#include "format/segment.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <unistd.h>
#include <unordered_map>

namespace reqscope {
namespace {
// IEEE CRC32 (Python's zlib.crc32), sliced by 8. A bitwise loop is a large
// share of the time spent reading a large trace, and every record the
// collector writes pays for it too.
struct Crc32Tables {
  uint32_t t[8][256] = {};
  constexpr Crc32Tables() {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k)
        c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1)));
      t[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; ++i)
      for (int s = 1; s < 8; ++s)
        t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xff];
  }
};
constexpr Crc32Tables crc_tables;
static_assert(std::endian::native == std::endian::little);
} // namespace
uint32_t crc32(std::string_view s) {
  const auto &t = crc_tables.t;
  auto p = reinterpret_cast<const unsigned char *>(s.data());
  size_t n = s.size();
  uint32_t c = 0xffffffffu;
  for (; n >= 8; p += 8, n -= 8) {
    uint32_t lo, hi;
    std::memcpy(&lo, p, 4);
    std::memcpy(&hi, p + 4, 4);
    lo ^= c;
    c = t[7][lo & 0xff] ^ t[6][(lo >> 8) & 0xff] ^ t[5][(lo >> 16) & 0xff] ^ t[4][lo >> 24] ^
        t[3][hi & 0xff] ^ t[2][(hi >> 8) & 0xff] ^ t[1][(hi >> 16) & 0xff] ^ t[0][hi >> 24];
  }
  for (; n; ++p, --n)
    c = (c >> 8) ^ t[0][(c ^ *p) & 0xff];
  return c ^ 0xffffffffu;
}
namespace {
void put32(char *b, uint32_t x) {
  for (int i = 0; i < 4; i++)
    b[i] = char(x >> (8 * i));
}
void write_all(int fd, const char *data, size_t length) {
  while (length) {
    auto n = ::write(fd, data, length);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      throw std::runtime_error("segment write failed");
    data += n;
    length -= size_t(n);
  }
}
uint32_t get32(const char *b) {
  uint32_t v = 0;
  for (int i = 0; i < 4; i++)
    v |= uint32_t(static_cast<unsigned char>(b[i])) << (8 * i);
  return v;
}
} // namespace
SegmentWriter::SegmentWriter(const std::filesystem::path &path, Json h) {
  fd_ = ::open(path.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd_ < 0)
    throw std::runtime_error("cannot create segment: " + path.string());
  try {
    write_all(fd_, "RQSSEG01", 8);
    h["type"] = "Header";
    h["format_version"] = 1;
    write(h);
  } catch (...) {
    ::close(fd_);
    fd_ = -1;
    throw;
  }
}
SegmentWriter::~SegmentWriter() {
  if (fd_ >= 0) {
    try {
      drain();
    } catch (...) {
    }
    ::close(fd_);
  }
}
size_t SegmentWriter::write(const Json &r) {
  const auto s = r.dump();
  if (s.size() > max_record_bytes)
    throw std::runtime_error("record exceeds 1 MiB");
  char frame[8];
  put32(frame, uint32_t(s.size()));
  put32(frame + 4, crc32(s));
  buffer_.append(frame, sizeof(frame));
  buffer_.append(s);
  if (buffer_.size() >= 1024 * 1024)
    drain();
  return sizeof(frame) + s.size();
}
void SegmentWriter::drain() {
  if (buffer_.empty())
    return;
  write_all(fd_, buffer_.data(), buffer_.size());
  buffer_.clear();
}
void SegmentWriter::flush() {
  drain();
  if (::fdatasync(fd_) != 0)
    throw std::runtime_error("segment flush failed");
}
namespace {
struct TraceLimit : std::runtime_error {
  using std::runtime_error::runtime_error;
};
} // namespace

Trace read_trace(const std::filesystem::path &path) {
  Trace t;
  size_t budget = max_trace_bytes;
  if (const auto configured = std::getenv("REQSCOPE_TRACE_BUDGET_MIB")) {
    const std::string value(configured);
    uint64_t mib = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), mib);
    if (error != std::errc{} || end != value.data() + value.size() || mib < 1 || mib > 65536)
      throw std::runtime_error("REQSCOPE_TRACE_BUDGET_MIB must be 1..65536");
    budget = size_t(mib) * 1024 * 1024;
  }
  std::vector<std::filesystem::path> files, collector_errors;
  if (std::filesystem::is_directory(path)) {
    for (auto &e : std::filesystem::directory_iterator(path)) {
      if (!e.is_regular_file() || e.is_symlink())
        continue;
      if (e.path().extension() == ".rqs")
        files.push_back(e.path());
      // A collector whose writer failed leaves collector-error-PID.txt.
      else if (e.path().filename().string().starts_with("collector-error-") &&
               e.path().extension() == ".txt")
        collector_errors.push_back(e.path());
    }
  } else if (std::filesystem::is_regular_file(path))
    files.push_back(path);
  else
    throw std::runtime_error("trace path not found: " + path.string());
  if (files.empty())
    throw std::runtime_error("no .rqs segments in trace");
  std::sort(files.begin(), files.end());
  if (files.size() > 4096)
    throw std::runtime_error("trace exceeds 4096 segment limit");
  size_t total = 0;
  size_t resident = 0;
  std::map<std::string, Json> producer_headers;
  // Each record's (producer, seq), noted while parsing, so ordering and the
  // sequence check never look inside the records again.
  std::vector<std::string> producer_names;
  std::unordered_map<std::string, uint32_t> producer_ids;
  std::vector<std::pair<uint32_t, uint64_t>> keys;
  auto issue = [&](const std::string &reason, const std::filesystem::path &f,
                   uint64_t offset, const std::string &producer = "") {
    t.partial = true;
    t.diagnostics.push_back(Json::object({{"reason", reason},
                                          {"file", f.string()},
                                          {"offset", offset},
                                          {"producer", producer}}));
  };
  std::sort(collector_errors.begin(), collector_errors.end());
  for (const auto &f : collector_errors) {
    std::string line;
    std::ifstream in(f);
    std::getline(in, line);
    issue("collector_failed", f, 0);
    t.diagnostics.back()["detail"] = line.substr(0, 512);
  }
  for (auto &f : files) {
    // One read per segment; frames are sliced from it, not read and
    // allocated one at a time.
    std::string data;
    if (std::ifstream in(f, std::ios::binary | std::ios::ate); in) {
      data.resize(size_t(in.tellg()));
      in.seekg(0);
      in.read(data.data(), std::streamsize(data.size()));
      data.resize(size_t(in.gcount()));
    }
    if (data.size() < 8 || std::memcmp(data.data(), "RQSSEG01", 8)) {
      issue("invalid_magic", f, 0);
      continue;
    }
    bool header = false;
    std::string producer;
    uint32_t producer_id = 0;
    uint64_t offset = 8;
    while (true) {
      const size_t left = data.size() - offset;
      if (left == 0)
        break;
      if (left < 8) {
        issue("truncated_frame_header", f, offset, producer);
        break;
      }
      const char *frame = data.data() + offset;
      uint32_t len = get32(frame), crc = get32(frame + 4);
      if (len == 0 || len > max_record_bytes) {
        issue("record_length_limit", f, offset, producer);
        break;
      }
      total += len;
      if (total > budget)
        throw std::runtime_error("trace exceeds configured parser byte budget");
      if (left - 8 < len) {
        issue("truncated_payload", f, offset, producer);
        break;
      }
      const std::string_view payload(frame + 8, len);
      if (crc32(payload) != crc) {
        issue("crc_mismatch", f, offset, producer);
        break;
      }
      try {
        Json r = Json::parse(payload);
        resident += r.estimated_memory();
        if (resident > budget)
          throw TraceLimit("decoded trace exceeds the parser memory budget; raise REQSCOPE_TRACE_BUDGET_MIB");
        if (!r.is_object())
          throw std::runtime_error("record must be object");
        if (!header) {
          if (r.string("type") != "Header" ||
              r.integer("format_version", 1) != 1 ||
              r.string("producer").empty())
            throw std::runtime_error("invalid header");
          producer = r.string("producer");
          const auto [id, added] = producer_ids.emplace(producer, uint32_t(producer_names.size()));
          if (added)
            producer_names.push_back(producer);
          producer_id = id->second;
          const auto canonical = r.string("boot_id") + ":" +
                                 std::to_string(r.integer("pid")) + ":" +
                                 std::to_string(r.integer("start_time_ticks"));
          if (r.string("boot_id").empty() || !r.integer("pid") ||
              !r.contains("start_time_ticks") || canonical != producer)
            throw std::runtime_error("producer/header identity mismatch");
          if (r.contains("clock_error_ns") && !r["clock_error_ns"].is_null())
            (void)r["clock_error_ns"].as_u64();
          if (producer_headers.contains(producer)) {
            const auto &old = producer_headers.at(producer);
            if (old["clock"] != r["clock"] ||
                old["clock_error_ns"] != r["clock_error_ns"])
              issue("inconsistent_producer_header", f, offset, producer);
          } else {
            producer_headers[producer] = r;
          }
          if (r.string("clock") != "CLOCK_MONOTONIC")
            issue("unsupported_clock", f, offset, producer);
          t.headers.push_back(r);
          header = true;
        } else {
          const auto *type = r.find("type"), *source = r.find("producer");
          if (!type || type->as_string().empty() || type->as_string() == "Header" ||
              !source || source->as_string() != producer || r.integer("seq") == 0)
            throw std::runtime_error("invalid record identity");
          if (t.records.size() >= max_trace_records)
            throw TraceLimit("trace exceeds the parser record limit");
          keys.emplace_back(producer_id, r.integer("seq"));
          t.records.push_back(std::move(r));
        }
      } catch (const TraceLimit &) {
        throw;
      } catch (const std::exception &e) {
        issue(std::string("invalid_record: ") + e.what(), f, offset, producer);
        break;
      }
      offset += 8 + len;
    }
    if (!header)
      issue("missing_header", f, 0);
  }
  // Stable order by (producer name, seq). Producers are ranked once by name,
  // so the tens of millions of comparisons a large trace needs copy no
  // producer strings.
  std::vector<uint32_t> rank(producer_names.size()), by_name(producer_names.size());
  std::iota(by_name.begin(), by_name.end(), 0u);
  std::sort(by_name.begin(), by_name.end(),
            [&](uint32_t a, uint32_t b) { return producer_names[a] < producer_names[b]; });
  for (uint32_t i = 0; i < by_name.size(); ++i)
    rank[by_name[i]] = i;
  std::vector<size_t> order(t.records.size());
  std::iota(order.begin(), order.end(), size_t(0));
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const auto ra = rank[keys[a].first], rb = rank[keys[b].first];
    return ra != rb ? ra < rb : keys[a].second < keys[b].second;
  });
  std::vector<Json> sorted;
  sorted.reserve(order.size());
  for (const auto i : order)
    sorted.push_back(std::move(t.records[i]));
  t.records = std::move(sorted);
  for (size_t i = 0; i < order.size(); ++i) {
    const auto [id, seq] = keys[order[i]];
    const auto &p = producer_names[id];
    if (i > 0 && keys[order[i - 1]].first == id) {
      const auto previous = keys[order[i - 1]].second;
      if (seq == previous)
        issue("duplicate_sequence", path, seq, p);
      else if (seq - previous != 1)
        issue("sequence_gap", path, seq, p);
    } else if (seq != 1)
      issue("initial_sequence_gap", path, seq, p);
    if (t.records[i]["type"].as_string() == "Loss")
      t.partial = true;
  }
  return t;
}
} // namespace reqscope
