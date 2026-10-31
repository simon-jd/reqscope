#include "format/json.hpp"
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace reqscope {
uint64_t Json::as_u64() const {
  if (auto p = std::get_if<uint64_t>(&data_))
    return *p;
  if (auto p = std::get_if<int64_t>(&data_); p && *p >= 0)
    return uint64_t(*p);
  throw std::runtime_error("expected nonnegative integer");
}
int64_t Json::as_i64() const {
  if (auto p = std::get_if<int64_t>(&data_))
    return *p;
  if (auto p = std::get_if<uint64_t>(&data_); p && *p <= uint64_t(INT64_MAX))
    return int64_t(*p);
  throw std::runtime_error("expected signed integer");
}
double Json::as_double() const {
  if (auto p = std::get_if<double>(&data_))
    return *p;
  if (auto p = std::get_if<uint64_t>(&data_))
    return double(*p);
  if (auto p = std::get_if<int64_t>(&data_))
    return double(*p);
  throw std::runtime_error("expected number");
}
size_t Json::estimated_memory() const {
  size_t bytes = sizeof(Json) + 32;
  if (is_string())
    bytes += as_string().capacity() + 1;
  if (is_array()) {
    bytes += (array_items().capacity() - array_items().size()) * sizeof(Json);
    for (const auto &item : array_items())
      bytes += item.estimated_memory();
  }
  if (is_object()) {
    bytes += (object_items().capacity() - object_items().size()) * sizeof(Object::value_type);
    for (const auto &[key, item] : object_items())
      bytes += sizeof(std::string) + key.capacity() + item.estimated_memory();
  }
  return bytes;
}
namespace {
struct Parser {
  std::string_view s;
  size_t i = 0;
  size_t nodes = 0;
  [[noreturn]] void fail() const {
    throw std::runtime_error("invalid JSON at byte " + std::to_string(i));
  }
  void ws() {
    while (i < s.size() &&
           (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t'))
      ++i;
  }
  bool take(char c) {
    ws();
    if (i < s.size() && s[i] == c) {
      ++i;
      return true;
    }
    return false;
  }
  unsigned hex4() {
    unsigned v = 0;
    for (int k = 0; k < 4; ++k) {
      if (i == s.size())
        fail();
      char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9')
        v += c - '0';
      else if (c >= 'a' && c <= 'f')
        v += c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
        v += c - 'A' + 10;
      else
        fail();
    }
    return v;
  }
  void utf8(std::string &out, unsigned cp) {
    if (cp < 0x80)
      out += char(cp);
    else if (cp < 0x800) {
      out += char(0xc0 | (cp >> 6));
      out += char(0x80 | (cp & 63));
    } else if (cp < 0x10000) {
      out += char(0xe0 | (cp >> 12));
      out += char(0x80 | ((cp >> 6) & 63));
      out += char(0x80 | (cp & 63));
    } else {
      out += char(0xf0 | (cp >> 18));
      out += char(0x80 | ((cp >> 12) & 63));
      out += char(0x80 | ((cp >> 6) & 63));
      out += char(0x80 | (cp & 63));
    }
  }
  std::string str() {
    if (!take('"'))
      fail();
    // Plain ASCII up to the closing quote is the usual case: copy it once,
    // at its exact size, instead of a character at a time.
    size_t end = i;
    for (; end < s.size(); ++end) {
      const auto c = static_cast<unsigned char>(s[end]);
      if (c < 32 || c >= 128 || c == '"' || c == '\\')
        break;
    }
    std::string out(s.substr(i, end - i));
    i = end;
    if (i < s.size() && s[i] == '"') {
      ++i;
      return out;
    }
    while (i < s.size()) {
      unsigned char c = s[i++];
      if (c == '"')
        return out;
      if (c < 32)
        fail();
      if (c != '\\') {
        if (c < 128) {
          out += char(c);
          continue;
        }
        unsigned cp;
        int n;
        if (c >= 0xc2 && c <= 0xdf) {
          cp = c & 31;
          n = 1;
        } else if (c >= 0xe0 && c <= 0xef) {
          cp = c & 15;
          n = 2;
        } else if (c >= 0xf0 && c <= 0xf4) {
          cp = c & 7;
          n = 3;
        } else
          fail();
        std::string raw(1, char(c));
        for (int k = 0; k < n; k++) {
          if (i >= s.size())
            fail();
          unsigned char d = s[i++];
          if ((d & 0xc0) != 0x80)
            fail();
          raw += char(d);
          cp = (cp << 6) | (d & 63);
        }
        if ((n == 1 && cp < 128) || (n == 2 && cp < 2048) ||
            (n == 3 && cp < 65536) || cp > 0x10ffff ||
            (cp >= 0xd800 && cp <= 0xdfff))
          fail();
        out += raw;
        continue;
      }
      if (i == s.size())
        fail();
      char e = s[i++];
      switch (e) {
      case '"':
      case '\\':
      case '/':
        out += e;
        break;
      case 'b':
        out += '\b';
        break;
      case 'f':
        out += '\f';
        break;
      case 'n':
        out += '\n';
        break;
      case 'r':
        out += '\r';
        break;
      case 't':
        out += '\t';
        break;
      case 'u': {
        unsigned cp = hex4();
        if (cp >= 0xd800 && cp <= 0xdbff) {
          if (i + 2 > s.size() || s[i++] != '\\' || s[i++] != 'u')
            fail();
          unsigned lo = hex4();
          if (lo < 0xdc00 || lo > 0xdfff)
            fail();
          cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
        } else if (cp >= 0xdc00 && cp <= 0xdfff)
          fail();
        utf8(out, cp);
        break;
      }
      default:
        fail();
      }
    }
    fail();
  }
  Json value(unsigned depth) {
    if (depth > 64 || ++nodes > 65536)
      fail();
    ws();
    if (i == s.size())
      fail();
    char c = s[i];
    if (c == '"')
      return str();
    if (take('{')) {
      Json o = Json::object();
      if (take('}'))
        return o;
      // Room for a typical record, then trimmed: a trace keeps millions of
      // these, and doubling growth would leave a third of each unused.
      o.object_items().reserve(16);
      do {
        auto k = str();
        if (o.contains(k) || !take(':'))
          fail();
        o[k] = value(depth + 1);
        if (take('}')) {
          o.object_items().shrink_to_fit();
          return o;
        }
      } while (take(','));
      fail();
    }
    if (take('[')) {
      Json a = Json::array();
      if (take(']'))
        return a;
      do {
        a.push_back(value(depth + 1));
        if (take(']'))
          return a;
      } while (take(','));
      fail();
    }
    for (auto lit : {std::string_view("true"), std::string_view("false"),
                     std::string_view("null")})
      if (s.substr(i, lit.size()) == lit) {
        i += lit.size();
        return lit == "null" ? Json(nullptr) : Json(lit == "true");
      }
    size_t begin = i;
    bool neg = false, real = false;
    if (s[i] == '-') {
      neg = true;
      if (++i == s.size())
        fail();
    }
    if (s[i] == '0')
      ++i;
    else {
      if (s[i] < '1' || s[i] > '9')
        fail();
      while (i < s.size() && s[i] >= '0' && s[i] <= '9')
        ++i;
    }
    if (i < s.size() && s[i] == '.') {
      real = true;
      ++i;
      size_t p = i;
      while (i < s.size() && s[i] >= '0' && s[i] <= '9')
        ++i;
      if (i == p)
        fail();
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
      real = true;
      ++i;
      if (i < s.size() && (s[i] == '+' || s[i] == '-'))
        ++i;
      size_t p = i;
      while (i < s.size() && s[i] >= '0' && s[i] <= '9')
        ++i;
      if (i == p)
        fail();
    }
    auto token = s.substr(begin, i - begin);
    if (real) {
      double v;
      auto [p, ec] =
          std::from_chars(token.data(), token.data() + token.size(), v);
      if (ec != std::errc{} || p != token.data() + token.size() ||
          !std::isfinite(v))
        fail();
      return v;
    }
    if (neg) {
      int64_t v;
      auto [p, ec] =
          std::from_chars(token.data(), token.data() + token.size(), v);
      if (ec != std::errc{})
        fail();
      return v;
    }
    uint64_t v;
    auto [p, ec] =
        std::from_chars(token.data(), token.data() + token.size(), v);
    if (ec != std::errc{})
      fail();
    return v;
  }
};
std::string quote(const std::string &s) {
  std::string out = "\"";
  const char *h = "0123456789abcdef";
  for (unsigned char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 32) {
        out += "\\u00";
        out += h[c >> 4];
        out += h[c & 15];
      } else
        out += char(c);
    }
  }
  return out + '"';
}
} // namespace
Json Json::parse(std::string_view s) {
  Parser p{s};
  auto v = p.value(0);
  p.ws();
  if (p.i != s.size())
    p.fail();
  return v;
}
std::string Json::dump() const {
  if (is_null())
    return "null";
  if (is_bool())
    return as_bool() ? "true" : "false";
  if (auto p = std::get_if<int64_t>(&data_))
    return std::to_string(*p);
  if (auto p = std::get_if<uint64_t>(&data_))
    return std::to_string(*p);
  if (auto p = std::get_if<double>(&data_)) {
    if (!std::isfinite(*p))
      throw std::runtime_error("nonfinite JSON number");
    std::ostringstream o;
    o << std::setprecision(17) << *p;
    return o.str();
  }
  if (is_string())
    return quote(as_string());
  std::string out = is_array() ? "[" : "{";
  bool first = true;
  if (is_array())
    for (auto &v : array_items()) {
      if (!first)
        out += ',';
      first = false;
      out += v.dump();
    }
  else
    for (auto &[k, v] : object_items()) {
      if (!first)
        out += ',';
      first = false;
      out += quote(k) + ':' + v.dump();
    }
  return out + (is_array() ? ']' : '}');
}
} // namespace reqscope
