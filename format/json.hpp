#pragma once
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace reqscope {
class Json {
public:
  using Array = std::vector<Json>;
  // Members sorted by key. A trace holds millions of small objects: one
  // allocation per object instead of one per member, and 40-byte values
  // instead of 56 (std::map is 48 bytes on its own).
  using Object = std::vector<std::pair<std::string, Json>>;
  using Storage = std::variant<std::nullptr_t, bool, int64_t, uint64_t, double,
                               std::string, Array, Object>;
  Json() : data_(nullptr) {}
  Json(std::nullptr_t) : data_(nullptr) {}
  Json(bool v) : data_(v) {}
  template <class T,
            std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>,
                             int> = 0>
  Json(T v) {
    if constexpr (std::is_signed_v<T>)
      data_ = int64_t(v);
    else
      data_ = uint64_t(v);
  }
  Json(double v) : data_(v) {}
  Json(const char *v) : data_(std::string(v)) {}
  Json(std::string v) : data_(std::move(v)) {}
  Json(std::string_view v) : data_(std::string(v)) {}
  Json(Array v) : data_(std::move(v)) {}
  Json(Object v) : data_(std::move(v)) {}
  static Json object(std::initializer_list<Object::value_type> v = {}) {
    Json o = Object{};
    for (const auto &[k, item] : v)
      if (!o.contains(k)) // as std::map: the first of repeated keys wins
        o[k] = item;
    return o;
  }
  static Json array(std::initializer_list<Json> v = {}) { return Array(v); }
  static Json parse(std::string_view input);
  std::string dump() const;
  size_t estimated_memory() const;
  bool is_null() const { return std::holds_alternative<std::nullptr_t>(data_); }
  bool is_bool() const { return std::holds_alternative<bool>(data_); }
  bool is_number() const { return data_.index() >= 2 && data_.index() <= 4; }
  bool is_string() const { return std::holds_alternative<std::string>(data_); }
  bool is_array() const { return std::holds_alternative<Array>(data_); }
  bool is_object() const { return std::holds_alternative<Object>(data_); }
  // The member named k, or nullptr when there is none or this is not an object.
  const Json *find(const std::string &k) const {
    if (!is_object())
      return nullptr;
    const auto &o = object_items();
    auto it = std::lower_bound(o.begin(), o.end(), k, before);
    return it != o.end() && it->first == k ? &it->second : nullptr;
  }
  bool contains(const std::string &k) const { return find(k) != nullptr; }
  Json &operator[](const std::string &k) {
    if (is_null())
      data_ = Object{};
    auto &o = std::get<Object>(data_);
    auto it = std::lower_bound(o.begin(), o.end(), k, before);
    if (it == o.end() || it->first != k)
      it = o.emplace(it, k, Json());
    return it->second;
  }
  const Json &operator[](const std::string &k) const {
    static const Json nil;
    const auto *found = find(k);
    return found ? *found : nil;
  }
  bool erase(const std::string &k) {
    auto &o = std::get<Object>(data_);
    auto it = std::lower_bound(o.begin(), o.end(), k, before);
    if (it == o.end() || it->first != k)
      return false;
    o.erase(it);
    return true;
  }
  Json &operator[](size_t i) { return std::get<Array>(data_).at(i); }
  const Json &operator[](size_t i) const {
    return std::get<Array>(data_).at(i);
  }
  Json &at(const std::string &k) {
    return const_cast<Json &>(std::as_const(*this).at(k));
  }
  const Json &at(const std::string &k) const {
    std::get<Object>(data_); // a non-object throws, as before
    if (const auto *found = find(k))
      return *found;
    throw std::out_of_range("no JSON member " + k);
  }
  Json value(const std::string &k, Json fallback = nullptr) const {
    const auto *found = find(k);
    return found ? *found : fallback;
  }
  std::string string(const std::string &k,
                     const std::string &fallback = "") const {
    const auto *found = find(k);
    return found ? found->as_string() : fallback;
  }
  uint64_t integer(const std::string &k, uint64_t fallback = 0) const {
    const auto *found = find(k);
    return found ? found->as_u64() : fallback;
  }
  double number(const std::string &k, double fallback = 0) const {
    const auto *found = find(k);
    return found ? found->as_double() : fallback;
  }
  bool boolean(const std::string &k, bool fallback = false) const {
    const auto *found = find(k);
    return found ? found->as_bool() : fallback;
  }
  const std::string &as_string() const { return std::get<std::string>(data_); }
  uint64_t as_u64() const;
  int64_t as_i64() const;
  double as_double() const;
  bool as_bool() const { return std::get<bool>(data_); }
  Array &array_items() { return std::get<Array>(data_); }
  const Array &array_items() const { return std::get<Array>(data_); }
  Object &object_items() { return std::get<Object>(data_); }
  const Object &object_items() const { return std::get<Object>(data_); }
  void push_back(Json v) {
    if (is_null())
      data_ = Array{};
    std::get<Array>(data_).push_back(std::move(v));
  }
  size_t size() const {
    if (is_array())
      return array_items().size();
    if (is_object())
      return object_items().size();
    if (is_string())
      return as_string().size();
    return 0;
  }
  bool operator==(const Json &) const = default;

private:
  static bool before(const Object::value_type &member, const std::string &k) {
    return member.first < k;
  }
  Storage data_;
};
} // namespace reqscope
