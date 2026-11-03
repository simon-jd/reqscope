#include "analyzer/analyzer.hpp"
#include "cli/otel.hpp"
#include "format/segment.hpp"
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef REQSCOPE_INSTALLED_LIBRARY_RELATIVE_PATH
#define REQSCOPE_INSTALLED_LIBRARY_RELATIVE_PATH "../lib/libreqscope.so"
#endif

namespace fs = std::filesystem;
using namespace reqscope;
namespace {
volatile sig_atomic_t child_pid = 0;
Json clock_anchor() {
  uint64_t best_width = UINT64_MAX, mono_mid = 0, unix_ns = 0;
  auto ns = [](const timespec &t) {
    return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
  };
  for (int sample = 0; sample < 100; ++sample) {
    timespec before{}, realtime{}, after{};
    if (clock_gettime(CLOCK_MONOTONIC, &before) ||
        clock_gettime(CLOCK_REALTIME, &realtime) ||
        clock_gettime(CLOCK_MONOTONIC, &after))
      throw std::runtime_error("clock_gettime failed");
    const auto width = ns(after) - ns(before);
    if (width < best_width) {
      best_width = width;
      mono_mid = ns(before) + width / 2;
      unix_ns = ns(realtime);
    }
  }
  return Json::object(
      {{"unix_ns", unix_ns},
       {"monotonic_ns", mono_mid},
       {"unix_minus_monotonic_ns", int64_t(unix_ns) - int64_t(mono_mid)},
       {"acquisition_error_ns", best_width},
       {"samples", 100},
       {"purpose", "OTel host-clock conversion; not GPU timestamp calibration"},
       {"limitation",
        "Measure again after capture and bound clock drift separately."}});
}
void forward_signal(int sig) {
  if (child_pid > 0)
    kill(-child_pid, sig);
}
uint64_t uint_arg(const std::string &s) {
  size_t pos = 0;
  if (s.empty() || s[0] == '-')
    throw std::runtime_error("expected positive integer");
  auto n = std::stoull(s, &pos);
  if (pos != s.size())
    throw std::runtime_error("invalid integer: " + s);
  return n;
}
std::string next(int &i, int argc, char **argv) {
  if (++i >= argc)
    throw std::runtime_error("missing option value");
  return argv[i];
}
uint64_t duration(const std::string &s) {
  uint64_t factor = 1000000000, n;
  std::string digits = s;
  if (s.ends_with("ms")) {
    factor = 1000000;
    digits = s.substr(0, s.size() - 2);
  } else if (s.ends_with("s"))
    digits = s.substr(0, s.size() - 1);
  n = uint_arg(digits);
  if (n > UINT64_MAX / factor)
    throw std::runtime_error("duration overflow");
  return n * factor;
}
void emit(const Json &v, const std::string &output = "") {
  if (output.empty()) {
    std::cout << v.dump() << '\n';
    return;
  }
  if (fs::exists(output))
    throw std::runtime_error("output already exists");
  std::ofstream f(output);
  f << v.dump() << '\n';
  if (!f)
    throw std::runtime_error("cannot write output");
}
int dump(int argc, char **argv) {
  std::string pid, output, sock;
  uint64_t last = 30000000000ULL;
  uint64_t timeout_seconds = 30;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--pid")
      pid = next(i, argc, argv);
    else if (a == "--output")
      output = next(i, argc, argv);
    else if (a == "--socket")
      sock = next(i, argc, argv);
    else if (a == "--last")
      last = duration(next(i, argc, argv));
    else if (a == "--timeout-seconds")
      timeout_seconds = uint_arg(next(i, argc, argv));
    else
      throw std::runtime_error("unknown dump option: " + a);
  }
  if (sock.empty()) {
    if (pid.empty() || uint_arg(pid) == 0)
      throw std::runtime_error("dump requires --pid or --socket");
    sock = "/tmp/reqscope-" + pid + ".sock";
  }
  if (output.empty())
    output = "reqscope-dump-" + std::to_string(getpid());
  if (!timeout_seconds || timeout_seconds > 300)
    throw std::runtime_error("dump timeout-seconds must be 1..300");
  output = fs::absolute(output).string();
  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0)
    throw std::runtime_error("socket failed");
  struct Guard {
    int f;
    ~Guard() { close(f); }
  } guard{fd};
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (sock.size() >= sizeof(addr.sun_path))
    throw std::runtime_error("socket path too long");
  std::strcpy(addr.sun_path, sock.c_str());
  timeval timeout{static_cast<time_t>(timeout_seconds), 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)))
    throw std::runtime_error(std::string("connect: ") + strerror(errno));
  auto msg =
      Json::object({{"command", "dump"}, {"output", output}, {"last_ns", last}})
          .dump() +
      "\n";
  size_t sent = 0;
  while (sent < msg.size()) {
    auto n = send(fd, msg.data() + sent, msg.size() - sent, MSG_NOSIGNAL);
    if (n <= 0)
      throw std::runtime_error("dump send failed");
    sent += n;
  }
  std::string response;
  char buf[1024];
  while (response.find('\n') == std::string::npos) {
    auto n = recv(fd, buf, sizeof(buf), 0);
    if (n <= 0)
      throw std::runtime_error("dump response timeout or disconnected");
    response.append(buf, n);
    if (response.size() > 1024 * 1024)
      throw std::runtime_error("dump response too large");
  }
  auto result = Json::parse(response.substr(0, response.find('\n')));
  emit(result);
  return result.boolean("ok") ? 0 : 1;
}
int run(int argc, char **argv) {
  std::string mode = "light", output, library;
  uint64_t ring = 30;
  int command = 0;
  bool manual = false;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--") {
      command = i + 1;
      break;
    }
    if (a == "--mode")
      mode = next(i, argc, argv);
    else if (a == "--output")
      output = next(i, argc, argv);
    else if (a == "--ring-seconds")
      ring = uint_arg(next(i, argc, argv));
    else if (a == "--library")
      library = next(i, argc, argv);
    else if (a == "--manual")
      manual = true;
    else
      throw std::runtime_error("unknown run option: " + a);
  }
  if (!command || command >= argc)
    throw std::runtime_error("run requires -- <command>");
  if (mode != "light" && mode != "trace" && mode != "sampled")
    throw std::runtime_error("mode must be light, trace or sampled");
  if (!ring || ring > 86400)
    throw std::runtime_error("ring-seconds must be 1..86400");
  if (output.empty())
    output = "reqscope-run-" + std::to_string(getpid());
  output = fs::absolute(output).string();
  if (fs::exists(output))
    throw std::runtime_error("run output must be a new directory");
  if (library.empty()) {
    if (auto e = getenv("REQSCOPE_LIBRARY"))
      library = e;
    else {
      auto exe = fs::canonical("/proc/self/exe");
      auto adjacent = exe.parent_path() / "libreqscope.so";
      library = fs::exists(adjacent)
                    ? adjacent.string()
                    : (exe.parent_path() / REQSCOPE_INSTALLED_LIBRARY_RELATIVE_PATH)
                          .lexically_normal()
                          .string();
    }
  }
  library = fs::absolute(library).string();
  if (!fs::is_regular_file(library))
    throw std::runtime_error("collector library not found: " + library);
  fs::create_directories(output);
  Json manifest = Json::object({{"schema_version", 1},
                                {"state", "Configured"},
                                {"mode", mode},
                                {"ring_seconds", ring},
                                {"evidence", "unvalidated"},
                                {"library", library}});
  auto save = [&] {
    std::ofstream f(fs::path(output) / "session.json");
    f << manifest.dump() << '\n';
    if (!f)
      throw std::runtime_error("session write failed");
  };
  save();
  struct sigaction action{}, oldint{}, oldterm{};
  action.sa_handler = forward_signal;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, &oldint);
  sigaction(SIGTERM, &action, &oldterm);
  pid_t pid = fork();
  if (pid < 0)
    throw std::runtime_error("fork failed");
  if (pid == 0) {
    setpgid(0, 0);
    sigaction(SIGINT, &oldint, nullptr);
    sigaction(SIGTERM, &oldterm, nullptr);
    setenv("CUDA_INJECTION64_PATH", library.c_str(), 1);
    setenv("REQSCOPE_LIBRARY", library.c_str(), 1);
    setenv("REQSCOPE_OUTPUT", output.c_str(), 1);
    setenv("REQSCOPE_MODE", mode.c_str(), 1);
    setenv("REQSCOPE_RING_SECONDS", std::to_string(ring).c_str(), 1);
    setenv("REQSCOPE_ENABLED", "1", 1);
    if (manual)
      setenv("REQSCOPE_BACKEND", "manual", 1);
    if (auto existing = getenv("VLLM_PLUGINS")) {
      std::string plugins = existing;
      std::istringstream names(plugins);
      std::string name;
      bool has_plugin = false;
      while (std::getline(names, name, ','))
        has_plugin = has_plugin || name == "reqscope";
      if (!has_plugin)
        plugins += (plugins.empty() ? "" : ",") + std::string("reqscope");
      setenv("VLLM_PLUGINS", plugins.c_str(), 1);
    }
    execvp(argv[command], argv + command);
    std::cerr << "reqscope: exec failed: " << strerror(errno) << '\n';
    _exit(127);
  }
  child_pid = pid;
  setpgid(pid, pid);
  manifest["pid"] = uint64_t(pid);
  manifest["state"] = "Attached";
  save();
  std::cerr << "reqscope: output " << output << ", child " << pid << '\n';
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR)
      throw std::runtime_error("waitpid failed");
  }
  child_pid = 0;
  sigaction(SIGINT, &oldint, nullptr);
  sigaction(SIGTERM, &oldterm, nullptr);
  int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  bool partial = code != 0;
  try {
    auto trace = read_trace(output);
    partial = partial || trace.partial;
    std::map<std::string, bool> finalized;
    for (const auto &header : trace.headers) {
      finalized.try_emplace(header.string("producer"), false);
      const auto backend = header.string("backend_status");
      partial = partial || header.string("session_state") == "Partial" ||
                backend == "unavailable" || backend == "failed";
    }
    for (const auto &record : trace.records) {
      auto producer = record.string("producer");
      if (record.string("type") == "Session") {
        const auto state = record.string("state");
        partial = partial || state == "Partial";
        finalized[producer] = state == "Finalized" || state == "Snapshot";
      } else if (record.string("type") != "Loss") {
        finalized[producer] = false;
      }
    }
    for (const auto &[producer, complete] : finalized)
      partial = partial || !complete;
  } catch (...) {
    partial = true;
  }
  manifest["exit_code"] = code;
  manifest["state"] = partial ? "Partial" : "Finalized";
  save();
  return code;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help") {
      std::cout << "reqscope 0.1.0\n  run --mode light|trace|sampled [--ring-seconds "
                   "N] [--output DIR] -- COMMAND...\n  dump --pid PID [--last "
                   "30s] [--timeout-seconds 30] [--output DIR]\n  inspect RUN [--output FILE]\n  analyze RUN [--output FILE]\n  "
                   "explain RUN --request ID [--output FILE]\n  blame RUN [--request ID] [--output FILE]\n  diff RUN_A "
                   "RUN_B [--by prompt-bucket] [--output FILE]\n  "
                   "export-perfetto RUN [--output FILE]\n  "
                   "import-otel RUN --input FILE --output DIR --producer ID "
                   "--unix-minus-monotonic-ns OFFSET --clock-error-ns BOUND [--client-log FILE]\n  clock-anchor\n";
      return 0;
    }
    std::string cmd = argv[1];
    if (cmd == "clock-anchor") {
      emit(clock_anchor());
      return 0;
    }
    if (cmd == "--version") {
      std::cout << "reqscope 0.1.0\n";
      return 0;
    }
    if (cmd == "run")
      return run(argc, argv);
    if (cmd == "dump")
      return dump(argc, argv);
    if (cmd == "import-otel")
      return import_otel(argc, argv);
    if (argc < 3)
      throw std::runtime_error("missing trace path");
    std::string output, request, by = "prompt-bucket";
    int option = cmd == "diff" ? 4 : 3;
    if (cmd == "diff" && argc < 4)
      throw std::runtime_error("diff requires two traces");
    for (int i = option; i < argc; i++) {
      std::string a = argv[i];
      if (a == "--output")
        output = next(i, argc, argv);
      else if (a == "--request")
        request = next(i, argc, argv);
      else if (a == "--by")
        by = next(i, argc, argv);
      else
        throw std::runtime_error("unknown option: " + a);
    }
    auto trace = read_trace(argv[2]);
    Json result;
    if (cmd == "inspect") {
      Json counts = Json::object();
      for (const auto &r : trace.records) {
        const auto kind = r.string("type");
        counts[kind] = counts.integer(kind) + 1;
      }
      result = Json::object({{"schema_version", 1},
                             {"raw_status", trace.partial ? "Partial" : "Valid"},
                             {"note", "Raw framing and sequence validity only; use analyze for attribution and clock evidence."},
                             {"record_counts", counts},
                             {"headers", trace.headers},
                             {"diagnostics", trace.diagnostics}});
    } else if (cmd == "analyze")
      result = analyze(trace);
    else if (cmd == "explain") {
      if (request.empty())
        throw std::runtime_error("explain requires --request");
      result = explain(trace, request);
    } else if (cmd == "blame")
      result = blame(trace, request);
    else if (cmd == "diff")
      result = diff(trace, read_trace(argv[3]), by);
    else if (cmd == "export-perfetto")
      result = perfetto(trace);
    else
      throw std::runtime_error("unknown command: " + cmd);
    emit(result, output);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "reqscope: " << e.what() << '\n';
    return 2;
  }
}
