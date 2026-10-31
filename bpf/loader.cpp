#include "bpf/sched_event.h"
#include "bpf/sched_state.hpp"
#include "bpf/options.hpp"
#include "format/segment.hpp"
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#ifdef REQSCOPE_HAS_LIBBPF
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#endif

namespace fs = std::filesystem;
using reqscope::Json;
using reqscope::SegmentWriter;
namespace {
volatile sig_atomic_t stop_requested = 0;
void stop(int) { stop_requested = 1; }
uint64_t now_ns() { timespec t{}; clock_gettime(CLOCK_MONOTONIC, &t); return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec; }
std::string line(const fs::path& path) { std::ifstream f(path); std::string s; std::getline(f,s); return s; }
uint64_t start_ticks(unsigned pid) {
    auto text = line("/proc/" + std::to_string(pid) + "/stat");
    auto end = text.rfind(')');
    if (end == std::string::npos) throw std::runtime_error("target process does not exist");
    std::istringstream fields(text.substr(end + 2)); std::string token;
    for (unsigned i=3; i<=22; ++i) { if (!(fields>>token)) throw std::runtime_error("invalid proc stat"); if (i==22) return reqscope::sched::unsigned_option(token,UINT64_MAX,"proc start ticks"); }
    return 0;
}
struct Sample { uint64_t runtime = 0, wait = 0, time = 0; };
struct Session {
    fs::path output;
    unsigned target;
    uint64_t target_start;
    std::string producer, target_producer;
    Json header;
    uint64_t seq = 0, segment = 0, count = 0, ring_ns;
    uint64_t created = 0, last_dropped_time = 0;
    std::unique_ptr<SegmentWriter> writer;
    std::deque<std::pair<fs::path,uint64_t>> segments;
    reqscope::sched::StateMachine states;
    std::map<unsigned, Sample> samples;
    bool callback_failed = false;
    bool capture_partial = false;
    Session(fs::path path, unsigned pid, uint64_t ring_seconds) : output(std::move(path)), target(pid),
        target_start(start_ticks(pid)), ring_ns(ring_seconds * 1000000000ULL) {
        fs::create_directories(output);
        auto boot = line("/proc/sys/kernel/random/boot_id"); auto self_start = start_ticks(getpid());
        producer = boot + ":" + std::to_string(getpid()) + ":" + std::to_string(self_start);
        target_producer = boot + ":" + std::to_string(pid) + ":" + std::to_string(target_start);
        header = Json::object({{"type","Header"},{"producer",producer},{"pid",uint64_t(getpid())},
            {"boot_id",boot},{"start_time_ticks",self_start},{"clock","CLOCK_MONOTONIC"},
            {"clock_error_ns",0},{"clock_calibrated",true},{"target_producer",target_producer},
            {"target_pid",uint64_t(pid)},{"session_state","Capturing"}});
    }
    void write(Json record) {
        if (!writer) {
            auto path = output / ("sched-" + std::to_string(getpid()) + "-" + std::to_string(++segment) + ".rqs");
            if (fs::exists(path)) throw std::runtime_error("refusing existing scheduler segment");
            writer = std::make_unique<SegmentWriter>(path,header);
            created = now_ns(); count = 0; segments.emplace_back(path,created);
        }
        record["producer"]=producer; record["target_producer"]=target_producer;
        record["seq"]=++seq; record["timestamp"]=now_ns();
        writer->write(record);
        if (++count>=4096 || now_ns()-created>=1000000000ULL) {
            writer->flush(); writer.reset();
            uint64_t removed=0;
            while (segments.size()>1 && (segments.size()>64 || now_ns()-segments.front().second>ring_ns)) {
                fs::remove(segments.front().first); segments.pop_front(); ++removed;
            }
            if (removed) {
                header["retention_partial"]=true;
                write(Json::object({{"type","Loss"},{"reason","ring_overwrite"},{"count",removed},
                    {"count_unit","segments"},{"time_range",Json::array({uint64_t(0),now_ns()})}}));
            }
        }
    }
    void event(const reqscope_sched_event& event) {
        const auto transition = states.observe(event);
        using reqscope::sched::Result;
        if (transition.result == Result::thread_limit || transition.result == Result::invalid_kind) {
            capture_partial=true;
            write(Json::object({{"type","Loss"},
                {"reason",transition.result == Result::thread_limit ? "scheduler_thread_limit" : "invalid_scheduler_event_kind"},{"count",1},
                {"count_unit","events"},{"range_known",false},
                {"detail",transition.result == Result::thread_limit ? "thread limit" : "invalid event kind"},
                {"time_range",Json::array({uint64_t(0),uint64_t(0)})}}));
            return;
        }
        if (transition.result == Result::out_of_order) {
            capture_partial=true;
            write(Json::object({{"type","Loss"},{"reason","out_of_order_scheduler_event"},{"count",1},
                {"tid",uint64_t(event.tid)},
                {"count_unit","events"},{"time_range",Json::array({event.timestamp,transition.previous_since})}}));
            // Do not overwrite a later observed state with an older event.
            return;
        }
        if (transition.closed) {
            const auto& old=transition.closed->state;
            write(Json::object({{"type","Sched"},{"tid",uint64_t(event.tid)},{"start",old.since},
                {"end",transition.closed->end},{"state",old.state},{"prev_state",old.previous},
                {"waker_tid",uint64_t(old.waker)},{"source","bpf-btf-tracepoint"}}));
        }
    }
    bool target_alive() const {
        try { return start_ticks(target)==target_start; } catch (...) { return false; }
    }
    void poll_error(int result,uint64_t since) {
        capture_partial=true;
        write(Json::object({{"type","Loss"},{"reason","bpf_poll_error"},{"count",1},
            {"error_code",int64_t(result)},{"time_range",Json::array({since,now_ns()})}}));
    }
    void poll_schedstat(uint64_t resolution) {
        std::error_code error;
        fs::directory_iterator tids("/proc/"+std::to_string(target)+"/task",error);
        if (error) return;
        // Only currently present TIDs remain in memory; short-lived threads
        // cannot grow this map indefinitely.
        std::map<unsigned, Sample> current_samples;
        for (const auto& entry:tids) {
            unsigned tid=static_cast<unsigned>(reqscope::sched::unsigned_option(
                entry.path().filename().string(),std::numeric_limits<int>::max(),"proc tid"));
            std::ifstream input(entry.path()/"schedstat");
            uint64_t runtime=0,wait=0,slices=0;
            if (!(input>>runtime>>wait>>slices)) continue;
            const auto now=now_ns();
            auto prior=samples.find(tid);
            if (prior!=samples.end() && runtime>=prior->second.runtime && wait>=prior->second.wait) {
                write(Json::object({{"type","SchedulerSample"},{"tid",uint64_t(tid)},
                    {"start",prior->second.time},{"end",now},
                    {"cpu_runtime_delta_ns",runtime-prior->second.runtime},
                    {"runqueue_delta_ns",wait-prior->second.wait},{"resolution_ns",resolution},
                    {"source","proc-schedstat"},{"instantaneous_state_known",false}}));
            }
            if (current_samples.size()<8192) current_samples[tid]={runtime,wait,now};
        }
        samples=std::move(current_samples);
    }
    void finalize() {
        if(capture_partial)write(Json::object({{"type","Session"},{"state","Partial"},{"reason","scheduler_capture_error"}}));
        write(Json::object({{"type","Session"},{"state","Finalized"}}));
        if(writer)writer->flush();
    }
};
#ifdef REQSCOPE_HAS_LIBBPF
int event_callback(void* context, void* bytes, size_t length) {
    auto* session=static_cast<Session*>(context);
    try {
        if (length!=sizeof(reqscope_sched_event)) {
            session->capture_partial=true;
            session->states.clear();
            session->write(Json::object({{"type","Loss"},{"reason","invalid_scheduler_event_size"},
                {"count",1},{"count_unit","records"},{"range_known",false},
                {"observed_size",uint64_t(length)},
                {"time_range",Json::array({uint64_t(0),uint64_t(0)})}}));
            return -1;
        }
        session->event(*static_cast<reqscope_sched_event*>(bytes));
    }
    catch (...) { session->callback_failed=true; return -1; }
    return 0;
}
bool run_bpf(Session& session,const std::string& path,uint64_t deadline,unsigned poll_ms) {
    bpf_object* object=bpf_object__open_file(path.c_str(),nullptr);
    if (libbpf_get_error(object)) return false;
    if (bpf_object__load(object)) { bpf_object__close(object); return false; }
    int targets=bpf_object__find_map_fd_by_name(object,"target_tgids");
    int dropped_fd=bpf_object__find_map_fd_by_name(object,"dropped");
    const int cpu_count=libbpf_num_possible_cpus();
    unsigned char yes=1;
    if (targets<0 || dropped_fd<0 || cpu_count<=0 || cpu_count>65536 ||
        bpf_map_update_elem(targets,&session.target,&yes,BPF_ANY)) {
        bpf_object__close(object);return false;
    }
    std::vector<bpf_link*> links;
    bpf_program* program;
    bpf_object__for_each_program(program,object) {
        auto* link=bpf_program__attach(program);
        if (libbpf_get_error(link)) {
            for (auto* old:links)bpf_link__destroy(old);
            bpf_object__close(object);return false;
        }
        links.push_back(link);
    }
    auto* ring=ring_buffer__new(bpf_object__find_map_fd_by_name(object,"events"),event_callback,&session,nullptr);
    if (!ring) { for(auto* link:links)bpf_link__destroy(link);bpf_object__close(object);return false; }
    session.header["cpu_backend"]="libbpf-core";
    session.write(Json::object({{"type","Session"},{"state","Capturing"},{"backend","libbpf-core"}}));
    std::vector<uint64_t> dropped(static_cast<size_t>(cpu_count));
    uint64_t previous=0,last=now_ns(); unsigned key=0;
    while (!stop_requested && now_ns()<deadline && session.target_alive()) {
        int result=ring_buffer__poll(ring,poll_ms);
        if (result<0 && result!=-EINTR) {
            session.poll_error(result,last);
            break;
        }
        if (!bpf_map_lookup_elem(dropped_fd,&key,dropped.data())) {
            uint64_t total=0;for(auto value:dropped)total+=value;
            if (total>previous) {
                session.capture_partial=true;
                session.write(Json::object({{"type","Loss"},{"reason","bpf_ringbuf_full"},{"count",total-previous},
                    {"count_unit","events"},{"range_known",false},
                    {"time_range",Json::array({uint64_t(0),uint64_t(0)})},
                    {"observed_time_range",Json::array({last,now_ns()})}}));
                session.states.clear();
            }
            previous=total;last=now_ns();
        } else {
            const int error=errno;
            session.capture_partial=true;
            session.states.clear();
            session.write(Json::object({{"type","Loss"},{"reason","bpf_loss_counter_unavailable"},
                {"count",1},{"count_unit","errors"},{"error_code",error},{"range_known",false},
                {"time_range",Json::array({uint64_t(0),uint64_t(0)})}}));
            break;
        }
    }
    ring_buffer__free(ring);
    for(auto* link:links)bpf_link__destroy(link);
    bpf_object__close(object);
    if (session.callback_failed) throw std::runtime_error("scheduler event writer failed");
    return true;
}
#endif
}
int main(int argc,char** argv) {
    try {
        unsigned pid=0,poll_ms=20;uint64_t duration=0,ring_seconds=30;
        fs::path output;std::string object="sched.bpf.o";bool fallback=false;
        for(int i=1;i<argc;++i) {
            std::string option=argv[i];
            if(option=="--allow-schedstat") { fallback=true;continue; }
            if(option=="--help") {
                std::cout<<"reqscope-bpf --pid PID --output DIR [--bpf-object FILE] [--allow-schedstat] [--poll-ms N] [--duration-seconds N] [--ring-seconds N]\n";return 0;
            }
            if(i+1>=argc)throw std::runtime_error("missing option value");
            std::string value=argv[++i];
            if(option=="--pid")pid=static_cast<unsigned>(reqscope::sched::unsigned_option(value,std::numeric_limits<int>::max(),"pid"));
            else if(option=="--output")output=value;
            else if(option=="--bpf-object")object=value;
            else if(option=="--poll-ms")poll_ms=static_cast<unsigned>(reqscope::sched::unsigned_option(value,10000,"poll-ms"));
            else if(option=="--duration-seconds")duration=reqscope::sched::unsigned_option(value,UINT64_MAX,"duration-seconds");
            else if(option=="--ring-seconds")ring_seconds=reqscope::sched::unsigned_option(value,86400,"ring-seconds");
            else throw std::runtime_error("unknown option: "+option);
        }
        if(!pid||output.empty()||poll_ms<1||poll_ms>10000||!ring_seconds||ring_seconds>86400)
            throw std::runtime_error("valid --pid and --output required; poll-ms 1..10000; ring-seconds 1..86400");
        const auto deadline=reqscope::sched::deadline_after_seconds(now_ns(),duration);
        Session session(output,pid,ring_seconds);
        std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
        bool captured=false;
#ifdef REQSCOPE_HAS_LIBBPF
        captured=run_bpf(session,object,deadline,poll_ms);
#endif
        if(!captured) {
            if(!fallback)throw std::runtime_error("BPF unavailable (build, object, BTF, or permission); --allow-schedstat enables explicitly coarse sampling");
            session.header["cpu_backend"]="proc-schedstat";
            session.header["session_state"]="Partial";
            session.header["resolution_ns"]=uint64_t(poll_ms)*1000000ULL;
            session.write(Json::object({{"type","Session"},{"state","Partial"},{"reason","BPF unavailable; coarse schedstat aggregates only"}}));
            while(!stop_requested && now_ns()<deadline && session.target_alive()) {
                session.poll_schedstat(uint64_t(poll_ms)*1000000ULL);
                std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
            }
        }
        session.finalize();return 0;
    } catch(const std::exception& error) { std::cerr<<"reqscope-bpf: "<<error.what()<<'\n';return 1; }
}
