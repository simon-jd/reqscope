#pragma once
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

#define CUDA_CHECK(call) do { cudaError_t status=(call); if(status!=cudaSuccess) \
    throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(status)); } while(0)
inline uint64_t timestamp_ns() { timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec; }
class Trace {
    using Emit=int(*)(const char*);using Push=int(*)(uint64_t);using Unary=int(*)();
    Emit emit_=nullptr;Push push_=nullptr;Unary pop_=nullptr,flush_=nullptr;
    uint64_t start_=0,end_=0,step_=0;
    std::string members_;
public:
    Trace() {
        const char* path=std::getenv("REQSCOPE_LIBRARY");
        if(!path)path=std::getenv("CUDA_INJECTION64_PATH");
        void* handle=path?dlopen(path,RTLD_NOW|RTLD_GLOBAL):RTLD_DEFAULT;
        if(path&&!handle)throw std::runtime_error(dlerror());
        auto init=reinterpret_cast<Unary>(dlsym(handle,"InitializeInjection"));
        if(init&&!init())throw std::runtime_error("collector initialization failed; inspect raw header");
        emit_=reinterpret_cast<Emit>(dlsym(handle,"reqscope_emit_json"));
        push_=reinterpret_cast<Push>(dlsym(handle,"reqscope_push_step"));
        pop_=reinterpret_cast<Unary>(dlsym(handle,"reqscope_pop_step"));
        flush_=reinterpret_cast<Unary>(dlsym(handle,"reqscope_flush"));
    }
    void begin(uint64_t step,std::string members) {
        step_=step;members_=std::move(members);start_=timestamp_ns();
        if(push_&&push_(step))throw std::runtime_error("external correlation push failed");
    }
    void end(unsigned padding=0) {
        uint64_t end=timestamp_ns();end_=end;
        if(pop_&&pop_())throw std::runtime_error("external correlation pop failed");
        auto record=std::string("{\"type\":\"Step\",\"step_id\":")+std::to_string(step_)+
            ",\"t_begin\":"+std::to_string(start_)+",\"t_end\":"+std::to_string(end)+
            ",\"tid\":"+std::to_string(syscall(SYS_gettid))+",\"members\":"+members_+
            ",\"padding\":"+std::to_string(padding)+",\"preempted\":[],\"finished\":[]}";
        if(emit_&&emit_(record.c_str()))throw std::runtime_error("step emission failed");
    }
    void clock_sample(uint64_t before,uint64_t after) {
        auto record=std::string("{\"type\":\"ClockSyncSample\",\"source\":\"stream-synchronize-return\",\"host_before\":")+
            std::to_string(before)+",\"sync_return_ns\":"+std::to_string(after)+",\"step_id\":"+
            std::to_string(step_)+",\"calibrated\":false}";
        if(emit_)emit_(record.c_str());
    }
    uint64_t last_begin_ns() const { return start_; }
    uint64_t last_end_ns() const { return end_; }
    void request(const std::string& id,uint64_t start,uint64_t end,unsigned tokens) {
        auto record=std::string("{\"type\":\"Request\",\"req_id\":\"")+id+"\",\"start\":"+
            std::to_string(start)+",\"end\":"+std::to_string(end)+",\"output_tokens\":"+std::to_string(tokens)+"}";
        if(emit_&&emit_(record.c_str()))throw std::runtime_error("request emission failed");
    }
    void finish() { if(flush_&&flush_())throw std::runtime_error("collector flush failed"); }
};
inline void truth(const std::string& document) {
    const char* path=std::getenv("REQSCOPE_TRUTH");
    if(path) { std::ofstream file(path);file<<document<<'\n';if(!file)throw std::runtime_error("cannot write fixture truth"); }
    std::puts(document.c_str());
}
