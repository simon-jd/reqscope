#include "common.hpp"
__global__ void stream_producer(int* data) { if(threadIdx.x==0)data[0]=21; }
__global__ void stream_consumer(int* data) { if(threadIdx.x==0)data[1]=2*data[0]; }
int main() {
    try {
        Trace trace;int* device=nullptr;cudaStream_t a{},b{};cudaEvent_t event{};
        CUDA_CHECK(cudaMalloc(&device,2*sizeof(int)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&a,cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&b,cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
        trace.begin(1,"[{\"req_id\":\"streams-request\",\"tokens\":1,\"phase\":\"decode\"}]");
        stream_producer<<<1,1,0,a>>>(device);CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(event,a));
        CUDA_CHECK(cudaStreamWaitEvent(b,event,0));
        stream_consumer<<<1,1,0,b>>>(device);CUDA_CHECK(cudaGetLastError());
        const auto before=timestamp_ns();CUDA_CHECK(cudaStreamSynchronize(b));trace.clock_sample(before,timestamp_ns());
        trace.end();trace.request("streams-request",trace.last_begin_ns(),trace.last_end_ns(),1);int result[2]{};
        CUDA_CHECK(cudaMemcpy(result,device,sizeof(result),cudaMemcpyDeviceToHost));
        if(result[0]!=21||result[1]!=42)throw std::runtime_error("stream DAG oracle failed");
        CUDA_CHECK(cudaEventDestroy(event));CUDA_CHECK(cudaStreamDestroy(a));CUDA_CHECK(cudaStreamDestroy(b));CUDA_CHECK(cudaFree(device));
        trace.finish();truth(R"({"fixture":"streams","nodes":["stream_producer","event_record","event_wait","stream_consumer"],"edges":[["stream_producer","event_record"],["event_record","event_wait"],["event_wait","stream_consumer"]],"expected_result":[21,42],"step_id":1})");
        return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"streams: %s\n",error.what());return 1;}
}
