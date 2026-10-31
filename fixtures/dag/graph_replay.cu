#include "common.hpp"
__global__ void graph_first(int* data) { if(threadIdx.x==0)data[0]+=1; }
__global__ void graph_second(int* data) { if(threadIdx.x==0)data[0]+=2; }
#if defined(REQSCOPE_ENABLE_CONDITIONAL_GRAPHS) && CUDART_VERSION >= 13000
__global__ void conditional_body(int* count,cudaGraphConditionalHandle handle) {
    if(threadIdx.x==0) { int n=++count[0];cudaGraphSetConditional(handle,n<3); }
}
static void conditional_fixture(Trace& trace,cudaStream_t stream) {
    int* count=nullptr;CUDA_CHECK(cudaMalloc(&count,sizeof(int)));CUDA_CHECK(cudaMemset(count,0,sizeof(int)));
    cudaGraph_t graph{};CUDA_CHECK(cudaGraphCreate(&graph,0));cudaGraphConditionalHandle handle{};
    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle,graph,1,cudaGraphCondAssignDefault));
    cudaGraphNodeParams parameters{};parameters.type=cudaGraphNodeTypeConditional;
    parameters.conditional.handle=handle;parameters.conditional.type=cudaGraphCondTypeWhile;parameters.conditional.size=1;
    cudaGraphNode_t conditional{};
    CUDA_CHECK(cudaGraphAddNode(&conditional,graph,nullptr,nullptr,0,&parameters));
    cudaKernelNodeParams body{};body.func=reinterpret_cast<void*>(conditional_body);body.gridDim=dim3(1);body.blockDim=dim3(1);
    void* args[]={&count,&handle};body.kernelParams=args;cudaGraphNode_t body_node{};
    CUDA_CHECK(cudaGraphAddKernelNode(&body_node,parameters.conditional.phGraph_out[0],nullptr,0,&body));
    cudaGraphExec_t executable{};CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
    trace.begin(100,"[{\"req_id\":\"conditional-request\",\"tokens\":1,\"phase\":\"decode\"}]");
    CUDA_CHECK(cudaGraphLaunch(executable,stream));CUDA_CHECK(cudaStreamSynchronize(stream));trace.end();
    int result=0;CUDA_CHECK(cudaMemcpy(&result,count,sizeof(int),cudaMemcpyDeviceToHost));
    if(result!=3)throw std::runtime_error("conditional graph oracle failed");
    CUDA_CHECK(cudaGraphExecDestroy(executable));CUDA_CHECK(cudaGraphDestroy(graph));CUDA_CHECK(cudaFree(count));
}
#endif
int main(int argc,char** argv) {
    try {
        bool conditional=argc>1&&std::string(argv[1])=="--conditional";
#if !defined(REQSCOPE_ENABLE_CONDITIONAL_GRAPHS) || CUDART_VERSION < 13000
        if(conditional)throw std::runtime_error("conditional graph fixture was not enabled with CUDA >=13.0");
#endif
        Trace trace;int* device=nullptr;cudaStream_t stream{};cudaGraph_t graph{};cudaGraphExec_t executable{};
        CUDA_CHECK(cudaMalloc(&device,sizeof(int)));CUDA_CHECK(cudaMemset(device,0,sizeof(int)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        graph_first<<<1,1,0,stream>>>(device);graph_second<<<1,1,0,stream>>>(device);
        CUDA_CHECK(cudaStreamEndCapture(stream,&graph));CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
        const char* members[]={
            "[{\"req_id\":\"A\",\"tokens\":1,\"phase\":\"decode\"}]",
            "[{\"req_id\":\"A\",\"tokens\":1,\"phase\":\"decode\"},{\"req_id\":\"B\",\"tokens\":2,\"phase\":\"prefill\"}]",
            "[{\"req_id\":\"B\",\"tokens\":1,\"phase\":\"decode\"}]"};
        uint64_t starts[3]{},ends[3]{};
        for(unsigned replay=0;replay<3;++replay) {
            trace.begin(replay+1,members[replay]);CUDA_CHECK(cudaGraphLaunch(executable,stream));
            const auto before=timestamp_ns();CUDA_CHECK(cudaStreamSynchronize(stream));trace.clock_sample(before,timestamp_ns());
            trace.end(replay==1?1:3);
            starts[replay]=trace.last_begin_ns();ends[replay]=trace.last_end_ns();
        }
        trace.request("A",starts[0],ends[1],2);trace.request("B",starts[1],ends[2],3);
        int result=0;CUDA_CHECK(cudaMemcpy(&result,device,sizeof(int),cudaMemcpyDeviceToHost));
        if(result!=9)throw std::runtime_error("graph replay DAG oracle failed");
#if defined(REQSCOPE_ENABLE_CONDITIONAL_GRAPHS) && CUDART_VERSION >= 13000
        if(conditional)conditional_fixture(trace,stream);
#endif
        CUDA_CHECK(cudaGraphExecDestroy(executable));CUDA_CHECK(cudaGraphDestroy(graph));CUDA_CHECK(cudaFree(device));CUDA_CHECK(cudaStreamDestroy(stream));trace.finish();
        truth(std::string(R"({"fixture":"graph_replay","graph_replays":3,"nodes_per_replay":["graph_first","graph_second"],"edges_per_replay":[["graph_first","graph_second"]],"steps":[{"step_id":1,"members":["A"],"padding":3},{"step_id":2,"members":["A","B"],"padding":1},{"step_id":3,"members":["B"],"padding":3}],"conditional_iterations":)")+(conditional?"3":"0")+"}");return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"graph_replay: %s\n",error.what());return 1;}
}
