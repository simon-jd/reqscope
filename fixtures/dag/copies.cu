#include "common.hpp"
__global__ void copy_transform(float* data,size_t count) { size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)data[i]=data[i]*2+1; }
int main() {
    try {
        Trace trace;constexpr size_t count=1<<20;float *host=nullptr,*device=nullptr;cudaStream_t stream{};
        CUDA_CHECK(cudaMallocHost(&host,count*sizeof(float)));CUDA_CHECK(cudaMalloc(&device,count*sizeof(float)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        for(size_t i=0;i<count;++i)host[i]=float(i%1024);
        trace.begin(1,"[{\"req_id\":\"copies-request\",\"tokens\":1,\"phase\":\"prefill\"}]");
        CUDA_CHECK(cudaMemcpyAsync(device,host,count*sizeof(float),cudaMemcpyHostToDevice,stream));
        copy_transform<<<(count+255)/256,256,0,stream>>>(device,count);CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(host,device,count*sizeof(float),cudaMemcpyDeviceToHost,stream));
        const auto before=timestamp_ns();CUDA_CHECK(cudaStreamSynchronize(stream));trace.clock_sample(before,timestamp_ns());trace.end();
        trace.request("copies-request",trace.last_begin_ns(),trace.last_end_ns(),1);
        for(size_t i=0;i<count;++i)if(host[i]!=float((i%1024)*2+1))throw std::runtime_error("copy DAG oracle failed");
        CUDA_CHECK(cudaStreamDestroy(stream));CUDA_CHECK(cudaFreeHost(host));CUDA_CHECK(cudaFree(device));trace.finish();
        truth(R"({"fixture":"copies","nodes":["H2D","copy_transform","D2H"],"edges":[["H2D","copy_transform"],["copy_transform","D2H"]],"bytes_each_copy":4194304,"step_id":1})");return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"copies: %s\n",error.what());return 1;}
}
