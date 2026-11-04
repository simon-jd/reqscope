// One CUDA graph of spin kernels replayed once per step, the way vLLM replays
// its decode graph: each step's graph is launched while the previous step's
// still runs, and the host then waits for the previous step only. Run under
// the collector in sampled mode to check what a traced step's kernel records
// hold when the steps after it are not traced.
//
//     dag_sampled_replay [steps]
#include "common.hpp"
#include <string>

constexpr unsigned kNodes = 8;
constexpr unsigned long long kSpinNs = 100000;

__global__ void spin(unsigned long long ns) {
    unsigned long long start = 0, now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); } while (now - start < ns);
}

int main(int argc, char** argv) {
    try {
        const unsigned steps = argc > 1 ? unsigned(std::stoul(argv[1])) : 48;
        Trace trace;
        cudaStream_t stream{};
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaGraph_t graph{};
        cudaGraphExec_t executable{};
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (unsigned node = 0; node < kNodes; ++node) spin<<<1, 1, 0, stream>>>(kSpinNs);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
        cudaEvent_t done[2]{};
        for (auto& event : done) CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        for (unsigned step = 1; step <= steps; ++step) {
            trace.begin(step, "[{\"req_id\":\"A\",\"tokens\":1,\"phase\":\"decode\"}]");
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaEventRecord(done[step % 2], stream));
            trace.end(0);
            if (step > 1) CUDA_CHECK(cudaEventSynchronize(done[(step - 1) % 2]));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (auto event : done) CUDA_CHECK(cudaEventDestroy(event));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
        trace.finish();
        truth("{\"fixture\":\"sampled_replay\",\"steps\":" + std::to_string(steps) +
              ",\"nodes_per_replay\":" + std::to_string(kNodes) + ",\"spin_ns\":" + std::to_string(kSpinNs) + "}");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "sampled_replay: %s\n", error.what());
        return 1;
    }
}
