#include "common.hpp"
__global__ void mutation_add(int* value, int increment) { if (!threadIdx.x) value[0] += increment; }
__global__ void mutation_multiply(int* value) { if (!threadIdx.x) value[0] *= 2; }
int main() {
    try {
        Trace trace;
        int* device = nullptr;
        CUDA_CHECK(cudaMalloc(&device, sizeof(int)));
        cudaStream_t stream{};
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaGraph_t graph{}, clone{};
        CUDA_CHECK(cudaGraphCreate(&graph, 0));
        cudaGraphNode_t add{}, multiply{};
        int increment = 1;
        void* add_args[] = {&device, &increment};
        void* multiply_args[] = {&device};
        cudaKernelNodeParams add_params{};
        add_params.func = reinterpret_cast<void*>(mutation_add);
        add_params.gridDim = add_params.blockDim = dim3(1);
        add_params.kernelParams = add_args;
        cudaKernelNodeParams multiply_params{};
        multiply_params.func = reinterpret_cast<void*>(mutation_multiply);
        multiply_params.gridDim = multiply_params.blockDim = dim3(1);
        multiply_params.kernelParams = multiply_args;
        CUDA_CHECK(cudaGraphAddKernelNode(&add, graph, nullptr, 0, &add_params));
        CUDA_CHECK(cudaGraphAddKernelNode(&multiply, graph, &add, 1, &multiply_params));
        cudaGraphExec_t executable{}, clone_executable{};
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
        auto replay = [&](uint64_t step, cudaGraphExec_t exec, int expected) {
            int input = 1;
            CUDA_CHECK(cudaMemcpy(device, &input, sizeof(input), cudaMemcpyHostToDevice));
            trace.begin(step, "[{\"req_id\":\"mutation\",\"tokens\":1,\"phase\":\"decode\"}]");
            CUDA_CHECK(cudaGraphLaunch(exec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            trace.end();
            int result = 0;
            CUDA_CHECK(cudaMemcpy(&result, device, sizeof(result), cudaMemcpyDeviceToHost));
            if (result != expected) throw std::runtime_error("graph mutation numerical oracle failed");
        };
        replay(1, executable, 4);
        increment = 3;
        CUDA_CHECK(cudaGraphExecKernelNodeSetParams(executable, add, &add_params));
        replay(2, executable, 8);
        CUDA_CHECK(cudaGraphClone(&clone, graph));
        cudaGraphNode_t clone_add{}, clone_multiply{};
        CUDA_CHECK(cudaGraphNodeFindInClone(&clone_add, add, clone));
        CUDA_CHECK(cudaGraphNodeFindInClone(&clone_multiply, multiply, clone));
        CUDA_CHECK(cudaGraphRemoveDependencies(clone, &clone_add, &clone_multiply, nullptr, 1));
        CUDA_CHECK(cudaGraphAddDependencies(clone, &clone_multiply, &clone_add, nullptr, 1));
        CUDA_CHECK(cudaGraphInstantiate(&clone_executable, clone, 0));
        replay(3, clone_executable, 3);
        CUDA_CHECK(cudaGraphExecDestroy(clone_executable));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(clone));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaFree(device));
        CUDA_CHECK(cudaStreamDestroy(stream));
        trace.finish();
        truth(R"({"fixture":"graph_mutation","steps":[{"step_id":1,"edge":["mutation_add","mutation_multiply"],"expected":4},{"step_id":2,"edge":["mutation_add","mutation_multiply"],"exec_kernel_params_changed":true,"expected":8},{"step_id":3,"edge":["mutation_multiply","mutation_add"],"graph_cloned_and_dependency_reversed":true,"expected":3}]})");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "graph_mutation: %s\n", error.what());
        return 1;
    }
}
