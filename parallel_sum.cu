#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void parallel_sum_k(float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // 1. Load current input
    float x = input[idx];
    __syncthreads();

    int max_depth = (int)__log2f(N);
    int max_scan = 1 << max_depth; // 2^max_depth
    for (int depth=0; depth < max_depth; depth++) {
        int stage_mod = 1 << depth; // 2^depth
        int prev_idx = idx - stage_mod;
        // pick this idx
        bool mask = prev_idx >= 0 && (idx + 1) % stage_mod == 0 && idx < max_scan;
        if (mask) x += input[prev_idx];
        __syncthreads();
        if (mask) input[idx] = x;
        __syncthreads();
    }

    // Process remainder

    if (idx == 0) {
        float final_scan_val = input[max_scan-1];
        for (int i=max_scan; i<N; i++) {
            final_scan_val += input[i];
        }
        *output = final_scan_val;
    }
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {
    float* input_D;
    float* output_D;

    cudaMalloc((void**) &input_D, N * sizeof(float));
    cudaMemcpy(input_D, input, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &output_D, sizeof(float));

    parallel_sum_k<<<1, N>>>(input_D, output_D, N);

    cudaMemcpy(output, output_D, sizeof(float), cudaMemcpyDeviceToHost);
}

int main() {
    const int N = 1000;
    float input[N];
    float output = 0.0f;

    srand(42);
    for (int i = 0; i < N; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2000.0f - 1000.0f;
    }

    float gt = 0.0;
    for (int i=0; i<N; i++) gt += input[i];

    solve(input, &output, N);

    float diff = fabs(gt - output);
    printf("Ground truth: %f | Output: %f | Diff %f\n", gt, output, diff);
    return 0;
}
