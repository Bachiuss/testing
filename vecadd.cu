#include "../../common/cuda_check.h"
#include <cmath>


__global__ void vecAdd(const float* a, const float *b, float *c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // total threads in the whole grid
    int stride = blockDim.x * gridDim.x;

    for(int i = idx; i < n; i = i + stride){
        c[i] = a[i] + b[i];
    }
}


int main(){
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);

    // host memory
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    for(int i = 0; i < n; ++i){
        h_a[i] = i * 0.5f;
    }
    for(int i = 0; i < n; ++i){
        h_b[i] = i * 2.0f;
    }

    // device memory
    float *d_a;
    float *d_b;
    float *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // launch the kernel
    int numSMs;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
    int numBlocks = numSMs * 4;
    int blockSize = 256;
    vecAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU -> CPU
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // check the correctness (CPU reference)
    bool ok = true;
    for(int i = 0; i < n; ++i){
        float expected = h_a[i] + h_b[i];
        if(fabs(h_c[i] - expected) > 1e-5f){
            printf("FAILED at index: %d\n", i);
            ok = false;
            break;
        }
    }
    if(ok) printf("PASS\n");

    // print the first 5 element
    for(int i = 0; i < 5; ++i){
        printf("h_c[%d]: %f\n", i, h_c[i]);
    }
    // print the last 5 element
    for(int i = n - 5; i < n; ++i){
        printf("h_c[%d]: %f\n", i, h_c[i]);
    }

    printf("\n");
    printf("number of SMs: %d\n", numSMs);
    printf("number of blocks: %d\n", numBlocks);
    printf("number of total threads: %d\n", blockSize * numBlocks);

    // free the memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}